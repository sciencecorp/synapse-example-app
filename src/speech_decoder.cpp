#include "apps/speech_decoder.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <span>
#include <thread>
#include <numeric> // accumulate
#include <limits>
#include <vector>

#include <spdlog/spdlog.h>
#include <synapse-app-sdk/middleware/conversions.hpp>

#include "api/datatype.pb.h"

namespace app
{
template <typename T>
T clamp(T value, T min, T max)
{
  return (value < min) ? min : (value > max) ? max : value;
}

SpeechDecoder::SpeechDecoder() : publish_rate_limiter_(kSpeechPublishRateSec)
{
  // Pre-compute causal Gaussian smoothing kernel (σ = 2 bins, length 15)
  smoothing_kernel_.resize(kSmoothingKernelLen);
  const float sigma_bins = 2.0f;
  float norm = 0.0f;
  for (size_t i = 0; i < kSmoothingKernelLen; ++i)
  {
    const float x = static_cast<float>(i);
    const float w = std::exp(-0.5f * (x * x) / (sigma_bins * sigma_bins));
    smoothing_kernel_[i] = w;
    norm += w;
  }
  // Normalise kernel to sum to 1
  for (auto &w : smoothing_kernel_)
    w /= norm;

  stride_bins_counter_ = 0; // initialize RNN patch stride counter
}

bool SpeechDecoder::setup()
{
  const uint32_t broadband_node_id = 1;
  if (!setup_reader(broadband_node_id))
  {
    spdlog::error("[SpeechDecoder] Failed to set up reader for broadband node {}", broadband_node_id);
    return false;
  }

  // Create taps
  if (!create_tap<synapse::Tensor>("features_out"))
  {
    spdlog::error("[SpeechDecoder] Failed to create 'features_out' tap");
    return false;
  }

  if (!create_tap<synapse::Tensor>("phonemes_out"))
  {
    spdlog::error("[SpeechDecoder] Failed to create 'phonemes_out' tap");
    return false;
  }

  return true;
}

std::vector<std::vector<int32_t>> frames_to_timeseries(const std::vector<synapse::BroadbandFrame> &frames) {
  if (frames.empty()) {
    spdlog::error("[SpeechDecoder] No frames to convert to timeseries");
    return {};
  }
  const size_t num_channels = frames[0].frame_data().size();
  const size_t num_frames = frames.size();

  std::vector<std::vector<int32_t>> timeseries(num_channels);
  for (size_t i = 0; i < num_channels; ++i) {
    timeseries[i].reserve(num_frames);
  }
  for (const auto &frame : frames) {
    const auto &data = frame.frame_data();
    for (size_t ch = 0; ch < num_channels; ++ch) {
      timeseries[ch].push_back(data[ch]);
    }
  }
  return timeseries;
}

void SpeechDecoder::main()
{
  size_t loop_count = 0;
  uint64_t tot_loop_time_us = 0;
  uint64_t tot_read_time_us = 0;
  uint64_t tot_feature_time_us = 0;
  // Storage for incoming frames ------------------------------------------------
  std::vector<synapse::BroadbandFrame> broadband_frames;

  // CAR per-array: we'll maintain mean per frame, apply per sample.
  const size_t channels_per_array = kChannelsPerArray;
  auto get_array_id = [channels_per_array](size_t ch) { return ch / channels_per_array; };

  const size_t num_channels = kChannelsPerArray * kNumArrays; // number of channels in the incoming frames
  const size_t samples_per_window = static_cast<size_t>(std::ceil(1 / (1000.0f / sample_rate_hz_)));
  const size_t windows_per_bin = static_cast<size_t>(bin_size_ms_);
  const size_t frames_to_read = samples_per_window;

  // Number of samples to accumulate in order to run parameter calibration
  const size_t calibration_required_samples = static_cast<size_t>(calibration_seconds_ * sample_rate_hz_);
  const size_t calibration_required_bins = calibration_required_samples / (samples_per_window * windows_per_bin);

  // Number of 1ms windows per bin.

  // Allocate state vectors
  thresholds_uv_.assign(num_channels, 0.0f);

  broadband_calibration_buffer.resize(num_channels);
  binned_feat_calibration_buffer.reserve(calibration_required_bins);
  for (size_t ch = 0; ch < num_channels; ++ch) {
    broadband_calibration_buffer[ch].reserve(calibration_required_samples);
  }

  // Create band-pass filters
  bandpass_filters_.clear();
  bandpass_filters_.reserve(num_channels);
  for (size_t ch = 0; ch < num_channels; ++ch)
  {
    auto fptr = synapse::create_bandpass_filter<kFilterOrder>(sample_rate_hz_, low_cut_hz_, high_cut_hz_);
    if (!fptr)
    {
      spdlog::error("[SpeechDecoder] Failed to create filter for ch {}", ch);
    }
    bandpass_filters_.push_back(std::move(fptr));
  }

  // Initialise 1-ms window buffers
  prev_window_.resize(num_channels);

  std::vector<float> sp_power_feature_bin(num_channels, 0); // 20-ms bin of spike power features
  std::vector<int16_t> threshold_crossings_feature_bin(num_channels, 0); // 20-ms bin of threshold crossings
  size_t windows_processed = 0; // how many 1-ms windows we have processed in a 20-ms bin
  spdlog::info("[SpeechDecoder] Reading {} frames per window ({} ms)", frames_to_read, samples_per_window);
  spdlog::info("[SpeechDecoder] Feature extractor initialised (channels={} sample_rate={} Hz)", num_channels, sample_rate_hz_);

  spdlog::set_level(spdlog::level::debug);


  while (node_running_) {
    const auto read_start = synapse::get_steady_clock_now();
    // BLOCK until we have a bin's worth of data --------------------------------
    if (!wait_for_frames(broadband_frames, frames_to_read))
    {
      continue; // try again
    }
    if (broadband_frames.empty())
    {
      spdlog::warn("[SpeechDecoder] No frames read");
      continue;
    }
    const uint64_t window_start_time = broadband_frames[0].timestamp_ns();

    const auto read_end = synapse::get_steady_clock_now();

    // Convert the incoming frames from [samples, channels] to [channels, samples]
    std::vector<std::vector<int32_t>> timeseries_window = frames_to_timeseries(broadband_frames);

    // ----------------------- I. Perform data pre-processing -----------------------

    // Filter the 1-ms windows of samples.
    filter_window(timeseries_window);

    // Apply CAR to the filtered 1-ms windows
    std::vector<std::vector<float>> car_window = apply_car(timeseries_window, kNumArrays, kChannelsPerArray);

    // ----------------------- II. Run feature extraction --------------------------

    // Features for the current 1ms window for all channels:
    // TODO: figure out when we want to initialize the spike thresholds. 
    std::vector<float> spike_powers = calc_spike_bandpower(car_window, 12500.0f);
    std::vector<int16_t> threshold_crossings = calc_threshold_crossings(car_window, thresholds_uv_);

    if (!calibration_finished_) {
      for (size_t ch_idx = 0; ch_idx < num_channels; ++ch_idx) {
        std::vector<float>& ch_calib_buffer = broadband_calibration_buffer[ch_idx];
        std::vector<float>& ch_new_window = car_window[ch_idx];
        ch_calib_buffer.insert(ch_calib_buffer.end(), ch_new_window.begin(), ch_new_window.end());
      }
    }

    // Update the 20-ms bin of features
    for (size_t ch = 0; ch < num_channels; ++ch)
    {
      sp_power_feature_bin[ch] += spike_powers[ch];
      threshold_crossings_feature_bin[ch] += threshold_crossings[ch];
    }
    windows_processed++;

    const auto feature_end = synapse::get_steady_clock_now();
    if (windows_processed >= windows_per_bin) {
      // Average the spike power features for the whole bin
      for (size_t ch = 0; ch < num_channels; ++ch) {
        sp_power_feature_bin[ch] /= static_cast<float>(windows_per_bin);
      }

      // The unnormalized features for one 20ms bin are finished at this point.
      std::vector<float> combined_features;
      combined_features.reserve(num_channels * 2);
      combined_features.insert(combined_features.end(), threshold_crossings_feature_bin.begin(), threshold_crossings_feature_bin.end());
      combined_features.insert(combined_features.end(), sp_power_feature_bin.begin(), sp_power_feature_bin.end());
      
      // Add data to calibration buffer
      if (binned_feat_calibration_buffer.size() < calibration_required_bins) {
        binned_feat_calibration_buffer.push_back(combined_features);

        // If we have enough data, we can calibrate the parameters.
        if (binned_feat_calibration_buffer.size() == calibration_required_bins) {
          calibrate_parameters(broadband_calibration_buffer, binned_feat_calibration_buffer);
          calibration_finished_ = true;
        }
        continue; // If calibration isnt finished, skip the rest of the loop.
      }

      // Now we apply the data pre-processing for blocks of features going into the RNN decoder

      // 1. Wait for 80ms of new features (4 bins). Buffer the last 14 bins of features.
      patch_window_.push_back(combined_features);
      if (patch_window_.size() > kPatchSizeBins) {
        patch_window_.pop_front();

        stride_bins_counter_++;
        if (stride_bins_counter_ >= kPatchStrideBins) {
          // 2. Z-score normalise the features using the mean and stddev of the data.
          std::vector<std::vector<float>> normalized_feat_frames = normalize_rnn_input_features(zscore_clip_); 
          const auto zscore_end = synapse::get_steady_clock_now();

          stride_bins_counter_ = 0; // reset stride counter
          // Output to tap
          if (!send_features_to_tap(normalized_feat_frames, "features_out")) {
            spdlog::warn("[SpeechDecoder] Failed to send features to tap");
          }
        }
      }
      windows_processed = 0; // reset window counter
      sp_power_feature_bin.assign(num_channels, 0); // reset the spike power feature bin
      threshold_crossings_feature_bin.assign(num_channels, 0); // reset the threshold crossings feature bin
    }

    // ---------------- Causal Gaussian smoothing ---------------------------
    // smoothing_buffer_.push_back(zscored_vec);
    // if (smoothing_buffer_.size() > kSmoothingKernelLen)
    //   smoothing_buffer_.pop_front();

    // std::vector<float> smoothed_vec(zscored_vec.size(), 0.0f);
    // float weight_sum = 0.0f;
    // // Iterate over buffer from newest (idx 0) to oldest, matching kernel
    // size_t w_idx = 0;
    // for (auto it = smoothing_buffer_.rbegin(); it != smoothing_buffer_.rend() && w_idx < smoothing_kernel_.size(); ++it, ++w_idx)
    // {
    //   const auto &vec = *it;
    //   float w = smoothing_kernel_[w_idx];
    //   weight_sum += w;
    //   for (size_t i = 0; i < vec.size(); ++i)
    //     smoothed_vec[i] += w * vec[i];
    // }
    // if (weight_sum > 0.0f)
    // {
    //   for (auto &v : smoothed_vec)
    //     v /= weight_sum;
    // }


    // -------------------------------------------------------------------------
    // TODO: Run ONNX inference. For now publish dummy phoneme index 0 ----------
    // -------------------------------------------------------------------------
    if (publish_rate_limiter_.reset_if_elapsed())
    {
      synapse::Tensor phoneme_tensor;
      const std::array<int32_t, 1> phoneme_shape = {1};
      phoneme_tensor.mutable_shape()->Add(phoneme_shape.begin(), phoneme_shape.end());
      phoneme_tensor.set_dtype(synapse::Tensor_DType_DT_INT32);
      phoneme_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

      const std::vector<int32_t> phoneme_data = {0}; // placeholder index
      const char *phon_ptr = reinterpret_cast<const char *>(phoneme_data.data());
      size_t phon_size = phoneme_data.size() * sizeof(int32_t);
      phoneme_tensor.set_data(std::string(phon_ptr, phon_size));
      phoneme_tensor.set_timestamp_ns(window_start_time);

      if (!publish_tap("phonemes_out", phoneme_tensor))
      {
        spdlog::warn("[SpeechDecoder] Failed to publish phonemes_out");
      }
    }

    uint64_t loop_time_us = (synapse::get_steady_clock_now() - read_start).count() / 1000; 
    tot_loop_time_us += loop_time_us;
    uint64_t read_time_us = (read_end - read_start).count() / 1000;
    tot_read_time_us += read_time_us;
    uint64_t feature_time_us = (feature_end - read_end).count() / 1000;
    tot_feature_time_us += feature_time_us;
    loop_count++;
    if (loop_count % 50 == 0) {
      double avg_loop_time_ms = static_cast<double>(tot_loop_time_us * 1e-3) / loop_count;
      double avg_read_time_us = static_cast<double>(tot_read_time_us) / loop_count;
      double avg_feature_time_us = static_cast<double>(tot_feature_time_us) / loop_count;
      spdlog::info("[SpeechDecoder] Loop time: {:.3} ms (avg: {:.3} ms) ----------", loop_time_us * 1e-3, avg_loop_time_ms);
      spdlog::info("[SpeechDecoder] Feature extraction: {} us (avg: {:.3} ms)", feature_time_us, avg_feature_time_us * 1e-3);
      spdlog::info("[SpeechDecoder] Read time avg: {:.3} ms ({} us inst)", avg_read_time_us * 1e-3, read_time_us);
    }
  }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------
bool SpeechDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame> &frames, size_t frames_to_read)
{
  if (frames_to_read <= 0)
  {
    spdlog::warn("[SpeechDecoder] Invalid bin size: {} frames", frames_to_read);
    return false;
  }


  frames.clear();
  frames.reserve(frames_to_read);
  size_t frames_read = 0;

  while (node_running_ && (frames_to_read > frames_read))
  {
    auto messages = data_reader_->receive_multipart();
    if (messages.empty()) 
    {
      continue;
    }

    for (auto &message : messages)
    {
      auto maybe_frame = synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(message));
      if (!maybe_frame.has_value())
      {
        spdlog::warn("[SpeechDecoder] Failed to parse broadband frame");
        continue;
      }

      const auto &frame = maybe_frame.value();

      // check for dropouts
      const auto dropped = detect_dropped_frames(last_sequence_number_, frame.sequence_number());
      if (dropped != 0)
        spdlog::warn("[SpeechDecoder] Dropped {} broadband frames", dropped);
      last_sequence_number_ = frame.sequence_number();
      frames.push_back(frame);
      frames_read++;
    }
  } 
  return true;
}

int SpeechDecoder::detect_dropped_frames(uint64_t last_seq, uint64_t current_seq)
{
  const auto expected = last_seq + 1;
  return static_cast<int>(current_seq - expected);
}

bool SpeechDecoder::send_features_to_tap(const std::vector<std::vector<float>>& feat_vec, const std::string &tap_name) {
  static synapse::Tensor features_tensor;

  features_tensor.Clear();
  features_tensor.add_shape(feat_vec.size());
  features_tensor.add_shape(feat_vec[0].size());
  features_tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
  features_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

  std::string* data = features_tensor.mutable_data();
  for (const auto& vec : feat_vec) {
    const char *ptr = reinterpret_cast<const char *>(vec.data());
    size_t bytes = vec.size() * sizeof(float);
    data->append(ptr, bytes);
  }
  // Placeholder
  features_tensor.set_timestamp_ns(synapse::get_steady_clock_now().count());

  // Publish the tensor to the tap
  if (!publish_tap(tap_name, features_tensor)) {
    spdlog::warn("[SpeechDecoder] Failed to publish features to tap {}", tap_name);
    return false;
  }
  // Reset the tensor for the next use
  return true;

}

// The algorithm used in UCD's python to pad data before filtering:
//    First concatenate: (1) dat (which may contain a few ms of hostoric data), 
//    (2) a small portion of flipped data to be filtered to avoid discontinuities at the edge, whilst preserving freq info
//    (3) mean padding of 1ms at the end 
//    (4) If no historic data is provided, add mean padding of 1ms at the beginning
// Instead of passing in historic data, we use the prev_window_ buffer to store the last 30 samples
std::vector<int32_t> SpeechDecoder::pad_data(std::vector<int32_t> &data, size_t ch_idx)
{
  const size_t n_samples_1ms = sample_rate_hz_ / 1000; // Number of samples 
  const size_t flipped_len = static_cast<size_t>(n_samples_1ms / 5);
  const int32_t mean_val = static_cast<float>(std::accumulate(data.begin(), data.end(), 0)) / data.size();
  std::vector<int32_t> padded;
  padded.reserve(data.size() + n_samples_1ms * 2 + flipped_len);
  // Append previous window (if not yet initialised, use mean value)
  if (!prev_window_[ch_idx].empty()) {
    padded.insert(padded.end(), prev_window_[ch_idx].begin(), prev_window_[ch_idx].end());
  }
  else {
    padded.insert(padded.end(), n_samples_1ms, mean_val);
  }
  padded.insert(padded.end(), data.begin(), data.end());

  // Append flipped data
  padded.insert(padded.end(), data.rbegin(), data.rbegin() + flipped_len);

  // Append mean padding
  padded.insert(padded.end(), n_samples_1ms, mean_val);

  // Save current window as previous window
  prev_window_[ch_idx] = data;
  return padded;
}

/**
 * Mirrors NeuralFeatureExtractor.filter_signal() from the python code.
 * Applies a zero-phase butterworth bandpass filter on the window of data. 
 * The data is forward-padded by the previous window of data, 
 * and end-padded by a reversed section of data, followed by 1ms of mean padding
 * Expects to operate on a 1 ms window of samples. 
 * Parameters:
 *    - ms_window: Input 1ms neural data window. Expects a shape of [channels, samples]. 
 *                 The processed samples are written back into this vector.
*/
void SpeechDecoder::filter_window(std::vector<std::vector<int32_t>>& ms_window) {
  if (ms_window.empty()) {
    spdlog::error("[SpeechDecoder] No data to filter");
    return;
  }
  for (size_t ch = 0; ch < ms_window.size(); ++ch) {
    std::vector<int32_t> padded = pad_data(ms_window[ch], ch);
    const auto& filter = bandpass_filters_[ch];
    for (size_t i = 0; i < padded.size(); ++i) {
      // Apply bandpass filter
      padded[i] = filter->filter(padded[i]);
    }
    filter->reset(); // reset filter state for the reverse pass
    // Reverse filter
    for (size_t i = 0; i < padded.size(); ++i) {
      padded[padded.size() - 1 - i] = filter->filter(padded[padded.size() - 1 - i]);
    }
    filter->reset(); // reset filter state for next window

    ms_window[ch].clear();
    ms_window[ch].insert(ms_window[ch].end(), padded.begin() + 30, padded.end() - 36); // Extract the 30 samples corresponding to curr window (indices 30..59)
  }
}

  
std::vector<std::vector<float>> SpeechDecoder::apply_car(const std::vector<std::vector<int32_t>> &ms_window, const size_t& n_arrays, const size_t& n_channels_per_array) {
  if (ms_window.empty()) {
    spdlog::error("[SpeechDecoder] No data to apply CAR");
    return {};
  }
  const size_t num_channels = ms_window.size();
  const size_t num_frames = ms_window[0].size();
  std::vector<std::vector<float>> car_window(num_channels, std::vector<float>(num_frames, 0.0f));  
  for (size_t array_i = 0; array_i < n_arrays; array_i++) {

    std::span<const std::vector<int32_t>> this_array_input(ms_window.begin() + (array_i * n_channels_per_array), n_channels_per_array );
    std::span<std::vector<float>> this_array_output(car_window.begin() + (array_i * n_channels_per_array), n_channels_per_array );

    // compute the mean reference signal for this array
    std::vector<float> array_mean_ref(num_frames, 0.0f);
    for (size_t fr_idx = 0; fr_idx < num_frames; fr_idx++) {
      for (size_t ch_idx = 0; ch_idx < n_channels_per_array; ch_idx++) {
        array_mean_ref[fr_idx] += this_array_input[ch_idx][fr_idx];
      }
      array_mean_ref[fr_idx] /= n_channels_per_array;
    }
    // subtract the mean reference signal from each channel
    for (size_t ch_idx = 0; ch_idx < n_channels_per_array; ch_idx++) {
      for (size_t fr_idx = 0; fr_idx < num_frames; fr_idx++) {
        this_array_output[ch_idx][fr_idx] = this_array_input[ch_idx][fr_idx] - array_mean_ref[fr_idx];
      }
    }
  }

  return car_window;
}

std::vector<float> SpeechDecoder::calc_spike_bandpower(const std::vector<std::vector<float>> &data, const float& clip_thresh) {
  const size_t num_channels = data.size();
  const size_t num_samples = data[0].size();

  std::vector<float> bandpower(num_channels, 0.0f);
  for (size_t ch = 0; ch < num_channels; ++ch) {
    float sum_sq = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
      const float& sample = data[ch][i];
      sum_sq += sample * sample;
    }
    bandpower[ch] = std::min(sum_sq / static_cast<float>(num_samples), clip_thresh);
  }

  return bandpower;
}

std::vector<int16_t> SpeechDecoder::calc_threshold_crossings(const std::vector<std::vector<float>> &data, const std::vector<float> &thresholds) {
  const size_t num_channels = data.size();
  const size_t num_samples = data[0].size();

  std::vector<int16_t> threshold_crossings(num_channels, 0);
  for (size_t ch = 0; ch < num_channels; ++ch) {
    const float& min_sample = *std::min_element(data[ch].begin(), data[ch].end());
    if (min_sample <= thresholds[ch]) {
      threshold_crossings[ch]++;
    }
  }
  return threshold_crossings;
}

// Mirrors NeuralFeatureExtractor.compute_thresholds() in the python code.
// Compute the spike thresholds for each channel based on the RMS of the data.
// Parameters:
//   - data: The input neural data window. shape of [num_channels][num_samples].
//   - thresh_mult: The multiplier for the RMS to set the threshold. Should normally be negative.
std::vector<float> SpeechDecoder::compute_thresholds(const std::vector<std::vector<float>> &data, const float& thresh_mult) {
  const size_t num_channels = data.size();
  const size_t num_samples = data[0].size();

  std::vector<float> thresholds(num_channels, 0.0f);
  // Compute the RMS for each channel and set the threshold based on the multiplier.
  for (size_t ch = 0; ch < num_channels; ++ch) {
    double sum_sq = 0.0f;
    for (size_t i = 0; i < num_samples; ++i) {
      const float& sample = data[ch][i];
      sum_sq += sample * sample;
    }
    float rms = std::sqrt(sum_sq / static_cast<float>(num_samples));
    thresholds[ch] = thresh_mult * rms; 
  }
  return thresholds;
}


bool SpeechDecoder::calibrate_parameters(const std::vector<std::vector<float>>& broadband_calibration_data, 
                              const std::vector<std::vector<float>>& binned_feat_calib_data) {
  // Compute per-channel spike threshold values.
  thresholds_uv_ = compute_thresholds(broadband_calibration_data, kThresholdMult);

  // Compute per-channel mean and stddev over the calibration data:
  const size_t num_channels = broadband_calibration_data.size();
  const size_t num_features = num_channels * 2; // 2 features per channel (spike power and threshold crossings)
  const size_t num_bins = binned_feat_calib_data.size();

  // Resize the means and stddevs vectors to match the number of features
  feature_means_.resize(num_features);
  feature_stddevs_.resize(num_features);

  for (size_t feat_idx = 0; feat_idx < num_features; ++feat_idx) {

    // Compute the mean.
    double sum = 0.0;
    for (size_t bin = 0; bin < num_bins; ++bin) {
      const float& sample = binned_feat_calib_data[bin][feat_idx];
      sum += sample;
    }
    feature_means_[feat_idx] = sum / static_cast<double>(num_bins);

    // Compute the stddev.
    double sum_sq_diffs = 0.0;
    for (size_t bin = 0; bin < num_bins; ++bin) {
      const float& sample = binned_feat_calib_data[bin][feat_idx];
      sum_sq_diffs += std::pow(sample - feature_means_[feat_idx], 2);
    }
    feature_stddevs_[feat_idx] = std::sqrt(sum_sq_diffs / static_cast<double>(num_bins));
  }

  return true;
}

/** 
 * Normalise the features using z-score normalisation. 
 * Currently the mean and stddev are not actually populated anywhere.
 * The supplemental info of the speech decoder paper states that mean and stddev were calculated based on 
 * "speech epochs of the previous 20 trials". Its not totally clear what one trial is. 
 * Im pretty sure its one attempt to reproduce a sentance.
 * So in practice this will probably be a value that is loaded from a config. 
*/
std::vector<std::vector<float>> SpeechDecoder::normalize_rnn_input_features(const double& zscore_clip) {
  std::vector<std::vector<float>> normalized_feat_frames;
  normalized_feat_frames.reserve(kPatchSizeBins);

  for (const auto& frame : patch_window_) {
    std::vector<float> normalized_frame(frame.size());
    for (size_t i = 0; i < frame.size(); ++i) {
      const double mean_centered = frame[i] - feature_means_[i];
      const double stddev = feature_stddevs_[i] + 1e-8; // Avoid division by zero
      double normalized_value = mean_centered / stddev; 
      normalized_value = std::clamp(normalized_value, -zscore_clip, zscore_clip); // Clip to zscore_clip
      normalized_frame[i] = static_cast<float>(normalized_value);
    }
    normalized_feat_frames.push_back(std::move(normalized_frame));
  }
  return normalized_feat_frames;
}


} // namespace app

int main(const int, const char **)
{
  return synapse::Entrypoint<app::SpeechDecoder>();
} 