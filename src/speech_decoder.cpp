#include "apps/speech_decoder.hpp"

#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <numeric> // accumulate
#include <limits>
#include <cmath>
#include <algorithm>

#include <synapse-app-sdk/middleware/conversions.hpp>

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

    return true;
  }

  void SpeechDecoder::main()
  {
    // Storage for incoming frames ------------------------------------------------
    std::vector<synapse::BroadbandFrame> broadband_frames;

    while (node_running_)
    {
      // BLOCK until we have a bin's worth of data --------------------------------
      if (!wait_for_frames(broadband_frames, bin_size_ms_))
      {
        continue; // try again
      }
      const auto loop_start_ns = synapse::get_steady_clock_now();


      // -------------------------------------------------------------------------
      // Initialise feature-extraction pipeline on first pass --------------------
      if (!feature_pipeline_initialized_)
      {
        auto& first_frame = broadband_frames.at(0);
        spdlog::info("Broadband 0: {}: {}", first_frame.DebugString(), first_frame.frame_data_size());
        if (first_frame.frame_data_size() < 256) {

          spdlog::error("Not equal to 256");
          for (auto& frame : broadband_frames) {
            // Calculate how many elements to add
            int padding_count = 256 - frame.frame_data_size();
            
            // Add padding elements (assuming frame_data contains numeric values)
            for (int i = 0; i < padding_count; i++) {
              frame.add_frame_data(0); // Add zeros or appropriate default value
            }
          }

        }
        const size_t num_channels = 256; //broadband_frames.at(0).frame_data_size();
        const float sample_rate_hz = 30000; //broadband_frames.at(0).sample_rate_hz();

        // Configure calibration sample threshold
        calibration_required_samples_ = static_cast<size_t>(calibration_seconds_ * sample_rate_hz);

        // Allocate state vectors
        thresholds_uv_.assign(num_channels, 0.0f);
        baseline_accum_.assign(num_channels, 0.0f);

        // Create band-pass filters
        bandpass_filters_.clear();
        bandpass_filters_.reserve(num_channels);
        for (size_t ch = 0; ch < num_channels; ++ch)
        {
          auto fptr = synapse::create_bandpass_filter<kFilterOrder>(sample_rate_hz, low_cut_hz_, high_cut_hz_);
          if (!fptr)
          {
            spdlog::error("[SpeechDecoder] Failed to create filter for ch {}", ch);
          }
          bandpass_filters_.push_back(std::move(fptr));
        }

        // Initialise 1-ms window buffers
        prev_window_.clear();
        curr_window_.assign(num_channels, std::vector<float>(30, 0.0f));
        sample_idx_in_window_ = 0;
        sample_rate_hz_ = sample_rate_hz;  // store for later use

        feature_pipeline_initialized_ = true;
        spdlog::info("[SpeechDecoder] Feature extractor initialised (channels={} sample_rate={} Hz)", num_channels, sample_rate_hz);
      }

      // --------- Feature computation over this bin ----------------------------
              auto& first_frame = broadband_frames.at(0);
        if (first_frame.frame_data_size() < 256) {

          for (auto& frame : broadband_frames) {
            // Calculate how many elements to add
            int padding_count = 256 - frame.frame_data_size();
            
            // Add padding elements (assuming frame_data contains numeric values)
            for (int i = 0; i < padding_count; i++) {
              frame.add_frame_data(0); // Add zeros or appropriate default value
            }
          }

        }
      const size_t num_channels = broadband_frames.at(0).frame_data_size();
      const size_t frames_in_bin = broadband_frames.size();

      // CAR per-array: we'll maintain mean per frame, apply per sample.
      const size_t channels_per_array = kChannelsPerArray;
      auto get_array_id = [channels_per_array](size_t ch) { return ch / channels_per_array; };

      // Per-channel accumulators over this 20-ms bin
      std::vector<float> sumsq(num_channels, 0.0f);              // accumulate mean(square) per 1-ms window
      std::vector<int32_t> thresh_cross_counts(num_channels, 0); // count threshold crossings across 1-ms windows

      size_t windows_processed = 0; // how many 1-ms windows we have processed in this 20-ms bin
  
      for (size_t idx = 0; idx < frames_in_bin; ++idx)
      {
        const auto &frame = broadband_frames[idx];
        const auto &data = frame.frame_data();

        // ------------------------------------------------------------------
        // Compute array-wise mean for CAR
        // float array_means[kNumArrays] = {0};
        // size_t counts[kNumArrays] = {0};
        // for (size_t ch = 0; ch < num_channels; ++ch)
        // {
        //   size_t aid = get_array_id(ch);
        //   array_means[aid] += data[ch];
        //   counts[aid]++;
        // }
        // for (size_t aid = 0; aid < kNumArrays; ++aid) {
        //   const auto value_aid = static_cast<float>(counts[aid]);
        //   if (value_aid == 0.0 || std::isnan(value_aid)) {
        //     array_means[aid] /= 0.01;
        //   } else {
        //     array_means[aid] /= static_cast<float>(counts[aid]);
        //   }

        // }

        // ------------------------------------------------------------------
        // For each channel: CAR (array-wise) and store into current 1-ms window buffer
        for (size_t ch = 0; ch < num_channels; ++ch)
        {
          //const float centred = data[ch] - array_means[get_array_id(ch)];
          const float centred = data[ch];
          // Store into current window buffer
          curr_window_[ch][sample_idx_in_window_] = centred;
        }

        // Advance sample index; when we have 30 samples, process window
        sample_idx_in_window_++;
        if (sample_idx_in_window_ >= 30)
        {
          process_one_ms_window(sumsq, thresh_cross_counts);
          windows_processed++;
        }
      }


      // Make sure we processed exactly 20 windows
      if (windows_processed == 0)
      {
        spdlog::warn("[SpeechDecoder] No 1-ms windows processed in bin – check frame rate/buffer sizes");
        continue;
      }

      // Finish calibration if ready
      if (!threshold_initialized_)
      {
        baseline_sample_counter_ += frames_in_bin;
        if (baseline_sample_counter_ >= calibration_required_samples_)
        {
          const float inv_samples = 1.0f / static_cast<float>(baseline_sample_counter_);
          for (size_t ch = 0; ch < thresholds_uv_.size(); ++ch)
          {
            float rms = std::sqrt(baseline_accum_[ch] * inv_samples);
            thresholds_uv_[ch] = -4.5f * rms; // negative threshold (match Python)
          }
          threshold_initialized_ = true;
          spdlog::info("[SpeechDecoder] Thresholds initialised after {:.2f} s ({} samples)", calibration_seconds_, baseline_sample_counter_);
        }
      }

      // Compute spike power per channel (mean of squares)
      std::vector<float> spike_power(num_channels, 0.0f);
      for (size_t ch = 0; ch < num_channels; ++ch)
      {
        spike_power[ch] = sumsq[ch] / static_cast<float>(windows_processed);
        // Clip to avoid extreme values (match Python 12 500 default)
        const float spike_power_clip = 12500.0f;
        if (spike_power[ch] > spike_power_clip)
          spike_power[ch] = spike_power_clip;
      }

      // Build feature vector for this 20-ms bin  [thresholdCounts | spikePower]
      std::vector<float> feature_vec;
      feature_vec.reserve(num_channels * 2);
      for (size_t ch = 0; ch < num_channels; ++ch)
        feature_vec.push_back(static_cast<float>(thresh_cross_counts[ch]));
      for (size_t ch = 0; ch < num_channels; ++ch)
        feature_vec.push_back(spike_power[ch]);

      // ---------------- Normalisation ---------------------------------------
      feature_window_.push_back(feature_vec);
      if (feature_window_.size() > kNormWindowBins)
        feature_window_.pop_front();

      std::vector<float> mean(feature_vec.size(), 0.0f);
      std::vector<float> stddev(feature_vec.size(), 0.0f);
      for (const auto &win_vec : feature_window_)
      {
        for (size_t i = 0; i < win_vec.size(); ++i)
          mean[i] += win_vec[i];
      }
      const float inv_cnt = 1.0f / static_cast<float>(feature_window_.size());
      for (size_t i = 0; i < mean.size(); ++i)
        mean[i] *= inv_cnt;

      for (const auto &win_vec : feature_window_)
      {
        for (size_t i = 0; i < win_vec.size(); ++i)
        {
          float diff = win_vec[i] - mean[i];
          stddev[i] += diff * diff;
        }
      }
      for (size_t i = 0; i < stddev.size(); ++i)
      {
        stddev[i] = std::sqrt(stddev[i] * inv_cnt);
        if (stddev[i] < 1e-4f)
          stddev[i] = 1e-4f;
      }

      std::vector<float> zscored_vec(feature_vec.size());
      for (size_t i = 0; i < feature_vec.size(); ++i)
      {
        float val = (feature_vec[i] - mean[i]) / stddev[i];
        if (val > zscore_clip_)
          val = zscore_clip_;
        else if (val < -zscore_clip_)
          val = -zscore_clip_;
        zscored_vec[i] = val;
      }

      // ---------------- Causal Gaussian smoothing ---------------------------
      smoothing_buffer_.push_back(zscored_vec);
      if (smoothing_buffer_.size() > kSmoothingKernelLen)
        smoothing_buffer_.pop_front();

      std::vector<float> smoothed_vec(zscored_vec.size(), 0.0f);
      float weight_sum = 0.0f;
      // Iterate over buffer from newest (idx 0) to oldest, matching kernel
      size_t w_idx = 0;
      for (auto it = smoothing_buffer_.rbegin(); it != smoothing_buffer_.rend() && w_idx < smoothing_kernel_.size(); ++it, ++w_idx)
      {
        const auto &vec = *it;
        float w = smoothing_kernel_[w_idx];
        weight_sum += w;
        for (size_t i = 0; i < vec.size(); ++i)
          smoothed_vec[i] += w * vec[i];
      }
      if (weight_sum > 0.0f)
      {
        for (auto &v : smoothed_vec) {
          v /= weight_sum;
        }
      }

      // ---------------- Rolling Z-score Normalisation ------------------------
      // --------------------------------------------------------------------

      // Add this smoothed 20-ms bin to patch window
      patch_window_.push_back(smoothed_vec);
      if (patch_window_.size() > kPatchSizeBins)
        patch_window_.pop_front();

      if (patch_window_.size() == kPatchSizeBins)
      {
        stride_bins_counter_++;
        if (stride_bins_counter_ >= kPatchStrideBins)
        {
          stride_bins_counter_ = 0;

          // Flatten 14×512 to contiguous vector
          std::vector<float> patch;
          patch.reserve(kPatchSizeBins * smoothed_vec.size());
          for (const auto &v : patch_window_)
            patch.insert(patch.end(), v.begin(), v.end());

          // Publish tensor -------------------------------------------------
          synapse::Tensor features_tensor;
          const std::array<int32_t, 2> feature_shape = {static_cast<int32_t>(kPatchSizeBins), static_cast<int32_t>(smoothed_vec.size())};
          features_tensor.mutable_shape()->Add(feature_shape.begin(), feature_shape.end());
          features_tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
          features_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

          const char *ptr = reinterpret_cast<const char *>(patch.data());
          size_t bytes = patch.size() * sizeof(float);
          features_tensor.set_data(std::string(ptr, bytes));
          features_tensor.set_timestamp_ns(loop_start_ns.count());

          if (!publish_tap("features_out", features_tensor))
          {
            spdlog::warn("[SpeechDecoder] Failed to publish features_out");
          }
        }
      }

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
        phoneme_tensor.set_timestamp_ns(loop_start_ns.count());

        // if (!publish_tap("phonemes_out", phoneme_tensor))
        // {
        //   spdlog::warn("[SpeechDecoder] Failed to publish phonemes_out");
        // }
      }

      // Housekeeping -----------------------------------------------------------
      // Update threshold refresh counter
      bins_since_threshold_refresh_++;
      if (bins_since_threshold_refresh_ >= kThresholdRefreshBins)
      {
        spdlog::info("[SpeechDecoder] Re-calibrating thresholds after refresh interval");
        threshold_initialized_ = false;
        std::fill(baseline_accum_.begin(), baseline_accum_.end(), 0.0f);
        baseline_sample_counter_ = 0;
        bins_since_threshold_refresh_ = 0;
      }

      const auto loop_dt_ns = synapse::get_steady_clock_now() - loop_start_ns;
      spdlog::debug("[SpeechDecoder] Loop latency: {} ms", loop_dt_ns.count() * 1e-6);
    }
  }

  // ---------------------------------------------------------------------------
  // Helper functions
  // ---------------------------------------------------------------------------
  bool SpeechDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame> &frames, float bin_size_ms)
  {
    int target_frames = 600;

    if (bin_size_ms <= 0)
    {
      spdlog::warn("[SpeechDecoder] Invalid bin size: {} ms", bin_size_ms);
      return false;
    }

    const uint64_t target_bin_ns = static_cast<uint64_t>(bin_size_ms * 1e6);

    frames.clear();
    uint64_t first_timestamp_ns = 0;

    while (node_running_)
    {
      auto messages = data_reader_->receive_multipart();
      if (messages.empty())
      {
        std::this_thread::sleep_for(std::chrono::microseconds(1));
        continue;
      }

      frames.reserve(frames.size() + messages.size());

      for (auto &message : messages)
      {
        auto maybe_frame = synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(message));
        if (!maybe_frame.has_value())
        {
          spdlog::warn("[SpeechDecoder] Failed to parse broadband frame");
          if (frames.empty())
            return false;
          return true;
        }

        const auto &frame = maybe_frame.value();

        // check for dropouts
        const auto dropped = detect_dropped_frames(last_sequence_number_, frame.sequence_number());
        if (dropped != 0)
          spdlog::warn("[SpeechDecoder] Dropped {} broadband frames", dropped);
        last_sequence_number_ = frame.sequence_number();

        if (frames.empty())
          first_timestamp_ns = frame.timestamp_ns();

        frames.push_back(frame);
      }

      //spdlog::info("Waiting from frames: frames.back().timestamp_ns(): {}, first_timestamp_ns: {} target_bin: {}", frames.back().timestamp_ns(), first_timestamp_ns,target_bin_ns );
      if (!frames.empty() && (frames.back().timestamp_ns() - first_timestamp_ns >= target_bin_ns))
      {
        return true;
      }

      if (frames.size() >= target_frames) {
        return true;
      }
    }
    return false;
  }

  int SpeechDecoder::detect_dropped_frames(uint64_t last_seq, uint64_t current_seq)
  {
    const auto expected = last_seq + 1;
    return static_cast<int>(current_seq - expected);
  }

  // ---------------------------------------------------------------------------
  // Helper: process a filled 1-ms (30-sample) window with zero-phase filtering
  // ---------------------------------------------------------------------------
  void SpeechDecoder::process_one_ms_window(std::vector<float>& sumsq, std::vector<int32_t>& thresh_cross_counts)
  {
    const size_t num_channels = curr_window_.size();
    if (num_channels == 0) return;

    const size_t n_samples = 30;               // 1 ms @ 30 kHz
    const size_t right_pad = 36;               // 1.2 ms mean padding

    for (size_t ch = 0; ch < num_channels; ++ch)
    {
      // Build padded buffer: prev 30 + curr 30 + mean pad 36
      std::vector<float> padded;
      padded.reserve(n_samples * 2 + right_pad);

      // Append previous window (if not yet initialised, use zeros)
      if (!prev_window_.empty())
        padded.insert(padded.end(), prev_window_[ch].begin(), prev_window_[ch].end());
      else
        padded.insert(padded.end(), n_samples, 0.0f);

      // Append current window samples
      padded.insert(padded.end(), curr_window_[ch].begin(), curr_window_[ch].end());

      // Mean pad
      float mean_val = std::accumulate(curr_window_[ch].begin(), curr_window_[ch].end(), 0.0f) / static_cast<float>(n_samples);
      padded.insert(padded.end(), right_pad, mean_val);

      // Forward filter
      auto fwd_filter = synapse::create_bandpass_filter<kFilterOrder>(sample_rate_hz_, low_cut_hz_, high_cut_hz_);
      std::vector<float> fwd_out(padded.size());
      for (size_t i = 0; i < padded.size(); ++i)
        fwd_out[i] = fwd_filter->filter(padded[i]);

      // Reverse filter
      auto rev_filter = synapse::create_bandpass_filter<kFilterOrder>(sample_rate_hz_, low_cut_hz_, high_cut_hz_);
      std::vector<float> rev_out(fwd_out.size());
      for (size_t i = 0; i < fwd_out.size(); ++i)
        rev_out[i] = rev_filter->filter(fwd_out[fwd_out.size() - 1 - i]);

      // Reverse back to forward order
      std::reverse(rev_out.begin(), rev_out.end());

      // Extract the 30 samples corresponding to curr window (indices 30..59)
      float min_val = std::numeric_limits<float>::max();
      float sum_sq_channel = 0.0f;
      const size_t start_idx = n_samples; // 30
      for (size_t i = 0; i < n_samples; ++i)
      {
        float sample = rev_out[start_idx + i];
        sum_sq_channel += sample * sample;
        if (sample < min_val) min_val = sample;
      }

      // Update accumulators
      sumsq[ch] += sum_sq_channel / static_cast<float>(n_samples); // mean square later scaled outside

      if (threshold_initialized_ && min_val <= thresholds_uv_[ch])
        thresh_cross_counts[ch]++;

      if (!threshold_initialized_)
      {
        baseline_accum_[ch] += sum_sq_channel;
      }
    }

    // After processing, move curr to prev and reset index
    if (prev_window_.empty())
    {
      prev_window_ = curr_window_;
    }
    else
    {
      prev_window_.swap(curr_window_);
    }
    // Clear curr_window_ for next fill
    for (auto &vec : curr_window_)
      std::fill(vec.begin(), vec.end(), 0.0f);

    sample_idx_in_window_ = 0;
    baseline_sample_counter_ += n_samples; // keep calibration sample count
  }

} // namespace app

int main(const int, const char **)
{
  return synapse::Entrypoint<app::SpeechDecoder>();
} 