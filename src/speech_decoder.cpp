#include "apps/speech_decoder.hpp"

#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <numeric> // accumulate
#include <limits>
#include <cmath>

#include <synapse-app-sdk/middleware/conversions.hpp>

namespace app
{
  template <typename T>
  T clamp(T value, T min, T max)
  {
    return (value < min) ? min : (value > max) ? max : value;
  }

  SpeechDecoder::SpeechDecoder() : publish_rate_limiter_(kSpeechPublishRateSec) {}

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
        const size_t num_channels = broadband_frames.at(0).frame_data_size();
        const float sample_rate_hz = broadband_frames.at(0).sample_rate_hz();

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

        feature_pipeline_initialized_ = true;
        spdlog::info("[SpeechDecoder] Feature extractor initialised (channels={} sample_rate={} Hz)", num_channels, sample_rate_hz);
      }

      // --------- Feature computation over this bin ----------------------------
      const size_t num_channels = broadband_frames.at(0).frame_data_size();
      const size_t frames_in_bin = broadband_frames.size();

      // CAR per-array: we'll maintain mean per frame, apply per sample.
      const size_t channels_per_array = kChannelsPerArray;
      auto get_array_id = [channels_per_array](size_t ch) { return ch / channels_per_array; };

      // Per-channel accumulators
      std::vector<float> sumsq(num_channels, 0.0f);
      std::vector<int32_t> thresh_cross_counts(num_channels, 0);

      // Track sub-windows of 1 ms (30 samples) to count spikes
      size_t sample_rate_hz = static_cast<size_t>(broadband_frames.at(0).sample_rate_hz());
      const size_t samples_per_ms = sample_rate_hz / 1000;
      size_t subwindow_counter = 0;

      // Storage for per-channel min within current 1-ms subwindow
      std::vector<float> subwindow_min(num_channels, std::numeric_limits<float>::max());

      for (size_t idx = 0; idx < frames_in_bin; ++idx)
      {
        const auto &frame = broadband_frames[idx];
        const auto &data = frame.frame_data();

        // ------------------------------------------------------------------
        // Compute array-wise mean for CAR
        float array_means[kNumArrays] = {0};
        size_t counts[kNumArrays] = {0};
        for (size_t ch = 0; ch < num_channels; ++ch)
        {
          size_t aid = get_array_id(ch);
          array_means[aid] += data[ch];
          counts[aid]++;
        }
        for (size_t aid = 0; aid < kNumArrays; ++aid)
          array_means[aid] /= static_cast<float>(counts[aid]);

        // ------------------------------------------------------------------
        // For each channel: CAR, filter, accumulate stats
        for (size_t ch = 0; ch < num_channels; ++ch)
        {
          const float centred = data[ch] - array_means[get_array_id(ch)];

          float filt_sample = centred;
          if (bandpass_filters_[ch])
            filt_sample = bandpass_filters_[ch]->filter(centred);

          // For calibration accumulate squares
          if (!threshold_initialized_)
          {
            baseline_accum_[ch] += filt_sample * filt_sample;
          }

          // For feature extraction this bin
          sumsq[ch] += filt_sample * filt_sample;

          // track minimum for threshold detection within current subwindow
          if (filt_sample < subwindow_min[ch])
            subwindow_min[ch] = filt_sample;
        }

        // Manage subwindow counter
        subwindow_counter++;
        if (subwindow_counter >= samples_per_ms)
        {
          // Evaluate threshold crossings for this 1-ms window
          if (threshold_initialized_)
          {
            for (size_t ch = 0; ch < num_channels; ++ch)
            {
              if (subwindow_min[ch] <= thresholds_uv_[ch])
                thresh_cross_counts[ch]++;
            }
          }

          // Reset for next subwindow
          std::fill(subwindow_min.begin(), subwindow_min.end(), std::numeric_limits<float>::max());
          subwindow_counter = 0;
        }
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
            thresholds_uv_[ch] = -4.0f * rms; // negative threshold
          }
          threshold_initialized_ = true;
          spdlog::info("[SpeechDecoder] Thresholds initialised after {:.2f} s ({} samples)", calibration_seconds_, baseline_sample_counter_);
        }
      }

      // Compute spike power per channel (mean of squares)
      std::vector<float> spike_power(num_channels, 0.0f);
      for (size_t ch = 0; ch < num_channels; ++ch)
      {
        spike_power[ch] = sumsq[ch] / static_cast<float>(frames_in_bin);
        // Clip to avoid extreme values (match Python 10,000 default)
        const float spike_power_clip = 10000.0f;
        if (spike_power[ch] > spike_power_clip)
          spike_power[ch] = spike_power_clip;
      }

      // Build feature vector [tx_counts | spike_power]
      std::vector<float> feature_vec_10ms;
      feature_vec_10ms.reserve(num_channels * 2);
      for (size_t ch = 0; ch < num_channels; ++ch)
        feature_vec_10ms.push_back(static_cast<float>(thresh_cross_counts[ch]));
      for (size_t ch = 0; ch < num_channels; ++ch)
        feature_vec_10ms.push_back(spike_power[ch]);

      // ---------------- Bin compression (20-ms) ------------------------------
      compression_buffer_.push_back(feature_vec_10ms);
      if (compression_buffer_.size() < kBinCompressionFactor)
      {
        // Wait until we have two 10-ms bins collected, then continue loop.
        continue;
      }

      // Aggregate counts (sum) and power (mean)
      std::vector<float> feature_vec(feature_vec_10ms.size(), 0.0f);
      // Sum across buffer
      for (const auto &vec : compression_buffer_)
      {
        for (size_t i = 0; i < vec.size(); ++i)
          feature_vec[i] += vec[i];
      }

      // For spike power part (second half) compute mean, and apply log.
      const size_t offset_power = num_channels; // index where power starts
      for (size_t i = offset_power; i < feature_vec.size(); ++i)
      {
        feature_vec[i] /= static_cast<float>(kBinCompressionFactor); // mean
        feature_vec[i] = std::log(feature_vec[i] + 1e-4f);           // log-transform
      }

      // Clear compression buffer for next cycle
      compression_buffer_.clear();

      // ---------------- Rolling Z-score Normalisation ------------------------
      feature_window_.push_back(feature_vec);
      if (feature_window_.size() > kRollingWindowBins)
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

      for (size_t i = 0; i < feature_vec.size(); ++i)
      {
        feature_vec[i] = (feature_vec[i] - mean[i]) / stddev[i];
        if (feature_vec[i] > zscore_clip_)
          feature_vec[i] = zscore_clip_;
        else if (feature_vec[i] < -zscore_clip_)
          feature_vec[i] = -zscore_clip_;
      }

      // Publish tensor -----------------------------------------------------
      synapse::Tensor features_tensor;
      const std::array<int32_t, 1> feature_shape = {static_cast<int32_t>(feature_vec.size())};
      features_tensor.mutable_shape()->Add(feature_shape.begin(), feature_shape.end());
      features_tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
      features_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

      const char *feat_ptr = reinterpret_cast<const char *>(feature_vec.data());
      size_t feat_size = feature_vec.size() * sizeof(float);
      features_tensor.set_data(std::string(feat_ptr, feat_size));
      features_tensor.set_timestamp_ns(loop_start_ns.count());

      if (!publish_tap("features_out", features_tensor))
      {
        spdlog::warn("[SpeechDecoder] Failed to publish features_out");
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

        if (!publish_tap("phonemes_out", phoneme_tensor))
        {
          spdlog::warn("[SpeechDecoder] Failed to publish phonemes_out");
        }
      }

      // Housekeeping -----------------------------------------------------------
      const auto loop_dt_ns = synapse::get_steady_clock_now() - loop_start_ns;
      spdlog::debug("[SpeechDecoder] Loop latency: {} ms", loop_dt_ns.count() * 1e-6);
    }
  }

  // ---------------------------------------------------------------------------
  // Helper functions
  // ---------------------------------------------------------------------------
  bool SpeechDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame> &frames, float bin_size_ms)
  {
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

      if (!frames.empty() && (frames.back().timestamp_ns() - first_timestamp_ns >= target_bin_ns))
      {
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

} // namespace app

int main(const int, const char **)
{
  return synapse::Entrypoint<app::SpeechDecoder>();
} 