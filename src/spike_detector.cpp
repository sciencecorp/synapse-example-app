#include "spike_detector.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <synapse-app-sdk/middleware/conversions.hpp>

namespace app
{
  // Helper function to clamp a value between min and max
  template <typename T>
  T clamp(T value, T min, T max)
  {
    return (value < min) ? min : (value > max) ? max
                                               : value;
  }

  SpikeDetectorApp::SpikeDetectorApp() = default;

  bool SpikeDetectorApp::setup()
  {
    const uint32_t broadband_node_id = 1;
    if (!setup_reader(broadband_node_id))
    {
      spdlog::error("Failed to set up broadband frame reader for node {}", broadband_node_id);
      return false;
    }

    // Create a tap that we will use to stream spike waveforms as Tensor messages.
    if (!create_tap<synapse::Tensor>("spike_waveforms"))
    {
      spdlog::error("Failed to create spike_waveforms tap");
      return false;
    }

    spdlog::info("SpikeDetectorApp set-up complete – waiting for broadband frames …");
    return true;
  }

  void SpikeDetectorApp::main()
  {
    constexpr float kBinMs = 10.0f;  // Gather 10 ms worth of frames for processing.
    std::vector<synapse::BroadbandFrame> frames;

    while (node_running_)
    {
      if (!wait_for_frames(frames, kBinMs))
      {
        // No frames available yet – small sleep to avoid a tight spin loop.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      }

      const auto& first_frame = frames.front();

      /* -------------------------- First-time initialisation ----------------- */
      if (!filters_ready_)
      {
        const size_t channel_count = first_frame.frame_data_size();
        sample_rate_hz_            = first_frame.sample_rate_hz();
        initialise_filters(channel_count, sample_rate_hz_);
        // Allocate per-channel detection state now that we know channel count.
        running_rms_.assign(channel_count, 1.0);
        sample_counter_.assign(channel_count, 0);
        last_spike_sample_idx_.assign(channel_count, 0);
        pre_buffers_.assign(channel_count, {});

        refractory_samples_ = static_cast<uint32_t>(sample_rate_hz_ * 0.001f);  // 1 ms

        detectors_ready_ = true;

        // Processing will restart on the next pass; discard frames collected so far so that
        // filter and detector state starts from clean data.
        frames.clear();
        continue;
      }

      if (!detectors_ready_)
      {
        continue;
      }

      /* ----------------- Filter and collect samples for this bin ----------- */
      const size_t channel_count = bandpass_filters_.size();
      std::vector<std::vector<float>> filtered_data(channel_count);
      for (auto& vec : filtered_data)
      {
        vec.reserve(frames.size());
      }

      for (const auto& frame : frames)
      {
        const auto& data_in = frame.frame_data();
        for (size_t ch = 0; ch < channel_count; ++ch)
        {
          // Convert raw counts to microvolts (Blackrock data units: 0.25 µV per count)
          float filtered_sample = bandpass_filters_[ch]->filter(data_in[ch]);
          filtered_data[ch].push_back(filtered_sample);
        }
      }

      /* --------------- Update RMS and run spike detection ------------------ */
      update_running_rms(filtered_data);
      const uint64_t bin_start_ts_ns = frames.front().timestamp_ns();
      detect_and_publish(filtered_data, bin_start_ts_ns);

      // Clear processed frames so that next iteration starts fresh.
      frames.clear();
    }
  }

  bool SpikeDetectorApp::wait_for_frames(std::vector<synapse::BroadbandFrame>& frames,
                                           float bin_size_ms)
  {
    const uint64_t target_span_ns = static_cast<uint64_t>(bin_size_ms * 1e6);
    uint64_t first_ts_ns = frames.empty() ? 0 : frames.front().timestamp_ns();

    while (node_running_)
    {
      auto multipart = data_reader_->receive_multipart();
      if (multipart.empty())
      {
        // No new data available right now – small sleep before retrying so we
        // don't spin-lock the CPU, but crucially *continue* accumulating any
        // frames we already have instead of discarding them.
        std::this_thread::sleep_for(std::chrono::microseconds(100));
        continue;
      }

      frames.reserve(frames.size() + multipart.size());

      for (auto& msg : multipart)
      {
        auto maybe_frame = synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(msg));
        if (!maybe_frame.has_value())
        {
          spdlog::warn("Failed to parse BroadbandFrame protobuf");
          continue;
        }
        const auto& frame = maybe_frame.value();

        // Dropout detection
        const auto dropped = detect_dropped_frames(last_sequence_number_, frame.sequence_number());
        if (dropped != 0)
        {
          spdlog::warn("Detected {} dropped frames", dropped);
        }
        last_sequence_number_ = frame.sequence_number();

        if (frames.empty())
        {
          first_ts_ns = frame.timestamp_ns();
        }

        frames.push_back(frame);
      }

      if (!frames.empty() && (frames.back().timestamp_ns() - first_ts_ns >= target_span_ns))
      {
        return true;
      }
    }
    return false;
  }

  int SpikeDetectorApp::detect_dropped_frames(uint64_t last_seq, uint64_t current_seq)
  {
    return static_cast<int64_t>(current_seq) - static_cast<int64_t>(last_seq) - 1;
  }

  void SpikeDetectorApp::initialise_filters(size_t channel_count, float sample_rate_hz)
  {
    bandpass_filters_.clear();
    bandpass_filters_.reserve(channel_count);

    for (size_t ch = 0; ch < channel_count; ++ch)
    {
      auto filter = synapse::create_bandpass_filter<kFilterOrder>(sample_rate_hz, low_cutoff_hz_, high_cutoff_hz_);
      if (!filter)
      {
        spdlog::error("Failed to create band-pass filter for channel {}", ch);
      }
      bandpass_filters_.push_back(std::move(filter));
    }

    filters_ready_ = true;
    spdlog::info("Initialised {} band-pass filters ({}–{} Hz, order {})", channel_count,
                 low_cutoff_hz_, high_cutoff_hz_, kFilterOrder);
  }

  void SpikeDetectorApp::update_running_rms(const std::vector<std::vector<float>>& filtered_data)
  {
    const float alpha = 0.01f;  // smoothing factor
    for (size_t ch = 0; ch < filtered_data.size(); ++ch)
    {
      const auto& samples = filtered_data[ch];
      if (samples.empty()) continue;

      // Compute RMS for this bin
      double sum_sq = std::inner_product(samples.begin(), samples.end(), samples.begin(), 0.0);
      double rms_bin = std::sqrt(sum_sq / samples.size());

      running_rms_[ch] = (1.0 - alpha) * running_rms_[ch] + alpha * rms_bin;
    }
  }

  void SpikeDetectorApp::detect_and_publish(const std::vector<std::vector<float>>& filtered_data,
                                            uint64_t bin_start_ts_ns)
  {
    const double sample_period_ns = 1e9 / sample_rate_hz_;

    for (size_t ch = 0; ch < filtered_data.size(); ++ch)
    {
      const auto& samples = filtered_data[ch];
      auto& pre_buf       = pre_buffers_[ch];

      for (size_t i = 0; i < samples.size(); ++i)
      {
        float x = samples[i];

        uint64_t global_idx = sample_counter_[ch] + i;

        bool candidate = (x < -threshold_std_ * running_rms_[ch] &&
                         (global_idx - last_spike_sample_idx_[ch]) > refractory_samples_ &&
                         i + (waveform_size_ - half_wave_ - 1) < samples.size()); // ensure enough post samples

        bool valid_spike = false;

        if (candidate)
        {
          // ---------------------------------------------
          // Optional spike validation to match offline pipeline
          // 1) Ensure a subsequent positive crossing within ~0.5 ms (~15 samples @ 30 kHz)
          const size_t kBiphasicSearchSamples = 15;  // ~0.5 ms @ 30 kHz
          size_t search_end = std::min(samples.size(), i + kBiphasicSearchSamples);
          for (size_t s = i + 1; s < search_end; ++s)
          {
            if (samples[s] > threshold_std_ * running_rms_[ch])
            {
              valid_spike = true;
              break;
            }
          }

          if (!valid_spike)
          {
            // fail biphasic check
            valid_spike = false;
          }
         
          if (valid_spike)
          {
            // Build waveform
            std::vector<float> waveform;
            waveform.reserve(waveform_size_);
            
            // Pre samples
            size_t pre_needed = half_wave_;
            if (pre_buf.size() < pre_needed)
            {
              waveform.insert(waveform.end(), pre_needed - pre_buf.size(), 0.0f);
              waveform.insert(waveform.end(), pre_buf.begin(), pre_buf.end());
            }
            else
            {
              waveform.insert(waveform.end(), pre_buf.end() - pre_needed, pre_buf.end());
            }

            // Current & post samples
            size_t post_count = waveform_size_ - waveform.size();
            for (size_t j = 0; j < post_count; ++j)
            {
              waveform.push_back(samples[i + j]);
            }
 
            // Amplitude gating – discard if waveform exceeds allowed range
            auto [min_it, max_it] = std::minmax_element(waveform.begin(), waveform.end());
            if (*max_it > max_amplitude_positive_ || *min_it < max_amplitude_negative_)
            {
              valid_spike = false;
            }

            if (valid_spike)
            {
              // ------------------------------------------------------------------
              // Accurate spike timestamp – avoid double-counting.
              // Use first-frame timestamp plus within-bin offset (i * period).
              // ------------------------------------------------------------------

              uint64_t ts_ns = bin_start_ts_ns + static_cast<uint64_t>(i * sample_period_ns);

              synapse::Tensor tensor;
              tensor.set_timestamp_ns(ts_ns);
              tensor.set_dtype(synapse::Tensor_DType_DT_INT16);
              tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

              std::vector<int16_t> payload;
              payload.reserve(waveform.size() + 2); // seq + channel + waveform
              payload.push_back(static_cast<int16_t>(spike_seq_++));  // sequence number
              payload.push_back(static_cast<int16_t>(ch));            // channel id
              payload.insert(payload.end(), waveform.begin(), waveform.end());

              // Set tensor shape (seq + channel + waveform)
              tensor.mutable_shape()->Clear();
              tensor.mutable_shape()->Add(waveform_size_ + 2);

              const char* data_ptr = reinterpret_cast<const char*>(payload.data());
              tensor.set_data(std::string(data_ptr, payload.size() * sizeof(int16_t)));

              if (!publish_tap("spike_waveforms", tensor))
              {
                spdlog::warn("Failed to publish spike tensor (ch {})", ch);
              }

              last_spike_sample_idx_[ch] = global_idx;
            }
          }
        }

        // Always push current sample into pre-buffer (for next iterations)
        pre_buf.push_back(x);
        if (pre_buf.size() > half_wave_)
          pre_buf.pop_front();
      }

      // Update sample counter for this channel
      sample_counter_[ch] += samples.size();
    }
  }
} // namespace app

int main(const int, const char **)
{
  return synapse::Entrypoint<app::SpikeDetectorApp>();
}
