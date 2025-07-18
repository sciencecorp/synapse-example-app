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

    // Create a tap for raw GPIO data.
    if (!create_tap<synapse::Tensor>("gpio"))
    {
      spdlog::error("Failed to create gpio tap");
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
        // Parse channel layout on the very first frame so we know which indices
        // correspond to electrode and GPIO channels.
        if (electrode_indices_.empty() && gpio_indices_.empty())
        {
          parse_channel_indices(first_frame);
        }

        const size_t channel_count = electrode_indices_.size();
        sample_rate_hz_            = first_frame.sample_rate_hz();
        initialise_filters(channel_count, sample_rate_hz_);
        // Allocate per-channel detection state now that we know channel count.
        running_rms_.assign(channel_count, 1.0);
        running_mean_.assign(channel_count, 0.0);
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

      /* ----------------- Zero-phase filter and collect samples ------------ */
      const size_t channel_count = electrode_indices_.size();
      const size_t gpio_count    = gpio_indices_.size();

      // Temporary container holding raw (µV) samples per channel for this bin.
      std::vector<std::vector<float>> raw_data(channel_count);
      for (auto& vec : raw_data)
      {
        vec.reserve(frames.size());
      }

      // Container for raw GPIO samples (int16) – one vector per GPIO pin.
      std::vector<std::vector<int16_t>> gpio_data(gpio_count);
      for (auto& vec : gpio_data)
      {
        vec.reserve(frames.size());
      }

      // ---------- Gather raw samples -------------------------------------
      for (const auto& frame : frames)
      {
        const auto& data_in = frame.frame_data();
        // Electrode channels
        for (size_t e = 0; e < channel_count; ++e)
        {
          raw_data[e].push_back(data_in[electrode_indices_[e]]);
        }
        // GPIO channels (raw counts, unprocessed)
        for (size_t g = 0; g < gpio_count; ++g)
        {
          gpio_data[g].push_back(static_cast<int16_t>(data_in[gpio_indices_[g]]));
        }
      }

      // ---------- Apply zero-phase band-pass filtering -------------------
      std::vector<std::vector<float>> filtered_data(channel_count);
      for (size_t ch = 0; ch < channel_count; ++ch)
      {
        filtered_data[ch] = zero_phase_filter(ch, raw_data[ch]);
      }

      /* --------------- Update RMS and run spike detection ------------------ */
      update_running_rms(filtered_data);
      const uint64_t bin_start_ts_ns = frames.front().timestamp_ns();
      detect_and_publish(filtered_data, bin_start_ts_ns);

      // ------------------------------------------------------------
      // Publish raw GPIO samples (if any)
      // ------------------------------------------------------------
      if (!gpio_indices_.empty())
      {
        synapse::Tensor gpio_tensor;
        gpio_tensor.set_timestamp_ns(bin_start_ts_ns);
        gpio_tensor.set_dtype(synapse::Tensor_DType_DT_INT16);
        gpio_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

        // Shape: [samples, gpio_pins]
        gpio_tensor.mutable_shape()->Clear();
        gpio_tensor.mutable_shape()->Add(static_cast<int32_t>(gpio_data[0].size()));
        gpio_tensor.mutable_shape()->Add(static_cast<int32_t>(gpio_data.size()));

        // Flatten in row-major order (sample-major)
        std::vector<int16_t> payload;
        payload.reserve(gpio_data[0].size() * gpio_data.size());
        for (size_t s = 0; s < gpio_data[0].size(); ++s)
        {
          for (size_t g = 0; g < gpio_data.size(); ++g)
          {
            payload.push_back(gpio_data[g][s]);
          }
        }
        const char* data_ptr = reinterpret_cast<const char*>(payload.data());
        gpio_tensor.set_data(std::string(data_ptr, payload.size() * sizeof(int16_t)));

        if (!publish_tap("gpio", gpio_tensor))
        {
          spdlog::warn("Failed to publish GPIO tensor");
        }
      }

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
              payload.push_back(static_cast<int16_t>(electrode_indices_[ch])); // channel id (original)
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

  std::vector<float> SpikeDetectorApp::zero_phase_filter(size_t channel_idx, const std::vector<float>& samples)
  {
    if (samples.empty()) return {};

    // Use reflection padding similar to scipy.signal.filtfilt.  Pad length must be
    // less than the input length; we cap it at 150 samples (≈5 ms @ 30 kHz) or
    // half the vector minus one if the bin is shorter.
    const size_t pad_len = std::min<size_t>(150, (samples.size() > 2 ? samples.size() / 2 - 1 : 0));

    // ----------------------------------------------------------------
    // Build the padded vector using reflection (… 3 2 1 | 1 2 3 4 5 | 5 4 3 …)
    // This removes the step at the boundaries and minimises start-up transients.
    // ----------------------------------------------------------------
    std::vector<float> padded;
    padded.reserve(pad_len + samples.size() + pad_len);

    // Front reflection: reverse of first pad_len samples (exclude the very first point)
    for (size_t i = pad_len; i > 0; --i)
    {
      padded.push_back(samples[i - 1]);
    }

    // Real samples
    padded.insert(padded.end(), samples.begin(), samples.end());

    // Back reflection: reverse of last pad_len samples (exclude last point)
    for (size_t i = 0; i < pad_len; ++i)
    {
      padded.push_back(samples[samples.size() - 2 - i]);
    }

    // ----------------------------------------------------------------
    // Forward pass
    // ----------------------------------------------------------------
    auto forward_filter = synapse::create_bandpass_filter<kFilterOrder>(sample_rate_hz_, low_cutoff_hz_, high_cutoff_hz_);
    std::vector<float> forward_out;
    forward_out.reserve(padded.size());
    for (float x : padded)
    {
      forward_out.push_back(forward_filter->filter(x));
    }

    // ----------------------------------------------------------------
    // Reverse & backward pass
    // ----------------------------------------------------------------
    std::reverse(forward_out.begin(), forward_out.end());
    auto backward_filter = synapse::create_bandpass_filter<kFilterOrder>(sample_rate_hz_, low_cutoff_hz_, high_cutoff_hz_);
    std::vector<float> backward_out;
    backward_out.reserve(forward_out.size());
    for (float x : forward_out)
    {
      backward_out.push_back(backward_filter->filter(x));
    }
    std::reverse(backward_out.begin(), backward_out.end());

    // ----------------------------------------------------------------
    // Remove padding and return the central (unpadded) portion.
    // ----------------------------------------------------------------
    std::vector<float> result(backward_out.begin() + pad_len, backward_out.begin() + pad_len + samples.size());
    return result;
  }

  void SpikeDetectorApp::parse_channel_indices(const synapse::BroadbandFrame& frame)
  {
    electrode_indices_.clear();
    gpio_indices_.clear();

    if (frame.channel_ranges_size() == 0)
    {
      // Legacy behaviour – assume *all but the last* entry are electrodes and
      // the final entry is a single GPIO pin.  This matches the current
      // firmware layout where GPIO samples (if any) are appended after all
      // electrode channels.

      const size_t total = frame.frame_data_size();
      if (total == 0)
        return;  // nothing to do

      // Electrode channels: indices 0 .. total-2  (if at least one electrode)
      if (total > 1)
      {
        electrode_indices_.resize(total - 1);
        std::iota(electrode_indices_.begin(), electrode_indices_.end(), 0);
      }

      // GPIO channel: last index
      gpio_indices_.push_back(total - 1);
      return;
    }

    size_t offset = 0;
    for (const auto& range : frame.channel_ranges())
    {
      for (uint32_t i = 0; i < range.count(); ++i)
      {
        if (range.type() == synapse::ELECTRODE)
        {
          electrode_indices_.push_back(offset + i);
        }
        else if (range.type() == synapse::GPIO)
        {
          gpio_indices_.push_back(offset + i);
        }
      }
      offset += range.count();
    }
  }
} // namespace app

int main(const int, const char **)
{
  return synapse::Entrypoint<app::SpikeDetectorApp>();
}
