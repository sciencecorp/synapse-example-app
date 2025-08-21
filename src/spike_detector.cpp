#include "spike_detector.hpp"

#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <filesystem>
#include <synapse-app-sdk/middleware/conversions.hpp>

namespace app
{
  const int kNumDecoderThreads = 4;

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
    // Parse config to determine whether to decode
    if (!get_app_config(
        [this](const synapse::ApplicationNodeConfig& configuration) {
          return validate_config(configuration);
        },
        application_config_)) {
      spdlog::error("Failed to get app config");
    }
    
    if (!parse_config(application_config_)) {
      spdlog::error("Failed to parse app config");
      return false;
    }

    const uint32_t broadband_node_id = 1;
    if (!setup_reader(broadband_node_id))
    {
      spdlog::error("Failed to set up broadband frame reader for node {}", broadband_node_id);
      return false;
    }

    // Create a tap to stream spike waveforms.
    if (!create_tap<synapse::Tensor>("spike_waveforms"))
    {
      spdlog::error("Failed to create spike_waveforms tap");
      return false;
    }

    // Create a tap to stream binned spikes.
    if (!create_tap<synapse::Tensor>("spike_counts"))
    {
      spdlog::error("Failed to create spike_counts tap");
      return false;
    }

    // Create a tap for raw GPIO data.
    if (!create_tap<synapse::Tensor>("gpio"))
    {
      spdlog::error("Failed to create gpio tap");
      return false;
    }

    if (enable_decode_) {
      spdlog::info("Decoding enabled, inferences will be published to 'inferences'.");
      if (!create_tap<synapse::Tensor>("inferences"))
      {
        spdlog::error("Failed to create inferences tap");
        return false;
      }
    }

    spdlog::info("SpikeDetectorApp set-up complete – waiting for broadband frames …");
    return true;
  }

  void SpikeDetectorApp::main()
  {
    constexpr float kBinMs = app::SpikeDetectorApp::kBinMs;  // 25-ms bins
    std::vector<synapse::BroadbandFrame> frames;

    // Initialize decoder if requested in config
    if (enable_decode_) {
      try {
        decoder_ = std::make_unique<Decoder>(model_path_, kNumDecoderThreads);
      } catch (const std::exception& e) {
        spdlog::error("Error while initializing decoder: {}", e.what());
      }

      try {
        history_bins_ = std::stoi(decoder_->GetMetadataValue("history_bins"));
      } catch (const std::exception& e) {
        spdlog::error("Error with parsing model metadata entry with key 'history_bins' with value \"{}\": {}", 
          decoder_->GetMetadataValue("history_bins"), e.what());
      }

      spdlog::info("Inferences will be performed with {} bins of history.", history_bins_);
      history_buffer_ = std::make_unique<CircularBuffer<std::vector<uint16_t>>>(history_bins_);
    }

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

      /* ----------------- Causal filter and collect samples --------------- */
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

      // ---------- Apply causal band-pass filtering (single forward pass) --
      std::vector<std::vector<float>> filtered_data(channel_count);
      for (size_t ch = 0; ch < channel_count; ++ch)
      {
        filtered_data[ch].reserve(raw_data[ch].size());
        for (float sample : raw_data[ch])
        {
          float y = bandpass_filters_[ch]->filter(sample); // causal IIR
          filtered_data[ch].push_back(y);
        }
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

        // Shape: [1] – single scalar representing first GPIO pin (0/1)
        gpio_tensor.mutable_shape()->Clear();
        gpio_tensor.mutable_shape()->Add(1); // single value

        // Payload: latest sample from first GPIO pin, converted to 0/1
        int16_t bit_value = gpio_data.empty() ? 0 : (gpio_data[0].back() != 0 ? 1 : 0);
        std::vector<int16_t> payload{bit_value};

        // Real-time log (for verification)
        spdlog::info("GPIO bit value: {}", bit_value);

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

    // Set artefact suppression window (~0.5 ms) once we know the sample rate.
    artefact_window_samples_ = static_cast<uint32_t>(std::round(sample_rate_hz_ * 0.0005f));
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

    const size_t channel_count = filtered_data.size();
    if (channel_count == 0) return;

    const size_t sample_count = filtered_data[0].size();

    // Collect at most one waveform per channel for this bin
    std::vector<std::vector<int8_t>> batched_waveforms;
    std::vector<uint8_t> batched_channel_ids;
    batched_waveforms.reserve(channel_count);
    batched_channel_ids.reserve(channel_count);

    // ---------------- Artefact mask (cross-channel) -------------------------
    std::vector<uint8_t> artefact(sample_count, 0); // 1 = artefact

    // Pre-compute per-channel thresholds
    std::vector<float> neg_threshold(channel_count);
    for (size_t ch = 0; ch < channel_count; ++ch)
    {
        neg_threshold[ch] = -threshold_std_ * static_cast<float>(running_rms_[ch]);
    }

    for (size_t s = 0; s < sample_count; ++s)
    {
        size_t crossings = 0;
        for (size_t ch = 0; ch < channel_count; ++ch)
        {
            if (filtered_data[ch][s] < neg_threshold[ch])
                ++crossings;
        }
        if (crossings > artefact_channel_ratio_ * channel_count)
        {
            artefact[s] = 1;
        }
    }

    // Expand artefact samples by ±window to cover residual transients
    if (artefact_window_samples_ > 0)
    {
        std::vector<uint8_t> expanded = artefact;
        const int w = static_cast<int>(artefact_window_samples_);
        for (size_t s = 0; s < sample_count; ++s)
        {
            if (!artefact[s]) continue;
            const int start = static_cast<int>(s) - w;
            const int end   = static_cast<int>(s) + w;
            for (int k = start; k <= end; ++k)
            {
                if (k >= 0 && k < static_cast<int>(sample_count)) expanded[k] = 1;
            }
        }
        artefact.swap(expanded);
    }

    // Initialise count vector for this bin
    const size_t channel_count_total = filtered_data.size();
    std::vector<uint16_t> bin_counts(channel_count_total, 0);

    for (size_t ch = 0; ch < channel_count_total; ++ch)
    {
      const auto& samples = filtered_data[ch];
      auto& pre_buf       = pre_buffers_[ch];

      for (size_t i = 0; i < samples.size(); ++i)
      {
        float x = samples[i];

        uint64_t global_idx = sample_counter_[ch] + i;

        bool candidate = (x < -threshold_std_ * running_rms_[ch] &&
                         (global_idx - last_spike_sample_idx_[ch]) > refractory_samples_ &&
                         i + (waveform_size_ - half_wave_ - 1) < samples.size() &&
                         !artefact[i]); // ensure enough post samples & not artefact

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

              // --------------------------------------------------------------
              // Increment per-bin spike count and capture one waveform per
              // channel for batched publishing later.
              // --------------------------------------------------------------
              if (bin_counts[ch] < std::numeric_limits<uint16_t>::max())
                  ++bin_counts[ch];

              // Ensure we only capture at most one waveform per channel in this bin
              const uint8_t ch_id_u8 = static_cast<uint8_t>(electrode_indices_[ch] & 0xFF);
              if (std::find(batched_channel_ids.begin(), batched_channel_ids.end(), ch_id_u8) == batched_channel_ids.end())
              {
                  // Quantise waveform to INT8 (-128…127)
                  std::vector<int8_t> qwf;
                  qwf.reserve(waveform.size());
                  for (float v : waveform)
                  {
                      int q = static_cast<int>(std::lround(v));
                      q = clamp<int>(q, -128, 127);
                      qwf.push_back(static_cast<int8_t>(q));
                  }

                  batched_waveforms.push_back(std::move(qwf));
                  batched_channel_ids.push_back(ch_id_u8);
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

    /* ---------------- Publish counts tensor --------------------------- */
    synapse::Tensor cnt_tensor;
    cnt_tensor.set_timestamp_ns(bin_start_ts_ns);
    cnt_tensor.set_dtype(synapse::Tensor_DType_DT_UINT16);
    cnt_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

    cnt_tensor.mutable_shape()->Clear();
    cnt_tensor.mutable_shape()->Add(static_cast<int32_t>(bin_counts.size()));

    const char* cnt_ptr = reinterpret_cast<const char*>(bin_counts.data());
    cnt_tensor.set_data(std::string(cnt_ptr, bin_counts.size() * sizeof(uint16_t)));

    if (enable_decode_) {
      // only need history if decoding
      history_buffer_->push(bin_counts);
    }

    if (!publish_tap("spike_counts", cnt_tensor))
    {
        spdlog::warn("Failed to publish spike_counts tensor");
    }

    /* ---------------- Publish batched waveforms --------------------------- */
    if (!batched_waveforms.empty())
    {
        synapse::Tensor waveform_tensor;
        waveform_tensor.set_timestamp_ns(bin_start_ts_ns);
        waveform_tensor.set_dtype(synapse::Tensor_DType_DT_INT8);
        waveform_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

        const int32_t row_len = static_cast<int32_t>(waveform_size_ + 2); // seq + ch + waveform

        // Shape: [N, W]
        waveform_tensor.mutable_shape()->Clear();
        waveform_tensor.mutable_shape()->Add(static_cast<int32_t>(batched_waveforms.size()));
        waveform_tensor.mutable_shape()->Add(row_len);

        // Payload: seq, channel, waveform
        std::vector<int8_t> payload;
        payload.reserve(batched_waveforms.size() * row_len);
        for (size_t idx = 0; idx < batched_waveforms.size(); ++idx)
        {
            payload.push_back(static_cast<int8_t>(spike_seq_ & 0xFF));
            payload.push_back(batched_channel_ids[idx]);
            const auto& qwf = batched_waveforms[idx];
            payload.insert(payload.end(), qwf.begin(), qwf.end());

            spike_seq_ = (spike_seq_ + 1) & 0xFF;
        }

        waveform_tensor.set_data(std::string(reinterpret_cast<const char*>(payload.data()), payload.size() * sizeof(int8_t)));

        if (!publish_tap("spike_waveforms", waveform_tensor))
        {
            spdlog::warn("Failed to publish batched spike waveforms");
        }
        batched_waveforms.clear();
        batched_channel_ids.clear();
    }

    /* ---------------- Publish inferences --------------------------- */
    if (enable_decode_ && history_buffer_->size() >= history_bins_) {
      // If decoding is enabled and we have enough history

      // Create input tensor from the history
      auto input_tensor = construct_decoder_input();

      // Infer with decoder
      auto infer_start = std::chrono::high_resolution_clock::now();
      auto inferences = decoder_->InferSingleInput(input_tensor);
      auto infer_end = std::chrono::high_resolution_clock::now();
      double infer_latency_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();

      // Convert std::vector<Ort::Value> to synapse::Tensor
      synapse::Tensor inferences_tensor = make_inference_tensor(inferences, bin_start_ts_ns);

      // Publish to tap
      if (!publish_tap("inferences", inferences_tensor)) {
        spdlog::warn("Failed to publish inferences");
      }
    }
    else {
      spdlog::info("not yet enough frames, skipping decode.");
    }

  }

  // Constructs an input tensor with batch size 1 with the one input tensor 
  // having shape [history_bins, channeL_count], but flattened
  std::vector<std::vector<float>> SpikeDetectorApp::construct_decoder_input() const {
    if (history_buffer_->size() < history_bins_) {
        // Not enough history yet, return empty or padded
        return {};
    }
    
    int channel_count = electrode_indices_.size();

    // Get the contents in correct order (oldest to newest)
    auto history_contents = history_buffer_->contents();
    
    // Flatten into 1D: [history_bins_ * channel_count]
    std::vector<float> flattened;
    flattened.reserve(history_bins_ * channel_count);
    
    for (const auto& bin : history_contents) {
        // Each bin is a vector<uint16_t> of size channel_count
        for (uint16_t count : bin) {
            flattened.push_back(static_cast<float>(count));
        }
    }

    // Reshape to 2D: [1, history_bins_ * channel_count]
    std::vector<std::vector<float>> input_tensor;
    input_tensor.resize(1);  // 1 row
    input_tensor[0] = std::move(flattened); 

    return input_tensor;
  }

  // Constructs the synapse;:Tensor object for a given input
  synapse::Tensor SpikeDetectorApp::make_inference_tensor(const std::vector<std::vector<float>>& inferences, uint64_t timestamp_ns) {
    synapse::Tensor tensor;
    tensor.set_timestamp_ns(timestamp_ns);
    tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
    tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);
    
    if (inferences.empty()) return tensor;
    
    // Get dimensions
    const int32_t rows = static_cast<int32_t>(inferences.size());
    const int32_t cols = rows ? static_cast<int32_t>(inferences.front().size()) : 0;
    
    // Set shape: [rows, cols]
    tensor.mutable_shape()->Clear();
    tensor.mutable_shape()->Add(rows);
    tensor.mutable_shape()->Add(cols);
    
    // Flatten row-major
    std::vector<float> flat;
    flat.reserve(static_cast<size_t>(rows) * cols);
    for (const auto& row : inferences) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    
    // Copy data
    const char* data_ptr = reinterpret_cast<const char*>(flat.data());
    tensor.set_data(std::string(data_ptr, flat.size() * sizeof(float)));
    
    return tensor;
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

  bool SpikeDetectorApp::validate_config(const synapse::ApplicationNodeConfig& configuration) {
    const auto& parameters = configuration.parameters();

    if (!parameters.contains("enable_decode")) {
      spdlog::error("enable_decode not found in configuration");
      return false;
    }
    // TODO: kind of ugly... maybe cleaner way to do this
    else if (parameters.at("enable_decode").bool_value() && !parameters.contains("model_path")) {
      // model_path must exist only if enable_decode is set to true
      spdlog::error("model_path not found in configuration when enable_decode=true");
      return false;
    }

    return true;
  }

  bool SpikeDetectorApp::parse_config(const synapse::ApplicationNodeConfig& configuration) {
    const auto& parameters = configuration.parameters();
    try {
      enable_decode_ = parameters.at("enable_decode").bool_value();
      model_path_    = parameters.at("model_path").string_value();
      if (enable_decode_ && !std::filesystem::exists(model_path_)) {
        spdlog::error("File for model_path {} not found when enable_decode=true", model_path_);
        return false;
      }
      return true;
    } catch (const std::exception& e) {
      spdlog::error("Failed to parse configuration {}", e.what());
      return false;
    }
  }
} // namespace app

int main(const int, const char **)
{
  return synapse::Entrypoint<app::SpikeDetectorApp>();
}
