#include "fixed_weight_decoder.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <algorithm>
#include <synapse-app-sdk/middleware/conversions.hpp>

namespace app {

template <typename T>
T clamp(T value, T min, T max) {
  return (value < min) ? min : (value > max) ? max : value;
}

FixedWeightDecoder::FixedWeightDecoder() : publish_rate_limiter_(kPublishRateSec) {}

bool FixedWeightDecoder::setup() {
  if (!get_app_config(
          [this](const synapse::ApplicationNodeConfig& configuration) {
            return validate_config(configuration);
          },
          application_config_)) {
    spdlog::error("Failed to get app config");
    return false;
  }

  if (!parse_config(application_config_)) {
    spdlog::error("Failed to parse app config");
    return false;
  }

  const auto reset_tap_ret = create_consumer_tap<google::protobuf::ListValue>(
      "set_cursor_channels",
      [this](const google::protobuf::ListValue& message) { handle_update_request(message); });
  if (!reset_tap_ret) {
    spdlog::error("Failed to set up consumer tap for set_cursor_channels");
    return false;
  }

  const uint32_t broadband_node_id = 1;
  if (!setup_reader(broadband_node_id)) {
    spdlog::warn("Failed to set up reader for controller");
    return 1;
  }

  if (!create_tap<synapse::Tensor>("joystick_out")) {
    spdlog::warn("Failed to create tap for joystick out");
    return false;
  }

  // Training data taps — non-fatal if unavailable
  if (!create_tap<synapse::Tensor>("spike_features")) {
    spdlog::warn("Failed to create spike_features tap");
  }
  if (!create_tap<synapse::Tensor>("controller_labels")) {
    spdlog::warn("Failed to create controller_labels tap");
  }

  if (enable_function_profiling_) {
    function_profiler_manager_.add("full_loop");
    function_profiler_manager_.add("inference");
    if (!enable_function_profiling(std::chrono::seconds(1))) {
      spdlog::error("Failed to enable function profile monitoring");
      return false;
    }
  }

  if (enable_inference_) {
    setup_inference();
  }

  return true;
}

void FixedWeightDecoder::publish_training_taps(const std::vector<uint32_t>& spike_counts,
                                               const synapse::BroadbandFrame& last_frame) {
  const auto ts_ns = synapse::get_steady_clock_now().count();

  // Always publish spike counts as float tensor
  {
    std::vector<float> spk_f(spike_counts.begin(), spike_counts.end());
    synapse::Tensor feat;
    feat.mutable_shape()->Add(static_cast<int>(spk_f.size()));
    feat.set_dtype(synapse::Tensor_DType_DT_FLOAT);
    feat.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);
    feat.set_data(synapse::pack_tensor_data(spk_f));
    feat.set_timestamp_ns(ts_ns);
    publish_tap("spike_features", feat);
  }

  // In training mode the encoder adds 2 label channels at the end (channels 32 & 33)
  // containing raw joystick X and joystick Y values respectively.
  const auto& fd = last_frame.frame_data();
  if (static_cast<size_t>(fd.size()) > kNeuralChannels + 1) {
    float lx = fd[kNeuralChannels];
    float ly = fd[kNeuralChannels + 1];
    synapse::Tensor lbl;
    lbl.mutable_shape()->Add(2);
    lbl.set_dtype(synapse::Tensor_DType_DT_FLOAT);
    lbl.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);
    lbl.set_data(synapse::pack_tensor_data({lx, ly}));
    lbl.set_timestamp_ns(ts_ns);
    publish_tap("controller_labels", lbl);
  }
}

void FixedWeightDecoder::main() {
  const float bin_size_ms = 10;
  std::vector<synapse::BroadbandFrame> broadband_frames;

  while (node_running_) {
    if (!wait_for_frames(broadband_frames, bin_size_ms)) {
      continue;
    }

    start_profile("full_loop");

    const auto broadband_frame = broadband_frames.at(0);
    if (!filters_initialized_) {
      const size_t channel_count = broadband_frame.frame_data_size();
      const float sample_rate_hz = broadband_frame.sample_rate_hz();
      sample_rate_hz_ = sample_rate_hz;
      initialize_filters(channel_count, sample_rate_hz, bin_size_ms);
      initialize_spike_detectors(channel_count);
      continue;
    }

    cleanup_spike_events();

    std::vector<std::vector<float>> filtered_channel_data;
    filtered_channel_data.resize(broadband_frames.at(0).frame_data_size());
    for (auto& channel_vector : filtered_channel_data) {
      channel_vector.reserve(broadband_frames.size());
    }

    std::vector<uint32_t> spike_counts(broadband_frames.at(0).frame_data_size(), 0);

    for (const auto& frame : broadband_frames) {
      const auto& frame_data = frame.frame_data();
      const uint64_t frame_timestamp_ns = frame.timestamp_ns();

      for (int channel_id = 0; channel_id < frame_data.size(); ++channel_id) {
        auto& channel_filter = bandpass_filters_.at(channel_id);
        const float filtered_data = channel_filter->filter(frame_data[channel_id]);
        filtered_channel_data.at(channel_id).push_back(filtered_data);

        if (spike_detectors_initialized_) {
          auto& spike_detector = spike_detectors_.at(channel_id);
          synapse::SpikeEvent* spike_event =
              spike_detector->detect(filtered_data, frame_timestamp_ns, channel_id);
          if (spike_event != nullptr) {
            detected_spikes_.push_back(spike_event);
            spike_counts[channel_id]++;
          }
        }
      }
    }

    spike_count_window_.push_back(spike_counts);
    if (spike_count_window_.size() > window_size_) {
      spike_count_window_.pop_front();
    }

    // Publish training taps every bin (not rate-limited)
    publish_training_taps(spike_counts, broadband_frames.back());

    // Publish joystick output at 30 Hz
    if (publish_rate_limiter_.reset_if_elapsed()) {
      float cursor_x = 0.0f;
      float cursor_y = 0.0f;

      if (spike_count_window_.size() == window_size_) {
        if (enable_inference_ && model_ && model_->is_ready()) {
          auto [x, y] = run_inference(spike_counts);
          cursor_x = x;
          cursor_y = y;
        } else {
          auto [x, y] = calculate_cursor_position(spike_counts);
          cursor_x = x;
          cursor_y = y;
        }
      }

      synapse::Tensor output_tensor;
      const auto tensor_shape = {2};
      output_tensor.mutable_shape()->Add(tensor_shape.begin(), tensor_shape.end());
      output_tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
      output_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);
      output_tensor.set_data(synapse::pack_tensor_data({cursor_x, cursor_y}));
      output_tensor.set_timestamp_ns(synapse::get_steady_clock_now().count());

      if (publish_tap("joystick_out", output_tensor)) {
        spdlog::info("Published tensor: [x,y]: [{},{}]", cursor_x, cursor_y);
      } else {
        spdlog::warn("Failed to publish tensor data");
      }

      stop_profile("full_loop");
      print_profile("full_loop");
    }
  }
}

std::pair<float, float> FixedWeightDecoder::calculate_cursor_position(
    const std::vector<uint32_t>& spike_counts) {
  std::array<float, 4> firing_rates = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int i = 0; i < 4; i++) {
    size_t ch = cursor_channels_[i];
    for (const auto& bin_counts : spike_count_window_) {
      firing_rates[i] += bin_counts[ch];
    }
    firing_rates[i] /= window_size_;
  }
  float cursor_x = firing_rates[1] - firing_rates[0];
  float cursor_y = firing_rates[3] - firing_rates[2];
  cursor_x = clamp(cursor_x / max_expected_rate_, -1.0f, 1.0f);
  cursor_y = clamp(cursor_y / max_expected_rate_, -1.0f, 1.0f);
  return {cursor_x, cursor_y};
}

void FixedWeightDecoder::setup_inference() {
  auto runtimes = synapse::get_available_runtimes();
  spdlog::info("Available inference runtimes:");
  for (const auto& rt : runtimes) {
    const char* name = "unknown";
    switch (rt) {
      case synapse::InferenceRuntime::kCpu: name = "CPU (ONNX Runtime)"; break;
      case synapse::InferenceRuntime::kGpu: name = "GPU (QNN)"; break;
      case synapse::InferenceRuntime::kDsp: name = "DSP (QNN HTP)"; break;
      case synapse::InferenceRuntime::kAuto: name = "Auto"; break;
    }
    spdlog::info("  - {}", name);
  }
  model_ = synapse::create_model(model_name_);
  if (model_ && model_->is_ready()) {
    spdlog::info("Inference model '{}' loaded successfully", model_name_);
    for (const auto& input : model_->get_input_info()) {
      spdlog::info("  Input: {} elements={}", input.name, input.element_count);
    }
  } else {
    spdlog::warn("Model '{}' not available - falling back to fixed-weight decoding", model_name_);
  }
}

std::pair<float, float> FixedWeightDecoder::run_inference(
    const std::vector<uint32_t>& spike_counts) {
  if (!model_ || !model_->is_ready()) return calculate_cursor_position(spike_counts);

  auto inputs = model_->get_input_info();
  if (inputs.empty()) return calculate_cursor_position(spike_counts);

  std::vector<float> input_features(inputs[0].element_count, 0.0f);
  for (size_t i = 0; i < spike_counts.size() && i < input_features.size(); ++i) {
    input_features[i] = static_cast<float>(spike_counts[i]);
  }

  start_profile("inference");
  auto result = model_->infer({input_features});
  stop_profile("inference");
  print_profile("inference");

  if (!result.success || result.outputs.empty()) {
    spdlog::warn("Inference failed, falling back to fixed-weight decoding");
    return calculate_cursor_position(spike_counts);
  }

  inference_count_++;
  inference_total_us_ += result.inference_time_us;
  inference_min_us_ = std::min(inference_min_us_, result.inference_time_us);
  inference_max_us_ = std::max(inference_max_us_, result.inference_time_us);

  if (inference_count_ % 100 == 0) {
    spdlog::info("Inference stats: count={}, avg={} us, min={} us, max={} us",
                  inference_count_, inference_total_us_ / inference_count_,
                  inference_min_us_, inference_max_us_);
  }

  const auto& output = result.outputs[0];
  float cursor_x = (output.size() > 0) ? clamp(output[0], -1.0f, 1.0f) : 0.0f;
  float cursor_y = (output.size() > 1) ? clamp(output[1], -1.0f, 1.0f) : 0.0f;
  return {cursor_x, cursor_y};
}

bool FixedWeightDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame>& frames,
                                         float bin_size_ms) {
  if (bin_size_ms <= 0) { spdlog::warn("invalid bin size of: {}", bin_size_ms); return false; }
  const float bin_size_sec = bin_size_ms / 1000;
  const size_t target_num_of_frames = bin_size_sec * sample_rate_hz_;
  frames.clear();

  while (node_running_) {
    auto messages = data_reader_->receive_multipart();
    if (messages.empty()) {
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }
    frames.reserve(frames.size() + messages.size());
    for (auto& message : messages) {
      const auto maybe_frame =
          synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(message));
      if (!maybe_frame.has_value()) {
        spdlog::warn("Failed to parse broadband frame");
        if (frames.empty()) return false;
        return true;
      }
      const auto& broadband_frame = maybe_frame.value();
      const auto dropped =
          detect_dropped_frames(last_sequence_number_, broadband_frame.sequence_number());
      if (dropped != 0) spdlog::warn("Dropped: {} frames", dropped);
      last_sequence_number_ = broadband_frame.sequence_number();
      frames.push_back(broadband_frame);
    }
    if (frames.size() >= target_num_of_frames) return true;
  }
  return false;
}

int FixedWeightDecoder::detect_dropped_frames(const uint64_t last_sequence_number,
                                              const uint64_t current_sequence_number) {
  const auto expected_sequence_number = last_sequence_number + 1;
  return (current_sequence_number - expected_sequence_number);
}

void FixedWeightDecoder::initialize_spike_detectors(const size_t channel_count) {
  spike_detectors_.clear();
  spike_detectors_.reserve(channel_count);
  for (size_t ch = 0; ch < channel_count; ++ch) {
    auto det = synapse::create_threshold_detector(spike_threshold_, waveform_size_,
                                                   refractory_period_us_, sample_rate_hz_);
    if (!det) spdlog::error("Failed to create spike detector for channel: {}", ch);
    spike_detectors_.push_back(std::move(det));
  }
  spdlog::info("Initialized spike detectors: threshold={} uV, rate={} Hz",
               spike_threshold_, sample_rate_hz_);
  spike_detectors_initialized_ = true;
}

void FixedWeightDecoder::cleanup_spike_events() {
  for (auto* ev : detected_spikes_) delete ev;
  detected_spikes_.clear();
}

void FixedWeightDecoder::initialize_filters(const size_t channel_count,
                                             const float sample_rate_hz,
                                             const float bin_size_ms) {
  if (!initialize_cursor_channels(channel_count)) return;
  spdlog::info("Initializing filters: rate={} Hz  channels={}  bin={} ms",
               sample_rate_hz, channel_count, bin_size_ms);
  bandpass_filters_.clear();
  bandpass_filters_.reserve(channel_count);
  for (size_t ch = 0; ch < channel_count; ++ch) {
    auto f = synapse::create_bandpass_filter<kSpectralFilterOrder>(
        sample_rate_hz, low_cutoff_hz_, high_cutoff_hz_);
    if (!f) spdlog::error("Failed to create filter for channel: {}", ch);
    bandpass_filters_.push_back(std::move(f));
  }
  spdlog::info("Initialized filters");
  filters_initialized_ = true;
}

bool FixedWeightDecoder::initialize_cursor_channels(const size_t channel_count) {
  if (channel_count < 4) {
    spdlog::warn("Need at least four channels for joystick control");
    return false;
  }
  std::stringstream ss;
  ss << "Using [";
  for (const auto& ch : cursor_channels_) ss << ch << ",";
  ss << "] for cursor control";
  spdlog::info("{}", ss.str());
  return true;
}

bool FixedWeightDecoder::validate_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& p = configuration.parameters();
  for (const auto& key : {"low_cutoff_hz", "high_cutoff_hz", "spike_threshold_uv",
                           "waveform_size", "refractory_period_us", "window_size",
                           "max_expected_rate", "cursor_channels", "enable_function_profiling"}) {
    if (!p.contains(key)) { spdlog::error("{} not found in configuration", key); return false; }
  }
  return true;
}

bool FixedWeightDecoder::parse_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& p = configuration.parameters();
  try {
    low_cutoff_hz_             = p.at("low_cutoff_hz").number_value();
    high_cutoff_hz_            = p.at("high_cutoff_hz").number_value();
    spike_threshold_           = p.at("spike_threshold_uv").number_value();
    waveform_size_             = p.at("waveform_size").number_value();
    refractory_period_us_      = p.at("refractory_period_us").number_value();
    window_size_               = p.at("window_size").number_value();
    max_expected_rate_         = p.at("max_expected_rate").number_value();
    enable_function_profiling_ = p.at("enable_function_profiling").bool_value();
    if (p.contains("enable_inference")) enable_inference_ = p.at("enable_inference").bool_value();
    if (p.contains("model_name"))       model_name_       = p.at("model_name").string_value();

    const auto& cc = p.at("cursor_channels").list_value().values();
    if (cc.size() != 4) { spdlog::error("cursor_channels must be a list of 4 integers"); return false; }
    for (size_t i = 0; i < 4; ++i) cursor_channels_[i] = cc[i].number_value();

    application_config_ = configuration;
    spdlog::info("Successfully parsed configuration: {}", application_config_.DebugString());
    return true;
  } catch (const std::exception& e) {
    spdlog::error("Failed to parse configuration: {}", e.what());
    return false;
  }
}

void FixedWeightDecoder::handle_update_request(const google::protobuf::ListValue& message) {
  try {
    const auto& values = message.values();
    if (values.size() != 4) {
      spdlog::warn("Got a reset request, but didn't see the current number of channels: {}",
                   message.DebugString());
      return;
    }
    for (const auto& value : values) {
      if (!value.has_number_value()) { spdlog::warn("Expected number value for cursor channel"); return; }
      const auto channel = value.number_value();
      if (channel < 0 || channel >= 32) { spdlog::warn("Got an out of range joystick channel: {}", channel); return; }
    }
    spdlog::info("Got a valid update request, setting new cursor channels");
    {
      std::lock_guard<std::mutex> lock(cursor_channel_mutex_);
      for (size_t i = 0; i < 4; ++i) cursor_channels_[i] = values[i].number_value();
    }
    initialize_cursor_channels(cursor_channels_.size());
  } catch (const std::exception& e) {
    spdlog::error("Got a reset request, but had trouble parsing. Why: {}", e.what());
  }
}

}  // namespace app

int main(const int, const char**) { return synapse::Entrypoint<app::FixedWeightDecoder>(); }
