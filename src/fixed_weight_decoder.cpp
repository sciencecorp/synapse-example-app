#include "fixed_weight_decoder.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <algorithm>                                   // for std::clamp
#include <synapse-app-sdk/middleware/conversions.hpp>  // for parse_protobuf_message

namespace app {
// Helper function to clamp a value between min and max
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

  // Setup a consumer tap to listen for reset commands
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

  // Setup our output tap
  if (!create_tap<synapse::Tensor>("joystick_out")) {
    spdlog::warn("Failed to create tap for joystick out");
    return false;
  }

  if (enable_function_profiling_) {
    // Enable performance monitoring
    add_profile("full_loop");
    add_profile("inference");
    // Publish loop stats every 1 second
    if (!enable_function_profiling(std::chrono::seconds(1))) {
      spdlog::error("Failed to enable function profile monitoring");
      return false;
    }
  }

  // Set up inference if enabled (optional - continues even if model not available)
  if (enable_inference_) {
    setup_inference();
  }

  return true;
}

void FixedWeightDecoder::main() {
  // Store our broadband frames here
  const float bin_size_ms = 10;
  std::vector<synapse::BroadbandFrame> broadband_frames;

  while (node_running_) {
    // Receive data from the node you configured
    if (!wait_for_frames(broadband_frames, bin_size_ms)) {
      // No frames just go wait again
      continue;
    }

    // Keep track of how long processing takes
    start_profile("full_loop");

    // You have a set of broadband frames now, you can do whatever you want
    // 1. Initialize the filters and our state on the first full set of frames
    const auto broadband_frame = broadband_frames.at(0);
    if (!filters_initialized_) {
      const size_t channel_count = broadband_frame.frame_data_size();
      const float sample_rate_hz = broadband_frame.sample_rate_hz();

      // Store the sample rate for later use
      sample_rate_hz_ = sample_rate_hz;

      initialize_filters(channel_count, sample_rate_hz, bin_size_ms);
      initialize_spike_detectors(channel_count);

      // Move to the next loop after init
      continue;
    }

    // Cleanup any previously detected spikes before processing new frames
    cleanup_spike_events();

    // 2. Filter the received frames
    // We have a mapping of channel to filtered data with size of the frames we got
    // TODO/NOTE: you could drop out early based on timestamps
    std::vector<std::vector<float>> filtered_channel_data;
    filtered_channel_data.resize(broadband_frames.at(0).frame_data_size());
    for (auto& channel_vector : filtered_channel_data) {
      channel_vector.reserve(broadband_frames.size());
    }

    // Create a vector to count spikes per channel in this batch
    std::vector<uint32_t> spike_counts(broadband_frames.at(0).frame_data_size(), 0);

    for (const auto& frame : broadband_frames) {
      const auto& frame_data = frame.frame_data();
      const uint64_t frame_timestamp_ns = frame.timestamp_ns();

      for (int channel_id = 0; channel_id < frame_data.size(); ++channel_id) {
        // TODO: bounds checking - but we might not even want this way of doing things
        auto& channel_filter = bandpass_filters_.at(channel_id);
        const float filtered_data = channel_filter->filter(frame_data[channel_id]);
        filtered_channel_data.at(channel_id).push_back(filtered_data);

        // 3. Detect spikes on the filtered data
        if (spike_detectors_initialized_) {
          auto& spike_detector = spike_detectors_.at(channel_id);

          // Pass the filtered data to the spike detector along with the frame timestamp
          // The detector handles the rest internally
          synapse::SpikeEvent* spike_event =
              spike_detector->detect(filtered_data, frame_timestamp_ns, channel_id);

          if (spike_event != nullptr) {
            // Store the detected spike for further processing
            detected_spikes_.push_back(spike_event);

            // Increment the spike count for this channel
            spike_counts[channel_id]++;
          }
        }
      }
    }

    // Add current binned spike counts to the window
    spike_count_window_.push_back(spike_counts);

    // Keep window at fixed size
    if (spike_count_window_.size() > window_size_) {
      spike_count_window_.pop_front();
    }

    // Calculate cursor position based on the binned spike counts
    float cursor_x = 0.0f;
    float cursor_y = 0.0f;

    // Only calculate cursor position if we have enough data in the window
    if (spike_count_window_.size() == window_size_) {
      if (enable_inference_ && model_ && model_->is_ready()) {
        // Use the inference model to decode cursor position from spike counts
        auto [x, y] = run_inference(spike_counts);
        cursor_x = x;
        cursor_y = y;
      } else {
        // Fixed-weight decoding: use differential firing rates across channel pairs
        auto [x, y] = calculate_cursor_position(spike_counts);
        cursor_x = x;
        cursor_y = y;
      }
    }

    // Create a tensor with the cursor position
    synapse::Tensor output_tensor;
    const auto tensor_shape = {2};
    output_tensor.mutable_shape()->Add(tensor_shape.begin(), tensor_shape.end());
    output_tensor.set_dtype(synapse::Tensor_DType_DT_FLOAT);
    output_tensor.set_endianness(synapse::Tensor_Endianness_TENSOR_LITTLE_ENDIAN);

    // Use the calculated cursor position instead of raw data values
    output_tensor.set_data(synapse::pack_tensor_data({cursor_x, cursor_y}));

    const auto current_time_ns = synapse::get_steady_clock_now();
    output_tensor.set_timestamp_ns(current_time_ns.count());

    // Then, send off your data using the publisher you configured earlier
    // In this demo, we use a ZMQ publisher over tcp
    if (publish_rate_limiter_.reset_if_elapsed()) {
      if (publish_tap("joystick_out", output_tensor)) {
        spdlog::info("Published tensor: [x,y]: [{},{}]", cursor_x, cursor_y);
      } else {
        spdlog::warn("Failed to publish tensor data");
      }
      stop_profile("full_loop");

      // We can also get a debug print of the output
      print_profile("full_loop");
    }

    // You can sleep here if you want,
    // We busy wait up at the top if there is no data, so you don't need to here
  }
}

std::pair<float, float> FixedWeightDecoder::calculate_cursor_position(
    const std::vector<uint32_t>& spike_counts) {
  // Calculate firing rates over the window for each cursor control channel
  std::array<float, 4> firing_rates = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int i = 0; i < 4; i++) {
    size_t ch = cursor_channels_[i];
    for (const auto& bin_counts : spike_count_window_) {
      firing_rates[i] += bin_counts[ch];
    }
    firing_rates[i] /= window_size_;  // Average over window
  }

  // Differential firing rates for x and y
  float cursor_x = firing_rates[1] - firing_rates[0];  // Positive = right, negative = left
  float cursor_y = firing_rates[3] - firing_rates[2];  // Positive = up, negative = down

  // Normalize to range [-1, 1]
  cursor_x = clamp(cursor_x / max_expected_rate_, -1.0f, 1.0f);
  cursor_y = clamp(cursor_y / max_expected_rate_, -1.0f, 1.0f);

  return {cursor_x, cursor_y};
}

void FixedWeightDecoder::setup_inference() {
  // Log available inference runtimes on this device
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

  // Load the model by name from /opt/scifi/data/models/
  // Deploy a model with: synapsectl deploy-model model.onnx --name <model_name> -u <device>
  model_ = synapse::create_model(model_name_);

  if (model_ && model_->is_ready()) {
    spdlog::info("Inference model '{}' loaded successfully", model_name_);

    auto inputs = model_->get_input_info();
    for (const auto& input : inputs) {
      std::string shape_str;
      for (size_t i = 0; i < input.shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(input.shape[i]);
      }
      spdlog::info("  Input: {} shape=[{}] elements={}", input.name, shape_str, input.element_count);
    }

    auto outputs = model_->get_output_info();
    for (const auto& output : outputs) {
      std::string shape_str;
      for (size_t i = 0; i < output.shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(output.shape[i]);
      }
      spdlog::info("  Output: {} shape=[{}] elements={}", output.name, shape_str,
                    output.element_count);
    }
  } else {
    spdlog::warn("Model '{}' not available - falling back to fixed-weight decoding", model_name_);
    spdlog::warn("Deploy a model with: synapsectl deploy-model <model.onnx> --name {} -u <device>",
                  model_name_);
  }
}

std::pair<float, float> FixedWeightDecoder::run_inference(
    const std::vector<uint32_t>& spike_counts) {
  if (!model_ || !model_->is_ready()) {
    return calculate_cursor_position(spike_counts);
  }

  auto inputs = model_->get_input_info();
  if (inputs.empty()) {
    return calculate_cursor_position(spike_counts);
  }

  // Convert spike counts to float input for the model
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

  // Update benchmarking stats
  inference_count_++;
  inference_total_us_ += result.inference_time_us;
  inference_min_us_ = std::min(inference_min_us_, result.inference_time_us);
  inference_max_us_ = std::max(inference_max_us_, result.inference_time_us);

  if (inference_count_ % 100 == 0) {
    uint64_t avg_us = inference_total_us_ / inference_count_;
    spdlog::info("Inference stats: count={}, avg={} us, min={} us, max={} us",
                  inference_count_, avg_us, inference_min_us_, inference_max_us_);
  }

  // Model output: expect at least 2 values [cursor_x, cursor_y]
  const auto& output = result.outputs[0];
  float cursor_x = (output.size() > 0) ? clamp(output[0], -1.0f, 1.0f) : 0.0f;
  float cursor_y = (output.size() > 1) ? clamp(output[1], -1.0f, 1.0f) : 0.0f;

  return {cursor_x, cursor_y};
}

bool FixedWeightDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame>& frames,
                                         float bin_size_ms) {
  if (bin_size_ms <= 0) {
    spdlog::warn("invalid bin size of: {}", bin_size_ms);
    return false;
  }

  const float bin_size_sec = bin_size_ms / 1000;
  const size_t target_num_of_frames = bin_size_sec * (sample_rate_hz_);

  // Prepare our output vector
  frames.clear();

  // TODO: We should consider having a timeout here
  while (node_running_) {
    // In this example, we are listening to BroadbandFrame data
    // TODO: the broadband node sends over the messages using multipart
    // Figure out why this is the case
    auto messages = data_reader_->receive_multipart();
    if (messages.empty()) {
      // Just keep trying
      // TODO: We should have better signaling on the read failure
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }

    // Reserve space for these frames
    frames.reserve(frames.size() + messages.size());

    // Process each received message in this multipart
    for (auto& message : messages) {
      // Parse the message into a BroadbandFrame
      const auto maybe_frame =
          synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(message));
      if (!maybe_frame.has_value()) {
        spdlog::warn("Failed to parse broadband frame");
        // If we have no frames at all, return false
        if (frames.empty()) {
          return false;
        }
        // Otherwise, return what we have so far
        return true;
      }

      const auto& broadband_frame = maybe_frame.value();

      // Check for dropped frames
      const auto dropped_frames =
          detect_dropped_frames(last_sequence_number_, broadband_frame.sequence_number());
      if (dropped_frames != 0) {
        spdlog::warn("Dropped: {} frames", dropped_frames);
      }
      last_sequence_number_ = broadband_frame.sequence_number();

      // Add the frame to our collection
      frames.push_back(broadband_frame);
    }

    // TODO: Instead, we could process the entire multipart?
    // After processing this multipart, check if we've reached the bin size
    if (frames.size() >= target_num_of_frames) {
      return true;
    }
  }
  return false;
}

int FixedWeightDecoder::detect_dropped_frames(const uint64_t last_sequence_number,
                                              const uint64_t current_sequence_number) {
  const auto expected_sequence_number = last_sequence_number + 1;
  return (current_sequence_number - expected_sequence_number);
}

void FixedWeightDecoder::initialize_spike_detectors(const size_t channel_count) {
  // Create spike detectors for each channel
  spike_detectors_.clear();
  spike_detectors_.reserve(channel_count);

  for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
    auto detector_ptr = synapse::create_threshold_detector(spike_threshold_, waveform_size_,
                                                           refractory_period_us_, sample_rate_hz_);

    if (detector_ptr == nullptr) {
      spdlog::error("Failed to create spike detector for channel: {}", channel_index);
    }
    spike_detectors_.push_back(std::move(detector_ptr));
  }

  spdlog::info("Initialized spike detectors with threshold: {} μV, sample rate: {} Hz",
               spike_threshold_, sample_rate_hz_);
  spike_detectors_initialized_ = true;
}

void FixedWeightDecoder::cleanup_spike_events() {
  // Free memory for all detected spike events
  for (auto spike_event : detected_spikes_) {
    delete spike_event;
  }
  detected_spikes_.clear();
}

void FixedWeightDecoder::initialize_filters(const size_t channel_count, const float sample_rate_hz,
                                            const float bin_size_ms) {
  if (!initialize_cursor_channels(channel_count)) {
    return;
  }

  // We have four channels selected, initialize our filters
  spdlog::info("Initializing\tsample_rate={} Hz\tchannels={}\tbin_size={} ms", sample_rate_hz,
               channel_count, bin_size_ms);

  // Create filters for each channel
  bandpass_filters_.clear();
  bandpass_filters_.reserve(channel_count);
  for (size_t channel_index = 0; channel_index < channel_count; ++channel_index) {
    auto filter_ptr = synapse::create_bandpass_filter<kSpectralFilterOrder>(
        sample_rate_hz, low_cutoff_hz_, high_cutoff_hz_);
    if (filter_ptr == nullptr) {
      spdlog::error("Failed to create filter for channel: {}", channel_index);
    }
    bandpass_filters_.push_back(std::move(filter_ptr));
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
  for (const auto& channel : cursor_channels_) {
    ss << channel << ",";
  }

  ss << "] for cursor control";
  spdlog::info("{}", ss.str());
  return true;
}

bool FixedWeightDecoder::validate_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& parameters = configuration.parameters();

  if (!parameters.contains("low_cutoff_hz")) {
    spdlog::error("low_cutoff_hz not found in configuration");
    return false;
  }

  if (!parameters.contains("high_cutoff_hz")) {
    spdlog::error("high_cutoff_hz not found in configuration");
    return false;
  }

  if (!parameters.contains("spike_threshold_uv")) {
    spdlog::error("spike_threshold_uv not found in configuration");
    return false;
  }

  if (!parameters.contains("waveform_size")) {
    spdlog::error("waveform_size not found in configuration");
    return false;
  }

  if (!parameters.contains("refractory_period_us")) {
    spdlog::error("refractory_period_us not found in configuration");
    return false;
  }

  if (!parameters.contains("window_size")) {
    spdlog::error("window_size not found in configuration");
    return false;
  }

  if (!parameters.contains("max_expected_rate")) {
    spdlog::error("max_expected_rate not found in configuration");
    return false;
  }

  if (!parameters.contains("cursor_channels")) {
    spdlog::error("cursor_channels not found in configuration");
    return false;
  }

  if (!parameters.contains("enable_function_profiling")) {
    spdlog::error("enable_function_profiling not found in configuration");
    return false;
  }

  return true;
}

bool FixedWeightDecoder::parse_config(const synapse::ApplicationNodeConfig& configuration) {
  const auto& parameters = configuration.parameters();
  try {
    low_cutoff_hz_ = parameters.at("low_cutoff_hz").number_value();
    high_cutoff_hz_ = parameters.at("high_cutoff_hz").number_value();
    spike_threshold_ = parameters.at("spike_threshold_uv").number_value();
    waveform_size_ = parameters.at("waveform_size").number_value();
    refractory_period_us_ = parameters.at("refractory_period_us").number_value();
    window_size_ = parameters.at("window_size").number_value();
    max_expected_rate_ = parameters.at("max_expected_rate").number_value();
    enable_function_profiling_ = parameters.at("enable_function_profiling").bool_value();

    // Inference parameters (optional - defaults to disabled)
    if (parameters.contains("enable_inference")) {
      enable_inference_ = parameters.at("enable_inference").bool_value();
    }
    if (parameters.contains("model_name")) {
      model_name_ = parameters.at("model_name").string_value();
    }

    const auto& cursor_channels = parameters.at("cursor_channels").list_value().values();
    if (cursor_channels.size() != 4) {
      spdlog::error("cursor_channels must be a list of 4 integers");
      return false;
    }

    for (size_t i = 0; i < 4; ++i) {
      cursor_channels_[i] = cursor_channels[i].number_value();
    }

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
    // Just handle the configuration where we want to change the cursor channels
    // Need to make sure we have exactly four channels
    const auto& values = message.values();
    if (values.size() != 4) {
      spdlog::warn("Got a reset request, but didn't see the current number of channels: {}",
                   message.DebugString());
      return;
    }

    // Make sure they are in a good range
    for (const auto& value : values) {
      if (!value.has_number_value()) {
        spdlog::warn("Expected number value for cursor channel");
        return;
      }

      const auto channel = value.number_value();
      if (channel < 0 || channel >= 32) {
        spdlog::warn("Got an out of range joystick channel: {}", channel);
        return;
      }
    }

    spdlog::info("Got a valid update request, setting new cursor channels");
    {
      std::lock_guard<std::mutex> lock(cursor_channel_mutex_);
      for (size_t i = 0; i < 4; ++i) {
        cursor_channels_[i] = values[i].number_value();
      }
    }
    initialize_cursor_channels(cursor_channels_.size());
  } catch (const std::exception& e) {
    spdlog::error("Got a reset request, but had trouble parsing. Why: {}", e.what());
  }
}
}  // namespace app

int main(const int, const char**) { return synapse::Entrypoint<app::FixedWeightDecoder>(); }
