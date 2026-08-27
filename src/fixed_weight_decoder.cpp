#include "fixed_weight_decoder.hpp"
#include <spdlog/spdlog.h>
#include <thread>
#include <chrono>
#include <algorithm>                                   // for std::clamp, std::sort
#include <map>
#include <mutex>
#include <utility>
#include <synapse-app-sdk/middleware/conversions.hpp>  // for parse_protobuf_message

namespace app {
// Helper function to clamp a value between min and max
template <typename T>
T clamp(T value, T min, T max) {
  return (value < min) ? min : (value > max) ? max : value;
}

FixedWeightDecoder::FixedWeightDecoder() {}

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

  // One reader per node the device configuration connects to us. That is a single broadband
  // source in the simplest chain, or several filter nodes fanning back in
  auto source_node_ids = setup_readers();
  if (source_node_ids.empty()) {
    spdlog::error("No input nodes are connected to this application node");
    return false;
  }
  std::sort(source_node_ids.begin(), source_node_ids.end());
  for (const auto node_id : source_node_ids) {
    auto source = std::make_unique<InputSource>();
    source->node_id = node_id;
    sources_.push_back(std::move(source));
  }
  spdlog::info("Reading from {} input node(s)", sources_.size());

  // Each stream publishes on its own taps, so a client can follow one probe without the others
  // being muxed into it
  for (auto& source : sources_) {
    source->joystick_tap = "joystick_out_" + std::to_string(source->node_id);
    source->packet_loss_tap = "packet_loss_" + std::to_string(source->node_id);
    source->loop_profile = "full_loop_" + std::to_string(source->node_id);
    source->inference_profile = "inference_" + std::to_string(source->node_id);

    if (!create_tap<synapse::Tensor>(source->joystick_tap)) {
      spdlog::error("Failed to create tap: {}", source->joystick_tap);
      return false;
    }
    if (!create_tap<google::protobuf::Struct>(source->packet_loss_tap)) {
      spdlog::error("Failed to create tap: {}", source->packet_loss_tap);
      return false;
    }
  }

  if (enable_function_profiling_) {
    // Enable performance monitoring, one label per stream
    for (const auto& source : sources_) {
      add_profile(source->loop_profile);
      add_profile(source->inference_profile);
    }
    // Publish loop stats every 1 second
    if (!enable_function_profiling(std::chrono::seconds(1))) {
      spdlog::error("Failed to enable function profile monitoring");
      return false;
    }
  }

  // Set up inference if enabled (optional - continues even if model not available).
  // Each stream loads its own model, three threads must not share one
  if (enable_inference_) {
    log_inference_runtimes();
    for (auto& source : sources_) {
      setup_inference(*source);
    }
  }

  return true;
}

void FixedWeightDecoder::main() {
  const float bin_size_ms = 10;

  // Every input node runs its own pipeline end to end on its own thread. Nothing is muxed and
  // no stream waits on another, so one slow or silent probe cannot stall the rest
  for (auto& source : sources_) {
    InputSource* source_ptr = source.get();
    source->thread =
        std::thread([this, source_ptr, bin_size_ms]() { process_source(*source_ptr, bin_size_ms); });
  }

  // The streams do the work, main just waits for the app to be stopped
  while (node_running_) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  for (auto& source : sources_) {
    if (source->thread.joinable()) {
      source->thread.join();
    }
  }
}

void FixedWeightDecoder::process_source(InputSource& source, const float bin_size_ms) {
  while (node_running_) {
    // Receive a bin worth of data from this node
    if (!wait_for_frames(source, bin_size_ms)) {
      // No frames just go wait again
      continue;
    }

    // Initialize the filters and this node's state on its first full set of frames
    if (!source.initialized) {
      initialize_source(source, bin_size_ms);

      // Move to the next bin after init
      continue;
    }

    // Keep track of how long processing takes. Every stream reports into the same label, so
    // the profile covers a bin of work whichever stream produced it
    start_profile(source.loop_profile);

    // Cleanup any previously detected spikes before processing new frames
    cleanup_spike_events(source);

    decode_and_publish(source, count_spikes(source));
    publish_packet_loss(source);

    stop_profile(source.loop_profile);
  }
}

void FixedWeightDecoder::decode_and_publish(InputSource& source,
                                            const std::vector<uint32_t>& spike_counts) {
  // Add current binned spike counts to this stream's window
  source.spike_count_window.push_back(spike_counts);

  // Keep window at fixed size
  if (source.spike_count_window.size() > static_cast<size_t>(window_size_)) {
    source.spike_count_window.pop_front();
  }

  // Calculate cursor position based on the binned spike counts
  float cursor_x = 0.0f;
  float cursor_y = 0.0f;

  // Only calculate cursor position if we have enough data in the window
  if (source.spike_count_window.size() == static_cast<size_t>(window_size_)) {
    if (enable_inference_ && source.model && source.model->is_ready()) {
      // Use the inference model to decode cursor position from spike counts
      auto [x, y] = run_inference(source, spike_counts);
      cursor_x = x;
      cursor_y = y;
    } else {
      // Fixed-weight decoding: use differential firing rates across channel pairs
      auto [x, y] = calculate_cursor_position(source, spike_counts);
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

  // Then, send off your data on this stream's own tap
  if (source.publish_rate_limiter.reset_if_elapsed()) {
    if (publish_tap(source.joystick_tap, output_tensor)) {
      spdlog::info("{}: [x,y]: [{},{}]", source.joystick_tap, cursor_x, cursor_y);
    } else {
      spdlog::warn("Failed to publish tensor data on {}", source.joystick_tap);
    }

    // We can also get a debug print of the output
    print_profile(source.loop_profile);
  }
}

void FixedWeightDecoder::publish_packet_loss(InputSource& source) {
  if (!source.loss_report_limiter.reset_if_elapsed()) {
    return;
  }

  const double loss_percent =
      source.frames_received == 0
          ? 0.0
          : (100.0 * static_cast<double>(source.frames_dropped) /
             static_cast<double>(source.frames_received + source.frames_dropped));

  if (source.frames_dropped == 0) {
    spdlog::info("node {}: no loss, {} frames received", source.node_id, source.frames_received);
  } else {
    spdlog::warn("node {}: dropped {} of {} expected frames ({:.4f}%)", source.node_id,
                 source.frames_dropped, source.frames_received + source.frames_dropped,
                 loss_percent);
  }

  google::protobuf::Struct stats;
  auto& fields = *stats.mutable_fields();
  fields["node_id"].set_number_value(source.node_id);
  fields["frames_received"].set_number_value(static_cast<double>(source.frames_received));
  fields["frames_dropped"].set_number_value(static_cast<double>(source.frames_dropped));
  fields["loss_percent"].set_number_value(loss_percent);
  fields["timestamp_ns"].set_number_value(
      static_cast<double>(synapse::get_steady_clock_now().count()));

  if (!publish_tap(source.packet_loss_tap, stats)) {
    spdlog::warn("Failed to publish packet loss on {}", source.packet_loss_tap);
  }
}

std::vector<uint32_t> FixedWeightDecoder::count_spikes(InputSource& source) {
  // Count spikes per channel on this node for this bin
  std::vector<uint32_t> spike_counts(source.channel_count, 0);
  const auto channel_count = static_cast<int>(source.channel_count);

  for (const auto& frame : source.frames) {
    const auto& frame_data = frame.frame_data();
    const uint64_t frame_timestamp_ns = frame.timestamp_ns();

    for (int channel_id = 0; channel_id < frame_data.size() && channel_id < channel_count;
         ++channel_id) {
      // Filter the sample
      auto& channel_filter = source.bandpass_filters.at(channel_id);
      const float filtered_data = channel_filter->filter(frame_data[channel_id]);

      // Detect spikes on the filtered data. Pass the filtered data to the spike detector along
      // with the frame timestamp, the detector handles the rest internally
      auto& spike_detector = source.spike_detectors.at(channel_id);
      synapse::SpikeEvent* spike_event =
          spike_detector->detect(filtered_data, frame_timestamp_ns, channel_id);

      if (spike_event != nullptr) {
        // Store the detected spike for further processing
        source.detected_spikes.push_back(spike_event);

        // Increment the spike count for this channel
        spike_counts[channel_id]++;
      }
    }
  }

  return spike_counts;
}

std::pair<float, float> FixedWeightDecoder::calculate_cursor_position(
    InputSource& source, const std::vector<uint32_t>& spike_counts) {
  // Calculate firing rates over the window for each cursor control channel
  std::array<float, 4> firing_rates = {0.0f, 0.0f, 0.0f, 0.0f};

  std::array<size_t, 4> channels;
  {
    // The control tap can re-steer the channels from another thread at any point
    std::lock_guard<std::mutex> lock(cursor_channel_mutex_);
    channels = cursor_channels_;
  }

  for (int i = 0; i < 4; i++) {
    size_t ch = channels[i];
    for (const auto& bin_counts : source.spike_count_window) {
      // Cursor channels index within one stream, so a config can name one past the end
      if (ch >= bin_counts.size()) {
        continue;
      }
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

void FixedWeightDecoder::log_inference_runtimes() {
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

}

void FixedWeightDecoder::setup_inference(InputSource& source) {
  // Load the model by name from /opt/scifi/data/models/
  // Deploy a model with: synapsectl deploy-model model.onnx --name <model_name> -u <device>
  // Every stream gets its own instance, BaseModel is not shared across threads
  source.model = synapse::create_model(model_name_);

  if (source.model && source.model->is_ready()) {
    spdlog::info("Node {}: inference model '{}' loaded successfully", source.node_id, model_name_);

    auto inputs = source.model->get_input_info();
    for (const auto& input : inputs) {
      std::string shape_str;
      for (size_t i = 0; i < input.shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(input.shape[i]);
      }
      spdlog::info("  Input: {} shape=[{}] elements={}", input.name, shape_str, input.element_count);
    }

    auto outputs = source.model->get_output_info();
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
    spdlog::warn("Node {}: model '{}' not available - falling back to fixed-weight decoding",
                 source.node_id, model_name_);
    spdlog::warn("Deploy a model with: synapsectl deploy-model <model.onnx> --name {} -u <device>",
                  model_name_);
  }
}

std::pair<float, float> FixedWeightDecoder::run_inference(
    InputSource& source, const std::vector<uint32_t>& spike_counts) {
  if (!source.model || !source.model->is_ready()) {
    return calculate_cursor_position(source, spike_counts);
  }

  auto inputs = source.model->get_input_info();
  if (inputs.empty()) {
    return calculate_cursor_position(source, spike_counts);
  }

  // Convert spike counts to float input for the model
  std::vector<float> input_features(inputs[0].element_count, 0.0f);
  for (size_t i = 0; i < spike_counts.size() && i < input_features.size(); ++i) {
    input_features[i] = static_cast<float>(spike_counts[i]);
  }

  start_profile(source.inference_profile);
  auto result = source.model->infer({input_features});
  stop_profile(source.inference_profile);
  print_profile(source.inference_profile);

  if (!result.success || result.outputs.empty()) {
    spdlog::warn("Node {}: inference failed, falling back to fixed-weight decoding",
                 source.node_id);
    return calculate_cursor_position(source, spike_counts);
  }

  // Update benchmarking stats
  source.inference_count++;
  source.inference_total_us += result.inference_time_us;
  source.inference_min_us = std::min(source.inference_min_us, result.inference_time_us);
  source.inference_max_us = std::max(source.inference_max_us, result.inference_time_us);

  if (source.inference_count % 100 == 0) {
    uint64_t avg_us = source.inference_total_us / source.inference_count;
    spdlog::info("Node {} inference stats: count={}, avg={} us, min={} us, max={} us",
                 source.node_id, source.inference_count, avg_us, source.inference_min_us,
                 source.inference_max_us);
  }

  // Model output: expect at least 2 values [cursor_x, cursor_y]
  const auto& output = result.outputs[0];
  float cursor_x = (output.size() > 0) ? clamp(output[0], -1.0f, 1.0f) : 0.0f;
  float cursor_y = (output.size() > 1) ? clamp(output[1], -1.0f, 1.0f) : 0.0f;

  return {cursor_x, cursor_y};
}

bool FixedWeightDecoder::wait_for_frames(InputSource& source, const float bin_size_ms) {
  if (bin_size_ms <= 0) {
    spdlog::warn("invalid bin size of: {}", bin_size_ms);
    return false;
  }

  auto* node_reader = reader(source.node_id);
  if (node_reader == nullptr) {
    spdlog::error("No reader set up for node: {}", source.node_id);
    return false;
  }

  const float bin_size_sec = bin_size_ms / 1000;
  const size_t target_num_of_frames = bin_size_sec * source.sample_rate_hz;

  // Prepare our output vector
  source.frames.clear();

  // TODO: We should consider having a timeout here
  while (node_running_) {
    // In this example, we are listening to BroadbandFrame data. A broadband source node sends a
    // batch of frames as one multipart message, a spectral filter node sends one frame per
    // message. Either way, every part is a BroadbandFrame
    auto messages = node_reader->receive_multipart();
    if (messages.empty()) {
      // Just keep trying
      // TODO: We should have better signaling on the read failure
      std::this_thread::sleep_for(std::chrono::microseconds(1));
      continue;
    }

    // Reserve space for these frames
    source.frames.reserve(source.frames.size() + messages.size());

    // Process each received message in this multipart
    for (auto& message : messages) {
      // Parse the message into a BroadbandFrame
      const auto maybe_frame =
          synapse::parse_protobuf_message<synapse::BroadbandFrame>(std::move(message));
      if (!maybe_frame.has_value()) {
        spdlog::warn("Failed to parse broadband frame from node {}", source.node_id);
        // If we have no frames at all, return false
        if (source.frames.empty()) {
          return false;
        }
        // Otherwise, return what we have so far
        return true;
      }

      const auto& broadband_frame = maybe_frame.value();

      // Track packet loss from gaps in the sequence numbers. The first frame we ever see only
      // seeds the counter, there is no earlier frame to compare it against
      source.frames_received++;
      if (source.have_sequence_number) {
        const auto dropped_frames =
            detect_dropped_frames(source.last_sequence_number, broadband_frame.sequence_number());
        if (dropped_frames > 0) {
          source.frames_dropped += static_cast<uint64_t>(dropped_frames);
        }
      }
      source.have_sequence_number = true;
      source.last_sequence_number = broadband_frame.sequence_number();

      // Add the frame to our collection
      source.frames.push_back(broadband_frame);
    }

    // TODO: Instead, we could process the entire multipart?
    // After processing this multipart, check if we've reached the bin size
    if (source.frames.size() >= target_num_of_frames) {
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

void FixedWeightDecoder::initialize_source(InputSource& source, const float bin_size_ms) {
  // Whatever this node sends us in its first frame is the shape of its stream
  const auto& first_frame = source.frames.at(0);
  source.channel_count = first_frame.frame_data_size();
  source.sample_rate_hz = first_frame.sample_rate_hz();

  spdlog::info("Node {} initializing\tsample_rate={} Hz\tchannels={}\tbin_size={} ms",
               source.node_id, source.sample_rate_hz, source.channel_count, bin_size_ms);

  // Create a filter and a spike detector for each of this node's channels
  source.bandpass_filters.clear();
  source.bandpass_filters.reserve(source.channel_count);
  source.spike_detectors.clear();
  source.spike_detectors.reserve(source.channel_count);

  for (size_t channel_index = 0; channel_index < source.channel_count; ++channel_index) {
    auto filter_ptr = synapse::create_bandpass_filter<kSpectralFilterOrder>(
        source.sample_rate_hz, low_cutoff_hz_, high_cutoff_hz_);
    if (filter_ptr == nullptr) {
      spdlog::error("Failed to create filter for node {} channel: {}", source.node_id,
                    channel_index);
    }
    source.bandpass_filters.push_back(std::move(filter_ptr));

    auto detector_ptr = synapse::create_threshold_detector(
        spike_threshold_, waveform_size_, refractory_period_us_, source.sample_rate_hz);
    if (detector_ptr == nullptr) {
      spdlog::error("Failed to create spike detector for node {} channel: {}", source.node_id,
                    channel_index);
    }
    source.spike_detectors.push_back(std::move(detector_ptr));
  }

  spdlog::info("Node {} initialized {} filters and spike detectors with threshold: {} uV",
               source.node_id, source.channel_count, spike_threshold_);
  source.initialized = true;

  // A cursor channel has to be valid on every stream, so track the smallest channel count
  size_t current = min_channel_count_.load();
  while ((current == 0 || source.channel_count < current) &&
         !min_channel_count_.compare_exchange_weak(current, source.channel_count)) {
  }

  initialize_cursor_channels(min_channel_count_.load());
}

void FixedWeightDecoder::cleanup_spike_events(InputSource& source) {
  // Free memory for all detected spike events
  for (auto spike_event : source.detected_spikes) {
    delete spike_event;
  }
  source.detected_spikes.clear();
}

bool FixedWeightDecoder::initialize_cursor_channels(const size_t channel_count) {
  if (channel_count < 4) {
    spdlog::warn("Need at least four channels for joystick control");
    return false;
  }

  for (const auto& channel : cursor_channels_) {
    if (channel >= channel_count) {
      spdlog::warn("Cursor channel {} is past the {} channels we are reading", channel,
                   channel_count);
    }
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

    // Cursor channels index within a stream and apply to all of them, so they have to be valid
    // on the narrowest one. We only know that once each node has sent us a bin
    const size_t channel_count = min_channel_count_;
    if (channel_count == 0) {
      spdlog::warn("Got an update request before any data arrived, ignoring it");
      return;
    }

    // Make sure they are in a good range
    for (const auto& value : values) {
      if (!value.has_number_value()) {
        spdlog::warn("Expected number value for cursor channel");
        return;
      }

      const auto channel = value.number_value();
      if (channel < 0 || channel >= static_cast<double>(channel_count)) {
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
    initialize_cursor_channels(min_channel_count_);
  } catch (const std::exception& e) {
    spdlog::error("Got a reset request, but had trouble parsing. Why: {}", e.what());
  }
}
}  // namespace app

int main(const int, const char**) { return synapse::Entrypoint<app::FixedWeightDecoder>(); }
