#include "fixed_weight_decoder.hpp"
#include <algorithm>  // for std::clamp
#include <chrono>
#include <spdlog/spdlog.h>
#include <synapse-app-sdk/middleware/conversions.hpp>  // for parse_protobuf_message
#include <thread>

namespace app {
// Helper function to clamp a value between min and max
template <typename T>
T clamp(T value, T min, T max) {
  return (value < min) ? min : (value > max) ? max : value;
}

FixedWeightDecoder::FixedWeightDecoder() : publish_rate_limiter_(kPublishRateSec) {}

bool FixedWeightDecoder::setup() {
  // Make sure our app configuration is valid
  if (!get_app_parameters(
          [this](const app::ExampleAppConfig& config) { return validate_configuration(config); },
          configuration_)) {
    spdlog::error("Failed to validate app parameters from app configuration");
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

  if (configuration_.enable_function_profiling()) {
    // Enable performance monitoring
    function_profiler_manager_.add("full_loop");

    // Publish loop stats every 1 second
    if (!enable_function_profiling(std::chrono::seconds(1))) {
      spdlog::error("Failed to enable function profile monitoring");
      return false;
    }
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
    // We have a mapping of channel to filtered data with size of the frames we
    // got
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
        // TODO: bounds checking - but we might not even want this way of doing
        // things
        auto& channel_filter = bandpass_filters_.at(channel_id);
        const float filtered_data = channel_filter->filter(frame_data[channel_id]);
        filtered_channel_data.at(channel_id).push_back(filtered_data);

        // 3. Detect spikes on the filtered data
        if (spike_detectors_initialized_) {
          auto& spike_detector = spike_detectors_.at(channel_id);

          // Pass the filtered data to the spike detector along with the frame
          // timestamp The detector handles the rest internally
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
    if (spike_count_window_.size() > configuration_.window_size()) {
      spike_count_window_.pop_front();
    }

    // Calculate cursor position based on the binned spike counts
    float cursor_x = 0.0f;
    float cursor_y = 0.0f;

    // Only calculate cursor position if we have enough data in the window
    if (spike_count_window_.size() == configuration_.window_size()) {
      // Calculate firing rates over the window for each cursor control channel
      std::array<float, 4> firing_rates = {0.0f, 0.0f, 0.0f, 0.0f};

      for (int i = 0; i < 4; i++) {
        size_t ch = cursor_channels_[i];
        for (const auto& bin_counts : spike_count_window_) {
          firing_rates[i] += bin_counts[ch];
        }
        firing_rates[i] /= configuration_.window_size();  // Average over window
      }

      // Calculate x-position based on first channel pair (differential)
      cursor_x = firing_rates[1] - firing_rates[0];  // Positive = right, negative = left

      // Calculate y-position based on second channel pair (differential)
      cursor_y = firing_rates[3] - firing_rates[2];  // Positive = up, negative = down

      // Normalize to reasonable range (-1 to 1)
      cursor_x = clamp(cursor_x / configuration_.max_expected_rate(), -1.0f, 1.0f);
      cursor_y = clamp(cursor_y / configuration_.max_expected_rate(), -1.0f, 1.0f);
    } else {
      // Not enough data in window yet, use default values
      cursor_x = 0.0f;
      cursor_y = 0.0f;
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

bool FixedWeightDecoder::wait_for_frames(std::vector<synapse::BroadbandFrame>& frames,
                                         float bin_size_ms) {
  if (bin_size_ms <= 0) {
    spdlog::warn("invalid bin size of: {}", bin_size_ms);
    return false;
  }

  const uint64_t target_bin_size_ns = static_cast<uint64_t>(bin_size_ms * 1e6);

  // Prepare our output vector
  frames.clear();

  // Get the first timestamp
  uint64_t first_timestamp_ns = 0;

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

      // Record the first timestamp if this is our first frame
      if (frames.empty()) {
        first_timestamp_ns = broadband_frame.timestamp_ns();
      }

      // Add the frame to our collection
      frames.push_back(broadband_frame);
    }

    // TODO: Instead, we could process the entire multipart?
    // After processing this multipart, check if we've reached the bin size
    if (!frames.empty()) {
      const auto& last_frame = frames.back();
      if (last_frame.timestamp_ns() - first_timestamp_ns >= target_bin_size_ns) {
        // We've collected enough frames to reach the bin size
        return true;
      }
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
    auto detector_ptr = synapse::create_threshold_detector(
        configuration_.spike_threshold_uv(), configuration_.waveform_size(),
        configuration_.refractory_period_us(), sample_rate_hz_);

    if (detector_ptr == nullptr) {
      spdlog::error("Failed to create spike detector for channel: {}", channel_index);
    }
    spike_detectors_.push_back(std::move(detector_ptr));
  }

  spdlog::info("Initialized spike detectors with threshold: {} μV, sample rate: {} Hz",
               configuration_.spike_threshold_uv(), sample_rate_hz_);
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
        sample_rate_hz, configuration_.low_cutoff_hz(), configuration_.high_cutoff_hz());
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

  // Copy values from the repeated field to the fixed-size array
  for (int i = 0; i < 4 && i < configuration_.cursor_channels_size(); i++) {
    cursor_channels_[i] = configuration_.cursor_channels(i);
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

bool FixedWeightDecoder::validate_configuration(const app::ExampleAppConfig& config) {
  // FIlters should be above zero, high cutoff should be above low cutoff
  if (config.low_cutoff_hz() <= 0 || config.high_cutoff_hz() <= 0 ||
      config.low_cutoff_hz() >= config.high_cutoff_hz()) {
    spdlog::error("Invalid filter configuration: low_cutoff={} Hz, high_cutoff={} Hz",
                  config.low_cutoff_hz(), config.high_cutoff_hz());
    return false;
  }

  // Spike threshold should be above zero
  if (config.spike_threshold_uv() <= 0) {
    spdlog::error("Invalid spike threshold: {}", config.spike_threshold_uv());
    return false;
  }

  // Refractory period should be above zero
  if (config.refractory_period_us() <= 0) {
    spdlog::error("Invalid refractory period: {}", config.refractory_period_us());
    return false;
  }

  // Window size should be above zero
  if (config.window_size() <= 0) {
    spdlog::error("Invalid window size: {}", config.window_size());
    return false;
  }

  // Max expected rate should be above zero
  if (config.max_expected_rate() <= 0) {
    spdlog::error("Invalid max expected rate: {}", config.max_expected_rate());
    return false;
  }

  // Should only have 4 cursor channels
  if (config.cursor_channels_size() != 4) {
    spdlog::error("Invalid number of cursor channels: {}, expected 4",
                  config.cursor_channels_size());
    return false;
  }

  return true;
}

}  // namespace app

int main(const int, const char**) { return synapse::Entrypoint<app::FixedWeightDecoder>(); }
