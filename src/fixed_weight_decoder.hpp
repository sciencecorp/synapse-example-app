#pragma once
#include <array>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <random>
#include <deque>
#include <thread>
#include <vector>

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>
#include <synapse-app-sdk/dsp/spike/threshold_detector.hpp>
#include <synapse-app-sdk/inference/model.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"

// For reset callbacks
#include <google/protobuf/struct.pb.h>

namespace app {
// 10 hz
constexpr auto kPublishRateSec = 1.0 / 10.0;
class FixedWeightDecoder : public synapse::App {
 public:
  FixedWeightDecoder();

  virtual bool setup() override;

 protected:
  virtual void main() override;

 private:
  // Everything the app needs to turn one input node's broadband stream into a bin of spike
  // counts. Each input gets a thread that owns its InputSource, so nothing in here is shared.
  struct InputSource {
    uint32_t node_id = 0;

    // Filled in from the first frame we read off this node
    bool initialized = false;
    float sample_rate_hz = 30000.0;
    size_t channel_count = 0;

    // Use this to detect if there is frame drops
    uint64_t last_sequence_number = 0;

    // One filter and one spike detector per channel on this node
    std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters;
    std::vector<std::unique_ptr<synapse::BaseSpikeDetector>> spike_detectors;

    // Collection of detected spikes
    std::vector<synapse::SpikeEvent*> detected_spikes;

    // The frames making up the bin currently being read
    std::vector<synapse::BroadbandFrame> frames;

    std::thread thread;
  };

  synapse::ApplicationNodeConfig application_config_;

  // One per node connected to us, ordered by node id
  std::vector<std::unique_ptr<InputSource>> sources_;

  // The most recent bin of spike counts from each node, keyed by node id. Written by the input
  // threads, drained by main() once every node has one
  std::mutex ready_bins_mutex_;
  std::condition_variable ready_bins_available_;
  std::map<uint32_t, std::vector<uint32_t>> ready_bins_;

  // A timer to provide a consistent publishing cadence for joystick commands
  synapse::Timer publish_rate_limiter_;

  float low_cutoff_hz_ = 200.0;
  float high_cutoff_hz_ = 5000.0;
  static constexpr int kSpectralFilterOrder = 2;

  // Spike detection configuration
  float spike_threshold_ = 50.0;          // Threshold in microvolts
  uint32_t waveform_size_ = 50;           // Total samples per waveform
  uint64_t refractory_period_us_ = 1000;  // 1ms refractory period

  // Channels from every node concatenated, known once every node has sent its first bin
  std::atomic<size_t> total_channel_count_{0};

  // Spike binning and cursor control parameters
  int window_size_ = 5;              // Number of bins to use for firing rate estimation
  float max_expected_rate_ = 10.0f;  // For normalization
  std::deque<std::vector<uint32_t>>
      spike_count_window_;  // Window buffer to store binned spike counts

  // We will select 4 channels randomly for cursor control
  std::mutex cursor_channel_mutex_;
  std::array<size_t, 4> cursor_channels_ = {0, 7, 16, 30};

  // Should function profiling be enabled?
  bool enable_function_profiling_ = false;

  // Inference: optional model for neural decoding
  bool enable_inference_ = false;
  std::string model_name_ = "decoder";
  std::unique_ptr<synapse::BaseModel> model_;

  // Inference benchmarking
  uint64_t inference_count_ = 0;
  uint64_t inference_total_us_ = 0;
  uint64_t inference_min_us_ = UINT64_MAX;
  uint64_t inference_max_us_ = 0;

  // Set up inference (loads model, logs runtimes)
  void setup_inference();

  // Run inference on spike count features and return decoded cursor position
  // Falls back to the fixed-weight calculation if inference is not available
  std::pair<float, float> run_inference(const std::vector<uint32_t>& spike_counts);

  // Reads, filters and spike detects one node's stream until the app stops. One of these runs
  // per input node
  void process_source(InputSource& source, const float bin_size_ms);

  // Waits until a set of broadband frames are read from the node into source.frames
  // Returns false if there was an error reading
  bool wait_for_frames(InputSource& source, const float bin_size_ms);

  // Filters source.frames and counts the spikes on each of that node's channels
  std::vector<uint32_t> count_spikes(InputSource& source);

  // Waits for a bin from every input node, then concatenates them in node id order so a channel
  // keeps the same index from bin to bin. Returns false if we stopped while waiting
  bool wait_for_bins(std::vector<uint32_t>& spike_counts);

  // If not zero, we dropped some frames, determine what to do
  int detect_dropped_frames(const uint64_t last_sequence_number,
                            const uint64_t current_sequence_number);

  // Randomly select channels to use for cursor control
  bool initialize_cursor_channels(const size_t channel_count);

  // Before starting, set up one node's filters and spike detectors.
  // We can use that node's first broadband frame to do this initialization
  void initialize_source(InputSource& source, const float bin_size_ms);

  // Clean up any allocated spike events
  void cleanup_spike_events(InputSource& source);

  // Calculate cursor position from spike counts
  std::pair<float, float> calculate_cursor_position(const std::vector<uint32_t>& spike_counts);

  // Validate the configuration
  bool validate_config(const synapse::ApplicationNodeConfig& configuration);

  // Parse the configuration
  bool parse_config(const synapse::ApplicationNodeConfig& configuration);

  // If we get a message on our update configuration tap, handle it
  // NOTE: if you are expecting frequent updates, you wouldn't handle the data in the callback
  //       you would instead add the message to a queue and run a process_callback() in your main
  //       loop
  void handle_update_request(const google::protobuf::ListValue& message);
};
}  // namespace app
