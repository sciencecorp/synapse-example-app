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
// Report packet loss once every 5 seconds
constexpr auto kLossReportRateSec = 5.0;
class FixedWeightDecoder : public synapse::App {
 public:
  FixedWeightDecoder();

  virtual bool setup() override;

 protected:
  virtual void main() override;

 private:
  // One input node's complete pipeline: read, filter, spike detect, decode, publish. Each node
  // gets a thread that owns its InputSource, so the streams share nothing and never wait on
  // each other. With one input node this is the pipeline the app has always run.
  struct InputSource {
    uint32_t node_id = 0;

    // This stream's own output taps
    std::string joystick_tap;
    std::string packet_loss_tap;

    // Profile labels are per stream: FunctionProfiler keeps one start_time_, so several
    // threads sharing a label would clobber each other's timing
    std::string loop_profile;
    std::string inference_profile;

    // Filled in from the first frame we read off this node
    bool initialized = false;
    float sample_rate_hz = 30000.0;
    size_t channel_count = 0;

    // One filter and one spike detector per channel on this node
    std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters;
    std::vector<std::unique_ptr<synapse::BaseSpikeDetector>> spike_detectors;

    // Collection of detected spikes
    std::vector<synapse::SpikeEvent*> detected_spikes;

    // The frames making up the bin currently being read
    std::vector<synapse::BroadbandFrame> frames;

    // Window buffer of binned spike counts, per stream since each decodes on its own
    std::deque<std::vector<uint32_t>> spike_count_window;

    // Packet loss from gaps in the sequence numbers. The first frame only seeds the sequence
    // number, there is nothing to compare it against yet
    bool have_sequence_number = false;
    uint64_t last_sequence_number = 0;
    uint64_t frames_received = 0;
    uint64_t frames_dropped = 0;

    // Inference is per stream so three threads never share one model
    std::unique_ptr<synapse::BaseModel> model;
    uint64_t inference_count = 0;
    uint64_t inference_total_us = 0;
    uint64_t inference_min_us = UINT64_MAX;
    uint64_t inference_max_us = 0;

    // Cadences, per stream now that each publishes on its own
    synapse::Timer publish_rate_limiter{kPublishRateSec};
    synapse::Timer loss_report_limiter{kLossReportRateSec};

    std::thread thread;
  };

  synapse::ApplicationNodeConfig application_config_;

  // One per node connected to us, ordered by node id
  std::vector<std::unique_ptr<InputSource>> sources_;

  float low_cutoff_hz_ = 200.0;
  float high_cutoff_hz_ = 5000.0;
  static constexpr int kSpectralFilterOrder = 2;

  // Spike detection configuration
  float spike_threshold_ = 50.0;          // Threshold in microvolts
  uint32_t waveform_size_ = 50;           // Total samples per waveform
  uint64_t refractory_period_us_ = 1000;  // 1ms refractory period

  // Smallest channel count across the streams, so a cursor channel that is accepted is valid
  // on every one of them. Known once each node has sent its first bin
  std::atomic<size_t> min_channel_count_{0};

  // Spike binning and cursor control parameters
  int window_size_ = 5;              // Number of bins to use for firing rate estimation
  float max_expected_rate_ = 10.0f;  // For normalization

  // Cursor channels index within a single stream and are shared by all of them, so one
  // set_cursor_channels message re-steers every stream at once
  std::mutex cursor_channel_mutex_;
  std::array<size_t, 4> cursor_channels_ = {0, 7, 16, 30};

  // Should function profiling be enabled?
  bool enable_function_profiling_ = false;

  // Inference: optional model for neural decoding, one instance per stream
  bool enable_inference_ = false;
  std::string model_name_ = "decoder";


  // Log the available runtimes once at startup
  void log_inference_runtimes();

  // Load this stream's own model, so three threads never share one
  void setup_inference(InputSource& source);

  // Run inference on spike count features and return decoded cursor position
  // Falls back to the fixed-weight calculation if inference is not available
  std::pair<float, float> run_inference(InputSource& source,
                                        const std::vector<uint32_t>& spike_counts);

  // Runs one node's whole pipeline until the app stops: read, filter, spike detect, decode and
  // publish. One of these runs per input node, and they never wait on each other
  void process_source(InputSource& source, const float bin_size_ms);

  // Waits until a set of broadband frames are read from the node into source.frames
  // Returns false if there was an error reading
  bool wait_for_frames(InputSource& source, const float bin_size_ms);

  // Filters source.frames and counts the spikes on each of that node's channels
  std::vector<uint32_t> count_spikes(InputSource& source);

  // Turns one bin of spike counts into a cursor position and publishes it on this stream's tap
  void decode_and_publish(InputSource& source, const std::vector<uint32_t>& spike_counts);

  // Publishes and logs this stream's cumulative packet loss
  void publish_packet_loss(InputSource& source);

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

  // Calculate cursor position from one stream's spike counts and its own window
  std::pair<float, float> calculate_cursor_position(InputSource& source,
                                                    const std::vector<uint32_t>& spike_counts);

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
