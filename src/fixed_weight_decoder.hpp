#pragma once
#include <array>
#include <deque>
#include <random>

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>
#include <synapse-app-sdk/dsp/spike/threshold_detector.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"

#include "example_app.pb.h"

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
  // Use this to detect if there is frame drops
  uint64_t last_sequence_number_ = 0;

  // A timer to provide a consistent publishing cadence for joystick commands
  synapse::Timer publish_rate_limiter_;

  // We want to filter the incoming broadband data, so do so here
  std::atomic<bool> filters_initialized_{false};

  // App parameters
  app::ExampleAppConfig configuration_;

  // Filter configuration
  static constexpr int kSpectralFilterOrder = 2;
  std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters_;

  // Spike detection configuration and detectors
  std::atomic<bool> spike_detectors_initialized_{false};
  float sample_rate_hz_ = 30000.0; // Will be updated during initialization
  std::vector<std::unique_ptr<synapse::BaseSpikeDetector>> spike_detectors_;

  // Collection of detected spikes
  std::vector<synapse::SpikeEvent *> detected_spikes_;

  // Spike binning and cursor control parameters
  std::deque<std::vector<uint32_t>>
      spike_count_window_; // Window buffer to store binned spike counts

  // We will select 4 channels randomly for cursor control
  // Default to a random selection
  std::array<size_t, 4> cursor_channels_ = {0, 7, 16, 30};

  // Waits until a set of broadband frames are read from the node
  // Returns false if there was an error reading
  bool wait_for_frames(std::vector<synapse::BroadbandFrame> &frames,
                       const float bin_size_ms);

  // If not zero, we dropped some frames, determine what to do
  int detect_dropped_frames(const uint64_t last_sequence_number,
                            const uint64_t current_sequence_number);

  // Randomly select channels to use for cursor control
  bool initialize_cursor_channels(const size_t channel_count);

  // Before starting, set up our filters.
  // We can use the first broadband frame to do this initialization
  void initialize_filters(const size_t channel_count,
                          const float sample_rate_hz, const float bin_size_ms);

  // Initialize spike detectors for each channel
  void initialize_spike_detectors(const size_t channel_count);

  // Clean up any allocated spike events
  void cleanup_spike_events();

  // Calculate cursor position from spike counts
  std::pair<float, float>
  calculate_cursor_position(const std::vector<uint32_t> &spike_counts);

  bool validate_configuration(const app::ExampleAppConfig &config);
};
} // namespace app
