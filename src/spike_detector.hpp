#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <deque>

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"

namespace app {

// Publishes every detected spike on the "spike_waveforms" tap.
// The spike message is a synapse::SpikeEvent protobuf containing:
//   * timestamp_ns : uint64
//   * channel_id   : uint32
//   * waveform     : repeated float (waveform samples in micro-volts)
//
// The application continuously streams BroadbandFrame data, applies a band-pass
// filter, runs threshold-based spike detection, and publishes each SpikeEvent
// immediately.
class SpikeDetectorApp : public synapse::App {
 public:
  SpikeDetectorApp();

  bool setup() override;

 protected:
  void main() override;

 private:
  // Wait until we have accumulated a contiguous block of broadband frames whose
  // span in time is at least |bin_size_ms|. Returns true and fills |frames| on
  // success. Returns false if the node stops running or an unrecoverable error
  // occurs.
  bool wait_for_frames(std::vector<synapse::BroadbandFrame>& frames,
                       float bin_size_ms);

  // Utility to detect dropped frames using frame sequence numbers.
  int detect_dropped_frames(uint64_t last_seq, uint64_t current_seq);

  // One-off initialisation of filter objects – called when we have the first
  // BroadbandFrame so we know the channel count and sample rate.
  void initialise_filters(size_t channel_count, float sample_rate_hz);

  // No built-in spike detector; we implement RMS detection directly.

  /* ----------------------------------------------------------------------- */
  // State
  /* ----------------------------------------------------------------------- */

  // Track last seen sequence number to warn about dropouts
  uint64_t last_sequence_number_ = 0;

  /* ------------------------- Filtering ----------------------------------- */
  std::atomic<bool> filters_ready_{false};
  static constexpr int kFilterOrder = 4;
  const float low_cutoff_hz_  = 300.0f;
  const float high_cutoff_hz_ = 6000.0f;
  float sample_rate_hz_       = 30000.0f;  // updated on first frame
  std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters_;

  /* ------------------------- Spike detection ----------------------------- */
  std::atomic<bool> detectors_ready_{false};

  // Match offline spike detector which uses 4 × RMS (operating in µV units)
  const float threshold_std_   = 3.0f;        // N × RMS
  const uint32_t waveform_size_ = 50;         // samples per waveform
  const uint32_t half_wave_     = waveform_size_ / 2;

  uint32_t refractory_samples_ = 0;  // set once we know sample rate

  // Per-channel state
  std::vector<double> running_rms_;               // exponentially-smoothed RMS estimate (µV)
  std::vector<uint64_t> sample_counter_;          // total samples processed per channel
  std::vector<uint64_t> last_spike_sample_idx_;   // last detected spike index (for refractory)
  std::vector<std::deque<float>> pre_buffers_;    // rolling buffer of previous half-wave samples

  // Helper to update RMS estimates
  void update_running_rms(const std::vector<std::vector<float>>& filtered_data);

  // Detect spikes in the current bin and publish them
  void detect_and_publish(const std::vector<std::vector<float>>& filtered_data,
                          uint64_t bin_start_timestamp_ns);

  const float max_amplitude_positive_ = 200.0f;  // µV
  const float max_amplitude_negative_ = -250.0f; // µV
};

}  // namespace app