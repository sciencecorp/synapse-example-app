#pragma once

#include <vector>
#include <memory>
#include <atomic>
#include <deque>
#include <string>

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"
#include "api/channel.pb.h"

#include "decoder.hpp"
#include "circular_buffer.hpp"

namespace app {

// Publishes every detected spike on the "spike_waveforms" tap.
// The spike message is a synapse::SpikeEvent protobuf containing:
//   * timestamp_ns : uint64
//   * channel_id   : uint32
//   * waveform     : repeated float (waveform samples in micro-volts)
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

  // Parse channel ranges in the first BroadbandFrame to populate electrode and GPIO indices.
  void parse_channel_indices(const synapse::BroadbandFrame& frame);

  // Ensure config keys exist
  bool validate_config(const synapse::ApplicationNodeConfig& configuration);

  // Parse config file to determine whether to decode
  bool parse_config(const synapse::ApplicationNodeConfig& configuration);
  
  /* ----------------------------------------------------------------------- */
  // State
  /* ----------------------------------------------------------------------- */
  synapse::ApplicationNodeConfig application_config_;
  
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

  // Match offline spike detector which uses 3 × RMS (operating in µV units)
  const float threshold_std_   = 3.0f;        // N × RMS
  const uint32_t waveform_size_ = 32;         // samples per waveform
  // Number of pre-threshold samples to capture (~1/3 of waveform)
  // This gives us roughly a 1:2 ratio of pre- to post-threshold samples
  const uint32_t half_wave_     = waveform_size_ / 3;

  uint32_t refractory_samples_ = 0;  // set once we know sample rate

  // Per-channel state
  std::vector<double> running_rms_;               // exponentially-smoothed RMS estimate (µV)
  std::vector<double> running_mean_;              // exponentially-smoothed mean estimate (µV) for padding
  std::vector<uint64_t> sample_counter_;          // total samples processed per channel
  std::vector<uint64_t> last_spike_sample_idx_;   // last detected spike index (for refractory)
  std::vector<std::deque<float>> pre_buffers_;    // rolling buffer of previous half-wave samples

  // Indices into frame_data for each channel type (filled on first frame)
  std::vector<size_t> electrode_indices_;
  std::vector<size_t> gpio_indices_;

  // Helper to update RMS estimates
  void update_running_rms(const std::vector<std::vector<float>>& filtered_data);

  // Detect spikes in the current bin and publish them
  // filtered_data: [channel][samples], bin_start_timestamp_ns: timestamp (ns)
  void detect_and_publish(const std::vector<std::vector<float>>& filtered_data,
                          uint64_t bin_start_timestamp_ns);

  std::vector<std::vector<float>> construct_decoder_input() const;
  synapse::Tensor make_inference_tensor(const std::vector<std::vector<float>>& outputs, uint64_t timestamp_ns);
  const float max_amplitude_positive_ = 200.0f;  // µV
  const float max_amplitude_negative_ = -250.0f; // µV

  // ---------------- Artefact rejection ----------------------------------
  // If more than this proportion of channels cross threshold simultaneously
  // we consider the sample an artefact and ignore spikes around it.
  const float artefact_channel_ratio_ = 0.5f;   // 50 %
  // Window (samples) around the artefact sample to suppress (±window/2).
  uint32_t artefact_window_samples_ = 6; // set once sample rate known (~0.2 ms)

  // Monotonically increasing sequence number for each published spike (wraps at 2^32)
  uint32_t spike_seq_ = 0;

  // Binning parameters
  static constexpr float kBinMs = 25.0f; // size of each spike-count bin

  // Decode parameters
  bool enable_decode_ = false; // whether to also perform inference and publish to `inferences` tap
  std::string model_path_; // ONNX model to decode with if we so choose to do so
  std::unique_ptr<Decoder> decoder_; // object for inference
  std::unique_ptr<CircularBuffer<std::vector<short unsigned int>>> history_buffer_; // buffer to store the last history_bins number of counts
  int history_bins_; // number of history bins for the history buffer, read from the ONNX model metadata
};

}  // namespace app