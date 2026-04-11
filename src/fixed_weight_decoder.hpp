#pragma once
#include <array>
#include <cstdint>
#include <random>
#include <deque>

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>
#include <synapse-app-sdk/dsp/spike/threshold_detector.hpp>
#include <synapse-app-sdk/inference/model.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"
#include <google/protobuf/struct.pb.h>

namespace app {

// 30 Hz publish rate → ~33 ms round-trip to game client
// DWELL_FRAMES=3 in bci_game.py → ~100 ms dwell time per selection
constexpr auto kPublishRateSec = 1.0 / 30.0;

// Number of true neural channels.
// In TRAINING mode the encoder appends 2 raw label channels at the end.
constexpr size_t kNeuralChannels = 32;

class FixedWeightDecoder : public synapse::App {
 public:
  FixedWeightDecoder();
  virtual bool setup() override;

 protected:
  virtual void main() override;

 private:
  synapse::ApplicationNodeConfig application_config_;
  uint64_t last_sequence_number_ = 0;
  synapse::Timer publish_rate_limiter_;

  std::atomic<bool> filters_initialized_{false};
  float low_cutoff_hz_ = 200.0;
  float high_cutoff_hz_ = 5000.0;
  static constexpr int kSpectralFilterOrder = 2;
  std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters_;

  std::atomic<bool> spike_detectors_initialized_{false};
  float spike_threshold_ = 50.0;
  uint32_t waveform_size_ = 50;
  uint64_t refractory_period_us_ = 1000;
  float sample_rate_hz_ = 32000.0;
  std::vector<std::unique_ptr<synapse::BaseSpikeDetector>> spike_detectors_;
  std::vector<synapse::SpikeEvent*> detected_spikes_;

  int window_size_ = 5;
  float max_expected_rate_ = 10.0f;
  std::deque<std::vector<uint32_t>> spike_count_window_;

  std::mutex cursor_channel_mutex_;
  std::array<size_t, 4> cursor_channels_ = {0, 7, 16, 30};

  bool enable_function_profiling_ = false;
  bool enable_inference_ = false;
  std::string model_name_ = "decoder";
  std::unique_ptr<synapse::BaseModel> model_;

  uint64_t inference_count_ = 0;
  uint64_t inference_total_us_ = 0;
  uint64_t inference_min_us_ = UINT64_MAX;
  uint64_t inference_max_us_ = 0;

  // Publishes spike counts every bin for training data collection.
  // Also publishes raw label channel values when in training mode (34 channels).
  void publish_training_taps(const std::vector<uint32_t>& spike_counts,
                              const synapse::BroadbandFrame& last_frame);

  void setup_inference();
  std::pair<float, float> run_inference(const std::vector<uint32_t>& spike_counts);
  bool wait_for_frames(std::vector<synapse::BroadbandFrame>& frames, const float bin_size_ms);
  int detect_dropped_frames(const uint64_t last_sequence_number,
                            const uint64_t current_sequence_number);
  bool initialize_cursor_channels(const size_t channel_count);
  void initialize_filters(const size_t channel_count, const float sample_rate_hz,
                          const float bin_size_ms);
  void initialize_spike_detectors(const size_t channel_count);
  void cleanup_spike_events();
  std::pair<float, float> calculate_cursor_position(const std::vector<uint32_t>& spike_counts);
  bool validate_config(const synapse::ApplicationNodeConfig& configuration);
  bool parse_config(const synapse::ApplicationNodeConfig& configuration);
  void handle_update_request(const google::protobuf::ListValue& message);
};

}  // namespace app
