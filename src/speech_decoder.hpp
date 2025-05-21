#pragma once

#include <vector>
#include <memory>
#include <deque>

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/dsp/filter/base_filter.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"


namespace app
{
  // NOTE: publish phoneme outputs at 10 Hz for now. This will be revisited once we know the model latency
  constexpr auto kSpeechPublishRateSec = 1.0 / 10.0;

  class SpeechDecoder : public synapse::App
  {
  public:
    SpeechDecoder();

    // Set up readers and taps
    virtual bool setup() override;

  protected:
    // Main processing loop
    virtual void main() override;

  private:
    // Helpers ---------------------------------------------------------------
    bool wait_for_frames(std::vector<synapse::BroadbandFrame> &frames, size_t frames_to_read);
    int detect_dropped_frames(uint64_t last_seq, uint64_t current_seq);

    // Data preprocessing helpers -----------------------------------------------
    void filter_window(std::vector<std::vector<int32_t>> &ms_window);
    std::vector<int32_t> pad_data(std::vector<int32_t> &data, size_t ch_idx);
    std::vector<std::vector<float>> apply_car(const std::vector<std::vector<int32_t>> &ms_window, const size_t& n_arrays, const size_t& n_channels_per_array);

    // Feature extraction helpers -------------------------------------------
    std::vector<float> calc_spike_bandpower(const std::vector<std::vector<float>> &data, const float& clip_thresh);
    std::vector<int16_t> calc_threshold_crossings(const std::vector<std::vector<float>> &data, const std::vector<float> &thresholds);
    
    std::vector<float> compute_thresholds(const std::vector<std::vector<float>> &data, const float& thresh_mult);

    bool calibrate_parameters(const std::vector<std::vector<float>>& broadband_calibration_data, 
                              const std::vector<std::vector<float>>& binned_feat_calib_data);
    // Feature normalization helpers -------------------------------------------
    std::vector<std::vector<float>> normalize_rnn_input_features(const double& zscore_clip);
    
    bool send_features_to_tap(const std::vector<std::vector<float>> &feat_vec, const std::string &tap_name);

    // ----------------------------------------------------------------------

    // Keep track of last sequence number to detect dropouts
    uint64_t last_sequence_number_ = 0;

    // Rate-limiter for phoneme output. Feature tensors are pushed every bin.
    synapse::Timer publish_rate_limiter_;

    // --- Parameters --------------------------------------------------------
    const float bin_size_ms_ = 20.0f; // 20-ms bins to match Python pipeline

    const float spike_power_clip = 50000.0f; 
    static constexpr float kThresholdMult = -4.5;

    // Per-channel dynamic state -------------------------------------------------
    std::vector<float> thresholds_uv_;     // negative threshold in µV (size = num_channels)

    // DSP filters – created after we know channel count / sample rate
    static constexpr int kFilterOrder = 4;
    std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters_;

    // Filter Parameters ---------------------------------------------------------------
    const float low_cut_hz_  = 250.0f;
    const float high_cut_hz_ = 5000.0f;

    // Calibration duration before thresholds are frozen (seconds)
    const float calibration_seconds_ = 1.0f;
    std::vector<std::vector<float>> broadband_calibration_buffer;
    std::vector<std::vector<float>> binned_feat_calibration_buffer;
    bool calibration_finished_ = false; // have we computed per-channel thresholds?

    // Array-wise common average referencing (4 arrays × 64 channels)
    static constexpr size_t kChannelsPerArray = 64;
    static constexpr size_t kNumArrays       = 4;

    // Rolling z-score & smoothing ---------------------------------------------
    std::vector<float> feature_means_;
    std::vector<float> feature_stddevs_;

    // Causal Gaussian smoothing (σ = 2 bins, kernel length 15, 160 ms lag)
    static constexpr size_t kSmoothingKernelLen = 15;
    std::vector<float> smoothing_kernel_;
    std::deque<std::vector<float>> smoothing_buffer_;

    // Buffers for 1-ms zero-phase filtering ----------------------------------
    std::vector<std::vector<int32_t>> prev_window_;   // last 30 centred samples per channel
    size_t sample_idx_in_window_ = 0;

    const float zscore_clip_ = 10.0f;

    // Sample rate (Hz) determined at runtime
    float sample_rate_hz_ = 30000.0f;

    // ----------------- Patch batching for RNN -----------------------------
    static constexpr size_t kPatchSizeBins   = 14; // 14 bins = 280 ms history
    static constexpr size_t kPatchStrideBins = 4;  // emit every 4 bins (80 ms)
    std::deque<std::vector<float>> patch_window_;  // store last kPatchSizeBins smoothed bins
    size_t stride_bins_counter_ = 0;
  };
} // namespace app

// Entrypoint ---------------------------------------------------------------
int main(const int argc, const char **argv); 