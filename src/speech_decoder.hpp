#pragma once

#include <synapse-app-sdk/app/app.hpp>
#include <synapse-app-sdk/utils/time/time.hpp>
#include <synapse-app-sdk/middleware/conversions.hpp>
#include <synapse-app-sdk/dsp/filter/bandpass.hpp>

#include "api/datatype.pb.h"
#include "api/nodes/broadband_source.pb.h"

#include <vector>
#include <memory>
#include <deque>

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
    bool wait_for_frames(std::vector<synapse::BroadbandFrame> &frames, float bin_size_ms);
    int detect_dropped_frames(uint64_t last_seq, uint64_t current_seq);

    // ----------------------------------------------------------------------
    std::atomic<bool> feature_pipeline_initialized_{false};

    // Keep track of last sequence number to detect dropouts
    uint64_t last_sequence_number_ = 0;

    // Rate-limiter for phoneme output. Feature tensors are pushed every bin.
    synapse::Timer publish_rate_limiter_;

    // --- Parameters --------------------------------------------------------
    const float bin_size_ms_ = 10.0f; // Assumed bin size for now (will match feature extractor)

    bool threshold_initialized_ = false; // have we computed per-channel thresholds?

    // Per-channel dynamic state -------------------------------------------------
    std::vector<float> thresholds_uv_;     // negative threshold in µV (size = num_channels)
    std::vector<float> baseline_accum_;    // accumulate squares for RMS estimation during calibration
    size_t baseline_sample_counter_ = 0;   // how many samples seen so far during calibration

    // DSP filters – created after we know channel count / sample rate
    static constexpr int kFilterOrder = 4;
    std::vector<std::unique_ptr<synapse::BaseFilter>> bandpass_filters_;

    // Parameters ---------------------------------------------------------------
    const float low_cut_hz_  = 200.0f;
    const float high_cut_hz_ = 5000.0f;

    // Calibration duration before thresholds are frozen (seconds)
    const float calibration_seconds_ = 1.0f;
    size_t calibration_required_samples_ = 0; // set after we know sample rate

    // Array-wise common average referencing (4 arrays × 64 channels)
    static constexpr size_t kChannelsPerArray = 64;
    static constexpr size_t kNumArrays       = 4;

    // Rolling z-score window ---------------------------------------------------
    static constexpr size_t kRollingWindowBins = 5; // 5 compressed bins (~100 ms)
    std::deque<std::vector<float>> feature_window_;
    const float zscore_clip_ = 10.0f;

    // Bin compression (aggregate two 10-ms bins into one 20-ms step)
    static constexpr size_t kBinCompressionFactor = 2;
    std::vector<std::vector<float>> compression_buffer_;
  };
} // namespace app

// Entrypoint ---------------------------------------------------------------
int main(const int argc, const char **argv); 