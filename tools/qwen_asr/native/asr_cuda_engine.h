#pragma once
// ============================================================================
// AsrCudaEngine — Phase 4.1 scaffold (top-level orchestration).
//
// Owns one AudioEncoderCudaEngine + one TalkerCudaEngine (text decoder, 28L
// Qwen3 — identical to TTS). The text decoder side reuses the existing
// CUDA kernels from tools/qwen_tts/native unmodified.
//
// Phase 4.1 scope: init from both GGUFs + smoke. Forward is Phase 4.4
// (split prefill in 4.3 first).
//
// Pipeline (full):
//   wav (16kHz F32) → MelSpectrogram (CPU)
//                  → AudioEncoderCudaEngine.encode  (Phase 4.2)
//                  → split prefill                   (Phase 4.3)
//                  → TalkerCudaEngine.forward_*      (existing)
//                  → BPE decode                      (Phase 4.4)
//                  → text
// ============================================================================

#include "audio_encoder_cuda_engine.h"
#include "../../qwen_tts/native/talker_cuda_engine.h"

#include "mel_spectrogram.h"

#include <memory>
#include <string>
#include <vector>

namespace ominix_cuda {

struct AsrCudaParams {
    std::string audio_encoder_gguf;
    std::string text_decoder_gguf;
    std::string vocab_path;        // vocab.json (Phase 4.4 BPE)
    std::string merges_path;       // merges.txt (Phase 4.4 BPE)
    std::string mel_filters_path;  // optional .npy with WhisperFE filters
    int device          = 0;
    int max_new_tokens  = 256;
};

class AsrCudaEngine {
public:
    AsrCudaEngine();
    ~AsrCudaEngine();

    bool init(const AsrCudaParams &params);
    bool is_ready() const { return ready_; }

    // Phase 4.4 forward path (stub returns empty in Phase 4.1).
    std::string transcribe(const float *wav_samples, int n_samples);

    AudioEncoderCudaEngine &audio_encoder() { return audio_encoder_; }
    TalkerCudaEngine       &text_decoder()  { return text_decoder_; }
    MelSpectrogram         &mel_spec()      { return mel_spec_; }

private:
    bool ready_ = false;

    MelSpectrogram          mel_spec_;
    AudioEncoderCudaEngine  audio_encoder_;
    TalkerCudaEngine        text_decoder_;

    int max_new_tokens_ = 256;
};

} // namespace ominix_cuda
