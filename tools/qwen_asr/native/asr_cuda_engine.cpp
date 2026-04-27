// ============================================================================
// AsrCudaEngine — Phase 4.1 scaffold.
//
// Drives MelSpectrogram (CPU) + AudioEncoderCudaEngine (CUDA) +
// TalkerCudaEngine (CUDA). Phase 4.1 covers init only; transcribe() returns
// an empty string and prints a Phase 4.4 stub message.
// ============================================================================

#include "asr_cuda_engine.h"
#include "../../qwen_tts/talker.h"   // TalkerConfig

#include <cstdio>

namespace ominix_cuda {

AsrCudaEngine::AsrCudaEngine() = default;
AsrCudaEngine::~AsrCudaEngine() = default;

bool AsrCudaEngine::init(const AsrCudaParams &params) {
    max_new_tokens_ = params.max_new_tokens;

    // Optional: load mel filterbank from .npy for exact-match parity with
    // the Python reference.
    if (!params.mel_filters_path.empty()) {
        if (!mel_spec_.load_mel_filterbank(params.mel_filters_path)) {
            fprintf(stderr,
                    "[asr_cuda] WARN: failed to load mel filterbank from %s "
                    "— falling back to slaney mel filters\n",
                    params.mel_filters_path.c_str());
        }
    }

    if (!audio_encoder_.init_from_gguf(params.audio_encoder_gguf,
                                       params.device)) {
        fprintf(stderr, "[asr_cuda] audio encoder init FAILED\n");
        return false;
    }

    // Text decoder: reuse the Phase 2.x TalkerCudaEngine verbatim. The
    // Qwen3-ASR text decoder shares the same 28-layer transformer body and
    // tokenizer space as the TTS talker.
    TalkerConfig tcfg;  // defaults: 28L / 16Q / 8KV / 2048 hidden / 6144 inter
    if (!text_decoder_.init_from_gguf(params.text_decoder_gguf, tcfg,
                                      params.device)) {
        fprintf(stderr, "[asr_cuda] text decoder init FAILED\n");
        return false;
    }

    fprintf(stderr,
            "[asr_cuda] init OK  audio_encoder.layers=%d  audio_encoder.d_model=%d  "
            "text_decoder.ready=%d\n",
            audio_encoder_.encoder_layers(), audio_encoder_.d_model(),
            (int)text_decoder_.is_ready());

    ready_ = true;
    return true;
}

std::string AsrCudaEngine::transcribe(const float * /*wav_samples*/,
                                      int /*n_samples*/) {
    fprintf(stderr,
            "[asr_cuda] transcribe() is a Phase 4.4 stub — returning empty "
            "string in Phase 4.1\n");
    return std::string();
}

} // namespace ominix_cuda
