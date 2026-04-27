// ============================================================================
// Phase 4.1 smoke test — AsrCudaEngine init.
//
// Usage:
//   test_qwen_asr_cuda_init <audio_encoder_gguf> <text_decoder_gguf>
//
// Init-only — does not call transcribe(). Prints the audio encoder
// hyperparameters + text decoder ready flag and exits 0 on success.
// ============================================================================

#include "asr_cuda_engine.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: test_qwen_asr_cuda_init <audio_encoder_gguf> "
                "<text_decoder_gguf>\n");
        return 2;
    }

    ominix_cuda::AsrCudaParams params;
    params.audio_encoder_gguf = argv[1];
    params.text_decoder_gguf  = argv[2];
    params.device             = 0;

    ominix_cuda::AsrCudaEngine eng;
    if (!eng.init(params)) {
        fprintf(stderr, "[smoke] AsrCudaEngine::init FAILED\n");
        return 1;
    }

    fprintf(stderr,
            "[smoke] Phase 4.1 ASR scaffold init PASS  "
            "audio_encoder.ready=%d  text_decoder.ready=%d  "
            "audio_encoder.layers=%d  audio_encoder.d_model=%d  "
            "audio_encoder.heads=%d  audio_encoder.out=%d\n",
            (int)eng.audio_encoder().is_ready(),
            (int)eng.text_decoder().is_ready(),
            eng.audio_encoder().encoder_layers(),
            eng.audio_encoder().d_model(),
            eng.audio_encoder().encoder_heads(),
            eng.audio_encoder().output_dim());
    return 0;
}
