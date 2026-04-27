// ============================================================================
// Phase 2.7a smoke harness — test_speech_tokenizer_decoder_init
//
// Drives:
//   1. SpeechTokenizerDecoderCudaEngine::init_from_gguf against the decoder
//      GGUF on zgx-3675 (qwen_tts_tokenizer_dec.gguf).
//   2. RVQ decode on a small dummy codes[16, 32] tensor; verifies the output
//      is [512, 32] F32, NaN/Inf free, and reports basic stats (min/max/mean/std).
//
// Phase 2.7a deliverable: this binary builds + runs on GB10 #1, prints
// scaffold-init OK with the configured decoder dims and RVQ output stats,
// and exits 0.
// ============================================================================

#include "speech_tokenizer_decoder_cuda_engine.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <path/to/qwen_tts_tokenizer_dec.gguf> [T=32]\n",
                argv[0]);
        return 2;
    }
    std::string gguf_path = argv[1];
    int T = (argc >= 3) ? std::atoi(argv[2]) : 32;
    if (T <= 0) T = 32;

    ominix_cuda::SpeechTokenizerDecoderCudaEngine eng;
    if (!eng.init_from_gguf(gguf_path, /*device=*/0)) {
        fprintf(stderr, "[smoke] init_from_gguf FAILED\n");
        return 1;
    }
    const auto &cfg = eng.config();
    printf("[smoke] init OK\n");
    printf("[smoke] config:\n");
    printf("        codebook_size       = %d\n", cfg.codebook_size);
    printf("        codebook_dim        = %d\n", cfg.codebook_dim);
    printf("        num_quantizers      = %d\n", cfg.num_quantizers);
    printf("        rvq_out_dim         = %d\n", cfg.rvq_out_dim);
    printf("        hidden_size         = %d\n", cfg.hidden_size);
    printf("        latent_dim          = %d\n", cfg.latent_dim);
    printf("        num_hidden_layers   = %d\n", cfg.num_hidden_layers);
    printf("        num_attention_heads = %d\n", cfg.num_attention_heads);
    printf("        sliding_window      = %d\n", cfg.sliding_window);
    printf("        decoder_dim         = %d\n", cfg.decoder_dim);
    printf("        output_sample_rate  = %d\n", cfg.output_sample_rate);
    printf("        decode_upsample     = %d\n", cfg.decode_upsample_rate);

    // ---- Build dummy codes[num_quantizers, T]. Use a deterministic pattern
    //      that exercises every codebook with a valid index (mod codebook_size).
    int n_q = cfg.num_quantizers;
    std::vector<int> codes((size_t)n_q * T);
    for (int q = 0; q < n_q; ++q) {
        for (int t = 0; t < T; ++t) {
            // Mix q + t with a couple of small primes to give each codebook a
            // distinct distribution of indices.
            int v = (q * 17 + t * 31 + 7) & 0x7FFFFFFF;
            codes[(size_t)q * T + t] = v % cfg.codebook_size;
        }
    }

    std::vector<float> out;
    if (!eng.rvq_decode(codes.data(), n_q, T, out)) {
        fprintf(stderr, "[smoke] rvq_decode FAILED\n");
        return 1;
    }

    size_t expected = (size_t)cfg.rvq_out_dim * T;
    if (out.size() != expected) {
        fprintf(stderr, "[smoke] rvq_decode bad size: got %zu, want %zu\n",
                out.size(), expected);
        return 1;
    }

    // ---- Stats: NaN/Inf check + min/max/mean/std ----
    int n_nan = 0, n_inf = 0;
    double sum = 0.0, sum2 = 0.0;
    float vmin = +1e30f, vmax = -1e30f;
    for (size_t i = 0; i < out.size(); ++i) {
        float v = out[i];
        if (std::isnan(v)) ++n_nan;
        if (std::isinf(v)) ++n_inf;
        sum  += v;
        sum2 += (double)v * v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }
    double n = (double)out.size();
    double mean = sum / n;
    double var  = std::max(0.0, sum2 / n - mean * mean);
    double std  = std::sqrt(var);

    printf("[smoke] rvq_decode OK: shape=[%d, %d] elems=%zu\n",
           cfg.rvq_out_dim, T, out.size());
    printf("[smoke] stats: nan=%d inf=%d min=%.4f max=%.4f mean=%.4f std=%.4f\n",
           n_nan, n_inf, vmin, vmax, mean, std);

    // First 8 values for eyeballing.
    printf("[smoke] head[0..8]: ");
    for (int i = 0; i < 8 && (size_t)i < out.size(); ++i) {
        printf("%.4f ", out[i]);
    }
    printf("\n");

    if (n_nan > 0 || n_inf > 0) {
        fprintf(stderr, "[smoke] FAIL: nan/inf detected\n");
        return 1;
    }
    if (vmax == 0.0f && vmin == 0.0f) {
        fprintf(stderr, "[smoke] FAIL: output is all zeros (suspicious)\n");
        return 1;
    }

    printf("[smoke] PASS\n");
    return 0;
}
