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
#include <cstdint>
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

    // ------------------------------------------------------------------
    // Phase 2.7b: decode() — RVQ -> pre_conv -> 2x upsample -> [4T, 1024].
    // Reuses the codes built above; upgrades the smoke from RVQ-only to the
    // full forward path through the upsample blocks (vocoder = 2.7c).
    // ------------------------------------------------------------------
    printf("[smoke] Phase 2.7b decode() (T=%d -> 4T=%d, latent=%d)\n",
           T, 4 * T, cfg.latent_dim);
    std::vector<float> dec = eng.decode(codes.data(), n_q, T);
    size_t expected_dec = (size_t)4 * T * cfg.latent_dim;
    if (dec.size() != expected_dec) {
        fprintf(stderr, "[smoke] decode bad size: got %zu, want %zu\n",
                dec.size(), expected_dec);
        return 1;
    }
    {
        int dn_nan = 0, dn_inf = 0;
        double dsum = 0.0, dsum2 = 0.0;
        float dvmin = +1e30f, dvmax = -1e30f;
        for (size_t i = 0; i < dec.size(); ++i) {
            float v = dec[i];
            if (std::isnan(v)) ++dn_nan;
            if (std::isinf(v)) ++dn_inf;
            dsum  += v;
            dsum2 += (double)v * v;
            if (v < dvmin) dvmin = v;
            if (v > dvmax) dvmax = v;
        }
        double dn = (double)dec.size();
        double dmean = dsum / dn;
        double dvar  = std::max(0.0, dsum2 / dn - dmean * dmean);
        double dstd  = std::sqrt(dvar);
        printf("[smoke] decode OK: shape=[%d, %d] elems=%zu\n",
               4 * T, cfg.latent_dim, dec.size());
        printf("[smoke] decode stats: nan=%d inf=%d min=%.4f max=%.4f "
               "mean=%.4f std=%.4f\n",
               dn_nan, dn_inf, dvmin, dvmax, dmean, dstd);
        printf("[smoke] decode head[0..8]: ");
        for (int i = 0; i < 8 && (size_t)i < dec.size(); ++i) {
            printf("%.4f ", dec[i]);
        }
        printf("\n");
        if (dn_nan > 0 || dn_inf > 0) {
            fprintf(stderr, "[smoke] FAIL: decode nan/inf\n");
            return 1;
        }
        if (dvmax == 0.0f && dvmin == 0.0f) {
            fprintf(stderr, "[smoke] FAIL: decode all zeros\n");
            return 1;
        }
    }

    // ------------------------------------------------------------------
    // Phase 2.7c: decode_audio() — full pipeline, audio waveform out.
    // ------------------------------------------------------------------
    printf("[smoke] Phase 2.7c decode_audio() (T=%d -> T_audio=%d, sr=%d)\n",
           T, T * cfg.decode_upsample_rate, cfg.output_sample_rate);
    std::vector<float> audio = eng.decode_audio(codes.data(), n_q, T);
    size_t expected_audio = (size_t)T * cfg.decode_upsample_rate;
    if (audio.size() != expected_audio) {
        fprintf(stderr, "[smoke] decode_audio bad size: got %zu, want %zu\n",
                audio.size(), expected_audio);
        return 1;
    }
    {
        int an_nan = 0, an_inf = 0;
        double asum = 0.0, asum2 = 0.0;
        float avmin = +1e30f, avmax = -1e30f;
        for (size_t i = 0; i < audio.size(); ++i) {
            float v = audio[i];
            if (std::isnan(v)) ++an_nan;
            if (std::isinf(v)) ++an_inf;
            asum  += v;
            asum2 += (double)v * v;
            if (v < avmin) avmin = v;
            if (v > avmax) avmax = v;
        }
        double an = (double)audio.size();
        double amean = asum / an;
        double avar  = std::max(0.0, asum2 / an - amean * amean);
        double astd  = std::sqrt(avar);
        printf("[smoke] decode_audio OK: shape=[%zu] elems=%zu\n",
               audio.size(), audio.size());
        printf("[smoke] audio stats: nan=%d inf=%d min=%.4f max=%.4f "
               "mean=%.4f std=%.4f\n",
               an_nan, an_inf, avmin, avmax, amean, astd);
        printf("[smoke] audio head[0..8]: ");
        for (int i = 0; i < 8 && (size_t)i < audio.size(); ++i) {
            printf("%.4f ", audio[i]);
        }
        printf("\n");
        if (an_nan > 0 || an_inf > 0) {
            fprintf(stderr, "[smoke] FAIL: audio nan/inf\n");
            return 1;
        }
        if (avmax == 0.0f && avmin == 0.0f) {
            fprintf(stderr, "[smoke] FAIL: audio all zeros\n");
            return 1;
        }
        // Tanh-bounded check (allow tiny epsilon over 1.0 due to fp roundoff).
        if (avmin < -1.001f || avmax > 1.001f) {
            fprintf(stderr,
                    "[smoke] WARN: audio out of [-1, 1] range (min=%.4f max=%.4f)\n",
                    avmin, avmax);
        }
        if (astd < 1e-3) {
            fprintf(stderr,
                    "[smoke] FAIL: audio is silent (std=%.6f < 1e-3)\n", astd);
            return 1;
        }

        // Save WAV (16-bit PCM) for eyeballing.
        const char *wav_path = "/tmp/qwen_tts_smoke.wav";
        FILE *fp = fopen(wav_path, "wb");
        if (fp) {
            uint32_t sr = cfg.output_sample_rate;
            uint32_t n_samples = (uint32_t)audio.size();
            uint32_t bytes_per_sample = 2;
            uint32_t data_bytes = n_samples * bytes_per_sample;
            uint32_t fmt_chunk_size = 16;
            uint32_t riff_size = 4 + (8 + fmt_chunk_size) + (8 + data_bytes);
            // RIFF header
            fwrite("RIFF", 1, 4, fp);
            fwrite(&riff_size, 4, 1, fp);
            fwrite("WAVE", 1, 4, fp);
            // fmt chunk
            fwrite("fmt ", 1, 4, fp);
            fwrite(&fmt_chunk_size, 4, 1, fp);
            uint16_t fmt = 1; fwrite(&fmt, 2, 1, fp);  // PCM
            uint16_t ch = 1;  fwrite(&ch, 2, 1, fp);    // mono
            fwrite(&sr, 4, 1, fp);
            uint32_t byte_rate = sr * ch * bytes_per_sample;
            fwrite(&byte_rate, 4, 1, fp);
            uint16_t block_align = ch * bytes_per_sample;
            fwrite(&block_align, 2, 1, fp);
            uint16_t bits_per_sample = 16;
            fwrite(&bits_per_sample, 2, 1, fp);
            // data chunk
            fwrite("data", 1, 4, fp);
            fwrite(&data_bytes, 4, 1, fp);
            // Convert float [-1,1] -> int16
            for (size_t i = 0; i < audio.size(); ++i) {
                float v = audio[i];
                if (v >  1.0f) v =  1.0f;
                if (v < -1.0f) v = -1.0f;
                int16_t s = (int16_t)(v * 32767.0f);
                fwrite(&s, 2, 1, fp);
            }
            fclose(fp);
            printf("[smoke] WAV saved: %s (%u samples @ %u Hz)\n",
                    wav_path, n_samples, sr);
        } else {
            fprintf(stderr, "[smoke] WARN: could not open %s for writing\n",
                    wav_path);
        }
    }

    printf("[smoke] PASS\n");
    return 0;
}
