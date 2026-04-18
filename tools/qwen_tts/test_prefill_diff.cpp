// ============================================================================
// test_prefill_diff.cpp — diagnostic driver to compare batched vs iterative
// forward_prefill output on the same synthetic embeddings. Sweeps seq_len
// and reports cos-sim between the two paths.
//
// We cannot simply flip TALKER_PREFILL_BATCHED in the middle of a call, so
// this test assumes a modified engine that exposes two explicit methods:
//   forward_prefill_batched(embeds, seq_len, start_pos, last_hidden)
//   forward_prefill_iterative(embeds, seq_len, start_pos, last_hidden)
// which we instrument directly in talker_cann_engine.cpp.
//
// For now, we work around that by using env var toggling: call
// forward_prefill twice with different env vars set. Not ideal but works.
// ============================================================================

#include "talker_cann_engine.h"
#include "talker.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <random>

static float rms(const std::vector<float> &x) {
    double s = 0;
    for (float v : x) s += (double)v * v;
    return (float)std::sqrt(s / (double)x.size());
}

static float cos_sim(const std::vector<float> &a, const std::vector<float> &b) {
    if (a.size() != b.size()) return 0.0f;
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * b[i];
        na  += (double)a[i] * a[i];
        nb  += (double)b[i] * b[i];
    }
    if (na <= 0 || nb <= 0) return 0.0f;
    return (float)(dot / (std::sqrt(na) * std::sqrt(nb)));
}

int main(int argc, char **argv) {
    const char *gguf_path = "gguf/qwen_tts_talker_llama.gguf";
    const char *input_bin = nullptr;   // optional: raw [seq_len, hidden] F32 file
    if (argc > 1) gguf_path = argv[1];
    if (argc > 2) input_bin = argv[2];

    printf("=== Test prefill batched vs iterative ===\n");
    printf("  GGUF: %s\n", gguf_path);
    if (input_bin) printf("  INPUT: %s\n", input_bin);

    {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CANN");
        if (!reg) { printf("FAIL: CANN backend not registered\n"); return 1; }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { printf("FAIL: ggml init failed\n"); return 1; }
    }

    TalkerConfig cfg;
    TalkerCannEngine eng;
    if (!eng.init_from_gguf(gguf_path, cfg, 0)) {
        printf("FAIL: init_from_gguf\n");
        return 1;
    }

    // If a real-input binary is provided (format: int32 seq_len, int32 hidden,
    // then seq_len * hidden F32s), run the comparison on it and exit.
    if (input_bin) {
        FILE *f = fopen(input_bin, "rb");
        if (!f) { printf("FAIL: cannot open %s\n", input_bin); return 1; }
        int32_t hdr[2] = {0, 0};
        if (fread(hdr, sizeof(int32_t), 2, f) != 2) {
            printf("FAIL: bad header\n"); fclose(f); return 1;
        }
        int seq_len = hdr[0], hidden_size = hdr[1];
        if (hidden_size != cfg.hidden_size) {
            printf("FAIL: hidden size mismatch: file=%d cfg=%d\n",
                   hidden_size, cfg.hidden_size);
            fclose(f); return 1;
        }
        std::vector<float> embeds((size_t)seq_len * hidden_size);
        if (fread(embeds.data(), sizeof(float), embeds.size(), f) != embeds.size()) {
            printf("FAIL: short read\n"); fclose(f); return 1;
        }
        fclose(f);
        printf("Loaded real embeddings: seq_len=%d hidden=%d\n", seq_len,
               hidden_size);

        // input stats
        double sum = 0, sumsq = 0, amin = embeds[0], amax = embeds[0];
        for (float v : embeds) {
            sum += v; sumsq += (double)v * v;
            if (v < amin) amin = v; if (v > amax) amax = v;
        }
        printf("  input rms=%.4f  min=%.4f  max=%.4f\n",
               std::sqrt(sumsq/embeds.size()), amin, amax);

        // Simulate production warmup: forward_prefill with batched, then
        // 5 decode steps (mimicking qwen_tts main.cpp warmup path with
        // max_new_tokens=5).
        eng.reset_kv_cache();
        setenv("TALKER_PREFILL_BATCHED", "1", 1);
        std::vector<float> warmup_hidden(cfg.hidden_size, 0.0f);
        eng.forward_prefill(embeds.data(), seq_len, 0, warmup_hidden.data());
        // Use last input row as a dummy step input (doesn't matter — we
        // just want state side-effects).
        std::vector<float> step_emb(embeds.begin() + (size_t)(seq_len - 1) * hidden_size,
                                     embeds.begin() + (size_t)seq_len * hidden_size);
        std::vector<float> step_out(cfg.hidden_size);
        for (int k = 0; k < 5; ++k) {
            eng.forward_decode(step_emb.data(), seq_len + k, step_out.data());
        }
        unsetenv("TALKER_PREFILL_BATCHED");

        // IMPORTANT: run batched FIRST (cold) then iterative, mirroring a
        // fresh qwen_tts process that enables TALKER_PREFILL_BATCHED=1.
        eng.reset_kv_cache();
        setenv("TALKER_PREFILL_BATCHED", "1", 1);
        std::vector<float> hidden_batch_cold(cfg.hidden_size, 0.0f);
        eng.forward_prefill(embeds.data(), seq_len, 0, hidden_batch_cold.data());
        unsetenv("TALKER_PREFILL_BATCHED");

        eng.reset_kv_cache();
        std::vector<float> hidden_iter(cfg.hidden_size, 0.0f);
        eng.forward_prefill(embeds.data(), seq_len, 0, hidden_iter.data());

        eng.reset_kv_cache();
        setenv("TALKER_PREFILL_BATCHED", "1", 1);
        std::vector<float> hidden_batch_warm(cfg.hidden_size, 0.0f);
        eng.forward_prefill(embeds.data(), seq_len, 0, hidden_batch_warm.data());
        unsetenv("TALKER_PREFILL_BATCHED");

        float cs_warm = cos_sim(hidden_iter, hidden_batch_warm);
        float cs_cold = cos_sim(hidden_iter, hidden_batch_cold);
        float cs_cw = cos_sim(hidden_batch_cold, hidden_batch_warm);
        printf("\nREAL INPUT seq_len=%d  iter_rms=%.4f  batch_cold_rms=%.4f  "
               "batch_warm_rms=%.4f\n",
               seq_len, rms(hidden_iter), rms(hidden_batch_cold),
               rms(hidden_batch_warm));
        printf("  cos(iter, batch_warm)=%.6f  cos(iter, batch_cold)=%.6f  "
               "cos(cold, warm)=%.6f\n", cs_warm, cs_cold, cs_cw);
        printf("iter       first16: ");
        for (int i = 0; i < 16; ++i) printf("%.4f ", hidden_iter[i]);
        printf("\nbatch_cold first16: ");
        for (int i = 0; i < 16; ++i) printf("%.4f ", hidden_batch_cold[i]);
        printf("\nbatch_warm first16: ");
        for (int i = 0; i < 16; ++i) printf("%.4f ", hidden_batch_warm[i]);
        printf("\n=== test done ===\n");
        return 0;
    }

    // Generate synthetic prefill input. Real Talker inputs are embedding-
    // table rows scaled by norm_scale (~sqrt(hidden_size)). Amplitude
    // matters: sigma=0.02 (original) hides the bug (cos-sim 0.999+);
    // sigma matching real embedding range (~1.0) reproduces the cos-sim
    // 0.28 divergence seen end-to-end.
    const float scales[] = {0.02f, 0.1f, 0.5f, 1.0f};
    const int seq_lens[] = {1, 2, 3, 4, 8, 16, 32, 64, 127};
    for (float sigma : scales) {
        printf("\n--- sigma=%.3f ---\n", sigma);
        std::mt19937 rng(42);
        std::normal_distribution<float> gauss(0.0f, sigma);
        for (int si = 0; si < (int)(sizeof(seq_lens) / sizeof(seq_lens[0])); ++si) {
            int seq_len = seq_lens[si];
            std::vector<float> embeds((size_t)seq_len * cfg.hidden_size);
            for (auto &v : embeds) v = gauss(rng);

            // Run iterative first (this is the reference).
            eng.reset_kv_cache();
            unsetenv("TALKER_PREFILL_BATCHED");
            std::vector<float> hidden_iter(cfg.hidden_size, 0.0f);
            eng.forward_prefill(embeds.data(), seq_len, 0, hidden_iter.data());

            // Run batched (set env var).
            eng.reset_kv_cache();
            setenv("TALKER_PREFILL_BATCHED", "1", 1);
            std::vector<float> hidden_batch(cfg.hidden_size, 0.0f);
            eng.forward_prefill(embeds.data(), seq_len, 0, hidden_batch.data());

            float cs = cos_sim(hidden_iter, hidden_batch);
            printf("seq_len=%3d  iter_rms=%.4f  batch_rms=%.4f  cos_sim=%.6f\n",
                   seq_len, rms(hidden_iter), rms(hidden_batch), cs);
        }
        unsetenv("TALKER_PREFILL_BATCHED");
    }

    printf("\n=== test done ===\n");
    return 0;
}
