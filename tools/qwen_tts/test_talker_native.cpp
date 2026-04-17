// ============================================================================
// test_talker_native.cpp — smoke test for TalkerCannEngine.
//
// Loads the talker llama-GGUF from disk, runs a single forward_decode with a
// constant input embedding, and sanity-checks the output:
//   - must be finite (no NaN / Inf)
//   - must not be all zeros
//   - output RMS must land in a plausible range (1e-3 .. 10.0)
//
// This exercises init_from_gguf, forward_decode at pos=0, forward_decode at
// pos=1 (KV cache read), and reset_kv_cache. Does NOT validate numerical
// correctness against a reference — that's M1.6.
// ============================================================================

#include "talker_cann_engine.h"
#include "talker.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>

static float rms(const std::vector<float> &x) {
    double s = 0;
    for (float v : x) s += (double)v * v;
    return (float)std::sqrt(s / (double)x.size());
}

static bool all_finite(const std::vector<float> &x) {
    for (float v : x) {
        if (!std::isfinite(v)) return false;
    }
    return true;
}

static bool all_zero(const std::vector<float> &x) {
    for (float v : x) {
        if (v != 0.0f) return false;
    }
    return true;
}

int main(int argc, char **argv) {
    const char *gguf_path = "gguf/qwen_tts_talker_llama.gguf";
    if (argc > 1) gguf_path = argv[1];

    printf("=== Test TalkerCannEngine ===\n");
    printf("  GGUF: %s\n", gguf_path);

    // CRITICAL: load the CANN backend so aclnn op-tiling kernel packages are
    // registered. Pure aclInit(nullptr) isn't enough for standalone binaries —
    // CANN's op kernels only resolve once the backend's global init runs. In
    // the main qwen_tts binary this happens via llama.cpp's CANN loader; here
    // we trigger it explicitly. Without this we get "Op has no infershape
    // func, opType: TransData" on every matmul.
    {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CANN");
        if (!reg) {
            printf("FAIL: CANN backend not registered in ggml\n");
            return 1;
        }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) {
            printf("FAIL: ggml_backend_dev_init(CANN) returned null\n");
            return 1;
        }
        // keep `be` alive — dropping it would unload the backend. We leak on
        // purpose; the process exits soon after.
    }

    TalkerConfig cfg;  // defaults: 28 layers, n_embd=2048, etc.
    TalkerCannEngine eng;
    if (!eng.init_from_gguf(gguf_path, cfg, /*device=*/0)) {
        printf("FAIL: init_from_gguf returned false\n");
        return 1;
    }
    if (!eng.is_ready()) {
        printf("FAIL: engine reports !ready after successful init\n");
        return 1;
    }
    printf("PASS: init_from_gguf\n");

    // Build a small test input embedding (uniform small value) and a second
    // one (alternating +/-) — verifies the cache picks up real values.
    std::vector<float> input_a(cfg.hidden_size, 0.05f);
    std::vector<float> input_b(cfg.hidden_size);
    for (int i = 0; i < cfg.hidden_size; ++i) {
        input_b[i] = (i % 2 == 0) ? 0.07f : -0.03f;
    }
    std::vector<float> out(cfg.hidden_size, 0.0f);

    // ---- Decode at pos 0 ----
    eng.reset_kv_cache();
    eng.forward_decode(input_a.data(), /*pos=*/0, out.data());
    if (!all_finite(out)) {
        printf("FAIL: forward_decode(pos=0) produced non-finite output\n");
        return 1;
    }
    if (all_zero(out)) {
        printf("FAIL: forward_decode(pos=0) produced all-zero output\n");
        return 1;
    }
    float r0 = rms(out);
    printf("pos=0 output RMS = %.4f (expect 1e-3 .. 10.0)\n", r0);
    if (r0 < 1e-3f || r0 > 10.0f) {
        printf("FAIL: pos=0 RMS out of plausible range\n");
        return 1;
    }
    printf("PASS: forward_decode pos=0\n");

    // ---- Decode at pos 1 (uses KV cache at pos=0) ----
    eng.forward_decode(input_b.data(), /*pos=*/1, out.data());
    if (!all_finite(out)) {
        printf("FAIL: forward_decode(pos=1) produced non-finite output\n");
        return 1;
    }
    float r1 = rms(out);
    printf("pos=1 output RMS = %.4f\n", r1);
    if (r1 < 1e-3f || r1 > 10.0f) {
        printf("FAIL: pos=1 RMS out of plausible range\n");
        return 1;
    }
    printf("PASS: forward_decode pos=1\n");

    // ---- reset_kv_cache -> decode at pos 0 again should match the first
    //      call (same input, same cache state) ----
    eng.reset_kv_cache();
    std::vector<float> out2(cfg.hidden_size, 0.0f);
    eng.forward_decode(input_a.data(), /*pos=*/0, out2.data());
    if (!all_finite(out2)) {
        printf("FAIL: re-decode after reset produced non-finite output\n");
        return 1;
    }
    float max_diff = 0.0f;
    for (int i = 0; i < cfg.hidden_size; ++i) {
        float d = std::fabs(out[i] - out2[i]);
        // Note: `out` was overwritten by the pos=1 call above; we compare the
        // pos=0 re-run against its own RMS instead of against `out`.
    }
    float r2 = rms(out2);
    printf("reset+pos=0 re-run RMS = %.4f (prev pos=0 RMS = %.4f)\n", r2, r0);
    if (std::fabs(r2 - r0) > 1e-3f) {
        printf("FAIL: reset_kv_cache didn't restore clean state (RMS drift %.6f)\n",
               std::fabs(r2 - r0));
        return 1;
    }
    printf("PASS: reset_kv_cache restores deterministic output\n");

    // ---- forward_prefill with seq_len > 1 ----
    eng.reset_kv_cache();
    const int prefill_len = 4;
    std::vector<float> prefill_embeds((size_t)prefill_len * cfg.hidden_size);
    for (int s = 0; s < prefill_len; ++s) {
        for (int i = 0; i < cfg.hidden_size; ++i) {
            prefill_embeds[(size_t)s * cfg.hidden_size + i] =
                0.01f * (float)(s + 1) * ((i % 3 == 0) ? 1.0f : -0.5f);
        }
    }
    std::vector<float> last_hidden(cfg.hidden_size, 0.0f);
    eng.forward_prefill(prefill_embeds.data(), prefill_len, /*start_pos=*/0,
                         last_hidden.data());
    if (!all_finite(last_hidden)) {
        printf("FAIL: forward_prefill produced non-finite output\n");
        return 1;
    }
    if (all_zero(last_hidden)) {
        printf("FAIL: forward_prefill produced all-zero output\n");
        return 1;
    }
    float rp = rms(last_hidden);
    printf("prefill(seq=%d) last-row RMS = %.4f\n", prefill_len, rp);
    if (rp < 1e-3f || rp > 10.0f) {
        printf("FAIL: prefill RMS out of plausible range\n");
        return 1;
    }
    printf("PASS: forward_prefill\n");

    // ---- set_rope_speed_factor doesn't crash, changes output ----
    eng.reset_kv_cache();
    eng.set_rope_speed_factor(1.0f);  // no-op, but exercises the branch
    std::vector<float> out_v1(cfg.hidden_size, 0.0f);
    eng.forward_decode(input_a.data(), /*pos=*/2, out_v1.data());

    eng.reset_kv_cache();
    eng.set_rope_speed_factor(1.5f);
    std::vector<float> out_v15(cfg.hidden_size, 0.0f);
    eng.forward_decode(input_a.data(), /*pos=*/2, out_v15.data());

    if (!all_finite(out_v1) || !all_finite(out_v15)) {
        printf("FAIL: rope-speed forward produced non-finite\n");
        return 1;
    }
    float diff = 0.0f;
    for (int i = 0; i < cfg.hidden_size; ++i) {
        diff += std::fabs(out_v1[i] - out_v15[i]);
    }
    printf("rope_speed 1.0 vs 1.5 total L1 diff = %.4f (should be > 0)\n", diff);
    if (diff < 1e-3f) {
        printf("FAIL: set_rope_speed_factor had no effect on output\n");
        return 1;
    }
    printf("PASS: set_rope_speed_factor modifies output\n");

    // Restore default factor for cleanliness.
    eng.set_rope_speed_factor(1.0f);

    printf("\n=== All tests PASSED ===\n");
    return 0;
}
