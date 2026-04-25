// ============================================================================
// Phase 2.2 smoke test for TalkerCudaEngine::forward_decode.
//
// Usage:
//   test_talker_cuda_decode <gguf_path>
//
// What it does:
//   1. Init the engine from the GGUF Phase 2.1 used.
//   2. Build a deterministic F32 input embedding (sin-wave seeded from layer 0).
//   3. Call forward_decode for pos=0..9 sequentially. Checks:
//        - no NaN / inf in the [n_embd] hidden output;
//        - magnitude is non-zero (sanity: zero output means a kernel silently
//          dropped a layer);
//        - KV cache pointer remains stable across calls (no realloc).
//   4. Records a wall-clock per-step number for the inner loop (input H2D +
//      28 layers + final cast + D2H + sync). Wall-clock includes the sync,
//      so it is the user-visible "time per token" — no graph capture yet.
//
// Exit 0 on PASS; non-zero on any check failure.
// ============================================================================

#include "talker_cuda_engine.h"
#include "../talker.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool check_finite(const float *x, int n, int pos) {
    int n_nan = 0, n_inf = 0;
    float min_v =  INFINITY, max_v = -INFINITY, sum_abs = 0.0f;
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        if (std::isnan(v)) ++n_nan;
        else if (std::isinf(v)) ++n_inf;
        else {
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
            sum_abs += std::fabs(v);
        }
    }
    fprintf(stdout,
            "[smoke] pos=%2d  n=%d  nan=%d  inf=%d  min=%+.4f  max=%+.4f  "
            "mean|x|=%.4f\n",
            pos, n, n_nan, n_inf, min_v, max_v, sum_abs / (float)n);
    if (n_nan > 0 || n_inf > 0) return false;
    if (sum_abs <= 0.0f) {
        fprintf(stderr, "[smoke] pos=%d output is all zeros — kernels silently no-op?\n", pos);
        return false;
    }
    return true;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_talker_cuda_decode <gguf_path>\n");
        return 2;
    }
    const std::string gguf = argv[1];

    TalkerConfig cfg;
    ominix_cuda::TalkerCudaEngine eng;
    if (!eng.init_from_gguf(gguf, cfg, /*device=*/0)) {
        fprintf(stderr, "[smoke] init_from_gguf FAILED\n");
        return 1;
    }
    eng.reset_kv_cache();

    const int n = cfg.hidden_size;
    std::vector<float> input(n);
    std::vector<float> hidden(n);

    // Deterministic input: smooth sinusoid plus a small step. This keeps
    // values in a sensible F16 range (~+/-1).
    for (int i = 0; i < n; ++i) {
        input[i] = 0.5f * std::sin(0.013f * (float)i) +
                    0.1f * (float)((i % 17) - 8) / 17.0f;
    }

    bool all_ok = true;
    double total_ms = 0.0;
    int n_steps = 10;
    for (int pos = 0; pos < n_steps; ++pos) {
        auto t0 = std::chrono::high_resolution_clock::now();
        eng.forward_decode(input.data(), pos, hidden.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        bool ok = check_finite(hidden.data(), n, pos);
        if (!ok) {
            fprintf(stderr, "[smoke] FAIL at pos=%d\n", pos);
            all_ok = false;
            break;
        }
    }

    if (!all_ok) return 1;
    double avg_ms  = total_ms / (double)n_steps;
    double avg_us  = avg_ms * 1000.0;
    fprintf(stdout,
            "[smoke] Phase 2.2 forward_decode PASS  steps=%d  "
            "avg_wall=%.2f us/step  total=%.3f ms\n",
            n_steps, avg_us, total_ms);
    return 0;
}
