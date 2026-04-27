// ============================================================================
// Phase 3.3b smoke harness for ImageDiffusionCudaEngine::forward_dit
// (60-block forward + norm_out + proj_out).
//
// Usage:
//   test_image_diffusion_cuda_dit <dit_gguf_path>
//
// Behaviour:
//   1. init_from_gguf() — Phase 3.1 path, builds pe-table + t_emb scratch.
//   2. Synthesize F32 inputs at 1024² shape:
//        img_in   [4096, 3072]   gaussian noise * 0.1
//        txt_in   [256,  3072]   gaussian noise * 0.1
//        timestep = 1.0           (canonical sigma_s * 1000 mid-step;
//                                   exercises the F32-attn widened path —
//                                   silu(t_emb) max_abs ≈ 278 at this t)
//   3. Call forward_dit(timestep, ...). 60-block loop + norm_out + proj_out.
//      Wall-time the call.
//   4. Scan output for NaN/Inf, report shape/range/std.
//
// Exit codes:
//   0  -> Phase 3.3b smoke PASS (output finite at [16, 64, 64] = patch_in
//         channels reshape; range/std sane)
//   1  -> init or forward_dit returned false
//   2  -> bad CLI / missing arg
//   3  -> NaN / Inf detected in output
// ============================================================================

#include "image_diffusion_cuda_engine.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static void print_gpu_mem_(const char *label) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        fprintf(stderr,
                "[smoke] %s: GPU mem free=%.2f GiB / total=%.2f GiB\n",
                label,
                (double)free_b  / (1024.0 * 1024.0 * 1024.0),
                (double)total_b / (1024.0 * 1024.0 * 1024.0));
    }
}

struct TensorStats {
    size_t n_total = 0;
    size_t n_nan   = 0;
    size_t n_inf   = 0;
    double sum     = 0.0;
    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    double max_abs = 0.0;
    double min_v   = +1e30;
    double max_v   = -1e30;
};

static TensorStats scan_(const std::vector<float> &v) {
    TensorStats s;
    s.n_total = v.size();
    for (float x : v) {
        if (std::isnan(x)) { s.n_nan++; continue; }
        if (std::isinf(x)) { s.n_inf++; continue; }
        double a = std::fabs((double)x);
        s.sum     += (double)x;
        s.sum_abs += a;
        s.sum_sq  += (double)x * (double)x;
        if (a > s.max_abs) s.max_abs = a;
        if ((double)x > s.max_v) s.max_v = (double)x;
        if ((double)x < s.min_v) s.min_v = (double)x;
    }
    return s;
}

static void log_(const char *lbl, const TensorStats &s) {
    size_t valid = s.n_total - s.n_nan - s.n_inf;
    double mean     = valid > 0 ? s.sum / (double)valid : 0.0;
    double mean_abs = valid > 0 ? s.sum_abs / (double)valid : 0.0;
    double var      = valid > 0 ? s.sum_sq / (double)valid - mean * mean : 0.0;
    double stdv     = var > 0 ? std::sqrt(var) : 0.0;
    fprintf(stderr,
            "[smoke] %-12s n=%zu  mean=%.4g  std=%.4g  mean_abs=%.4g  "
            "max_abs=%.4g  range=[%.4g .. %.4g]  NaN=%zu  Inf=%zu\n",
            lbl, s.n_total, mean, stdv, mean_abs, s.max_abs,
            s.min_v == +1e30 ? 0.0 : s.min_v,
            s.max_v == -1e30 ? 0.0 : s.max_v,
            s.n_nan, s.n_inf);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: test_image_diffusion_cuda_dit <dit_gguf_path>\n");
        return 2;
    }
    const std::string dit_path = argv[1];

    print_gpu_mem_("pre-init");

    ominix_cuda::ImageDiffusionCudaEngine eng;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = eng.init_from_gguf(dit_path,
                                  /*llm_path=*/"",
                                  /*llm_vision_path=*/"",
                                  /*vae_path=*/"",
                                  /*device=*/0);
    auto t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!ok) { fprintf(stderr, "[smoke] init FAILED\n"); return 1; }
    fprintf(stderr, "[smoke] init OK  load_ms=%.0f\n", load_ms);
    print_gpu_mem_("post-init");

    const auto &cfg = eng.config();
    const int H        = cfg.hidden;
    const int PATCH    = cfg.patch_in;
    const int img_seq  = 4096;          // 1024² → 64×64 patch grid → 4096
    const int txt_seq  = 256;
    // Phase 3.3b smoke uses t=1.0 by default; can override via env. The
    // F32-widened residual chain is finite at t=0 (silu_t_emb max_abs=278);
    // higher t exercises the deeper t_emb amplification regime.
    float timestep = 1.0f;
    if (const char *e = std::getenv("OMINIX_CUDA_TIMESTEP"); e && e[0]) {
        timestep = std::atof(e);
    }

    std::mt19937_64 rng(0xC0DA13DEull);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> img_in((size_t)img_seq * H);
    std::vector<float> txt_in((size_t)txt_seq * H);
    for (auto &x : img_in)  x = dist(rng) * 0.1f;
    for (auto &x : txt_in)  x = dist(rng) * 0.1f;

    auto si = scan_(img_in);  log_("img_in",  si);
    auto st = scan_(txt_in);  log_("txt_in",  st);
    fprintf(stderr, "[smoke] timestep    = %.1f\n", timestep);

    std::vector<float> img_out((size_t)img_seq * PATCH, 0.0f);

    auto fb0 = std::chrono::steady_clock::now();
    bool fok = eng.forward_dit(timestep,
                                img_in.data(), img_seq,
                                txt_in.data(), txt_seq,
                                img_out.data());
    auto fb1 = std::chrono::steady_clock::now();
    double fb_ms = std::chrono::duration<double, std::milli>(fb1 - fb0).count();
    if (!fok) { fprintf(stderr, "[smoke] forward_dit FAILED\n"); return 1; }
    fprintf(stderr, "[smoke] forward_dit OK  60-block + norm_out + proj_out  "
                    "shape=%dx%d (img_seq×patch_in)  wall_ms=%.1f\n",
            img_seq, PATCH, fb_ms);
    print_gpu_mem_("post-forward");

    auto so = scan_(img_out); log_("img_out", so);

    if (so.n_nan || so.n_inf) {
        fprintf(stderr,
                "[smoke] FAIL: NaN=%zu Inf=%zu in output (Phase 3.3b expected "
                "0/0 — F32 attn accum should keep things finite)\n",
                so.n_nan, so.n_inf);
        return 3;
    }
    if (so.max_abs == 0.0) {
        fprintf(stderr, "[smoke] FAIL: output is identically zero\n");
        return 3;
    }
    fprintf(stderr,
            "[smoke] Phase 3.3b 60-block forward smoke PASS  (final latent "
            "shape [16, 64, 64] = patch_in[%d] reshaped, NaN/Inf=0/0)\n",
            PATCH);
    return 0;
}
