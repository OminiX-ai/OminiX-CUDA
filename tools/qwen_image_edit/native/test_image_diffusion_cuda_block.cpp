// ============================================================================
// Phase 3.2 smoke harness for ImageDiffusionCudaEngine::forward_block.
//
// Usage:
//   test_image_diffusion_cuda_block <dit_gguf_path>
//
// Behaviour:
//   1. init_from_gguf() (Phase 3.1 path).
//   2. Synthesize F32 inputs at 1024² shape:
//        img_in   [1, 4096, 3072]   gaussian noise * 0.1
//        txt_in   [1,  256, 3072]   gaussian noise * 0.1
//        mod_vec  [12 * 3072]       gaussian noise * 0.05
//   3. Call forward_block(0, ...). Wall-time the call.
//   4. Scan outputs for NaN/Inf, report stats.
//
// Exit codes:
//   0  -> Phase 3.2 smoke PASS (output finite, stats sane)
//   1  -> init or forward_block returned false
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
    double sum_abs = 0.0;
    double max_abs = 0.0;
};

static TensorStats scan_(const std::vector<float> &v) {
    TensorStats s;
    s.n_total = v.size();
    for (float x : v) {
        if (std::isnan(x)) { s.n_nan++; continue; }
        if (std::isinf(x)) { s.n_inf++; continue; }
        double a = std::fabs((double)x);
        s.sum_abs += a;
        if (a > s.max_abs) s.max_abs = a;
    }
    return s;
}

static void log_(const char *lbl, const TensorStats &s) {
    size_t valid = s.n_total - s.n_nan - s.n_inf;
    double mean_abs = valid > 0 ? s.sum_abs / (double)valid : 0.0;
    fprintf(stderr,
            "[smoke] %-12s n=%zu  mean_abs=%.4g  max_abs=%.4g  NaN=%zu  Inf=%zu\n",
            lbl, s.n_total, mean_abs, s.max_abs, s.n_nan, s.n_inf);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: test_image_diffusion_cuda_block <dit_gguf_path>\n");
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
    const int img_seq  = 4096;          // 1024² → 64×64 patch grid → 4096 tokens
    const int txt_seq  = 256;
    const int mod_chunks = 12;

    std::mt19937_64 rng(0xC0DA13DEull);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> img_in((size_t)img_seq * H);
    std::vector<float> txt_in((size_t)txt_seq * H);
    std::vector<float> mod_vec((size_t)mod_chunks * H);
    for (auto &x : img_in)  x = dist(rng) * 0.1f;
    for (auto &x : txt_in)  x = dist(rng) * 0.1f;
    for (auto &x : mod_vec) x = dist(rng) * 0.05f;

    auto si = scan_(img_in);  log_("img_in",  si);
    auto st = scan_(txt_in);  log_("txt_in",  st);
    auto sm = scan_(mod_vec); log_("mod_vec", sm);

    std::vector<float> img_out(img_in.size(), 0.0f);
    std::vector<float> txt_out(txt_in.size(), 0.0f);

    auto fb0 = std::chrono::steady_clock::now();
    bool fok = eng.forward_block(/*block_idx=*/0,
                                  img_in.data(),  img_seq,
                                  txt_in.data(),  txt_seq,
                                  mod_vec.data(),
                                  img_out.data(), txt_out.data());
    auto fb1 = std::chrono::steady_clock::now();
    double fb_ms = std::chrono::duration<double, std::milli>(fb1 - fb0).count();
    if (!fok) { fprintf(stderr, "[smoke] forward_block FAILED\n"); return 1; }
    fprintf(stderr, "[smoke] forward_block OK  block=0  shape=1x%dx%d (img) / 1x%dx%d (txt)  wall_ms=%.1f\n",
            img_seq, H, txt_seq, H, fb_ms);
    print_gpu_mem_("post-forward");

    auto so_img = scan_(img_out); log_("img_out", so_img);
    auto so_txt = scan_(txt_out); log_("txt_out", so_txt);

    if (so_img.n_nan || so_img.n_inf || so_txt.n_nan || so_txt.n_inf) {
        fprintf(stderr, "[smoke] FAIL: NaN / Inf in output\n");
        return 3;
    }
    if (so_img.max_abs == 0.0 || so_txt.max_abs == 0.0) {
        fprintf(stderr, "[smoke] FAIL: output is identically zero\n");
        return 3;
    }
    if (so_img.max_abs > 1.0e6 || so_txt.max_abs > 1.0e6) {
        fprintf(stderr, "[smoke] WARN: output magnitude very large "
                          "(img max_abs=%.4g txt max_abs=%.4g)\n",
                so_img.max_abs, so_txt.max_abs);
    }

    fprintf(stderr, "[smoke] Phase 3.2 1-block 1024² smoke PASS\n");
    return 0;
}
