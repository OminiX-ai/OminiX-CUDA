// ============================================================================
// Phase 3.1 smoke test for ImageDiffusionCudaEngine.
//
// Usage:
//   test_image_diffusion_cuda_init <dit_gguf_path>
//
// Behaviour:
//   - Constructs ImageDiffusionCudaEngine, calls init_from_gguf() against the
//     QIE-Edit-2511 (or 2509) GGUF passed on argv[1], and prints
//     scaffold-init OK with all 60 transformer-block weights uploaded.
//   - Reports n_blocks / n_heads / head_dim / hidden / mlp_inter (must match
//     the contract: 60 / 24 / 128 / 3072 / 12288).
//   - Reports uploaded byte count (HBM headroom check).
//   - Reports non-finite element count (must be zero on a clean GGUF).
//
// Exit codes:
//   0  -> Phase 3.1 init smoke PASS
//   1  -> init_from_gguf failed
//   2  -> bad CLI / missing arg
// ============================================================================

#include "image_diffusion_cuda_engine.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

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

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: test_image_diffusion_cuda_init <dit_gguf_path>\n");
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

    if (!ok) {
        fprintf(stderr, "[smoke] init_from_gguf FAILED\n");
        return 1;
    }

    print_gpu_mem_("post-init");

    const auto &cfg = eng.config();
    fprintf(stderr,
            "[smoke] dims  n_blocks=%d  hidden=%d  n_heads=%d  head_dim=%d  "
            "mlp_inter=%d  mod_dim=%d  text_hidden=%d\n",
            cfg.n_blocks, cfg.hidden, cfg.n_heads, cfg.head_dim,
            cfg.mlp_inter, cfg.mod_dim, cfg.text_hidden);
    fprintf(stderr,
            "[smoke] uploaded=%.2f GiB  nonfinite=%zu  load_ms=%.0f\n",
            (double)eng.total_weight_bytes() / (1024.0 * 1024.0 * 1024.0),
            eng.nonfinite_weight_count(),
            load_ms);

    bool dims_ok = (cfg.n_blocks  == 60)
                && (cfg.hidden    == 3072)
                && (cfg.n_heads   == 24)
                && (cfg.head_dim  == 128)
                && (cfg.mlp_inter == 12288);
    if (!dims_ok) {
        fprintf(stderr,
                "[smoke] FAIL: dims do not match QIE-Edit contract\n");
        return 1;
    }
    if (eng.nonfinite_weight_count() != 0) {
        fprintf(stderr,
                "[smoke] FAIL: %zu non-finite weight elements detected\n",
                eng.nonfinite_weight_count());
        return 1;
    }
    if (!eng.is_ready()) {
        fprintf(stderr, "[smoke] FAIL: engine not ready\n");
        return 1;
    }

    fprintf(stderr, "[smoke] Phase 3.1 init PASS\n");
    return 0;
}
