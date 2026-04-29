// ============================================================================
// Phase 3.3c E2E harness — denoise() → final latent.
//
// Mirrors Ascend's denoise_full smoke flow with pre-computed text encoder
// hidden state from a CLI Ascend run (see §5.5.28). VAE decode is performed
// externally — this binary only emits /tmp/qie_3p3c_latent.f32.bin which
// the caller pipes through a Python diffusers AutoencoderKL or rsync's to
// a working VAE-capable backend.
//
// Usage:
//   test_image_diffusion_cuda_e2e <dit_gguf_path>
//
// Reads from /tmp:
//   qie_3p3c_noised_init.f32.bin  F32 [128, 128, 16, 1]    1024² noised init
//   qie_3p3c_cond_text.f32.bin    F32 [3584, 212, 1, 1]    cond text encoder
//
// Writes:
//   /tmp/qie_3p3c_latent.f32.bin  F32 [128, 128, 16, 1]    final latent
// ============================================================================

#include "image_diffusion_cuda_engine.h"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static long file_size_(const char *path) {
    FILE *f = std::fopen(path, "rb");
    if (!f) return -1;
    std::fseek(f, 0, SEEK_END);
    long s = std::ftell(f);
    std::fclose(f);
    return s;
}

static bool slurp_(const char *path, std::vector<float> &out, size_t expected_n) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[e2e] open %s FAILED\n", path);
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if ((size_t)bytes != expected_n * sizeof(float)) {
        fprintf(stderr, "[e2e] %s size %ld != expected %zu\n",
                path, bytes, expected_n * sizeof(float));
        std::fclose(f);
        return false;
    }
    out.assign(expected_n, 0.0f);
    size_t nread = std::fread(out.data(), sizeof(float), expected_n, f);
    std::fclose(f);
    return nread == expected_n;
}

static void scan_(const char *lbl, const std::vector<float> &v) {
    size_t n = v.size(), nan_c = 0, inf_c = 0;
    double sum = 0, sum_sq = 0, max_abs = 0;
    double mn = +1e30, mx = -1e30;
    for (float x : v) {
        if (std::isnan(x)) { nan_c++; continue; }
        if (std::isinf(x)) { inf_c++; continue; }
        sum    += (double)x;
        sum_sq += (double)x * (double)x;
        double a = std::fabs((double)x);
        if (a > max_abs) max_abs = a;
        if ((double)x < mn) mn = (double)x;
        if ((double)x > mx) mx = (double)x;
    }
    size_t valid = n - nan_c - inf_c;
    double mean = valid ? sum / valid : 0.0;
    double var  = valid ? sum_sq / valid - mean * mean : 0.0;
    double stdv = var > 0 ? std::sqrt(var) : 0.0;
    fprintf(stderr,
            "[e2e] %-22s  n=%zu  mean=%.4g  std=%.4g  max_abs=%.4g  "
            "range=[%.4g .. %.4g]  NaN=%zu Inf=%zu\n",
            lbl, n, mean, stdv, max_abs,
            (mn == +1e30 ? 0.0 : mn), (mx == -1e30 ? 0.0 : mx),
            nan_c, inf_c);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_image_diffusion_cuda_e2e <dit_gguf_path>\n");
        return 2;
    }
    const std::string dit_path = argv[1];

    // Fixture shape via env (default 1024² → 128×128).
    int W_lat = 128, H_lat = 128;
    if (const char *e = std::getenv("OMINIX_CUDA_LAT_W"); e && e[0]) W_lat = std::atoi(e);
    if (const char *e = std::getenv("OMINIX_CUDA_LAT_H"); e && e[0]) H_lat = std::atoi(e);
    const int C_lat = 16;
    const int joint_dim = 3584;
    const size_t latent_n = (size_t)W_lat * H_lat * C_lat;

    // Discover txt_seq from cond_text file size: bytes / (joint_dim * 4).
    long ct_bytes = file_size_("/tmp/qie_3p3c_cond_text.f32.bin");
    if (ct_bytes <= 0) {
        fprintf(stderr, "[e2e] /tmp/qie_3p3c_cond_text.f32.bin missing\n");
        return 1;
    }
    if (ct_bytes % (joint_dim * 4) != 0) {
        fprintf(stderr, "[e2e] cond_text bytes %ld not multiple of %d\n",
                ct_bytes, joint_dim * 4);
        return 1;
    }
    const int txt_seq = (int)(ct_bytes / (joint_dim * 4));
    const size_t txt_n = (size_t)txt_seq * joint_dim;
    fprintf(stderr, "[e2e] cond_text shape: [txt_seq=%d, joint_dim=%d]\n",
            txt_seq, joint_dim);

    std::vector<float> noised_init, cond_text, ref_latent;
    if (!slurp_("/tmp/qie_3p3c_noised_init.f32.bin", noised_init, latent_n)) return 1;
    if (!slurp_("/tmp/qie_3p3c_cond_text.f32.bin",  cond_text,   txt_n))    return 1;
    bool has_ref = false;
    if (const char *e = std::getenv("OMINIX_CUDA_USE_REF"); e && e[0] == '1') {
        if (slurp_("/tmp/qie_3p3c_ref_latent.f32.bin", ref_latent, latent_n)) {
            has_ref = true;
        } else {
            fprintf(stderr, "[e2e] ref requested but slurp failed; continuing without\n");
        }
    }
    scan_("noised_init", noised_init);
    scan_("cond_text",   cond_text);
    if (has_ref) scan_("ref_latent", ref_latent);

    // Sigma schedule — DiscreteFlowDenoiser w/ shift=3 (QIE-Edit canonical).
    //   t_max = 999 (TIMESTEPS-1)
    //   step  = t_max / (n-1)
    //   for i in [0,n):  t = t_max - step*i;   sigma_i = time_snr_shift(3, (t+1)/1000)
    //   sigma_n = 0     (final)
    // time_snr_shift(α, t) = α*t / (1 + (α-1)*t)
    int n_steps = 20;
    if (const char *e = std::getenv("OMINIX_CUDA_NSTEPS"); e && e[0]) {
        n_steps = std::atoi(e);
        if (n_steps < 1) n_steps = 1;
    }
    std::vector<float> sigmas(n_steps + 1, 0.0f);
    {
        const float alpha = 3.0f;
        const float t_max = 999.0f;
        if (n_steps == 1) {
            float t = t_max;
            float x = (t + 1.0f) / 1000.0f;
            sigmas[0] = alpha * x / (1.0f + (alpha - 1.0f) * x);
        } else {
            float step = t_max / (float)(n_steps - 1);
            for (int i = 0; i < n_steps; ++i) {
                float t = t_max - step * (float)i;
                float x = (t + 1.0f) / 1000.0f;
                sigmas[i] = alpha * x / (1.0f + (alpha - 1.0f) * x);
            }
        }
        sigmas[n_steps] = 0.0f;
    }
    fprintf(stderr, "[e2e] sigmas[%d]: ", (int)sigmas.size());
    for (size_t i = 0; i < sigmas.size(); ++i) fprintf(stderr, "%.4f ", sigmas[i]);
    fprintf(stderr, "\n");

    ominix_cuda::ImageDiffusionCudaEngine eng;
    auto t_init_0 = std::chrono::steady_clock::now();
    bool ok = eng.init_from_gguf(dit_path);
    auto t_init_1 = std::chrono::steady_clock::now();
    if (!ok) { fprintf(stderr, "[e2e] init FAILED\n"); return 1; }
    double init_ms = std::chrono::duration<double, std::milli>(t_init_1 - t_init_0).count();
    fprintf(stderr, "[e2e] init OK  init_ms=%.0f\n", init_ms);

    std::vector<float> out_latent(latent_n, 0.0f);
    std::vector<double> per_step_ms(n_steps, 0.0);

    auto t_d_0 = std::chrono::steady_clock::now();
    ok = eng.denoise(noised_init.data(),
                      has_ref ? ref_latent.data() : nullptr,
                      W_lat, H_lat, C_lat,
                      cond_text.data(), txt_seq, joint_dim,
                      sigmas.data(), n_steps,
                      out_latent.data(), per_step_ms.data());
    auto t_d_1 = std::chrono::steady_clock::now();
    if (!ok) { fprintf(stderr, "[e2e] denoise FAILED\n"); return 1; }
    double total_s = std::chrono::duration<double>(t_d_1 - t_d_0).count();
    fprintf(stderr, "[e2e] denoise OK  total=%.1fs  avg_step=%.0fms\n",
            total_s, total_s * 1000.0 / n_steps);

    scan_("final_latent", out_latent);

    FILE *f = std::fopen("/tmp/qie_3p3c_latent.f32.bin", "wb");
    if (!f) {
        fprintf(stderr, "[e2e] open out FAILED\n"); return 1;
    }
    std::fwrite(out_latent.data(), sizeof(float), out_latent.size(), f);
    std::fclose(f);
    fprintf(stderr, "[e2e] wrote /tmp/qie_3p3c_latent.f32.bin (%zu F32)\n",
            out_latent.size());
    return 0;
}
