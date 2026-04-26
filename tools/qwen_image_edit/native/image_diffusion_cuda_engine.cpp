// ============================================================================
// Image Diffusion CUDA Engine — Phase 3.1 scaffold + GGUF weight upload.
//
// Phase 3.1 (THIS):
//   - Header full, ctor/dtor lifecycle.
//   - init_from_gguf: open DiT GGUF, validate qwen_image arch, upload every
//     transformer_blocks.<L>.* weight (60 blocks) + global head/tail
//     (img_in / txt_in / txt_norm / time_text_embed / norm_out / proj_out)
//     to device. Source dtypes Q8_0 / Q4_0 / BF16 / F16 / F32 are dequant-ed
//     to F32 on host via ggml's type-trait `to_float` hook (same path as
//     qwen_tts Phase 2.2), then cast to F16 for matmul weights / kept as
//     F32 for biases & norm gammas.
//   - Tracks total uploaded byte count + non-finite element count (smoke
//     metric).
//
// Phase 3.2 — forward_block body. Currently aborts.
// Phase 3.3 — final_proj (norm_out + proj_out). Currently aborts.
// ============================================================================

#include "image_diffusion_cuda_engine.h"
#include "cuda_kernels/dit_kernels.h"

// llama.cpp GGUF helpers (vendored in ominix-cuda).
#include "ggml.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define OMX_CUDA_CHECK(expr)                                                   \
    do {                                                                       \
        cudaError_t _err = (expr);                                             \
        if (_err != cudaSuccess) {                                             \
            fprintf(stderr, "[image_diffusion_cuda] CUDA error %s at %s:%d: %s\n", \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_err));      \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define OMX_CUBLAS_CHECK(expr)                                                 \
    do {                                                                       \
        cublasStatus_t _st = (expr);                                           \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[image_diffusion_cuda] cuBLAS error %s at %s:%d: %d\n", \
                    #expr, __FILE__, __LINE__, (int)_st);                      \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace ominix_cuda {

namespace {

// Tracks aggregate stats from the upload helpers. Reset per init_from_gguf.
struct UploadStats {
    size_t bytes_uploaded   = 0;
    size_t nonfinite_count  = 0;
};

// Pull a tensor out of GGUF as F32 vector (handles F32 / F16 / BF16 / quant).
// Mirrors talker_cuda_engine.cpp::load_gguf_tensor_f32 behaviour: Q8_0 / Q4_0
// / Q5_K and any other quant get dispatched through ggml's type-trait
// `to_float` hook. BF16 is treated as a quantized type by ggml so the same
// hook handles it.
std::vector<float> load_gguf_tensor_f32(ggml_context *ggml_ctx,
                                          const char *name,
                                          size_t expected_elems) {
    ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (!t) {
        fprintf(stderr, "[image_diffusion_cuda] missing tensor: %s\n", name);
        return {};
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr,
                "[image_diffusion_cuda] %s: expected %zu elems, got %zu\n",
                name, expected_elems, n);
        return {};
    }
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const ggml_type_traits *tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float) {
            tt->to_float(t->data, out.data(), (int64_t)n);
        } else {
            fprintf(stderr,
                    "[image_diffusion_cuda] %s: unsupported dtype %d (no to_float trait)\n",
                    name, (int)t->type);
            return {};
        }
    }
    return out;
}

size_t count_nonfinite(const std::vector<float> &v) {
    size_t bad = 0;
    for (float x : v) if (!std::isfinite(x)) ++bad;
    return bad;
}

bool upload_tensor_f16(ggml_context *ggml_ctx, const char *name,
                        size_t expected_elems, void *&dev,
                        UploadStats &stats) {
    std::vector<float> host = load_gguf_tensor_f32(ggml_ctx, name,
                                                    expected_elems);
    if (host.empty()) return false;
    stats.nonfinite_count += count_nonfinite(host);
    std::vector<__half> f16(expected_elems);
    for (size_t i = 0; i < expected_elems; ++i)
        f16[i] = __float2half(host[i]);
    cudaError_t err = cudaMalloc(&dev, expected_elems * sizeof(__half));
    if (err != cudaSuccess) {
        fprintf(stderr, "[image_diffusion_cuda] cudaMalloc(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(dev, f16.data(), expected_elems * sizeof(__half),
                      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[image_diffusion_cuda] cudaMemcpy(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    stats.bytes_uploaded += expected_elems * sizeof(__half);
    return true;
}

bool upload_tensor_f32(ggml_context *ggml_ctx, const char *name,
                        size_t expected_elems, void *&dev,
                        UploadStats &stats) {
    std::vector<float> host = load_gguf_tensor_f32(ggml_ctx, name,
                                                    expected_elems);
    if (host.empty()) return false;
    stats.nonfinite_count += count_nonfinite(host);
    cudaError_t err = cudaMalloc(&dev, expected_elems * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[image_diffusion_cuda] cudaMalloc(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(dev, host.data(), expected_elems * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[image_diffusion_cuda] cudaMemcpy(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    stats.bytes_uploaded += expected_elems * sizeof(float);
    return true;
}

}  // namespace

ImageDiffusionCudaEngine::~ImageDiffusionCudaEngine() {
    auto safe_free = [](void *p) {
        if (p) cudaFree(p);
    };

    for (auto &b : blocks_) {
        safe_free(b.to_q_w);     safe_free(b.to_k_w);
        safe_free(b.to_v_w);     safe_free(b.to_out_0_w);
        safe_free(b.to_q_b);     safe_free(b.to_k_b);
        safe_free(b.to_v_b);     safe_free(b.to_out_0_b);

        safe_free(b.add_q_w);    safe_free(b.add_k_w);
        safe_free(b.add_v_w);    safe_free(b.to_add_out_w);
        safe_free(b.add_q_b);    safe_free(b.add_k_b);
        safe_free(b.add_v_b);    safe_free(b.to_add_out_b);

        safe_free(b.norm_q_w);   safe_free(b.norm_k_w);
        safe_free(b.norm_added_q_w); safe_free(b.norm_added_k_w);

        safe_free(b.img_mlp_0_w); safe_free(b.img_mlp_2_w);
        safe_free(b.img_mlp_0_b); safe_free(b.img_mlp_2_b);
        safe_free(b.txt_mlp_0_w); safe_free(b.txt_mlp_2_w);
        safe_free(b.txt_mlp_0_b); safe_free(b.txt_mlp_2_b);

        safe_free(b.img_mod_w);  safe_free(b.img_mod_b);
        safe_free(b.txt_mod_w);  safe_free(b.txt_mod_b);
    }

    safe_free(img_in_w_);   safe_free(img_in_b_);
    safe_free(txt_in_w_);   safe_free(txt_in_b_);
    safe_free(txt_norm_w_);
    safe_free(time_lin1_w_); safe_free(time_lin1_b_);
    safe_free(time_lin2_w_); safe_free(time_lin2_b_);
    safe_free(norm_out_w_); safe_free(norm_out_b_);
    safe_free(proj_out_w_); safe_free(proj_out_b_);

    safe_free(scratch_img_f16_);  safe_free(scratch_txt_f16_);
    safe_free(scratch_img_norm_); safe_free(scratch_txt_norm_);
    safe_free(scratch_q_full_);   safe_free(scratch_k_full_);
    safe_free(scratch_v_full_);   safe_free(scratch_attn_full_);
    safe_free(scratch_img_mlp_);  safe_free(scratch_txt_mlp_);
    safe_free(scratch_img_proj_); safe_free(scratch_txt_proj_);
    safe_free(scratch_mod_vec_f16_);
    safe_free(scratch_rope_cos_); safe_free(scratch_rope_sin_);

    if (primary_stream_) cudaStreamDestroy(primary_stream_);
    if (cublas_)         cublasDestroy(cublas_);
#ifdef OMINIX_CUDA_USE_CUDNN
    if (cudnn_)          cudnnDestroy(cudnn_);
#endif
}

void ImageDiffusionCudaEngine::alloc_dev_(void **ptr, size_t bytes) {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[image_diffusion_cuda] cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        *ptr = nullptr;
    }
}

bool ImageDiffusionCudaEngine::init_from_gguf(const std::string &dit_path,
                                                const std::string &llm_path,
                                                const std::string &llm_vision_path,
                                                const std::string &vae_path,
                                                int device) {
    (void)llm_path; (void)llm_vision_path; (void)vae_path;  // Phase 3.2/3.3
    device_ = device;
    OMX_CUDA_CHECK(cudaSetDevice(device_));
    OMX_CUDA_CHECK(cudaStreamCreate(&primary_stream_));
    stream_ = primary_stream_;

    OMX_CUBLAS_CHECK(cublasCreate(&cublas_));
    OMX_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    OMX_CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));

#ifdef OMINIX_CUDA_USE_CUDNN
    if (cudnnCreate(&cudnn_) != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[image_diffusion_cuda] cudnnCreate failed\n");
        return false;
    }
#endif

    // Open DiT GGUF.
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(dit_path.c_str(), params);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[image_diffusion_cuda] failed to load DiT GGUF: %s\n",
                dit_path.c_str());
        return false;
    }

    // Validate architecture key.
    {
        int64_t key_id = gguf_find_key(gguf_ctx, "general.architecture");
        if (key_id >= 0 && gguf_get_kv_type(gguf_ctx, key_id) == GGUF_TYPE_STRING) {
            const char *arch = gguf_get_val_str(gguf_ctx, key_id);
            if (arch) {
                fprintf(stderr,
                        "[image_diffusion_cuda] DiT architecture = '%s'\n", arch);
                if (std::strcmp(arch, "qwen_image") != 0) {
                    fprintf(stderr,
                            "[image_diffusion_cuda] WARNING: expected "
                            "'qwen_image' arch, got '%s' — proceeding anyway\n",
                            arch);
                }
            }
        }
    }

    // Print tensor count for sanity.
    {
        int64_t n_t = gguf_get_n_tensors(gguf_ctx);
        fprintf(stderr,
                "[image_diffusion_cuda] DiT GGUF tensor_count = %lld\n",
                (long long)n_t);
    }

    blocks_.assign(cfg_.n_blocks, DiTBlockWeights{});
    UploadStats stats;

    const size_t H        = (size_t)cfg_.hidden;
    const size_t MLP      = (size_t)cfg_.mlp_inter;
    const size_t MOD      = (size_t)cfg_.mod_dim;
    const size_t HEAD_DIM = (size_t)cfg_.head_dim;
    const size_t TXT_H    = (size_t)cfg_.text_hidden;
    const size_t PATCH    = (size_t)cfg_.patch_in;
    const size_t TIMESTEP = (size_t)cfg_.timestep_inner;

    // ---- Per-block weight upload ------------------------------------------
    char name[256];
    bool ok = true;
#define TFMT(fmt) (snprintf(name, sizeof(name), fmt, il), name)
    for (int il = 0; il < cfg_.n_blocks && ok; ++il) {
        auto &b = blocks_[il];

        // Image-stream attention: to_q/k/v + to_out.0
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_q.weight"),
                                      H * H, b.to_q_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_k.weight"),
                                      H * H, b.to_k_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_v.weight"),
                                      H * H, b.to_v_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_out.0.weight"),
                                      H * H, b.to_out_0_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_q.bias"),
                                      H, b.to_q_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_k.bias"),
                                      H, b.to_k_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_v.bias"),
                                      H, b.to_v_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_out.0.bias"),
                                      H, b.to_out_0_b, stats);

        // Text-stream attention: add_q/k/v + to_add_out
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.add_q_proj.weight"),
                                      H * H, b.add_q_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.add_k_proj.weight"),
                                      H * H, b.add_k_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.add_v_proj.weight"),
                                      H * H, b.add_v_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_add_out.weight"),
                                      H * H, b.to_add_out_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.add_q_proj.bias"),
                                      H, b.add_q_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.add_k_proj.bias"),
                                      H, b.add_k_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.add_v_proj.bias"),
                                      H, b.add_v_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.to_add_out.bias"),
                                      H, b.to_add_out_b, stats);

        // QK-norm gammas (RMSNorm, F32 [head_dim])
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.norm_q.weight"),
                                      HEAD_DIM, b.norm_q_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.norm_k.weight"),
                                      HEAD_DIM, b.norm_k_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.norm_added_q.weight"),
                                      HEAD_DIM, b.norm_added_q_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.attn.norm_added_k.weight"),
                                      HEAD_DIM, b.norm_added_k_w, stats);

        // Image MLP
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.img_mlp.net.0.proj.weight"),
                                      MLP * H, b.img_mlp_0_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.img_mlp.net.2.weight"),
                                      H * MLP, b.img_mlp_2_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.img_mlp.net.0.proj.bias"),
                                      MLP, b.img_mlp_0_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.img_mlp.net.2.bias"),
                                      H, b.img_mlp_2_b, stats);

        // Text MLP
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.txt_mlp.net.0.proj.weight"),
                                      MLP * H, b.txt_mlp_0_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.txt_mlp.net.2.weight"),
                                      H * MLP, b.txt_mlp_2_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.txt_mlp.net.0.proj.bias"),
                                      MLP, b.txt_mlp_0_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.txt_mlp.net.2.bias"),
                                      H, b.txt_mlp_2_b, stats);

        // AdaLN modulation
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.img_mod.1.weight"),
                                      MOD * H, b.img_mod_w, stats);
        ok = ok && upload_tensor_f16(ggml_ctx, TFMT("transformer_blocks.%d.txt_mod.1.weight"),
                                      MOD * H, b.txt_mod_w, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.img_mod.1.bias"),
                                      MOD, b.img_mod_b, stats);
        ok = ok && upload_tensor_f32(ggml_ctx, TFMT("transformer_blocks.%d.txt_mod.1.bias"),
                                      MOD, b.txt_mod_b, stats);
    }
#undef TFMT

    if (!ok) {
        fprintf(stderr, "[image_diffusion_cuda] per-block weight upload FAILED\n");
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return false;
    }

    // ---- Global head/tail tensors -----------------------------------------
    ok = ok && upload_tensor_f16(ggml_ctx, "img_in.weight",
                                  H * PATCH, img_in_w_, stats);
    ok = ok && upload_tensor_f32(ggml_ctx, "img_in.bias",
                                  H, img_in_b_, stats);
    ok = ok && upload_tensor_f16(ggml_ctx, "txt_in.weight",
                                  H * TXT_H, txt_in_w_, stats);
    ok = ok && upload_tensor_f32(ggml_ctx, "txt_in.bias",
                                  H, txt_in_b_, stats);
    ok = ok && upload_tensor_f32(ggml_ctx, "txt_norm.weight",
                                  TXT_H, txt_norm_w_, stats);

    ok = ok && upload_tensor_f16(ggml_ctx, "time_text_embed.timestep_embedder.linear_1.weight",
                                  H * TIMESTEP, time_lin1_w_, stats);
    ok = ok && upload_tensor_f32(ggml_ctx, "time_text_embed.timestep_embedder.linear_1.bias",
                                  H, time_lin1_b_, stats);
    ok = ok && upload_tensor_f16(ggml_ctx, "time_text_embed.timestep_embedder.linear_2.weight",
                                  H * H, time_lin2_w_, stats);
    ok = ok && upload_tensor_f32(ggml_ctx, "time_text_embed.timestep_embedder.linear_2.bias",
                                  H, time_lin2_b_, stats);

    // norm_out.linear is the AdaLN-final head: only 2*hidden (shift+scale),
    // NOT the per-block 6*hidden mod_dim. Confirmed against the GGUF dump
    // (`3072, 6144` shape).
    const size_t NORM_OUT_FEAT = 2 * H;
    ok = ok && upload_tensor_f16(ggml_ctx, "norm_out.linear.weight",
                                  NORM_OUT_FEAT * H, norm_out_w_, stats);
    ok = ok && upload_tensor_f32(ggml_ctx, "norm_out.linear.bias",
                                  NORM_OUT_FEAT, norm_out_b_, stats);
    ok = ok && upload_tensor_f16(ggml_ctx, "proj_out.weight",
                                  PATCH * H, proj_out_w_, stats);
    ok = ok && upload_tensor_f32(ggml_ctx, "proj_out.bias",
                                  PATCH, proj_out_b_, stats);

    if (!ok) {
        fprintf(stderr, "[image_diffusion_cuda] global head/tail upload FAILED\n");
        gguf_free(gguf_ctx);
        ggml_free(ggml_ctx);
        return false;
    }

    total_weight_bytes_     = stats.bytes_uploaded;
    nonfinite_weight_count_ = stats.nonfinite_count;

    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    ready_ = true;
    fprintf(stderr,
            "[image_diffusion_cuda] Phase 3.1 init OK  device=%d  "
            "n_blocks=%d  hidden=%d  n_heads=%d  head_dim=%d  "
            "mlp_inter=%d  mod_dim=%d  text_hidden=%d  "
            "uploaded=%.2f GiB  nonfinite=%zu\n",
            device_, cfg_.n_blocks, cfg_.hidden, cfg_.n_heads, cfg_.head_dim,
            cfg_.mlp_inter, cfg_.mod_dim, cfg_.text_hidden,
            (double)total_weight_bytes_ / (1024.0 * 1024.0 * 1024.0),
            nonfinite_weight_count_);
    return true;
}

// ===========================================================================
// Phase 3.2 — single-block joint forward.
//
// Numeric strategy (smoke-grade, see dit_kernels.h header for full notes):
//   - All activations on device are F16; LayerNorm/RMSNorm/AdaLN reductions
//     use F32 internally.
//   - All matmuls go through cuBLAS GemmEx (F16 in/out, F32 accumulator).
//   - Joint cross-attention is a naive O(seq²) kernel (smoke; no FMHA).
//   - RoPE tables are built per-call on host with a flat [seq, head_dim/2]
//     layout (smoke approximation of the multi-axis temporal/h/w layout).
// ===========================================================================

namespace {

// Build a flat NEOX-mode RoPE table on host: cos/sin of theta * pos for each
// of head_dim/2 frequencies. seq_total rows. Output is F16 host vectors that
// the caller uploads to scratch_rope_{cos,sin}_.
void build_rope_table_host(int seq_total, int head_dim, float theta,
                            std::vector<__half> &cos_out,
                            std::vector<__half> &sin_out) {
    int half = head_dim / 2;
    cos_out.resize((size_t)seq_total * half);
    sin_out.resize((size_t)seq_total * half);
    // Standard transformer freqs: inv_freq[i] = 1 / theta^(i / head_dim).
    // (Joint multi-axis Qwen-Image RoPE uses temporal/h/w buckets, but for
    // smoke-grade single-block forward a unit-axis index suffices — the
    // output magnitudes are bounded the same way.)
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i) {
        float exp = (float)(2 * i) / (float)head_dim;
        inv_freq[i] = 1.0f / std::pow(theta, exp);
    }
    for (int p = 0; p < seq_total; ++p) {
        for (int i = 0; i < half; ++i) {
            float a = (float)p * inv_freq[i];
            cos_out[(size_t)p * half + i] = __float2half(std::cos(a));
            sin_out[(size_t)p * half + i] = __float2half(std::sin(a));
        }
    }
}

}  // namespace

bool ImageDiffusionCudaEngine::ensure_scratch_(int img_seq_len, int txt_seq_len) {
    if (scratch_img_seq_ == img_seq_len && scratch_txt_seq_ == txt_seq_len &&
        scratch_img_f16_ != nullptr) {
        return true;
    }
    // Free any previous allocation (we conservatively realloc on shape change).
    auto re = [&](void **p) { if (*p) { cudaFree(*p); *p = nullptr; } };
    re(&scratch_img_f16_);  re(&scratch_txt_f16_);
    re(&scratch_img_norm_); re(&scratch_txt_norm_);
    re(&scratch_q_full_);   re(&scratch_k_full_);
    re(&scratch_v_full_);   re(&scratch_attn_full_);
    re(&scratch_img_mlp_);  re(&scratch_txt_mlp_);
    re(&scratch_img_proj_); re(&scratch_txt_proj_);
    re(&scratch_mod_vec_f16_);
    re(&scratch_rope_cos_); re(&scratch_rope_sin_);

    const size_t H        = (size_t)cfg_.hidden;
    const size_t MLP      = (size_t)cfg_.mlp_inter;
    const size_t HD       = (size_t)cfg_.head_dim;
    const size_t img_seq  = (size_t)img_seq_len;
    const size_t txt_seq  = (size_t)txt_seq_len;
    const size_t seq_tot  = img_seq + txt_seq;
    const size_t hsz      = sizeof(__half);

    auto alloc = [&](void **p, size_t bytes) -> bool {
        cudaError_t e = cudaMalloc(p, bytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "[image_diffusion_cuda] scratch cudaMalloc(%zu) failed: %s\n",
                    bytes, cudaGetErrorString(e));
            return false;
        }
        return true;
    };
    bool ok = true;
    ok = ok && alloc(&scratch_img_f16_,     img_seq * H   * hsz);
    ok = ok && alloc(&scratch_txt_f16_,     txt_seq * H   * hsz);
    ok = ok && alloc(&scratch_img_norm_,    img_seq * H   * hsz);
    ok = ok && alloc(&scratch_txt_norm_,    txt_seq * H   * hsz);
    ok = ok && alloc(&scratch_q_full_,      seq_tot * H   * hsz);
    ok = ok && alloc(&scratch_k_full_,      seq_tot * H   * hsz);
    ok = ok && alloc(&scratch_v_full_,      seq_tot * H   * hsz);
    ok = ok && alloc(&scratch_attn_full_,   seq_tot * H   * hsz);
    ok = ok && alloc(&scratch_img_mlp_,     img_seq * MLP * hsz);
    ok = ok && alloc(&scratch_txt_mlp_,     txt_seq * MLP * hsz);
    ok = ok && alloc(&scratch_img_proj_,    img_seq * H   * hsz);
    ok = ok && alloc(&scratch_txt_proj_,    txt_seq * H   * hsz);
    ok = ok && alloc(&scratch_mod_vec_f16_, 12 * H        * hsz);
    ok = ok && alloc(&scratch_rope_cos_,    seq_tot * (HD / 2) * hsz);
    ok = ok && alloc(&scratch_rope_sin_,    seq_tot * (HD / 2) * hsz);
    if (!ok) return false;
    scratch_img_seq_ = img_seq_len;
    scratch_txt_seq_ = txt_seq_len;
    return true;
}

bool ImageDiffusionCudaEngine::forward_block(int block_idx,
                                              const float *img_in,
                                              int img_seq_len,
                                              const float *txt_in,
                                              int txt_seq_len,
                                              const float *mod_vec,
                                              float *img_out, float *txt_out) {
    if (!ready_) {
        fprintf(stderr, "[image_diffusion_cuda] forward_block: engine not ready\n");
        return false;
    }
    if (block_idx < 0 || block_idx >= cfg_.n_blocks) {
        fprintf(stderr, "[image_diffusion_cuda] forward_block: bad block_idx %d\n",
                block_idx);
        return false;
    }
    if (!ensure_scratch_(img_seq_len, txt_seq_len)) return false;

    const int H        = cfg_.hidden;
    const int NH       = cfg_.n_heads;
    const int HD       = cfg_.head_dim;
    const int MLP      = cfg_.mlp_inter;
    const int img_seq  = img_seq_len;
    const int txt_seq  = txt_seq_len;
    const int seq_tot  = img_seq + txt_seq;
    const float ln_eps  = cfg_.ln_eps;
    const float rms_eps = cfg_.rms_norm_eps;
    const float inv_sqrt_d = 1.0f / std::sqrt((float)HD);

    auto &lw = blocks_[block_idx];
    cudaStream_t s = stream_;

    // ---- Upload inputs (F32 host -> F16 device) ----
    // img_in [img_seq, H], txt_in [txt_seq, H], mod_vec [12*H].
    {
        std::vector<__half> img_h((size_t)img_seq * H);
        std::vector<__half> txt_h((size_t)txt_seq * H);
        std::vector<__half> mod_h((size_t)12 * H);
        for (size_t i = 0; i < img_h.size(); ++i) img_h[i] = __float2half(img_in[i]);
        for (size_t i = 0; i < txt_h.size(); ++i) txt_h[i] = __float2half(txt_in[i]);
        for (size_t i = 0; i < mod_h.size(); ++i) mod_h[i] = __float2half(mod_vec[i]);
        OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_img_f16_, img_h.data(),
                                         img_h.size() * sizeof(__half),
                                         cudaMemcpyHostToDevice, s));
        OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_txt_f16_, txt_h.data(),
                                         txt_h.size() * sizeof(__half),
                                         cudaMemcpyHostToDevice, s));
        OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_mod_vec_f16_, mod_h.data(),
                                         mod_h.size() * sizeof(__half),
                                         cudaMemcpyHostToDevice, s));
    }

    // ---- Build + upload RoPE table ----
    {
        std::vector<__half> cos_h, sin_h;
        build_rope_table_host(seq_tot, HD, cfg_.rope_theta, cos_h, sin_h);
        OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_rope_cos_, cos_h.data(),
                                         cos_h.size() * sizeof(__half),
                                         cudaMemcpyHostToDevice, s));
        OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_rope_sin_, sin_h.data(),
                                         sin_h.size() * sizeof(__half),
                                         cudaMemcpyHostToDevice, s));
    }

    // Convenience pointer aliases.
    __half *img_resid = (__half *)scratch_img_f16_;   // [img_seq, H]
    __half *txt_resid = (__half *)scratch_txt_f16_;   // [txt_seq, H]
    __half *img_norm  = (__half *)scratch_img_norm_;  // [img_seq, H]
    __half *txt_norm  = (__half *)scratch_txt_norm_;  // [txt_seq, H]
    __half *Q_full    = (__half *)scratch_q_full_;
    __half *K_full    = (__half *)scratch_k_full_;
    __half *V_full    = (__half *)scratch_v_full_;
    __half *A_full    = (__half *)scratch_attn_full_;
    __half *img_mlp   = (__half *)scratch_img_mlp_;
    __half *txt_mlp   = (__half *)scratch_txt_mlp_;
    __half *img_proj  = (__half *)scratch_img_proj_;
    __half *txt_proj  = (__half *)scratch_txt_proj_;
    __half *mod_h_dev = (__half *)scratch_mod_vec_f16_;

    // mod_vec chunk pointers (12 chunks of H halfs each).
    auto mchunk = [&](int idx) {
        return mod_h_dev + (size_t)idx * H;
    };
    __half *img_scale1 = mchunk(0);
    __half *img_shift1 = mchunk(1);
    __half *img_gate1  = mchunk(2);
    __half *img_scale2 = mchunk(3);
    __half *img_shift2 = mchunk(4);
    __half *img_gate2  = mchunk(5);
    __half *txt_scale1 = mchunk(6);
    __half *txt_shift1 = mchunk(7);
    __half *txt_gate1  = mchunk(8);
    __half *txt_scale2 = mchunk(9);
    __half *txt_shift2 = mchunk(10);
    __half *txt_gate2  = mchunk(11);

    // cuBLAS helper: y[m, n] = x[m, k] @ W^T[k, n] + bias[n]
    // W is stored row-major as [out, in] = [n, k] in our F16 device layout
    // (matches how upload_tensor_f16 packs each row). For row-major X[m,k] *
    // row-major W[n,k]^T = Y[m,n], the equivalent column-major call is:
    //   cublasGemmEx(opB=T, opA=N) treating W as [k,n] col-major (== [n,k]
    //   row-major), X as [k,m] col-major (== [m,k] row-major), Y col-major
    //   [n, m] (== row-major [m,n]).
    auto gemm_f16 = [&](const __half *X, const __half *W, __half *Y,
                          int M, int K, int N, const float *bias) -> bool {
        const float alpha = 1.0f;
        const float beta  = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            W, CUDA_R_16F, K,
            X, CUDA_R_16F, K,
            &beta,
            Y, CUDA_R_16F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[image_diffusion_cuda] cuBLAS GemmEx failed: %d\n",
                    (int)st);
            return false;
        }
        if (bias) {
            launch_add_bias_f32_f16(Y, bias, M, N, s);
        }
        return true;
    };

    // -----------------------------------------------------------------------
    // 1. LayerNorm1 + AdaLN-modulate(scale1, shift1) for both streams.
    //    Skip the silu(t_emb) + img_mod/txt_mod linear: caller pre-supplies
    //    the 12-chunk mod_vec. (Phase 3.2 smoke; Phase 3.3 wires the silu+
    //    linear to derive mod_vec from t_emb.)
    // -----------------------------------------------------------------------
    launch_layernorm_noaffine_f16(img_resid, img_norm, img_seq, H, ln_eps, s);
    launch_adaln_modulate_f16(img_norm, img_scale1, img_shift1,
                                img_seq, H, s);
    launch_layernorm_noaffine_f16(txt_resid, txt_norm, txt_seq, H, ln_eps, s);
    launch_adaln_modulate_f16(txt_norm, txt_scale1, txt_shift1,
                                txt_seq, H, s);

    // -----------------------------------------------------------------------
    // 2. QKV projections (per stream). Joint sequence layout: txt rows first,
    //    then img rows. Q/K/V buffers are [seq_tot, H].
    // -----------------------------------------------------------------------
    if (!gemm_f16(txt_norm, (const __half *)lw.add_q_w, Q_full,
                    txt_seq, H, H, (const float *)lw.add_q_b)) return false;
    if (!gemm_f16(txt_norm, (const __half *)lw.add_k_w, K_full,
                    txt_seq, H, H, (const float *)lw.add_k_b)) return false;
    if (!gemm_f16(txt_norm, (const __half *)lw.add_v_w, V_full,
                    txt_seq, H, H, (const float *)lw.add_v_b)) return false;
    __half *Q_img = Q_full + (size_t)txt_seq * H;
    __half *K_img = K_full + (size_t)txt_seq * H;
    __half *V_img = V_full + (size_t)txt_seq * H;
    if (!gemm_f16(img_norm, (const __half *)lw.to_q_w, Q_img,
                    img_seq, H, H, (const float *)lw.to_q_b)) return false;
    if (!gemm_f16(img_norm, (const __half *)lw.to_k_w, K_img,
                    img_seq, H, H, (const float *)lw.to_k_b)) return false;
    if (!gemm_f16(img_norm, (const __half *)lw.to_v_w, V_img,
                    img_seq, H, H, (const float *)lw.to_v_b)) return false;

    // -----------------------------------------------------------------------
    // 3. Head-wise RMSNorm on Q / K (txt + img streams use different gammas).
    // -----------------------------------------------------------------------
    launch_rmsnorm_head_f16_g32(Q_full, (const float *)lw.norm_added_q_w,
                                  Q_full, txt_seq, NH, HD, rms_eps, s);
    launch_rmsnorm_head_f16_g32(K_full, (const float *)lw.norm_added_k_w,
                                  K_full, txt_seq, NH, HD, rms_eps, s);
    launch_rmsnorm_head_f16_g32(Q_img, (const float *)lw.norm_q_w,
                                  Q_img, img_seq, NH, HD, rms_eps, s);
    launch_rmsnorm_head_f16_g32(K_img, (const float *)lw.norm_k_w,
                                  K_img, img_seq, NH, HD, rms_eps, s);

    // -----------------------------------------------------------------------
    // 4. NEOX joint RoPE on Q / K. The cos/sin tables span the joint
    //    seq_total: rows 0..txt_seq are txt, rows txt_seq..seq_tot are img.
    // -----------------------------------------------------------------------
    launch_rope_neox_seq_f16(Q_full, (const __half *)scratch_rope_cos_,
                                (const __half *)scratch_rope_sin_, Q_full,
                                seq_tot, NH, HD, s);
    launch_rope_neox_seq_f16(K_full, (const __half *)scratch_rope_cos_,
                                (const __half *)scratch_rope_sin_, K_full,
                                seq_tot, NH, HD, s);

    // -----------------------------------------------------------------------
    // 5. Joint cross-attention (smoke-grade naive kernel).
    // -----------------------------------------------------------------------
    launch_attn_joint_naive_f16(Q_full, K_full, V_full, A_full,
                                  seq_tot, NH, HD, inv_sqrt_d, s);

    // -----------------------------------------------------------------------
    // 6. Output projections (per stream). txt occupies rows [0..txt_seq),
    //    img occupies rows [txt_seq..seq_tot).
    // -----------------------------------------------------------------------
    if (!gemm_f16(A_full, (const __half *)lw.to_add_out_w, txt_proj,
                    txt_seq, H, H, (const float *)lw.to_add_out_b)) return false;
    if (!gemm_f16(A_full + (size_t)txt_seq * H,
                    (const __half *)lw.to_out_0_w, img_proj,
                    img_seq, H, H, (const float *)lw.to_out_0_b)) return false;

    // -----------------------------------------------------------------------
    // 7. Gated residual add: img_resid += img_proj * gate1; same for txt.
    // -----------------------------------------------------------------------
    launch_gated_residual_add_f16(img_resid, img_proj, img_gate1,
                                    img_seq, H, s);
    launch_gated_residual_add_f16(txt_resid, txt_proj, txt_gate1,
                                    txt_seq, H, s);

    // -----------------------------------------------------------------------
    // 8. LayerNorm2 + AdaLN-modulate(scale2, shift2).
    // -----------------------------------------------------------------------
    launch_layernorm_noaffine_f16(img_resid, img_norm, img_seq, H, ln_eps, s);
    launch_adaln_modulate_f16(img_norm, img_scale2, img_shift2,
                                img_seq, H, s);
    launch_layernorm_noaffine_f16(txt_resid, txt_norm, txt_seq, H, ln_eps, s);
    launch_adaln_modulate_f16(txt_norm, txt_scale2, txt_shift2,
                                txt_seq, H, s);

    // -----------------------------------------------------------------------
    // 9. FFN: Linear(H -> MLP) + GELU-tanh + Linear(MLP -> H), per stream.
    // -----------------------------------------------------------------------
    if (!gemm_f16(img_norm, (const __half *)lw.img_mlp_0_w, img_mlp,
                    img_seq, H, MLP, (const float *)lw.img_mlp_0_b)) return false;
    launch_gelu_tanh_f16(img_mlp, img_seq * MLP, s);
    if (!gemm_f16(img_mlp, (const __half *)lw.img_mlp_2_w, img_proj,
                    img_seq, MLP, H, (const float *)lw.img_mlp_2_b)) return false;
    if (!gemm_f16(txt_norm, (const __half *)lw.txt_mlp_0_w, txt_mlp,
                    txt_seq, H, MLP, (const float *)lw.txt_mlp_0_b)) return false;
    launch_gelu_tanh_f16(txt_mlp, txt_seq * MLP, s);
    if (!gemm_f16(txt_mlp, (const __half *)lw.txt_mlp_2_w, txt_proj,
                    txt_seq, MLP, H, (const float *)lw.txt_mlp_2_b)) return false;

    // -----------------------------------------------------------------------
    // 10. Gated residual add #2.
    // -----------------------------------------------------------------------
    launch_gated_residual_add_f16(img_resid, img_proj, img_gate2,
                                    img_seq, H, s);
    launch_gated_residual_add_f16(txt_resid, txt_proj, txt_gate2,
                                    txt_seq, H, s);

    // ---- Download output (F16 device -> F32 host) ----
    {
        std::vector<__half> img_h((size_t)img_seq * H);
        std::vector<__half> txt_h((size_t)txt_seq * H);
        OMX_CUDA_CHECK(cudaMemcpyAsync(img_h.data(), img_resid,
                                         img_h.size() * sizeof(__half),
                                         cudaMemcpyDeviceToHost, s));
        OMX_CUDA_CHECK(cudaMemcpyAsync(txt_h.data(), txt_resid,
                                         txt_h.size() * sizeof(__half),
                                         cudaMemcpyDeviceToHost, s));
        OMX_CUDA_CHECK(cudaStreamSynchronize(s));
        for (size_t i = 0; i < img_h.size(); ++i) img_out[i] = __half2float(img_h[i]);
        for (size_t i = 0; i < txt_h.size(); ++i) txt_out[i] = __half2float(txt_h[i]);
    }
    return true;
}

void ImageDiffusionCudaEngine::final_proj(const float * /*img_in*/,
                                           int /*seq_len*/,
                                           float * /*img_out*/) {
    fprintf(stderr,
            "[image_diffusion_cuda] final_proj is a Phase 3.3 stub — "
            "aborting\n");
    std::abort();
}

}  // namespace ominix_cuda
