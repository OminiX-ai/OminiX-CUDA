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

void ImageDiffusionCudaEngine::forward_block(int /*block_idx*/,
                                              const float * /*img_in*/,
                                              int /*img_seq_len*/,
                                              const float * /*txt_in*/,
                                              int /*txt_seq_len*/,
                                              const float * /*mod_vec*/,
                                              float * /*img_out*/,
                                              float * /*txt_out*/) {
    fprintf(stderr,
            "[image_diffusion_cuda] forward_block is a Phase 3.2 stub — "
            "aborting\n");
    std::abort();
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
