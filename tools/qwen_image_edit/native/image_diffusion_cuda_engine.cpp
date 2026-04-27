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
    safe_free(scratch_img_resid_f32_); safe_free(scratch_txt_resid_f32_);
    safe_free(scratch_img_norm_f32_);  safe_free(scratch_txt_norm_f32_);
    safe_free(scratch_q_full_);   safe_free(scratch_k_full_);
    safe_free(scratch_v_full_);   safe_free(scratch_attn_full_);
    safe_free(scratch_q_full_f32_); safe_free(scratch_k_full_f32_);
    safe_free(scratch_v_full_f32_); safe_free(scratch_attn_f32_);
    safe_free(scratch_img_mlp_);  safe_free(scratch_txt_mlp_);
    safe_free(scratch_img_proj_); safe_free(scratch_txt_proj_);
    safe_free(scratch_img_mlp_f32_); safe_free(scratch_txt_mlp_f32_);
    safe_free(scratch_img_proj_f32_); safe_free(scratch_txt_proj_f32_);
    safe_free(scratch_mod_vec_f16_);
    // Phase 3.3a: pe-table + t_emb buffers.
    safe_free(pe_cos_dev_);       safe_free(pe_sin_dev_);
    safe_free(t_emb_in_f16_);     safe_free(t_emb_mid_f16_);
    safe_free(silu_t_emb_f16_);

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

    // ---- Phase 3.3a: precompute multi-axis RoPE pe-table (host → device) ---
    if (!build_pe_table_()) {
        fprintf(stderr, "[image_diffusion_cuda] build_pe_table_ FAILED\n");
        return false;
    }

    // ---- Phase 3.3a: allocate persistent t_emb scratch buffers -------------
    {
        const size_t H = (size_t)cfg_.hidden;
        const size_t T = (size_t)cfg_.timestep_inner;
        OMX_CUDA_CHECK(cudaMalloc(&t_emb_in_f16_,   T * sizeof(__half)));
        OMX_CUDA_CHECK(cudaMalloc(&t_emb_mid_f16_, H * sizeof(__half)));
        OMX_CUDA_CHECK(cudaMalloc(&silu_t_emb_f16_, H * sizeof(__half)));
    }

    ready_ = true;
    fprintf(stderr,
            "[image_diffusion_cuda] Phase 3.3a init OK  device=%d  "
            "n_blocks=%d  hidden=%d  n_heads=%d  head_dim=%d  "
            "mlp_inter=%d  mod_dim=%d  text_hidden=%d  "
            "uploaded=%.2f GiB  nonfinite=%zu  pe_total_pos=%d\n",
            device_, cfg_.n_blocks, cfg_.hidden, cfg_.n_heads, cfg_.head_dim,
            cfg_.mlp_inter, cfg_.mod_dim, cfg_.text_hidden,
            (double)total_weight_bytes_ / (1024.0 * 1024.0 * 1024.0),
            nonfinite_weight_count_, pe_total_pos_);
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

// ----------------------------------------------------------------------------
// Phase 3.3a multi-axis NEOX RoPE table (Qwen-Image temporal=16 + h=56 + w=56).
//
// Output layout (caller passes vectors that get uploaded as F16):
//   cos_out / sin_out : [pe_total_pos, head_dim/2]
//   pe_total_pos       = max_txt_seq + (h_len * w_len)         (h=w=64 → 4096)
//   pe rows 0..max_txt_seq               → txt positions (diagonal t=h=w)
//   pe rows max_txt_seq..pe_total_pos    → img positions ((t=0, h=row, w=col))
//
// Mirrors `compute_qwen_rope_pe_host` in the Ascend reference
// (image_diffusion_engine.cpp:510-624). Per-axis frequency convention follows
// Qwen::Rope::linspace (rope.hpp:44-56):
//   linspace(0, (axis_dim - 2) / axis_dim, axis_dim/2) → omega = 1/theta^scale
// Each axis contributes axis_dim/2 d_pairs sequentially in the head_dim/2
// packing: [0..axes_t/2-1]=T, [axes_t/2..axes_t/2+axes_h/2-1]=H, [...rest..]=W.
// scale_rope=true in qwen_image: h ids = [-h_len/2..h_len/2-1], same for w.
// txt id starts at max(h_len, w_len) so the txt range never collides with img.
// ----------------------------------------------------------------------------
void build_qwen_rope_pe_host_3axis(const ominix_cuda::ImageDiffusionConfig &cfg,
                                    std::vector<__half> &cos_out,
                                    std::vector<__half> &sin_out,
                                    int &pe_total_pos_out) {
    const int axes_t = cfg.rope_axes_temporal;
    const int axes_h = cfg.rope_axes_h;
    const int axes_w = cfg.rope_axes_w;
    const int head_dim = cfg.head_dim;
    const int half = head_dim / 2;

    // Patch grid corresponding to max_img_seq (default 4096 → 64×64).
    int h_len = (int)std::lround(std::sqrt((double)cfg.max_img_seq));
    int w_len = h_len;
    while (h_len * w_len < cfg.max_img_seq) ++h_len;
    const int img_tokens = h_len * w_len;
    const int ctx_len    = cfg.max_txt_seq;
    const int txt_start  = std::max(h_len, w_len);

    pe_total_pos_out = ctx_len + img_tokens;
    cos_out.assign((size_t)pe_total_pos_out * half, __float2half(1.0f));
    sin_out.assign((size_t)pe_total_pos_out * half, __float2half(0.0f));

    // axis_omega: per-axis frequencies. omega[i] = 1 / theta^scale where
    // scale = linspace(0, (d-2)/d, d/2)[i] = end_scale * i/(half_axis-1).
    auto axis_omega = [&](int axis_dim, std::vector<float> &omega) {
        const int half_axis = axis_dim / 2;
        omega.assign(half_axis, 0.0f);
        if (half_axis == 0) return;
        if (half_axis == 1) { omega[0] = 1.0f; return; }
        const float end_scale = (axis_dim - 2.0f) / (float)axis_dim;
        for (int i = 0; i < half_axis; ++i) {
            const float scale = end_scale * (float)i / (float)(half_axis - 1);
            omega[i] = 1.0f / std::pow(cfg.rope_theta, scale);
        }
    };
    std::vector<float> omega_t, omega_h, omega_w;
    axis_omega(axes_t, omega_t);
    axis_omega(axes_h, omega_h);
    axis_omega(axes_w, omega_w);

    auto pe_set = [&](int pos, int dpair, float a) {
        size_t idx = (size_t)pos * half + (size_t)dpair;
        cos_out[idx] = __float2half(std::cos(a));
        sin_out[idx] = __float2half(std::sin(a));
    };

    // Txt positions: diagonal t=h=w=(txt_start + i).
    for (int i = 0; i < ctx_len; ++i) {
        const float p = (float)(txt_start + i);
        int dp = 0;
        for (size_t j = 0; j < omega_t.size(); ++j, ++dp) pe_set(i, dp, p * omega_t[j]);
        for (size_t j = 0; j < omega_h.size(); ++j, ++dp) pe_set(i, dp, p * omega_h[j]);
        for (size_t j = 0; j < omega_w.size(); ++j, ++dp) pe_set(i, dp, p * omega_w[j]);
    }

    // Img positions: (t=0, h=h_start+r, w=w_start+c). scale_rope=true.
    const int h_start = -h_len / 2;
    const int w_start = -w_len / 2;
    for (int r = 0; r < h_len; ++r) {
        const float h_id = (float)(h_start + r);
        for (int c = 0; c < w_len; ++c) {
            const float w_id = (float)(w_start + c);
            const int pos = ctx_len + r * w_len + c;
            if (pos >= pe_total_pos_out) break;
            int dp = 0;
            const float t_id = 0.0f;
            for (size_t j = 0; j < omega_t.size(); ++j, ++dp) pe_set(pos, dp, t_id * omega_t[j]);
            for (size_t j = 0; j < omega_h.size(); ++j, ++dp) pe_set(pos, dp, h_id * omega_h[j]);
            for (size_t j = 0; j < omega_w.size(); ++j, ++dp) pe_set(pos, dp, w_id * omega_w[j]);
        }
    }
}

// Host sinusoidal timestep embedding [dim]. Matches the ggml CPU reference
// (`ggml_compute_forward_timestep_embedding_f32`) and the Ascend port's
// `host_timestep_embedding_f32` (image_diffusion_engine.cpp:4621-4636):
//   out = [cos(arg_0), .., cos(arg_half-1), sin(arg_0), .., sin(arg_half-1)]
//   arg_j = timestep * exp(-log(max_period) * j / half),  half = dim/2
void host_timestep_embedding_f32(float timestep, int dim, int max_period,
                                  std::vector<float> &out) {
    out.assign((size_t)dim, 0.0f);
    const int half = dim / 2;
    for (int j = 0; j < half; ++j) {
        float freq = std::exp(-std::log((float)max_period) * (float)j /
                               (float)half);
        float arg  = timestep * freq;
        out[(size_t)j]        = std::cos(arg);
        out[(size_t)j + half] = std::sin(arg);
    }
    if (dim % 2 != 0) out[(size_t)2 * half] = 0.0f;
}

}  // namespace

bool ImageDiffusionCudaEngine::build_pe_table_() {
    std::vector<__half> cos_h, sin_h;
    int total_pos = 0;
    build_qwen_rope_pe_host_3axis(cfg_, cos_h, sin_h, total_pos);
    pe_total_pos_ = total_pos;
    const size_t bytes = cos_h.size() * sizeof(__half);
    OMX_CUDA_CHECK(cudaMalloc(&pe_cos_dev_, bytes));
    OMX_CUDA_CHECK(cudaMalloc(&pe_sin_dev_, bytes));
    OMX_CUDA_CHECK(cudaMemcpy(pe_cos_dev_, cos_h.data(), bytes,
                                cudaMemcpyHostToDevice));
    OMX_CUDA_CHECK(cudaMemcpy(pe_sin_dev_, sin_h.data(), bytes,
                                cudaMemcpyHostToDevice));
    return true;
}

bool ImageDiffusionCudaEngine::ensure_scratch_(int img_seq_len, int txt_seq_len) {
    if (scratch_img_seq_ == img_seq_len && scratch_txt_seq_ == txt_seq_len &&
        scratch_img_f16_ != nullptr) {
        return true;
    }
    // Free any previous allocation (we conservatively realloc on shape change).
    auto re = [&](void **p) { if (*p) { cudaFree(*p); *p = nullptr; } };
    re(&scratch_img_f16_);  re(&scratch_txt_f16_);
    re(&scratch_img_norm_); re(&scratch_txt_norm_);
    re(&scratch_img_resid_f32_); re(&scratch_txt_resid_f32_);
    re(&scratch_img_norm_f32_);  re(&scratch_txt_norm_f32_);
    re(&scratch_q_full_);   re(&scratch_k_full_);
    re(&scratch_v_full_);   re(&scratch_attn_full_);
    re(&scratch_q_full_f32_); re(&scratch_k_full_f32_);
    re(&scratch_v_full_f32_); re(&scratch_attn_f32_);
    re(&scratch_img_mlp_);  re(&scratch_txt_mlp_);
    re(&scratch_img_proj_); re(&scratch_txt_proj_);
    re(&scratch_img_mlp_f32_); re(&scratch_txt_mlp_f32_);
    re(&scratch_img_proj_f32_); re(&scratch_txt_proj_f32_);
    re(&scratch_mod_vec_f16_);

    const size_t H        = (size_t)cfg_.hidden;
    const size_t MLP      = (size_t)cfg_.mlp_inter;
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
    ok = ok && alloc(&scratch_img_resid_f32_, img_seq * H * sizeof(float));
    ok = ok && alloc(&scratch_txt_resid_f32_, txt_seq * H * sizeof(float));
    ok = ok && alloc(&scratch_img_norm_f32_,  img_seq * H * sizeof(float));
    ok = ok && alloc(&scratch_txt_norm_f32_,  txt_seq * H * sizeof(float));
    ok = ok && alloc(&scratch_q_full_,      seq_tot * H   * hsz);
    ok = ok && alloc(&scratch_k_full_,      seq_tot * H   * hsz);
    ok = ok && alloc(&scratch_v_full_,      seq_tot * H   * hsz);
    ok = ok && alloc(&scratch_attn_full_,   seq_tot * H   * hsz);
    // Phase 3.3b — F32 mirrors for the widened attention path.
    ok = ok && alloc(&scratch_q_full_f32_,  seq_tot * H   * sizeof(float));
    ok = ok && alloc(&scratch_k_full_f32_,  seq_tot * H   * sizeof(float));
    ok = ok && alloc(&scratch_v_full_f32_,  seq_tot * H   * sizeof(float));
    ok = ok && alloc(&scratch_attn_f32_,    seq_tot * H   * sizeof(float));
    ok = ok && alloc(&scratch_img_mlp_,     img_seq * MLP * hsz);
    ok = ok && alloc(&scratch_txt_mlp_,     txt_seq * MLP * hsz);
    ok = ok && alloc(&scratch_img_proj_,    img_seq * H   * hsz);
    ok = ok && alloc(&scratch_txt_proj_,    txt_seq * H   * hsz);
    ok = ok && alloc(&scratch_img_mlp_f32_, img_seq * MLP * sizeof(float));
    ok = ok && alloc(&scratch_txt_mlp_f32_, txt_seq * MLP * sizeof(float));
    ok = ok && alloc(&scratch_img_proj_f32_, img_seq * H   * sizeof(float));
    ok = ok && alloc(&scratch_txt_proj_f32_, txt_seq * H   * sizeof(float));
    ok = ok && alloc(&scratch_mod_vec_f16_, 12 * H        * hsz);
    if (!ok) return false;
    scratch_img_seq_ = img_seq_len;
    scratch_txt_seq_ = txt_seq_len;
    return true;
}

// Phase 3.3a: derive silu_t_emb_f16_ from a scalar timestep.
// Pipeline: sinusoidal[256] (host) → t_emb_in_f16_ (dev, F16) → time_lin1
// (cuBLAS GEMM, +F32 bias) → t_emb_mid_f16_ → silu in-place → time_lin2
// → silu_t_emb_f16_ → silu in-place. Output = silu(t_emb) ready for the
// per-block img_mod / txt_mod GEMMs.
bool ImageDiffusionCudaEngine::compute_t_emb_(float timestep) {
    const int H = cfg_.hidden;
    const int T = cfg_.timestep_inner;

    // 1. Sinusoidal embedding on host, upload to device, cast F32→F16.
    std::vector<float> t_sinu_f32;
    host_timestep_embedding_f32(timestep, T, /*max_period=*/10000, t_sinu_f32);
    std::vector<__half> t_sinu_f16(T);
    for (int i = 0; i < T; ++i) t_sinu_f16[i] = __float2half(t_sinu_f32[i]);
    OMX_CUDA_CHECK(cudaMemcpyAsync(t_emb_in_f16_, t_sinu_f16.data(),
                                     T * sizeof(__half),
                                     cudaMemcpyHostToDevice, stream_));

    // 2. time_lin1: [1, T] @ time_lin1_w[H, T]^T → [1, H] (+ F32 bias).
    auto gemm = [&](const __half *X, const __half *W, __half *Y,
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
            fprintf(stderr, "[image_diffusion_cuda] t_emb GemmEx failed: %d\n",
                    (int)st);
            return false;
        }
        if (bias) launch_add_bias_f32_f16(Y, bias, M, N, stream_);
        return true;
    };

    if (!gemm((const __half *)t_emb_in_f16_,
              (const __half *)time_lin1_w_,
              (__half *)t_emb_mid_f16_,
              /*M=*/1, /*K=*/T, /*N=*/H,
              (const float *)time_lin1_b_)) return false;

    // 3. SiLU in place.
    launch_silu_f16((__half *)t_emb_mid_f16_, H, stream_);

    // 4. time_lin2: [1, H] → [1, H] (+ F32 bias).
    if (!gemm((const __half *)t_emb_mid_f16_,
              (const __half *)time_lin2_w_,
              (__half *)silu_t_emb_f16_,
              /*M=*/1, /*K=*/H, /*N=*/H,
              (const float *)time_lin2_b_)) return false;

    // 5. SiLU in place — this is the final silu(t_emb) the per-block
    //    img_mod / txt_mod GEMMs consume.
    launch_silu_f16((__half *)silu_t_emb_f16_, H, stream_);

    // Phase 3.3a smoke probe: dump silu_t_emb stats to understand
    // mod_vec amplification at a given timestep.
    if (const char *e = std::getenv("OMINIX_CUDA_DUMP_TEMB"); e && e[0] == '1') {
        std::vector<__half> h(H);
        cudaStreamSynchronize(stream_);
        cudaMemcpy(h.data(), silu_t_emb_f16_, H * sizeof(__half),
                    cudaMemcpyDeviceToHost);
        double sum_abs = 0.0, max_abs = 0.0;
        size_t n_nan = 0, n_inf = 0;
        for (auto v : h) {
            float f = __half2float(v);
            if (std::isnan(f)) { ++n_nan; continue; }
            if (std::isinf(f)) { ++n_inf; continue; }
            double a = std::fabs((double)f);
            sum_abs += a;
            if (a > max_abs) max_abs = a;
        }
        size_t valid = (size_t)H - n_nan - n_inf;
        fprintf(stderr,
                "[image_diffusion_cuda] silu_t_emb t=%.2f  mean_abs=%.4g "
                " max_abs=%.4g  NaN=%zu Inf=%zu\n",
                timestep, valid > 0 ? sum_abs / valid : 0.0, max_abs,
                n_nan, n_inf);
    }
    return true;
}

bool ImageDiffusionCudaEngine::forward_block(int block_idx,
                                              const float *img_in,
                                              int img_seq_len,
                                              const float *txt_in,
                                              int txt_seq_len,
                                              float timestep,
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
    if (img_seq_len > cfg_.max_img_seq) {
        fprintf(stderr, "[image_diffusion_cuda] img_seq_len=%d > max_img_seq=%d\n",
                img_seq_len, cfg_.max_img_seq);
        return false;
    }
    if (txt_seq_len > cfg_.max_txt_seq) {
        fprintf(stderr, "[image_diffusion_cuda] txt_seq_len=%d > max_txt_seq=%d\n",
                txt_seq_len, cfg_.max_txt_seq);
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

    // ---- Upload inputs (F32 host -> F32 device residual) ----
    // Phase 3.3b: residual chain widened to F32 (Ascend §5.5.46 BF16 analog
    // applied to the whole double-stream residual, not just Q/K/V).
    OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_img_resid_f32_, img_in,
                                     (size_t)img_seq * H * sizeof(float),
                                     cudaMemcpyHostToDevice, s));
    OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_txt_resid_f32_, txt_in,
                                     (size_t)txt_seq * H * sizeof(float),
                                     cudaMemcpyHostToDevice, s));

    // ---- Phase 3.3a Gate B: compute silu(t_emb) from timestep ----
    if (!compute_t_emb_(timestep)) return false;

    // Convenience pointer aliases.
    float  *img_resid_f32 = (float *)scratch_img_resid_f32_;
    float  *txt_resid_f32 = (float *)scratch_txt_resid_f32_;
    float  *img_norm_f32  = (float *)scratch_img_norm_f32_;
    float  *txt_norm_f32  = (float *)scratch_txt_norm_f32_;
    __half *img_norm  = (__half *)scratch_img_norm_;  // F16 staging for GEMM input
    __half *txt_norm  = (__half *)scratch_txt_norm_;
    // F16 attention output buffer (fed into the F16 attention output
    // projection). Q/K/V used by the widened-F32 attention path live in
    // scratch_q_full_f32_ / k_full_f32_ / v_full_f32_; the F16 mirrors
    // are retained for any downstream Phase 3.3c FMHA fallback.
    __half *A_full    = (__half *)scratch_attn_full_;
    float  *Q_full_f32 = (float *)scratch_q_full_f32_;
    float  *K_full_f32 = (float *)scratch_k_full_f32_;
    float  *V_full_f32 = (float *)scratch_v_full_f32_;
    float  *A_full_f32 = (float *)scratch_attn_f32_;
    (void)scratch_q_full_; (void)scratch_k_full_; (void)scratch_v_full_;
    __half *img_mlp   = (__half *)scratch_img_mlp_;
    __half *txt_mlp   = (__half *)scratch_txt_mlp_;
    float  *img_mlp_f32  = (float *)scratch_img_mlp_f32_;
    float  *txt_mlp_f32  = (float *)scratch_txt_mlp_f32_;
    float  *img_proj_f32 = (float *)scratch_img_proj_f32_;
    float  *txt_proj_f32 = (float *)scratch_txt_proj_f32_;
    (void)scratch_img_proj_; (void)scratch_txt_proj_;
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
    // 0. Phase 3.3a Gate C — derive mod_vec from silu(t_emb).
    //    img_mod_w  : F16 [mod_dim=6H, H]  →  silu_t_emb @ img_mod_w^T
    //                + img_mod_b  →  scratch_mod_vec_f16_[0..6H]
    //    txt_mod_w  : F16 [mod_dim=6H, H]  →  silu_t_emb @ txt_mod_w^T
    //                + txt_mod_b  →  scratch_mod_vec_f16_[6H..12H]
    //    The 6 chunks per stream lay out as
    //    [scale1, shift1, gate1, scale2, shift2, gate2] (matches the
    //    `mchunk()` accessor below — img occupies chunks 0..5, txt 6..11).
    // -----------------------------------------------------------------------
    if (!gemm_f16((const __half *)silu_t_emb_f16_,
                    (const __half *)lw.img_mod_w,
                    mod_h_dev,
                    /*M=*/1, /*K=*/H, /*N=*/cfg_.mod_dim,
                    (const float *)lw.img_mod_b)) return false;
    if (!gemm_f16((const __half *)silu_t_emb_f16_,
                    (const __half *)lw.txt_mod_w,
                    mod_h_dev + (size_t)cfg_.mod_dim,
                    /*M=*/1, /*K=*/H, /*N=*/cfg_.mod_dim,
                    (const float *)lw.txt_mod_b)) return false;

    // -----------------------------------------------------------------------
    // 1. LayerNorm1 + AdaLN-modulate(scale1, shift1) for both streams.
    //    F32 residual → F32 LN → F32 AdaLN → F16 cast for GEMM input.
    // -----------------------------------------------------------------------
    launch_layernorm_noaffine_f32(img_resid_f32, img_norm_f32, img_seq, H, ln_eps, s);
    launch_adaln_modulate_f32(img_norm_f32, img_scale1, img_shift1,
                                img_seq, H, s);
    launch_cast_f32_to_f16_2d(img_norm_f32, img_norm, img_seq * H, s);
    launch_layernorm_noaffine_f32(txt_resid_f32, txt_norm_f32, txt_seq, H, ln_eps, s);
    launch_adaln_modulate_f32(txt_norm_f32, txt_scale1, txt_shift1,
                                txt_seq, H, s);
    launch_cast_f32_to_f16_2d(txt_norm_f32, txt_norm, txt_seq * H, s);

    // -----------------------------------------------------------------------
    // 2. Phase 3.3b — QKV projections via F16-input / F32-output cuBLAS GEMM.
    //    The widened F32 Q/K/V storage is the CUDA equivalent of Ascend
    //    §5.5.46 BF16 widening: AdaLN-modulated img_norm/txt_norm (already
    //    near the F16 ceiling under high-magnitude mod_vec) projected by
    //    F16 weights would saturate to Inf in F16 storage. F32 output keeps
    //    the dynamic range intact through attention.
    // -----------------------------------------------------------------------
    auto gemm_f16in_f32out = [&](const __half *X, const __half *W, float *Y,
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
            Y, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[image_diffusion_cuda] cuBLAS GemmEx (f32 out) failed: %d\n",
                    (int)st);
            return false;
        }
        if (bias) {
            launch_add_bias_f32_f32(Y, bias, M, N, s);
        }
        return true;
    };
    if (!gemm_f16in_f32out(txt_norm, (const __half *)lw.add_q_w, Q_full_f32,
                    txt_seq, H, H, (const float *)lw.add_q_b)) return false;
    if (!gemm_f16in_f32out(txt_norm, (const __half *)lw.add_k_w, K_full_f32,
                    txt_seq, H, H, (const float *)lw.add_k_b)) return false;
    if (!gemm_f16in_f32out(txt_norm, (const __half *)lw.add_v_w, V_full_f32,
                    txt_seq, H, H, (const float *)lw.add_v_b)) return false;
    float *Q_img_f32 = Q_full_f32 + (size_t)txt_seq * H;
    float *K_img_f32 = K_full_f32 + (size_t)txt_seq * H;
    float *V_img_f32 = V_full_f32 + (size_t)txt_seq * H;
    if (!gemm_f16in_f32out(img_norm, (const __half *)lw.to_q_w, Q_img_f32,
                    img_seq, H, H, (const float *)lw.to_q_b)) return false;
    if (!gemm_f16in_f32out(img_norm, (const __half *)lw.to_k_w, K_img_f32,
                    img_seq, H, H, (const float *)lw.to_k_b)) return false;
    if (!gemm_f16in_f32out(img_norm, (const __half *)lw.to_v_w, V_img_f32,
                    img_seq, H, H, (const float *)lw.to_v_b)) return false;

    // -----------------------------------------------------------------------
    // 3. Head-wise RMSNorm on Q / K (F32, in-place).
    // -----------------------------------------------------------------------
    launch_rmsnorm_head_f32_g32(Q_full_f32, (const float *)lw.norm_added_q_w,
                                  Q_full_f32, txt_seq, NH, HD, rms_eps, s);
    launch_rmsnorm_head_f32_g32(K_full_f32, (const float *)lw.norm_added_k_w,
                                  K_full_f32, txt_seq, NH, HD, rms_eps, s);
    launch_rmsnorm_head_f32_g32(Q_img_f32, (const float *)lw.norm_q_w,
                                  Q_img_f32, img_seq, NH, HD, rms_eps, s);
    launch_rmsnorm_head_f32_g32(K_img_f32, (const float *)lw.norm_k_w,
                                  K_img_f32, img_seq, NH, HD, rms_eps, s);

    // -----------------------------------------------------------------------
    // 4. Multi-axis NEOX RoPE on Q / K (F32 in-place).
    // -----------------------------------------------------------------------
    {
        const __half *pe_cos = (const __half *)pe_cos_dev_;
        const __half *pe_sin = (const __half *)pe_sin_dev_;
        launch_rope_neox_3axis_f32(Q_full_f32, pe_cos, pe_sin, Q_full_f32,
                                     txt_seq, NH, HD, /*pe_off=*/0, s);
        launch_rope_neox_3axis_f32(K_full_f32, pe_cos, pe_sin, K_full_f32,
                                     txt_seq, NH, HD, /*pe_off=*/0, s);
        launch_rope_neox_3axis_f32(Q_img_f32, pe_cos, pe_sin, Q_img_f32,
                                     img_seq, NH, HD,
                                     /*pe_off=*/cfg_.max_txt_seq, s);
        launch_rope_neox_3axis_f32(K_img_f32, pe_cos, pe_sin, K_img_f32,
                                     img_seq, NH, HD,
                                     /*pe_off=*/cfg_.max_txt_seq, s);
    }

    // -----------------------------------------------------------------------
    // 5. Joint cross-attention (F32 in/out, F32 accumulator throughout).
    // -----------------------------------------------------------------------
    launch_attn_joint_naive_f32(Q_full_f32, K_full_f32, V_full_f32, A_full_f32,
                                  seq_tot, NH, HD, inv_sqrt_d, s);

    // Cast attention output back to F16 for the output projection. The
    // output projection re-expresses the attention output back into the
    // residual hidden space; the residual itself is bounded by AdaLN gate
    // (≪ 1) and the per-block residual add, so F16 storage is safe again.
    launch_cast_f32_to_f16_2d(A_full_f32, A_full, (int)((size_t)seq_tot * H), s);

    // Phase 3.3b probe (gated by env): scan F32 attn output for NaN/Inf
    // *before* the F16 cast to localize where overflow originates.
    if (const char *e = std::getenv("OMINIX_CUDA_PROBE_ATTN"); e && e[0] == '1') {
        cudaStreamSynchronize(s);
        std::vector<float> probe((size_t)seq_tot * H);
        cudaMemcpy(probe.data(), A_full_f32,
                    probe.size() * sizeof(float),
                    cudaMemcpyDeviceToHost);
        size_t nan = 0, inf = 0;
        double max_abs = 0.0, sum_abs = 0.0;
        for (float v : probe) {
            if (std::isnan(v)) { ++nan; continue; }
            if (std::isinf(v)) { ++inf; continue; }
            double a = std::fabs((double)v);
            sum_abs += a;
            if (a > max_abs) max_abs = a;
        }
        size_t valid = probe.size() - nan - inf;
        fprintf(stderr,
                "[image_diffusion_cuda] probe(attn_f32)  n=%zu  mean_abs=%.4g  "
                "max_abs=%.4g  NaN=%zu  Inf=%zu\n",
                probe.size(), valid > 0 ? sum_abs / valid : 0.0,
                max_abs, nan, inf);
    }

    // -----------------------------------------------------------------------
    // 6. Output projections (per stream). F16-in, F32-out (Phase 3.3b).
    //    txt occupies rows [0..txt_seq), img occupies rows [txt_seq..seq_tot).
    // -----------------------------------------------------------------------
    if (!gemm_f16in_f32out(A_full, (const __half *)lw.to_add_out_w, txt_proj_f32,
                    txt_seq, H, H, (const float *)lw.to_add_out_b)) return false;
    if (!gemm_f16in_f32out(A_full + (size_t)txt_seq * H,
                    (const __half *)lw.to_out_0_w, img_proj_f32,
                    img_seq, H, H, (const float *)lw.to_out_0_b)) return false;

    // -----------------------------------------------------------------------
    // 7. Gated residual add (F32 resid += F32 delta * F16 gate).
    // -----------------------------------------------------------------------
    launch_gated_residual_add_f32_delta(img_resid_f32, img_proj_f32, img_gate1,
                                          img_seq, H, s);
    launch_gated_residual_add_f32_delta(txt_resid_f32, txt_proj_f32, txt_gate1,
                                          txt_seq, H, s);

    // -----------------------------------------------------------------------
    // 8. LayerNorm2 + AdaLN-modulate(scale2, shift2).
    // -----------------------------------------------------------------------
    launch_layernorm_noaffine_f32(img_resid_f32, img_norm_f32, img_seq, H, ln_eps, s);
    launch_adaln_modulate_f32(img_norm_f32, img_scale2, img_shift2,
                                img_seq, H, s);
    launch_cast_f32_to_f16_2d(img_norm_f32, img_norm, img_seq * H, s);
    launch_layernorm_noaffine_f32(txt_resid_f32, txt_norm_f32, txt_seq, H, ln_eps, s);
    launch_adaln_modulate_f32(txt_norm_f32, txt_scale2, txt_shift2,
                                txt_seq, H, s);
    launch_cast_f32_to_f16_2d(txt_norm_f32, txt_norm, txt_seq * H, s);

    // -----------------------------------------------------------------------
    // 9. FFN — F32 widened path (Phase 3.3b).
    //    MLP_0 (F16-in F32-out) → GELU-tanh F32 → cast to F16 →
    //    MLP_2 (F16-in F32-out). The cast in the middle is safe because
    //    GELU is bounded by its input (no amplification), and the post-AdaLN
    //    F16 input to MLP_0 was bounded by the F32 cast precondition.
    // -----------------------------------------------------------------------
    if (!gemm_f16in_f32out(img_norm, (const __half *)lw.img_mlp_0_w, img_mlp_f32,
                    img_seq, H, MLP, (const float *)lw.img_mlp_0_b)) return false;
    launch_gelu_tanh_f32(img_mlp_f32, img_seq * MLP, s);
    launch_cast_f32_to_f16_2d(img_mlp_f32, img_mlp, img_seq * MLP, s);
    if (!gemm_f16in_f32out(img_mlp, (const __half *)lw.img_mlp_2_w, img_proj_f32,
                    img_seq, MLP, H, (const float *)lw.img_mlp_2_b)) return false;
    if (!gemm_f16in_f32out(txt_norm, (const __half *)lw.txt_mlp_0_w, txt_mlp_f32,
                    txt_seq, H, MLP, (const float *)lw.txt_mlp_0_b)) return false;
    launch_gelu_tanh_f32(txt_mlp_f32, txt_seq * MLP, s);
    launch_cast_f32_to_f16_2d(txt_mlp_f32, txt_mlp, txt_seq * MLP, s);
    if (!gemm_f16in_f32out(txt_mlp, (const __half *)lw.txt_mlp_2_w, txt_proj_f32,
                    txt_seq, MLP, H, (const float *)lw.txt_mlp_2_b)) return false;

    // -----------------------------------------------------------------------
    // 10. Gated residual add #2 (F32 resid += F32 delta * F16 gate).
    // -----------------------------------------------------------------------
    launch_gated_residual_add_f32_delta(img_resid_f32, img_proj_f32, img_gate2,
                                          img_seq, H, s);
    launch_gated_residual_add_f32_delta(txt_resid_f32, txt_proj_f32, txt_gate2,
                                          txt_seq, H, s);

    // ---- Download output (F32 device -> F32 host) ----
    OMX_CUDA_CHECK(cudaMemcpyAsync(img_out, img_resid_f32,
                                     (size_t)img_seq * H * sizeof(float),
                                     cudaMemcpyDeviceToHost, s));
    OMX_CUDA_CHECK(cudaMemcpyAsync(txt_out, txt_resid_f32,
                                     (size_t)txt_seq * H * sizeof(float),
                                     cudaMemcpyDeviceToHost, s));
    OMX_CUDA_CHECK(cudaStreamSynchronize(s));
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

// Phase 3.3b — full 60-block DiT forward (Gate C). Mirrors Ascend
// `denoise_full` per-step body for a single timestep:
//   - Internalised t_emb chain runs once per call (block 0 reuses it).
//   - Loop blk=0..59: forward_block(blk, ...).
//   - Final tail: SiLU(t_emb) → norm_out.linear → split [scale, shift] →
//     LayerNorm(img_resid) → modulate(scale, shift) → proj_out → F16
//     [img_seq, patch_in].
//
// The engine does not own the patchify/unpatchify reshape — the caller
// is responsible for converting [img_seq, patch_in] back to a latent
// volume with whatever grid shape its denoiser expects.
bool ImageDiffusionCudaEngine::forward_dit(float timestep,
                                            const float *img_in,
                                            int img_seq_len,
                                            const float *txt_in,
                                            int txt_seq_len,
                                            float *img_out) {
    if (!ready_) {
        fprintf(stderr, "[image_diffusion_cuda] forward_dit: engine not ready\n");
        return false;
    }
    if (img_in == nullptr || txt_in == nullptr || img_out == nullptr) {
        fprintf(stderr, "[image_diffusion_cuda] forward_dit: null input/output\n");
        return false;
    }
    const int H        = cfg_.hidden;
    const int PATCH    = cfg_.patch_in;
    const int img_seq  = img_seq_len;
    const float ln_eps  = cfg_.ln_eps;

    // We thread img/txt through forward_block via host scratch (the existing
    // F32-host -> F16-device path). For 60 blocks this is ~60*(img_seq*H +
    // txt_seq*H) F32 host bytes copied each iter — acceptable for a smoke
    // verdict; Phase 3.3c will keep activations resident on device.
    std::vector<float> img_buf_a(img_in, img_in + (size_t)img_seq_len * H);
    std::vector<float> img_buf_b((size_t)img_seq_len * H, 0.0f);
    std::vector<float> txt_buf_a(txt_in, txt_in + (size_t)txt_seq_len * H);
    std::vector<float> txt_buf_b((size_t)txt_seq_len * H, 0.0f);

    float *img_src = img_buf_a.data();
    float *img_dst = img_buf_b.data();
    float *txt_src = txt_buf_a.data();
    float *txt_dst = txt_buf_b.data();

    bool dump_blocks = false;
    if (const char *e = std::getenv("OMINIX_CUDA_DUMP_BLOCKS"); e && e[0] == '1') {
        dump_blocks = true;
    }
    for (int blk = 0; blk < cfg_.n_blocks; ++blk) {
        if (!forward_block(blk,
                            img_src, img_seq_len,
                            txt_src, txt_seq_len,
                            timestep,
                            img_dst, txt_dst)) {
            fprintf(stderr,
                    "[image_diffusion_cuda] forward_dit: block %d FAILED\n", blk);
            return false;
        }
        if (dump_blocks) {
            size_t nan_i = 0, inf_i = 0; double max_i = 0.0;
            for (size_t i = 0; i < (size_t)img_seq_len * H; ++i) {
                float v = img_dst[i];
                if (std::isnan(v)) { ++nan_i; continue; }
                if (std::isinf(v)) { ++inf_i; continue; }
                double a = std::fabs((double)v);
                if (a > max_i) max_i = a;
            }
            fprintf(stderr,
                    "[image_diffusion_cuda] forward_dit blk=%d  img max_abs=%.4g  NaN=%zu  Inf=%zu\n",
                    blk, max_i, nan_i, inf_i);
        }
        std::swap(img_src, img_dst);
        std::swap(txt_src, txt_dst);
    }
    // After the loop, img_src points at the final residual (img_seq, H).

    // ---- Tail: norm_out (AdaLN-final) + proj_out -------------------------
    // 1. compute_t_emb_ already populated silu_t_emb_f16_; but forward_block
    //    overwrites it each call. Re-compute fresh so the tail uses the
    //    same timestep.
    if (!compute_t_emb_(timestep)) return false;

    cudaStream_t s = stream_;

    // 2. norm_out.linear: [1, H] @ norm_out_w[2H, H]^T + norm_out_b → [1, 2H].
    //    Reuse scratch_mod_vec_f16_ buffer (>= 12*H halfs, fits 2H trivially).
    __half *adaln_emb_f16 = (__half *)scratch_mod_vec_f16_;
    {
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            /*N=*/2 * H, /*M=*/1, /*K=*/H,
            &alpha,
            norm_out_w_, CUDA_R_16F, H,
            silu_t_emb_f16_, CUDA_R_16F, H,
            &beta,
            adaln_emb_f16, CUDA_R_16F, 2 * H,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[image_diffusion_cuda] forward_dit: norm_out GEMM failed: %d\n", (int)st);
            return false;
        }
        launch_add_bias_f32_f16(adaln_emb_f16, (const float *)norm_out_b_,
                                 1, 2 * H, s);
    }
    // diffusers convention: chunk is [scale, shift] (first H = scale, second H = shift).
    __half *norm_out_scale = adaln_emb_f16;
    __half *norm_out_shift = adaln_emb_f16 + H;

    // 3. Upload final img_resid (img_src host F32) to F32 device buffer.
    if (!ensure_scratch_(img_seq_len, txt_seq_len)) return false;
    OMX_CUDA_CHECK(cudaMemcpyAsync(scratch_img_resid_f32_, img_src,
                                     (size_t)img_seq * H * sizeof(float),
                                     cudaMemcpyHostToDevice, s));
    float  *img_resid_f32 = (float *)scratch_img_resid_f32_;
    float  *img_norm_f32  = (float *)scratch_img_norm_f32_;
    __half *img_norm  = (__half *)scratch_img_norm_;

    // 4. LayerNorm(img_resid) → img_norm in F32.
    launch_layernorm_noaffine_f32(img_resid_f32, img_norm_f32, img_seq, H, ln_eps, s);

    // 5. AdaLN-modulate(img_norm, scale, shift) in F32.
    //    x = x * (1 + scale) + shift.
    launch_adaln_modulate_f32(img_norm_f32, norm_out_scale, norm_out_shift,
                                img_seq, H, s);

    // 5b. Cast to F16 for proj_out GEMM input (post-AdaLN values are
    //     bounded ~hundreds, fits F16).
    launch_cast_f32_to_f16_2d(img_norm_f32, img_norm, img_seq * H, s);

    // 6. proj_out: img_norm[img_seq, H] @ proj_out_w[PATCH, H]^T +
    //    proj_out_b → [img_seq, PATCH] F16. Reuse scratch_img_proj_ for
    //    the patch-sized output (img_seq * PATCH halfs ≤ img_seq * H halfs).
    __half *patch_out_f16 = (__half *)scratch_img_proj_;
    {
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            /*N=*/PATCH, /*M=*/img_seq, /*K=*/H,
            &alpha,
            proj_out_w_, CUDA_R_16F, H,
            img_norm, CUDA_R_16F, H,
            &beta,
            patch_out_f16, CUDA_R_16F, PATCH,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[image_diffusion_cuda] forward_dit: proj_out GEMM failed: %d\n", (int)st);
            return false;
        }
        launch_add_bias_f32_f16(patch_out_f16, (const float *)proj_out_b_,
                                 img_seq, PATCH, s);
    }

    // 7. Download patch_out F16 → host F32.
    {
        std::vector<__half> h((size_t)img_seq * PATCH);
        OMX_CUDA_CHECK(cudaMemcpyAsync(h.data(), patch_out_f16,
                                         h.size() * sizeof(__half),
                                         cudaMemcpyDeviceToHost, s));
        OMX_CUDA_CHECK(cudaStreamSynchronize(s));
        for (size_t i = 0; i < h.size(); ++i) img_out[i] = __half2float(h[i]);
    }
    return true;
}

}  // namespace ominix_cuda
