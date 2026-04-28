// ============================================================================
// Talker CUDA Engine — Phase 2.1 scaffold + Phase 2.2 forward decode body.
//
// Phase 2.1 (LANDED): header full, ctor/dtor lifecycle, init_from_gguf
// parses metadata, allocates per-layer KV cache + activation scratch,
// builds NEOX RoPE tables. Reset/setters work.
//
// Phase 2.2 (THIS): per-token autoregressive forward_decode body wired up.
//   - GGUF weight upload: F16 matmul weights + F32 norm gammas, schema
//     matching the TalkerCannEngine reference (`blk.<L>.attn_q.weight`,
//     `blk.<L>.attn_norm.weight`, `output_norm.weight`, ...).
//   - run_decode_ops_: 28-layer body (RmsNorm -> Q/K/V proj -> QK-norm
//     -> RoPE -> KV write -> GQA attn -> O proj -> residual -> ffn_norm
//     -> SwiGLU -> down -> residual), final RmsNorm + F16->F32 cast.
//   - forward_decode: H2D input + cast + body + D2H + sync.
//
// Phase 2.3+ remain stubs:
//   - forward_prefill (S>1) — std::abort with a clear message.
//   - int8_calibrate_weight_ (Phase 2.6).
//
// CUDA error-handling macros are local (no global cuda_check.hpp yet to keep
// the scaffold self-contained).
// ============================================================================

#include "talker_cuda_engine.h"

#include "../talker.h"  // TalkerConfig
#include "cuda_kernels/cuda_kernels.h"

// llama.cpp GGUF helpers (vendored in ominix-cuda).
#include "ggml.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

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
            fprintf(stderr, "[talker_cuda] CUDA error %s at %s:%d: %s\n",      \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_err));      \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define OMX_CUBLAS_CHECK(expr)                                                 \
    do {                                                                       \
        cublasStatus_t _st = (expr);                                           \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[talker_cuda] cuBLAS error %s at %s:%d: %d\n",    \
                    #expr, __FILE__, __LINE__, (int)_st);                      \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace ominix_cuda {

namespace {

// Pull a tensor out of GGUF as F32 vector (handles F32 / F16 sources).
std::vector<float> load_gguf_tensor_f32(ggml_context *ggml_ctx,
                                          const char *name,
                                          size_t expected_elems) {
    ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (!t) {
        fprintf(stderr, "[talker_cuda] missing tensor: %s\n", name);
        return {};
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr,
                "[talker_cuda] %s: expected %zu elems, got %zu\n",
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
        // Generic dequant path: handles Q8_0, Q4_0, Q5_K, and any other
        // quantized dtype ggml supports via its type-trait to_float hook.
        // Phase 2.2: zgx-3675 ships only Q8_0 GGUF; this unblocks smoke
        // and keeps the loader future-proof for additional quants.
        const ggml_type_traits *tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float) {
            tt->to_float(t->data, out.data(), (int64_t)n);
        } else {
            fprintf(stderr,
                    "[talker_cuda] %s: unsupported dtype %d (no to_float trait)\n",
                    name, (int)t->type);
            return {};
        }
    }
    return out;
}

bool upload_tensor_f16(ggml_context *ggml_ctx, const char *name,
                        size_t expected_elems, void *&dev) {
    std::vector<float> host = load_gguf_tensor_f32(ggml_ctx, name,
                                                    expected_elems);
    if (host.empty()) return false;
    std::vector<__half> f16(expected_elems);
    for (size_t i = 0; i < expected_elems; ++i)
        f16[i] = __float2half(host[i]);
    cudaError_t err = cudaMalloc(&dev, expected_elems * sizeof(__half));
    if (err != cudaSuccess) {
        fprintf(stderr, "[talker_cuda] cudaMalloc(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(dev, f16.data(), expected_elems * sizeof(__half),
                      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[talker_cuda] cudaMemcpy(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool upload_tensor_f32(ggml_context *ggml_ctx, const char *name,
                        size_t expected_elems, void *&dev) {
    std::vector<float> host = load_gguf_tensor_f32(ggml_ctx, name,
                                                    expected_elems);
    if (host.empty()) return false;
    cudaError_t err = cudaMalloc(&dev, expected_elems * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[talker_cuda] cudaMalloc(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(dev, host.data(), expected_elems * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[talker_cuda] cudaMemcpy(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// dtor / setters
// ---------------------------------------------------------------------------

TalkerCudaEngine::~TalkerCudaEngine() {
    // Best-effort teardown. Order: free per-layer weight blobs first, then
    // scratch, then handles, then streams.
    auto safe_free = [](void *p) {
        if (p) cudaFree(p);
    };

    for (auto &lw : layer_w_) {
        safe_free(lw.q_proj_w);    safe_free(lw.k_proj_w);
        safe_free(lw.v_proj_w);    safe_free(lw.o_proj_w);
        safe_free(lw.q_norm_w);    safe_free(lw.k_norm_w);
        safe_free(lw.gate_proj_w); safe_free(lw.up_proj_w);
        safe_free(lw.down_proj_w);
        safe_free(lw.input_ln_w);  safe_free(lw.post_ln_w);

        safe_free(lw.q_proj_w_i8);    safe_free(lw.k_proj_w_i8);
        safe_free(lw.v_proj_w_i8);    safe_free(lw.o_proj_w_i8);
        safe_free(lw.gate_proj_w_i8); safe_free(lw.up_proj_w_i8);
        safe_free(lw.down_proj_w_i8);
        safe_free(lw.q_proj_scale);    safe_free(lw.k_proj_scale);
        safe_free(lw.v_proj_scale);    safe_free(lw.o_proj_scale);
        safe_free(lw.gate_proj_scale); safe_free(lw.up_proj_scale);
        safe_free(lw.down_proj_scale);

        safe_free(lw.q_proj_w_fp8);    safe_free(lw.k_proj_w_fp8);
        safe_free(lw.v_proj_w_fp8);    safe_free(lw.o_proj_w_fp8);
        safe_free(lw.gate_proj_w_fp8); safe_free(lw.up_proj_w_fp8);
        safe_free(lw.down_proj_w_fp8);
    }
    safe_free(final_norm_w_dev_);

    safe_free(cur_dev_);   safe_free(residual_dev_); safe_free(normed_dev_);
    safe_free(q_dev_);     safe_free(k_dev_);        safe_free(v_dev_);
    safe_free(attn_out_dev_); safe_free(o_out_dev_);
    safe_free(gate_dev_);  safe_free(up_dev_);       safe_free(ffn_out_dev_);

    safe_free(cur_batch_dev_);   safe_free(residual_batch_dev_);
    safe_free(normed_batch_dev_);
    safe_free(q_batch_dev_);     safe_free(k_batch_dev_);  safe_free(v_batch_dev_);
    safe_free(attn_out_batch_dev_); safe_free(o_out_batch_dev_);
    safe_free(gate_batch_dev_);  safe_free(up_batch_dev_); safe_free(ffn_out_batch_dev_);
    safe_free(causal_mask_dev_);

    safe_free(rope_cos_dev_); safe_free(rope_sin_dev_);
    safe_free(rstd_dev_);

    for (auto *p : k_cache_dev_) safe_free(p);
    for (auto *p : v_cache_dev_) safe_free(p);

    safe_free(input_stage_f32_dev_);
    safe_free(output_stage_f32_dev_);

    // Phase 2.9 device LM-head buffers.
    safe_free(lm_head_w_f16_dev_);
    safe_free(lm_head_hidden_f16_dev_);
    safe_free(lm_head_logits_f32_dev_);

    // Phase 2.6 FP8 LM-head lane buffers + cublasLt resources. Kept in
    // dedicated cleanup function below to keep cublasLt headers out of
    // this dtor block (it's defined just above the FP8 entry points).
    teardown_fp8_lm_head_();

    for (auto exec : decode_graph_execs_) {
        if (exec) cudaGraphExecDestroy(exec);
    }

    if (decode_done_event_) cudaEventDestroy(decode_done_event_);
    if (stream_b_)       cudaStreamDestroy(stream_b_);
    if (primary_stream_) cudaStreamDestroy(primary_stream_);
    if (cublas_)         cublasDestroy(cublas_);
#ifdef OMINIX_CUDA_USE_CUDNN
    if (cudnn_)          cudnnDestroy(cudnn_);
#endif
}

void TalkerCudaEngine::reset_kv_cache() { kv_cache_len_ = 0; }

void TalkerCudaEngine::set_rope_speed_factor(float factor) {
    if (factor <= 0.0f) factor = 1.0f;
    if (factor == rope_speed_factor_) return;
    rope_speed_factor_ = factor;
    if (ready_) build_rope_tables_();
}

void TalkerCudaEngine::set_use_mrope_xvec_layout(bool enable) {
    if (enable && mrope_temporal_section_ == 0) {
        fprintf(stderr, "[talker_cuda] xvec/customvoice MRoPE layout requested "
                        "but mrope_temporal_section is zero in GGUF — refusing\n");
        return;
    }
    if (use_mrope_xvec_layout_ == enable) return;
    use_mrope_xvec_layout_ = enable;
    if (ready_) build_rope_tables_();
}

void TalkerCudaEngine::alloc_dev_(void **ptr, size_t bytes) {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[talker_cuda] cudaMalloc(%zu) failed: %s\n",
                bytes, cudaGetErrorString(err));
        *ptr = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Phase 2.1: GGUF parse + handle/stream alloc + RoPE table build (host-only).
//
// Weight upload (cudaMemcpy of every projection buffer) is wired in this
// scaffold but kept guarded by env OMINIX_CUDA_TALKER_LOAD=1 so a CI-style
// build can run init without a full GGUF in tree. Default behavior is to
// run the full upload — same convention as the Ascend reference.
// ---------------------------------------------------------------------------

bool TalkerCudaEngine::init_from_gguf(const std::string &gguf_path,
                                       const TalkerConfig &cfg, int device) {
    device_ = device;
    OMX_CUDA_CHECK(cudaSetDevice(device_));
    OMX_CUDA_CHECK(cudaStreamCreate(&primary_stream_));
    stream_ = primary_stream_;
    OMX_CUDA_CHECK(cudaStreamCreate(&stream_b_));
    OMX_CUDA_CHECK(cudaEventCreateWithFlags(&decode_done_event_,
                                             cudaEventDisableTiming));

    OMX_CUBLAS_CHECK(cublasCreate(&cublas_));
    OMX_CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
    // Default math mode: TF32 / FP16 tensor cores enabled.
    OMX_CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));

#ifdef OMINIX_CUDA_USE_CUDNN
    if (cudnnCreate(&cudnn_) != CUDNN_STATUS_SUCCESS) {
        fprintf(stderr, "[talker_cuda] cudnnCreate failed\n");
        return false;
    }
#endif

    // Cache dims from config.
    n_embd_   = cfg.hidden_size;
    n_heads_  = cfg.num_attention_heads;
    n_kv_     = cfg.num_key_value_heads;
    head_dim_ = cfg.head_dim;
    q_dim_    = n_heads_ * head_dim_;
    kv_dim_   = n_kv_    * head_dim_;
    inter_    = cfg.intermediate_size;
    n_layers_ = cfg.num_hidden_layers;
    eps_      = cfg.rms_norm_eps;
    rope_theta_ = cfg.rope_theta;
    rope_speed_factor_ = 1.0f;

    // Open GGUF.
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[talker_cuda] failed to load GGUF: %s\n",
                gguf_path.c_str());
        return false;
    }

    // ---- Read mrope_section (B6.1 parity) --------------------------------
    {
        mrope_temporal_section_ = 0;
        const char *keys[] = {
            "qwen3.rope.dimension_sections",
            "rope_scaling.mrope_section",
        };
        int64_t sec[4] = {0, 0, 0, 0};
        int n_read = 0;
        for (const char *k : keys) {
            int64_t key_id = gguf_find_key(gguf_ctx, k);
            if (key_id < 0) continue;
            enum gguf_type t = gguf_get_kv_type(gguf_ctx, key_id);
            if (t != GGUF_TYPE_ARRAY) continue;
            enum gguf_type at = gguf_get_arr_type(gguf_ctx, key_id);
            size_t n = gguf_get_arr_n(gguf_ctx, key_id);
            if (n == 0) continue;
            size_t to_read = n < 4 ? n : 4;
            const void *data = gguf_get_arr_data(gguf_ctx, key_id);
            n_read = (int)to_read;
            for (size_t i = 0; i < to_read; ++i) {
                switch (at) {
                    case GGUF_TYPE_INT32:  sec[i] = ((const int32_t  *)data)[i]; break;
                    case GGUF_TYPE_UINT32: sec[i] = ((const uint32_t *)data)[i]; break;
                    case GGUF_TYPE_INT64:  sec[i] = ((const int64_t  *)data)[i]; break;
                    case GGUF_TYPE_UINT64: sec[i] = (int64_t)((const uint64_t *)data)[i]; break;
                    case GGUF_TYPE_INT16:  sec[i] = ((const int16_t  *)data)[i]; break;
                    case GGUF_TYPE_UINT16: sec[i] = ((const uint16_t *)data)[i]; break;
                    default: n_read = 0; i = to_read; break;
                }
            }
            if (n_read > 0) {
                printf("[talker_cuda] mrope_section from '%s' = "
                       "[%lld, %lld, %lld, %lld]\n",
                       k, (long long)sec[0], (long long)sec[1],
                       (long long)sec[2], (long long)sec[3]);
                break;
            }
        }
        if (n_read > 0) {
            mrope_temporal_section_ = (int)sec[0];
            if (mrope_temporal_section_ * 2 > head_dim_) {
                fprintf(stderr,
                        "[talker_cuda] FATAL: mrope_temporal_section %d "
                        "exceeds head_dim/2 (%d)\n",
                        mrope_temporal_section_, head_dim_ / 2);
                gguf_free(gguf_ctx);
                ggml_free(ggml_ctx);
                return false;
            }
        }
    }

    // ---- Per-layer weight upload (Phase 2.2). Mirrors the Ascend reference's
    //      `load_proj_weight` / `upload_tensor_f32` two-helper pattern: every
    //      matmul weight goes up as F16 (bit-cast on host, single
    //      cudaMemcpy), every norm gamma goes up as F32. Tensor names
    //      match the GGUF schema emitted by `tools/qwen_tts/export_talker_llama.py`.
    layer_w_.resize(n_layers_);
    {
        char name[128];
        bool ok = true;
        for (int il = 0; il < n_layers_ && ok; ++il) {
            auto &lw = layer_w_[il];
#define TFMT(fmt) (snprintf(name, sizeof(name), fmt, il), name)
            ok = ok && upload_tensor_f16(ggml_ctx, TFMT("blk.%d.attn_q.weight"),
                                          (size_t)q_dim_  * n_embd_, lw.q_proj_w);
            ok = ok && upload_tensor_f16(ggml_ctx, TFMT("blk.%d.attn_k.weight"),
                                          (size_t)kv_dim_ * n_embd_, lw.k_proj_w);
            ok = ok && upload_tensor_f16(ggml_ctx, TFMT("blk.%d.attn_v.weight"),
                                          (size_t)kv_dim_ * n_embd_, lw.v_proj_w);
            ok = ok && upload_tensor_f16(ggml_ctx, TFMT("blk.%d.attn_output.weight"),
                                          (size_t)n_embd_ * q_dim_,  lw.o_proj_w);
            ok = ok && upload_tensor_f16(ggml_ctx, TFMT("blk.%d.ffn_gate.weight"),
                                          (size_t)inter_  * n_embd_, lw.gate_proj_w);
            ok = ok && upload_tensor_f16(ggml_ctx, TFMT("blk.%d.ffn_up.weight"),
                                          (size_t)inter_  * n_embd_, lw.up_proj_w);
            ok = ok && upload_tensor_f16(ggml_ctx, TFMT("blk.%d.ffn_down.weight"),
                                          (size_t)n_embd_ * inter_,  lw.down_proj_w);

            ok = ok && upload_tensor_f32(ggml_ctx, TFMT("blk.%d.attn_q_norm.weight"),
                                          (size_t)head_dim_, lw.q_norm_w);
            ok = ok && upload_tensor_f32(ggml_ctx, TFMT("blk.%d.attn_k_norm.weight"),
                                          (size_t)head_dim_, lw.k_norm_w);
            ok = ok && upload_tensor_f32(ggml_ctx, TFMT("blk.%d.attn_norm.weight"),
                                          (size_t)n_embd_,   lw.input_ln_w);
            ok = ok && upload_tensor_f32(ggml_ctx, TFMT("blk.%d.ffn_norm.weight"),
                                          (size_t)n_embd_,   lw.post_ln_w);
#undef TFMT
        }
        ok = ok && upload_tensor_f32(ggml_ctx, "output_norm.weight",
                                      (size_t)n_embd_, final_norm_w_dev_);
        if (!ok) {
            fprintf(stderr, "[talker_cuda] weight upload FAILED\n");
            gguf_free(gguf_ctx);
            ggml_free(ggml_ctx);
            return false;
        }
    }

    // ---- Pre-alloc activation scratch (S=1) ------------------------------
    const size_t f16_n_embd = (size_t)n_embd_ * sizeof(__half);
    const size_t f16_q_dim  = (size_t)q_dim_  * sizeof(__half);
    const size_t f16_kv_dim = (size_t)kv_dim_ * sizeof(__half);
    const size_t f16_inter  = (size_t)inter_  * sizeof(__half);

    alloc_dev_(&cur_dev_,      f16_n_embd);
    alloc_dev_(&residual_dev_, f16_n_embd);
    alloc_dev_(&normed_dev_,   f16_n_embd);
    alloc_dev_(&q_dev_,        f16_q_dim);
    alloc_dev_(&k_dev_,        f16_kv_dim);
    alloc_dev_(&v_dev_,        f16_kv_dim);
    alloc_dev_(&attn_out_dev_, f16_q_dim);
    alloc_dev_(&o_out_dev_,    f16_n_embd);
    alloc_dev_(&gate_dev_,     f16_inter);
    alloc_dev_(&up_dev_,       f16_inter);
    alloc_dev_(&ffn_out_dev_,  f16_n_embd);

    // ---- Pre-alloc activation scratch (prefill, S<=MAX_PREFILL) -----------
    alloc_dev_(&cur_batch_dev_,      (size_t)MAX_PREFILL * f16_n_embd);
    alloc_dev_(&residual_batch_dev_, (size_t)MAX_PREFILL * f16_n_embd);
    alloc_dev_(&normed_batch_dev_,   (size_t)MAX_PREFILL * f16_n_embd);
    alloc_dev_(&q_batch_dev_,        (size_t)MAX_PREFILL * f16_q_dim);
    alloc_dev_(&k_batch_dev_,        (size_t)MAX_PREFILL * f16_kv_dim);
    alloc_dev_(&v_batch_dev_,        (size_t)MAX_PREFILL * f16_kv_dim);
    alloc_dev_(&attn_out_batch_dev_, (size_t)MAX_PREFILL * f16_q_dim);
    alloc_dev_(&o_out_batch_dev_,    (size_t)MAX_PREFILL * f16_n_embd);
    alloc_dev_(&gate_batch_dev_,     (size_t)MAX_PREFILL * f16_inter);
    alloc_dev_(&up_batch_dev_,       (size_t)MAX_PREFILL * f16_inter);
    alloc_dev_(&ffn_out_batch_dev_,  (size_t)MAX_PREFILL * f16_n_embd);
    alloc_dev_(&causal_mask_dev_,
               (size_t)MAX_PREFILL * (size_t)MAX_SEQ * sizeof(__half));

    alloc_dev_(&rstd_dev_, (size_t)MAX_PREFILL * (size_t)n_heads_ * sizeof(float));

    // ---- KV cache: per-layer F16 [MAX_SEQ, kv_dim] ------------------------
    k_cache_dev_.assign(n_layers_, nullptr);
    v_cache_dev_.assign(n_layers_, nullptr);
    const size_t kv_layer_bytes = (size_t)MAX_SEQ * f16_kv_dim;
    for (int il = 0; il < n_layers_; ++il) {
        alloc_dev_(&k_cache_dev_[il], kv_layer_bytes);
        alloc_dev_(&v_cache_dev_[il], kv_layer_bytes);
    }

    // ---- I/O staging ------------------------------------------------------
    alloc_dev_(&input_stage_f32_dev_,
               (size_t)MAX_PREFILL * (size_t)n_embd_ * sizeof(float));
    alloc_dev_(&output_stage_f32_dev_, (size_t)n_embd_ * sizeof(float));

    // ---- RoPE tables (precomputed on host, uploaded F16) ------------------
    cos_host_.assign((size_t)MAX_SEQ * (size_t)head_dim_, 0.0f);
    sin_host_.assign((size_t)MAX_SEQ * (size_t)head_dim_, 0.0f);
    alloc_dev_(&rope_cos_dev_,
               (size_t)MAX_SEQ * (size_t)head_dim_ * sizeof(__half));
    alloc_dev_(&rope_sin_dev_,
               (size_t)MAX_SEQ * (size_t)head_dim_ * sizeof(__half));
    build_rope_tables_();

    // ---- Decode-graph cache placeholder (Phase 2.5) -----------------------
    decode_graph_execs_.assign(MAX_SEQ, nullptr);

    // Phase 2.5 — env toggle. Mirrors Ascend TALKER_CP_ACLGRAPH=1 lever.
    // Engine setter still works programmatically; this just gives smoke
    // tests a no-source-edit way to flip it on.
    if (const char *e = std::getenv("TALKER_USE_CUDA_GRAPHS")) {
        if (e[0] == '1' || e[0] == 'y' || e[0] == 'Y' || e[0] == 't' || e[0] == 'T') {
            use_cuda_graphs_ = true;
            fprintf(stderr,
                    "[talker_cuda] CUDA Graphs enabled via "
                    "TALKER_USE_CUDA_GRAPHS=%s\n", e);
        }
    }

    // Free GGUF context — Phase 2.2 re-opens it inside a dedicated weight-
    // upload helper that does the per-layer cudaMemcpyAsync.
    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    ready_ = true;
    fprintf(stderr,
            "[talker_cuda] Phase 2.1 scaffold init OK  device=%d  "
            "n_embd=%d  n_heads=%d  n_kv=%d  head_dim=%d  inter=%d  "
            "n_layers=%d  MAX_SEQ=%d  MAX_PREFILL=%d\n",
            device_, n_embd_, n_heads_, n_kv_, head_dim_, inter_,
            n_layers_, MAX_SEQ, MAX_PREFILL);
    return true;
}

// ---------------------------------------------------------------------------
// build_rope_tables_  — host precompute then H2D F16 upload.
//
// Layout matches the Ascend reference: [MAX_SEQ, head_dim] with the two halves
// (rotate_half) duplicated so a NEOX-mode RoPE kernel reads cos[r, d] and
// sin[r, d] symmetrically. Sector-aware xvec layout zeroes pair-indices >=
// mrope_temporal_section_.
// ---------------------------------------------------------------------------

void TalkerCudaEngine::build_rope_tables_() {
    if (!rope_cos_dev_ || !rope_sin_dev_) return;
    const int half = head_dim_ / 2;
    const float theta = rope_theta_;
    for (int pos = 0; pos < MAX_SEQ; ++pos) {
        const float scaled_pos = (float)pos * rope_speed_factor_;
        for (int j = 0; j < half; ++j) {
            float inv_freq = 1.0f / std::pow(theta, (float)(2 * j) / (float)head_dim_);
            float c = std::cos(scaled_pos * inv_freq);
            float s = std::sin(scaled_pos * inv_freq);
            const bool zero_pair = use_mrope_xvec_layout_ &&
                                   j >= mrope_temporal_section_;
            if (zero_pair) { c = 1.0f; s = 0.0f; }
            cos_host_[(size_t)pos * head_dim_ + j]        = c;
            cos_host_[(size_t)pos * head_dim_ + j + half] = c;
            sin_host_[(size_t)pos * head_dim_ + j]        = s;
            sin_host_[(size_t)pos * head_dim_ + j + half] = s;
        }
    }

    // F32 -> F16 staging then H2D. Doing the cast on host is fine — this is
    // off the hot path.
    std::vector<__half> tmp_cos(cos_host_.size());
    std::vector<__half> tmp_sin(sin_host_.size());
    for (size_t i = 0; i < cos_host_.size(); ++i) {
        tmp_cos[i] = __float2half(cos_host_[i]);
        tmp_sin[i] = __float2half(sin_host_[i]);
    }
    cudaMemcpyAsync(rope_cos_dev_, tmp_cos.data(),
                    tmp_cos.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(rope_sin_dev_, tmp_sin.data(),
                    tmp_sin.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, stream_);
    cudaStreamSynchronize(stream_);
}

void TalkerCudaEngine::build_causal_mask_() {
    // Wired in Phase 2.2 alongside FMHA. Left as a stub — masks are recomputed
    // per prefill chunk on Ascend, and we keep that contract here.
}

// ---------------------------------------------------------------------------
// Per-token forward — Phase 2.2 (S=1 decode body).
//
// The per-layer op sequence mirrors the Ascend reference (talker_cann_engine
// run_decode_body_) one-for-one:
//   pre_norm -> Q/K/V proj -> QK-norm -> RoPE Q/K -> KV cache write
//     -> GQA attention -> O proj -> residual
//   ffn_norm -> gate proj + up proj -> SwiGLU -> down proj -> residual
// then a final RmsNorm and an F16->F32 cast into output_stage_f32_dev_.
//
// Shape conventions:
//   weights are stored row-major [out, in] in the GGUF and uploaded
//   bit-cast to F16. cuBLAS treats them as column-major [in, out] of the
//   same bytes, so we use op_A = T (transpose) when computing y = W @ x:
//     y_col[out,1] = (W_col[in,out])^T @ x_col[in,1]
//
// All ops issue on stream_; the inner loop has no host syncs.
// ---------------------------------------------------------------------------

namespace {

// Helper: y[out] = W[out, in] @ x[in], all F16. W is stored row-major
// [out, in] which cuBLAS sees as column-major [in, out] -> we use op_A = T.
// Compute type is F32 (alpha/beta MUST be float per cuBLAS doc); I/O F16.
inline cublasStatus_t gemm_rowmajor_matvec_f16(cublasHandle_t cublas,
                                                 const __half *W,
                                                 const __half *x,
                                                 __half *y,
                                                 int out, int in_dim) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    // cuBLAS column-major:
    //   C[out, 1] = alpha * op(A)[out, in] @ op(B)[in, 1] + beta * C
    //   op(A) = T over A_storage [in, out] with lda = in.
    //   B = x with ldb = in. C = y with ldc = out. m = out, n = 1, k = in.
    return cublasGemmEx(
        cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/out, /*n=*/1, /*k=*/in_dim,
        &alpha,
        W, CUDA_R_16F, /*lda=*/in_dim,
        x, CUDA_R_16F, /*ldb=*/in_dim,
        &beta,
        y, CUDA_R_16F, /*ldc=*/out,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

}  // namespace

void TalkerCudaEngine::run_decode_ops_(int pos) {
    // Per-position cos/sin row pointers into the precomputed RoPE table.
    const __half *rope_cos_row = (const __half *)rope_cos_dev_ +
                                  (size_t)pos * head_dim_;
    const __half *rope_sin_row = (const __half *)rope_sin_dev_ +
                                  (size_t)pos * head_dim_;

    const int seq_len_total = pos + 1;
    const float inv_sqrt_d  = 1.0f / std::sqrt((float)head_dim_);

    for (int il = 0; il < n_layers_; ++il) {
        const auto &lw = layer_w_[il];

        // residual <- cur (D2D F16 copy).
        cudaMemcpyAsync(residual_dev_, cur_dev_,
                         (size_t)n_embd_ * sizeof(__half),
                         cudaMemcpyDeviceToDevice, stream_);

        // 1. Input RmsNorm (F16 x, F32 gamma) -> normed.
        launch_rmsnorm_f16_g32((const __half *)cur_dev_,
                                 (const float *)lw.input_ln_w,
                                 (__half *)normed_dev_,
                                 /*rows=*/1, /*cols=*/n_embd_, eps_, stream_);

        // 2. Q/K/V projections.
        gemm_rowmajor_matvec_f16(cublas_,
            (const __half *)lw.q_proj_w, (const __half *)normed_dev_,
            (__half *)q_dev_, q_dim_,  n_embd_);
        gemm_rowmajor_matvec_f16(cublas_,
            (const __half *)lw.k_proj_w, (const __half *)normed_dev_,
            (__half *)k_dev_, kv_dim_, n_embd_);
        gemm_rowmajor_matvec_f16(cublas_,
            (const __half *)lw.v_proj_w, (const __half *)normed_dev_,
            (__half *)v_dev_, kv_dim_, n_embd_);

        // 3. QK-norm (per-head RmsNorm; shared gamma [head_dim]). In place.
        launch_rmsnorm_f16_g32((const __half *)q_dev_,
                                 (const float *)lw.q_norm_w,
                                 (__half *)q_dev_,
                                 /*rows=*/n_heads_, /*cols=*/head_dim_,
                                 eps_, stream_);
        launch_rmsnorm_f16_g32((const __half *)k_dev_,
                                 (const float *)lw.k_norm_w,
                                 (__half *)k_dev_,
                                 /*rows=*/n_kv_, /*cols=*/head_dim_,
                                 eps_, stream_);

        // 4a. RoPE on Q (writes to attn_out_dev_ — non-aliasing).
        launch_rope_neox_f16((const __half *)q_dev_,
                              rope_cos_row, rope_sin_row,
                              (__half *)attn_out_dev_,
                              n_heads_, head_dim_, stream_);

        // 4b. RoPE on K -> directly into KV cache slot at this `pos`.
        __half *k_slot = (__half *)k_cache_dev_[il] + (size_t)pos * kv_dim_;
        launch_rope_neox_f16((const __half *)k_dev_,
                              rope_cos_row, rope_sin_row,
                              k_slot,
                              n_kv_, head_dim_, stream_);

        // 5. V -> KV cache slot.
        __half *v_slot = (__half *)v_cache_dev_[il] + (size_t)pos * kv_dim_;
        cudaMemcpyAsync(v_slot, v_dev_, (size_t)kv_dim_ * sizeof(__half),
                         cudaMemcpyDeviceToDevice, stream_);

        // 6. GQA attention (S=1 decode). Q comes from attn_out_dev_ (post-RoPE);
        //    output overwrites q_dev_ as a contiguous [n_heads, head_dim]
        //    so the next O projection sees a tightly packed buffer.
        launch_attn_decode_gqa_f16(
            (const __half *)attn_out_dev_,
            (const __half *)k_cache_dev_[il],
            (const __half *)v_cache_dev_[il],
            (__half *)q_dev_,
            seq_len_total, n_heads_, n_kv_, head_dim_, inv_sqrt_d,
            stream_);

        // 7. O projection: o_out = W_o @ q_attn.
        gemm_rowmajor_matvec_f16(cublas_,
            (const __half *)lw.o_proj_w, (const __half *)q_dev_,
            (__half *)o_out_dev_, n_embd_, q_dim_);

        // 8. cur = residual + o_out.
        launch_add_f16((const __half *)residual_dev_,
                        (const __half *)o_out_dev_,
                        (__half *)cur_dev_, n_embd_, stream_);

        // residual <- cur (for FFN).
        cudaMemcpyAsync(residual_dev_, cur_dev_,
                         (size_t)n_embd_ * sizeof(__half),
                         cudaMemcpyDeviceToDevice, stream_);

        // 9. Post-attention RmsNorm.
        launch_rmsnorm_f16_g32((const __half *)cur_dev_,
                                 (const float *)lw.post_ln_w,
                                 (__half *)normed_dev_,
                                 /*rows=*/1, /*cols=*/n_embd_, eps_, stream_);

        // 10. FFN gate + up projections (parallelizable; same input).
        gemm_rowmajor_matvec_f16(cublas_,
            (const __half *)lw.gate_proj_w, (const __half *)normed_dev_,
            (__half *)gate_dev_, inter_, n_embd_);
        gemm_rowmajor_matvec_f16(cublas_,
            (const __half *)lw.up_proj_w, (const __half *)normed_dev_,
            (__half *)up_dev_, inter_, n_embd_);

        // SwiGLU: gate <- silu(gate) * up.
        launch_swiglu_f16((const __half *)gate_dev_,
                           (const __half *)up_dev_,
                           (__half *)gate_dev_, inter_, stream_);

        // 11. Down projection: ffn_out = W_down @ swiglu.
        gemm_rowmajor_matvec_f16(cublas_,
            (const __half *)lw.down_proj_w, (const __half *)gate_dev_,
            (__half *)ffn_out_dev_, n_embd_, inter_);

        // 12. cur = residual + ffn_out.
        launch_add_f16((const __half *)residual_dev_,
                        (const __half *)ffn_out_dev_,
                        (__half *)cur_dev_, n_embd_, stream_);
    }

    // Final RmsNorm.
    launch_rmsnorm_f16_g32((const __half *)cur_dev_,
                             (const float *)final_norm_w_dev_,
                             (__half *)normed_dev_,
                             /*rows=*/1, /*cols=*/n_embd_, eps_, stream_);

    // F16 normed -> F32 staging.
    launch_cast_f16_to_f32((const __half *)normed_dev_,
                             (float *)output_stage_f32_dev_,
                             n_embd_, stream_);
}

void TalkerCudaEngine::forward_decode(const float *input_embed,
                                       int pos, float *hidden_out) {
    if (!ready_) {
        fprintf(stderr, "[talker_cuda] forward_decode called before init\n");
        std::abort();
    }
    if (pos < 0 || pos >= MAX_SEQ) {
        fprintf(stderr, "[talker_cuda] forward_decode: pos %d out of range [0,%d)\n",
                pos, MAX_SEQ);
        std::abort();
    }

    // 1. Upload F32 input embedding to staging (sync H2D — small, off the
    //    inner loop). Then cast F32 -> F16 cur_dev_ on stream_.
    //    Kept outside CUDA-graph capture: H2D from pageable host memory
    //    is not always capturable, and the input pointer changes per call.
    cudaMemcpyAsync(input_stage_f32_dev_, input_embed,
                     (size_t)n_embd_ * sizeof(float),
                     cudaMemcpyHostToDevice, stream_);
    launch_cast_f32_to_f16((const float *)input_stage_f32_dev_,
                             (__half *)cur_dev_, n_embd_, stream_);

    // 2. Run the 28-layer body. All ops on stream_.
    //    Phase 2.5: when use_cuda_graphs_ is on, capture the body once per
    //    `pos` (KV-slot/RoPE-row pointers + seq_len_total are pos-baked),
    //    then replay on every subsequent visit. cur_dev_/residual_dev_/
    //    normed_dev_/etc. are stable buffers refilled per call by the
    //    upstream cast above, so replay sees fresh inputs each time.
    if (use_cuda_graphs_) {
        cudaGraphExec_t exec = decode_graph_execs_[pos];
        if (exec == nullptr) {
            // First visit at this pos — capture.
            cudaGraph_t graph = nullptr;
            cudaError_t cerr = cudaStreamBeginCapture(
                stream_, cudaStreamCaptureModeThreadLocal);
            if (cerr != cudaSuccess) {
                fprintf(stderr,
                        "[talker_cuda] cudaStreamBeginCapture failed at pos=%d: "
                        "%s — falling back to eager\n",
                        pos, cudaGetErrorString(cerr));
                run_decode_ops_(pos);
            } else {
                run_decode_ops_(pos);
                cerr = cudaStreamEndCapture(stream_, &graph);
                if (cerr != cudaSuccess || graph == nullptr) {
                    fprintf(stderr,
                            "[talker_cuda] cudaStreamEndCapture failed at "
                            "pos=%d: %s — falling back to eager\n",
                            pos, cudaGetErrorString(cerr));
                } else {
                    cerr = cudaGraphInstantiate(&exec, graph,
                                                 nullptr, nullptr, 0);
                    cudaGraphDestroy(graph);
                    if (cerr != cudaSuccess || exec == nullptr) {
                        fprintf(stderr,
                                "[talker_cuda] cudaGraphInstantiate failed at "
                                "pos=%d: %s — falling back to eager\n",
                                pos, cudaGetErrorString(cerr));
                        exec = nullptr;
                    } else {
                        decode_graph_execs_[pos] = exec;
                        cudaError_t lerr =
                            cudaGraphLaunch(exec, stream_);
                        if (lerr != cudaSuccess) {
                            fprintf(stderr,
                                    "[talker_cuda] cudaGraphLaunch (post-cap) "
                                    "failed at pos=%d: %s\n",
                                    pos, cudaGetErrorString(lerr));
                        }
                    }
                }
            }
        } else {
            // Replay path — the hot loop.
            cudaError_t lerr = cudaGraphLaunch(exec, stream_);
            if (lerr != cudaSuccess) {
                fprintf(stderr,
                        "[talker_cuda] cudaGraphLaunch failed at pos=%d: "
                        "%s — falling back to eager\n",
                        pos, cudaGetErrorString(lerr));
                run_decode_ops_(pos);
            }
        }
    } else {
        run_decode_ops_(pos);
    }

    // 3. D2H download of F32 output. Synchronizes so the caller can read
    //    `hidden_out` immediately (this matches the Ascend
    //    forward_decode_fetch behaviour for the eager / non-graph path).
    cudaMemcpyAsync(hidden_out, output_stage_f32_dev_,
                     (size_t)n_embd_ * sizeof(float),
                     cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    if (kv_cache_len_ < pos + 1) kv_cache_len_ = pos + 1;
}

void TalkerCudaEngine::forward_prefill(const float * /*input_embeds*/,
                                        int /*seq_len*/, int /*start_pos*/,
                                        float * /*last_hidden_out*/) {
    fprintf(stderr,
            "[talker_cuda] forward_prefill not yet implemented (Phase 2.2 — "
            "decode-only deliverable; prefill lands in Phase 2.3)\n");
    std::abort();
}

bool TalkerCudaEngine::int8_calibrate_weight_(const float * /*host_w*/,
                                                int64_t /*rows*/, int64_t /*cols*/,
                                                void *& /*weight_i8_dev*/,
                                                void *& /*scale_dev*/) {
    fprintf(stderr,
            "[talker_cuda] int8_calibrate_weight_ not yet implemented (Phase 2.6)\n");
    return false;
}

// ============================================================================
// Phase 2.9 — device LM-head + per-step logits via cuBLAS.
//
// Replaces the host-side O(vocab * n_embd) matvec that the predictor
// performed at every group step. For the predictor that's
// 30720 * 1024 = ~31 M scalar ops per step on the CPU; on a 5-layer GPU
// path that single host matvec was an order of magnitude more expensive
// than the entire device decode it followed. After uploading the
// LM-head weights once, we run cuBLAS GEMM on device and only D2H the
// `vocab` floats of logits per step (~120 KB instead of needing 4 KB
// hidden D2H + 31 M host FLOPs).
// ============================================================================

bool TalkerCudaEngine::upload_lm_head_weights(const float *lm_head_w_f32,
                                              int vocab) {
    if (!ready_) {
        fprintf(stderr,
                "[talker_cuda] upload_lm_head_weights called before init\n");
        return false;
    }
    if (lm_head_w_f32 == nullptr || vocab <= 0) {
        fprintf(stderr,
                "[talker_cuda] upload_lm_head_weights: bad args (w=%p, vocab=%d)\n",
                (const void *)lm_head_w_f32, vocab);
        return false;
    }
    cudaSetDevice(device_);

    // Free any previous upload (allow re-binding).
    auto safe_free = [](void *&p) {
        if (p) { cudaFree(p); p = nullptr; }
    };
    safe_free(lm_head_w_f16_dev_);
    safe_free(lm_head_hidden_f16_dev_);
    safe_free(lm_head_logits_f32_dev_);

    const size_t n_w = (size_t)vocab * (size_t)n_embd_;
    std::vector<__half> w_f16(n_w);
    for (size_t i = 0; i < n_w; ++i) w_f16[i] = __float2half(lm_head_w_f32[i]);

    alloc_dev_(&lm_head_w_f16_dev_, n_w * sizeof(__half));
    cudaError_t err = cudaMemcpy(lm_head_w_f16_dev_, w_f16.data(),
                                  n_w * sizeof(__half),
                                  cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "[talker_cuda] upload_lm_head_weights: cudaMemcpy failed: %s\n",
                cudaGetErrorString(err));
        return false;
    }
    alloc_dev_(&lm_head_hidden_f16_dev_, (size_t)n_embd_ * sizeof(__half));
    alloc_dev_(&lm_head_logits_f32_dev_, (size_t)vocab * sizeof(float));
    lm_head_vocab_ = vocab;
    fprintf(stderr,
            "[talker_cuda] Phase 2.9 LM-head uploaded: vocab=%d hidden=%d "
            "(F16, %.2f MB)\n",
            vocab, n_embd_,
            (double)(n_w * sizeof(__half)) / (1024.0 * 1024.0));
    return true;
}

namespace {

// Helper: F32 logits[out] = W_f16[out, in] @ x_f16[in], cuBLAS GemmEx with
// F32 accumulation. Mirrors gemm_rowmajor_matvec_f16 above but emits F32.
inline cublasStatus_t gemm_lm_head_f16_to_f32(cublasHandle_t cublas,
                                                const __half *W,
                                                const __half *x,
                                                float *y,
                                                int out, int in_dim) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    return cublasGemmEx(
        cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        /*m=*/out, /*n=*/1, /*k=*/in_dim,
        &alpha,
        W, CUDA_R_16F, /*lda=*/in_dim,
        x, CUDA_R_16F, /*ldb=*/in_dim,
        &beta,
        y, CUDA_R_32F, /*ldc=*/out,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

}  // namespace

void TalkerCudaEngine::forward_decode_with_logits(const float *input_embed,
                                                   int pos,
                                                   float *logits_out_f32) {
    if (!ready_) {
        fprintf(stderr,
                "[talker_cuda] forward_decode_with_logits before init\n");
        std::abort();
    }
    if (lm_head_w_f16_dev_ == nullptr) {
        fprintf(stderr,
                "[talker_cuda] forward_decode_with_logits: LM-head not uploaded "
                "(call upload_lm_head_weights first)\n");
        std::abort();
    }
    if (pos < 0 || pos >= MAX_SEQ) {
        fprintf(stderr,
                "[talker_cuda] forward_decode_with_logits: pos %d out of range\n",
                pos);
        std::abort();
    }

    // Stage input (same as forward_decode).
    cudaMemcpyAsync(input_stage_f32_dev_, input_embed,
                     (size_t)n_embd_ * sizeof(float),
                     cudaMemcpyHostToDevice, stream_);
    launch_cast_f32_to_f16((const float *)input_stage_f32_dev_,
                             (__half *)cur_dev_, n_embd_, stream_);

    // Run the decode body. Reuse the per-pos CUDA-graph cache if enabled —
    // identical body to forward_decode(), so the cache is shared and a
    // single capture covers both call paths once both have visited a
    // given pos.
    if (use_cuda_graphs_) {
        cudaGraphExec_t exec = decode_graph_execs_[pos];
        if (exec == nullptr) {
            cudaGraph_t graph = nullptr;
            cudaError_t cerr = cudaStreamBeginCapture(
                stream_, cudaStreamCaptureModeThreadLocal);
            if (cerr != cudaSuccess) {
                run_decode_ops_(pos);
            } else {
                run_decode_ops_(pos);
                cerr = cudaStreamEndCapture(stream_, &graph);
                if (cerr == cudaSuccess && graph != nullptr) {
                    cerr = cudaGraphInstantiate(&exec, graph, nullptr,
                                                 nullptr, 0);
                    cudaGraphDestroy(graph);
                    if (cerr == cudaSuccess && exec != nullptr) {
                        decode_graph_execs_[pos] = exec;
                        cudaGraphLaunch(exec, stream_);
                    }
                }
            }
        } else {
            cudaError_t lerr = cudaGraphLaunch(exec, stream_);
            if (lerr != cudaSuccess) run_decode_ops_(pos);
        }
    } else {
        run_decode_ops_(pos);
    }

    if (use_fp8_lm_head_ && lm_head_w_fp8_dev_ != nullptr) {
        // -------- FP8 lane (Phase 2.6) ---------------------------------
        // Quantize hidden state F32 -> FP8 E4M3 (inv_scale = 1.0; weight
        // scale is composed by cublasLt via A-scale pointer).
        launch_cast_f32_to_fp8_e4m3_scaled(
            (const float *)output_stage_f32_dev_,
            lm_head_x_fp8_dev_,
            /*inv_scale=*/1.0f,
            n_embd_, stream_);

        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasLtMatmul(
            (cublasLtHandle_t)lm_head_lt_handle_,
            (cublasLtMatmulDesc_t)lm_head_lt_desc_,
            &alpha,
            lm_head_w_fp8_dev_,                              // A: W FP8 [in,out]^T
            (cublasLtMatrixLayout_t)lm_head_lt_layout_a_,
            lm_head_x_fp8_dev_,                              // B: x FP8 [in,1]
            (cublasLtMatrixLayout_t)lm_head_lt_layout_b_,
            &beta,
            lm_head_logits_f16_dev_,                         // C (unused, beta=0)
            (cublasLtMatrixLayout_t)lm_head_lt_layout_d_,
            lm_head_logits_f16_dev_,                         // D: F16 [vocab,1]
            (cublasLtMatrixLayout_t)lm_head_lt_layout_d_,
            &((cublasLtMatmulHeuristicResult_t *)lm_head_lt_algo_blob_)->algo,
            lm_head_lt_workspace_, lm_head_lt_ws_bytes_,
            stream_);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[talker_cuda] FP8 LM-head cublasLtMatmul failed (status=%d)\n",
                    (int)st);
            std::abort();
        }
        // Widen F16 logits -> F32 (consumers expect F32).
        launch_cast_f16_to_f32((const __half *)lm_head_logits_f16_dev_,
                                 (float *)lm_head_logits_f32_dev_,
                                 lm_head_vocab_, stream_);
    } else {
        // -------- F16 lane (Phase 2.9 reference) -----------------------
        launch_cast_f32_to_f16((const float *)output_stage_f32_dev_,
                                 (__half *)lm_head_hidden_f16_dev_,
                                 n_embd_, stream_);
        cublasStatus_t st = gemm_lm_head_f16_to_f32(cublas_,
                                  (const __half *)lm_head_w_f16_dev_,
                                  (const __half *)lm_head_hidden_f16_dev_,
                                  (float *)lm_head_logits_f32_dev_,
                                  lm_head_vocab_, n_embd_);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[talker_cuda] LM-head cuBLAS GemmEx failed (status=%d)\n",
                    (int)st);
            std::abort();
        }
    }

    // D2H the F32 logits — this is the only sync per step (~vocab*4 bytes).
    cudaMemcpyAsync(logits_out_f32, lm_head_logits_f32_dev_,
                     (size_t)lm_head_vocab_ * sizeof(float),
                     cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    if (kv_cache_len_ < pos + 1) kv_cache_len_ = pos + 1;
}

// ============================================================================
// Phase 2.6 — FP8 E4M3 LM-head lane.
//
// Microbench (Apr 2026) on Blackwell GB10 sm_121a + CUDA 13.0.88:
//   shape                      | f16 us | fp8 us | speedup
//   pred_lm_head [30720,1024]  |  320   |  133   |  2.40x   <- only big win
//   talker_lm_head [3072,2048] |   14   |   20   |  0.73x
//   ffn shapes (matvec K=1)    |   ~30  |   ~35  |  ~0.85x
//
// Therefore we plumb FP8 ONLY into forward_decode_with_logits (predictor
// LM-head fires 1920 times per request). The Talker LM-head runs on the
// host side anyway in the production e2e harness, so it stays out of
// scope. FFN matmuls remain F16.
//
// Numerics:
//   weight: per-tensor amax-derived F32 scale_w = amax(W) / E4M3_MAX
//           (E4M3_MAX = 448.0). Stored compressed: w_fp8 = E4M3(W / scale_w).
//   activ : input range varies, we use a fixed scale_a = 1.0 (E4M3_MAX is
//           large enough for hidden state values; if needed, future work
//           tracks running amax per request). The activation cast also
//           applies inv_scale_a = 1.0.
//   GEMM  : cublasLtMatmul with A_scale=&scale_w, B_scale=&one. The library
//           computes D[M,N] = scale_w * 1.0 * (W_fp8^T · x_fp8) and emits F16.
//   We then widen F16 D -> F32 via launch_cast_f16_to_f32 and D2H to host.
//
// Fallback policy: if upload_lm_head_weights_fp8() fails to find a cublasLt
// algo for the shape, the FP8 lane stays unpopulated and use_fp8_lm_head()
// silently keeps returning false; the caller falls back to F16 GEMM.
// ============================================================================

namespace {

// E4M3 representable max (per CUDA fp8 spec).
constexpr float kE4M3_MAX = 448.0f;

// Helper: free a cublasLt handle/desc/layout/preference set, in safe order.
void destroy_lt_resources(cublasLtHandle_t  &handle,
                           cublasLtMatmulDesc_t &desc,
                           cublasLtMatrixLayout_t &aL,
                           cublasLtMatrixLayout_t &bL,
                           cublasLtMatrixLayout_t &dL,
                           cublasLtMatmulPreference_t &pref) {
    if (pref) { cublasLtMatmulPreferenceDestroy(pref); pref = nullptr; }
    if (aL)   { cublasLtMatrixLayoutDestroy(aL);  aL = nullptr; }
    if (bL)   { cublasLtMatrixLayoutDestroy(bL);  bL = nullptr; }
    if (dL)   { cublasLtMatrixLayoutDestroy(dL);  dL = nullptr; }
    if (desc) { cublasLtMatmulDescDestroy(desc);  desc = nullptr; }
    if (handle) { cublasLtDestroy(handle); handle = nullptr; }
}

}  // namespace

void TalkerCudaEngine::teardown_fp8_lm_head_() {
    if (lm_head_w_fp8_dev_)      { cudaFree(lm_head_w_fp8_dev_);      lm_head_w_fp8_dev_ = nullptr; }
    if (lm_head_x_fp8_dev_)      { cudaFree(lm_head_x_fp8_dev_);      lm_head_x_fp8_dev_ = nullptr; }
    if (lm_head_logits_f16_dev_) { cudaFree(lm_head_logits_f16_dev_); lm_head_logits_f16_dev_ = nullptr; }
    if (lm_head_scale_a_dev_)    { cudaFree(lm_head_scale_a_dev_);    lm_head_scale_a_dev_ = nullptr; }
    if (lm_head_scale_b_dev_)    { cudaFree(lm_head_scale_b_dev_);    lm_head_scale_b_dev_ = nullptr; }
    if (lm_head_lt_workspace_)   { cudaFree(lm_head_lt_workspace_);   lm_head_lt_workspace_ = nullptr; }

    auto pref = (cublasLtMatmulPreference_t)lm_head_lt_pref_;
    auto desc = (cublasLtMatmulDesc_t)lm_head_lt_desc_;
    auto aL   = (cublasLtMatrixLayout_t)lm_head_lt_layout_a_;
    auto bL   = (cublasLtMatrixLayout_t)lm_head_lt_layout_b_;
    auto dL   = (cublasLtMatrixLayout_t)lm_head_lt_layout_d_;
    auto lt   = (cublasLtHandle_t)lm_head_lt_handle_;
    destroy_lt_resources(lt, desc, aL, bL, dL, pref);
    lm_head_lt_handle_   = nullptr;
    lm_head_lt_desc_     = nullptr;
    lm_head_lt_layout_a_ = nullptr;
    lm_head_lt_layout_b_ = nullptr;
    lm_head_lt_layout_d_ = nullptr;
    lm_head_lt_pref_     = nullptr;
    if (lm_head_lt_algo_blob_) {
        std::free(lm_head_lt_algo_blob_);
        lm_head_lt_algo_blob_ = nullptr;
    }
    use_fp8_lm_head_ = false;
}

bool TalkerCudaEngine::upload_lm_head_weights_fp8(const float *lm_head_w_f32,
                                                   int vocab) {
    if (!ready_) {
        fprintf(stderr,
                "[talker_cuda] upload_lm_head_weights_fp8 before init\n");
        return false;
    }
    if (lm_head_w_f32 == nullptr || vocab <= 0) {
        fprintf(stderr,
                "[talker_cuda] upload_lm_head_weights_fp8: bad args\n");
        return false;
    }
    if (vocab != lm_head_vocab_ && lm_head_vocab_ != 0) {
        // Caller mixed F16 and FP8 vocab sizes — bail.
        fprintf(stderr,
                "[talker_cuda] upload_lm_head_weights_fp8: vocab mismatch (had %d new %d)\n",
                lm_head_vocab_, vocab);
        return false;
    }

    teardown_fp8_lm_head_();

    const size_t n_w = (size_t)vocab * (size_t)n_embd_;

    // 1) Compute per-tensor amax over the F32 weight (host).
    float amax = 0.0f;
    for (size_t i = 0; i < n_w; ++i) {
        float a = std::fabs(lm_head_w_f32[i]);
        if (a > amax) amax = a;
    }
    if (amax <= 0.0f) {
        fprintf(stderr,
                "[talker_cuda] upload_lm_head_weights_fp8: zero weight tensor\n");
        return false;
    }
    const float scale_w = amax / kE4M3_MAX;
    const float inv_scale_w = 1.0f / scale_w;

    // 2) Quantize weights to E4M3 on host, copy to device.
    std::vector<__nv_fp8_e4m3> w_fp8(n_w);
    for (size_t i = 0; i < n_w; ++i) {
        w_fp8[i] = __nv_fp8_e4m3(lm_head_w_f32[i] * inv_scale_w);
    }
    void *w_fp8_dev = nullptr;
    if (cudaMalloc(&w_fp8_dev, n_w) != cudaSuccess) return false;
    if (cudaMemcpy(w_fp8_dev, w_fp8.data(), n_w, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(w_fp8_dev);
        return false;
    }

    // 3) Allocate FP8 input staging + F16 logits buffer + scale buffers.
    void *x_fp8_dev = nullptr;
    if (cudaMalloc(&x_fp8_dev, (size_t)n_embd_) != cudaSuccess) {
        cudaFree(w_fp8_dev); return false;
    }
    void *logits_f16_dev = nullptr;
    if (cudaMalloc(&logits_f16_dev, (size_t)vocab * sizeof(__half)) != cudaSuccess) {
        cudaFree(w_fp8_dev); cudaFree(x_fp8_dev); return false;
    }
    float *scale_a_dev = nullptr, *scale_b_dev = nullptr;
    if (cudaMalloc(&scale_a_dev, sizeof(float)) != cudaSuccess) {
        cudaFree(w_fp8_dev); cudaFree(x_fp8_dev); cudaFree(logits_f16_dev); return false;
    }
    if (cudaMalloc(&scale_b_dev, sizeof(float)) != cudaSuccess) {
        cudaFree(w_fp8_dev); cudaFree(x_fp8_dev); cudaFree(logits_f16_dev);
        cudaFree(scale_a_dev); return false;
    }
    const float scale_b_host = 1.0f;
    cudaMemcpy(scale_a_dev, &scale_w,      sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(scale_b_dev, &scale_b_host, sizeof(float), cudaMemcpyHostToDevice);

    // 4) cublasLt setup. Layouts: A is row-major [vocab, n_embd] which we
    //    feed as col-major [n_embd, vocab] with op_A=T. B is [n_embd, 1].
    //    D is [vocab, 1] in F16.
    cublasLtHandle_t lt = nullptr;
    if (cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
        cudaFree(w_fp8_dev); cudaFree(x_fp8_dev); cudaFree(logits_f16_dev);
        cudaFree(scale_a_dev); cudaFree(scale_b_dev);
        return false;
    }

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t aL = nullptr, bL = nullptr, dL = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;

    auto cleanup_fail = [&]() {
        destroy_lt_resources(lt, desc, aL, bL, dL, pref);
        cudaFree(w_fp8_dev); cudaFree(x_fp8_dev); cudaFree(logits_f16_dev);
        cudaFree(scale_a_dev); cudaFree(scale_b_dev);
    };

    if (cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
            != CUBLAS_STATUS_SUCCESS) { cleanup_fail(); return false; }
    cublasOperation_t op_A = CUBLAS_OP_T;
    cublasOperation_t op_B = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(op_A));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(op_B));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                    &scale_a_dev, sizeof(scale_a_dev));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                    &scale_b_dev, sizeof(scale_b_dev));

    // A col-major view: rows=in_dim (n_embd), cols=out (vocab), ld=in_dim.
    if (cublasLtMatrixLayoutCreate(&aL, CUDA_R_8F_E4M3, n_embd_, vocab, n_embd_)
            != CUBLAS_STATUS_SUCCESS) { cleanup_fail(); return false; }
    if (cublasLtMatrixLayoutCreate(&bL, CUDA_R_8F_E4M3, n_embd_, 1, n_embd_)
            != CUBLAS_STATUS_SUCCESS) { cleanup_fail(); return false; }
    if (cublasLtMatrixLayoutCreate(&dL, CUDA_R_16F, vocab, 1, vocab)
            != CUBLAS_STATUS_SUCCESS) { cleanup_fail(); return false; }

    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
        cleanup_fail(); return false;
    }
    size_t ws_bytes = 16ull * 1024ull * 1024ull;  // 16 MB
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &ws_bytes, sizeof(ws_bytes));

    cublasLtMatmulHeuristicResult_t *algo_blob =
        (cublasLtMatmulHeuristicResult_t *)std::malloc(sizeof(cublasLtMatmulHeuristicResult_t));
    int n_results = 0;
    cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(
        lt, desc, aL, bL, dL, dL, pref, 1, algo_blob, &n_results);
    if (hst != CUBLAS_STATUS_SUCCESS || n_results == 0) {
        fprintf(stderr,
                "[talker_cuda] FP8 LM-head: no cublasLt algo for shape "
                "[%d,%d] (status=%d, n=%d) — falling back to F16\n",
                vocab, n_embd_, (int)hst, n_results);
        std::free(algo_blob);
        cleanup_fail();
        return false;
    }

    void *workspace = nullptr;
    if (cudaMalloc(&workspace, ws_bytes) != cudaSuccess) {
        std::free(algo_blob); cleanup_fail();
        return false;
    }

    // Persist resources.
    lm_head_w_fp8_dev_      = w_fp8_dev;
    lm_head_x_fp8_dev_      = x_fp8_dev;
    lm_head_logits_f16_dev_ = logits_f16_dev;
    lm_head_scale_a_dev_    = scale_a_dev;
    lm_head_scale_b_dev_    = scale_b_dev;
    lm_head_scale_a_host_   = scale_w;
    lm_head_lt_handle_      = lt;
    lm_head_lt_desc_        = desc;
    lm_head_lt_layout_a_    = aL;
    lm_head_lt_layout_b_    = bL;
    lm_head_lt_layout_d_    = dL;
    lm_head_lt_pref_        = pref;
    lm_head_lt_workspace_   = workspace;
    lm_head_lt_ws_bytes_    = ws_bytes;
    lm_head_lt_algo_blob_   = algo_blob;
    lm_head_vocab_          = vocab;

    // Allocate the F32 output buffer if the F16 lane hasn't already.
    if (lm_head_logits_f32_dev_ == nullptr) {
        alloc_dev_(&lm_head_logits_f32_dev_, (size_t)vocab * sizeof(float));
    }

    fprintf(stderr,
            "[talker_cuda] Phase 2.6 FP8 LM-head uploaded: vocab=%d hidden=%d  "
            "scale_w=%.6g (amax=%.4f, %.2f MB E4M3 + %.2f MB workspace)\n",
            vocab, n_embd_, scale_w, amax,
            (double)n_w / (1024.0 * 1024.0),
            (double)ws_bytes / (1024.0 * 1024.0));
    return true;
}

}  // namespace ominix_cuda
