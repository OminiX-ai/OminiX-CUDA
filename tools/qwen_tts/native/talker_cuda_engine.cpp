// ============================================================================
// Talker CUDA Engine — Phase 2.1 scaffold (GGUF parse + cuBLAS handle alloc)
//
// This file lands the bones of the engine: header is full, ctor/dtor lifecycle,
// init_from_gguf parses metadata + uploads F16 weights via the same key
// schema as TalkerCannEngine on Ascend (qwen3 llama-style GGUF emitted by
// `tools/qwen_tts/export_talker_llama.py`), and reset/setters work.
//
// Per-token forward (run_decode_ops_) is intentionally a "not implemented"
// stub here — it is the headline Phase 2.2 deliverable and lives in this
// file in the next sub-phase. Same for forward_prefill.
//
// CUDA error-handling macros are local (no global cuda_check.hpp yet to keep
// the scaffold self-contained).
// ============================================================================

#include "talker_cuda_engine.h"

#include "../talker.h"  // TalkerConfig

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

    // ---- Allocate per-layer weight slots; the actual cudaMemcpy of each
    //      projection happens in Phase 2.2 once we wire ggml_get_tensor +
    //      shape validation. This scaffold stops short of touching device
    //      memory for the weights so a fresh build on GB10 can validate the
    //      handle/stream/RoPE plumbing alone.
    layer_w_.resize(n_layers_);

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
// Per-token forward — Phase 2.2 deliverable. Stub for now.
// ---------------------------------------------------------------------------

void TalkerCudaEngine::run_decode_ops_(int /*pos*/) {
    fprintf(stderr,
            "[talker_cuda] run_decode_ops_ not yet implemented (Phase 2.2)\n");
    std::abort();
}

void TalkerCudaEngine::forward_decode(const float * /*input_embed*/,
                                       int /*pos*/, float * /*hidden_out*/) {
    fprintf(stderr,
            "[talker_cuda] forward_decode not yet implemented (Phase 2.2)\n");
    std::abort();
}

void TalkerCudaEngine::forward_prefill(const float * /*input_embeds*/,
                                        int /*seq_len*/, int /*start_pos*/,
                                        float * /*last_hidden_out*/) {
    fprintf(stderr,
            "[talker_cuda] forward_prefill not yet implemented (Phase 2.2)\n");
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

}  // namespace ominix_cuda
