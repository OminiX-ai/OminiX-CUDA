// ============================================================================
// SpeechTokenizerDecoderCudaEngine — Phase 2.7a implementation.
//
// Scope (this commit): GGUF parse + RVQ decode only. The pre_conv,
// pre_transformer, upsample, and vocoder stages are scaffolded but abort
// with std::abort() in decode(); they land in 2.7b and 2.7c.
//
// RVQ decode runs as a host-side gather + cuBLAS GEMM:
//   1. For each timestep t and each codebook i, look up the F32 normalized
//      embedding row `embedding_sum[code, :] / cluster_usage[code]` (256-d).
//   2. Sum embeddings across codebooks within each RVQ group (1 first, 15 rest).
//   3. Project each group sum through its output_proj (Conv1d k=1 == matmul
//      of shape [rvq_out_dim, codebook_dim] @ [codebook_dim, T]).
//   4. Add the two group projections → [rvq_out_dim, T] F32.
//
// The Ascend reference fuses step 1 with `ggml_get_rows(embedding, codes)`
// over a precomputed `embedding_sum / cluster_usage` divide. We mirror that
// by precomputing `*_norm` tables at init time, so the hot path is a tight
// gather + accumulate.
//
// For Phase 2.7a the gather + accumulate runs on the host. The cuBLAS GEMM
// takes the device-uploaded sum as input and produces the device-side
// projection result, which is copied back as F32. This keeps the engine
// CUDA-resident at the boundary (device_ owns the cuBLAS handle + scratch)
// without committing to a fused CUDA gather kernel until 2.7b — at which
// point pre_conv consumes the device buffer in place.
// ============================================================================

#include "speech_tokenizer_decoder_cuda_engine.h"
#include "cuda_kernels/cuda_kernels.h"

#include "ggml.h"
#include "gguf.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <chrono>
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
            fprintf(stderr, "[stdec_cuda] CUDA error %s at %s:%d: %s\n",       \
                    #expr, __FILE__, __LINE__, cudaGetErrorString(_err));      \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define OMX_CUBLAS_CHECK(expr)                                                 \
    do {                                                                       \
        cublasStatus_t _st = (expr);                                           \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                    \
            fprintf(stderr, "[stdec_cuda] cuBLAS error %s at %s:%d: %d\n",     \
                    #expr, __FILE__, __LINE__, (int)_st);                      \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace ominix_cuda {

namespace {

// Read GGUF u32/f32/i32 KV with optional default tracking. Mirrors the
// qwen_common::ModelLoader pattern (cf. tools/qwen_common/model_loader.cpp).
bool gguf_read_u32(gguf_context *gctx, const char *key, int &out) {
    int64_t i = gguf_find_key(gctx, key);
    if (i < 0) return false;
    enum gguf_type t = gguf_get_kv_type(gctx, i);
    if (t == GGUF_TYPE_UINT32) {
        out = (int)gguf_get_val_u32(gctx, i);
        return true;
    } else if (t == GGUF_TYPE_INT32) {
        out = (int)gguf_get_val_i32(gctx, i);
        return true;
    }
    return false;
}

bool gguf_read_f32(gguf_context *gctx, const char *key, float &out) {
    int64_t i = gguf_find_key(gctx, key);
    if (i < 0) return false;
    enum gguf_type t = gguf_get_kv_type(gctx, i);
    if (t == GGUF_TYPE_FLOAT32) {
        out = gguf_get_val_f32(gctx, i);
        return true;
    }
    return false;
}

// Read array of int32 / uint32 (used for upsample_rates).
bool gguf_read_arr_i32(gguf_context *gctx, const char *key,
                        int *dst, int max_n, int &n_read) {
    int64_t i = gguf_find_key(gctx, key);
    if (i < 0) return false;
    enum gguf_type t = gguf_get_kv_type(gctx, i);
    if (t != GGUF_TYPE_ARRAY) return false;
    enum gguf_type at = gguf_get_arr_type(gctx, i);
    size_t n = gguf_get_arr_n(gctx, i);
    const void *data = gguf_get_arr_data(gctx, i);
    n_read = (int)((n < (size_t)max_n) ? n : (size_t)max_n);
    for (int k = 0; k < n_read; ++k) {
        switch (at) {
            case GGUF_TYPE_INT32:  dst[k] = ((const int32_t  *)data)[k]; break;
            case GGUF_TYPE_UINT32: dst[k] = (int)((const uint32_t *)data)[k]; break;
            case GGUF_TYPE_INT64:  dst[k] = (int)((const int64_t *)data)[k]; break;
            case GGUF_TYPE_UINT64: dst[k] = (int)((const uint64_t *)data)[k]; break;
            default: n_read = 0; return false;
        }
    }
    return true;
}

// Pull a tensor out of GGUF as F32 vector (handles F32 / F16 / quantized).
bool load_tensor_f32(ggml_context *gctx, const char *name,
                      size_t expected_elems, std::vector<float> &out) {
    ggml_tensor *t = ggml_get_tensor(gctx, name);
    if (!t) {
        fprintf(stderr, "[stdec_cuda] missing tensor: %s\n", name);
        return false;
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr, "[stdec_cuda] %s: expected %zu elems, got %zu\n",
                name, expected_elems, n);
        return false;
    }
    out.resize(n);
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
                    "[stdec_cuda] %s: unsupported dtype %d (no to_float trait)\n",
                    name, (int)t->type);
            return false;
        }
    }
    return true;
}

// Load one RVQ codebook layer (cluster_usage + embedding_sum). Returns false
// on a missing-tensor or shape-mismatch failure.
bool load_codebook(ggml_context *gctx, const std::string &prefix,
                    int codebook_dim, int codebook_size,
                    RVQCodebookHost &cb) {
    if (!load_tensor_f32(gctx, (prefix + "cluster_usage").c_str(),
                          codebook_size, cb.cluster_usage)) return false;
    if (!load_tensor_f32(gctx, (prefix + "embedding_sum").c_str(),
                          (size_t)codebook_dim * codebook_size,
                          cb.embedding_sum)) return false;
    return true;
}

// Precompute normalized codebook embedding: row r (length codebook_dim) is
// embedding_sum[:, r] / cluster_usage[r]. Output `norm` is laid out
// [codebook_size, codebook_dim] row-major (codebook_dim is fastest), so
// `&norm[code * codebook_dim]` is a contiguous codebook_dim slice — what
// rvq_decode wants for the per-timestep gather.
//
// embedding_sum is stored in GGUF as [codebook_dim, codebook_size] with
// codebook_dim fastest (i.e. the same row-of-256 layout we want), so the
// precompute is just an elementwise divide by the broadcast cluster_usage
// scalar for each row.
void precompute_codebook_norm(const RVQCodebookHost &cb,
                                int codebook_dim, int codebook_size,
                                std::vector<float> &norm) {
    norm.resize((size_t)codebook_size * codebook_dim);
    const float *emb = cb.embedding_sum.data();
    const float *use = cb.cluster_usage.data();
    for (int r = 0; r < codebook_size; ++r) {
        float u = use[r];
        // Guard against zero usage (matches the implicit ggml semantics:
        // ggml_div would produce inf/nan; we substitute 1.0 to keep the
        // sum finite. Real codebooks never have zero usage, but fixturing
        // can hit it). The Ascend reference relies on the trained model
        // never hitting 0; mirror that contract loudly.
        if (u == 0.0f) u = 1.0f;
        float inv = 1.0f / u;
        const float *src = emb + (size_t)r * codebook_dim;
        float *dst = norm.data() + (size_t)r * codebook_dim;
        for (int c = 0; c < codebook_dim; ++c) {
            dst[c] = src[c] * inv;
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Persistent per-instance handles. cuBLAS handle + a single primary stream
// are created in init_from_gguf and torn down in the dtor. Phase 2.7a uses
// these only for the RVQ output_proj GEMM; later phases reuse them.
// ---------------------------------------------------------------------------

struct UpsampleBlockDev {
    // ConvTranspose1d k=2 s=2 (rearranged to [K, C_out, C_in] row-major,
    // K outer, C_in fastest — matches launch_conv_transpose1d_k2s2_f32).
    float *up_w = nullptr;   // [2, C_out, C_in] = [2, 1024, 1024]
    float *up_b = nullptr;   // [C_out]

    // ConvNeXt: dwconv k=7 (rearranged to [K, C] row-major, k outer, c fastest).
    float *dw_w = nullptr;       // [7, 1024]
    float *dw_b = nullptr;       // [1024]
    // LayerNorm gamma/beta.
    float *norm_w = nullptr;     // [1024]
    float *norm_b = nullptr;     // [1024]
    // pwconv1: nn.Linear(1024 -> 4096); GGUF stores PyTorch [4096, 1024]
    // row-major bytes (so col-major view = [1024, 4096], lda=1024 in our gemm).
    float *pw1_w = nullptr;      // [4096, 1024] PyTorch row-major
    float *pw1_b = nullptr;      // [4096]
    // pwconv2: nn.Linear(4096 -> 1024); PyTorch [1024, 4096] row-major.
    float *pw2_w = nullptr;      // [1024, 4096] PyTorch row-major
    float *pw2_b = nullptr;      // [1024]
    // ConvNeXt residual scale.
    float *gamma = nullptr;      // [1024]
};

// ---- Phase 2.7c: vocoder weights ------------------------------------------
// Per-block channel widths: 1536 -> 768 -> 384 -> 192 -> 96
// Strides:                         8     5     4    3   (k = 2*S each)
// Each VocoderResUnit: SnakeBeta -> Conv1d(k=7, dilation D[j]) -> SnakeBeta -> Conv1d(k=1)
struct VocoderResUnitDev {
    float *act1_alpha = nullptr;   // [C]
    float *act1_beta  = nullptr;   // [C]
    float *conv1_w    = nullptr;   // rearranged [K*C, C] row-major (K=7)
    float *conv1_b    = nullptr;   // [C]
    float *act2_alpha = nullptr;
    float *act2_beta  = nullptr;
    float *conv2_w    = nullptr;   // [C, C] row-major (k=1; just a Linear)
    float *conv2_b    = nullptr;
};

struct VocoderBlockDev {
    int C_in    = 0;          // input channels (e.g. 1536 for block 0)
    int C_out   = 0;          // output channels after upsample (e.g. 768)
    int stride  = 0;          // upsample rate
    int kernel  = 0;          // transconv kernel (= 2*stride)
    float *snake_alpha = nullptr;  // [C_in]
    float *snake_beta  = nullptr;  // [C_in]
    float *up_w        = nullptr;  // [K, C_out, C_in] row-major (C_in fastest)
    float *up_b        = nullptr;  // [C_out]
    VocoderResUnitDev res[3];
};

struct DecoderCudaContext {
    cublasHandle_t cublas = nullptr;
    cudaStream_t stream = nullptr;

    // Device scratch for RVQ output_proj GEMM.
    void *first_sum_dev   = nullptr;  // F32 [codebook_dim, T]
    void *rest_sum_dev    = nullptr;  // F32 [codebook_dim, T]
    void *first_proj_dev  = nullptr;  // F32 [rvq_out_dim, T]
    void *rest_proj_dev   = nullptr;  // F32 [rvq_out_dim, T]
    void *first_proj_w_dev = nullptr; // F32 [rvq_out_dim, codebook_dim]
    void *rest_proj_w_dev  = nullptr; // F32 [rvq_out_dim, codebook_dim]
    int scratch_T = 0;                // current allocated T

    // ---- Phase 2.7b: pre_conv weights (rearranged on host before upload) ----
    // pre_conv: causal Conv1d 512 -> 1024, kernel=3.
    // Stored as row-major [K*C_in, C_out] (k outer, ic fastest within k-block,
    // oc fastest globally) so the cuBLAS gemm against im2col [T, K*C_in] is
    // a straight Sgemm(N, N, C_out, T, K*C_in, ...).
    float *pre_conv_w = nullptr;    // [K*C_in, C_out] = [1536, 1024]
    float *pre_conv_b = nullptr;    // [C_out=1024]

    // ---- Phase 2.7b: 2x upsample block weights ----
    UpsampleBlockDev up[2];

    // ---- Phase 2.7b: device scratch buffers for the forward path ----
    // Two ping-pong I/O buffers sized for the worst case [4*T_max, 4096]
    // (the pwconv1 expansion). T grows 1 -> 4x through 2 upsample stages.
    float *buf_a = nullptr;     // [4*T, 4096]
    float *buf_b = nullptr;     // [4*T, 4096]
    float *im2col_buf = nullptr;// [T, K*C_in]  (pre_conv only, T worst case)
    int scratch_T_fwd = 0;      // T used to size buf_a / buf_b / im2col_buf

    // ---- Phase 2.7c: vocoder weights -----------------------------------
    float *voc_init_w = nullptr;   // rearranged [K*1024, 1536] (k=7)
    float *voc_init_b = nullptr;   // [1536]
    VocoderBlockDev voc_blocks[4];
    float *voc_final_alpha = nullptr;  // [96]
    float *voc_final_beta  = nullptr;  // [96]
    float *voc_final_w     = nullptr;  // rearranged [K*96, 1] (k=7)
    float *voc_final_b     = nullptr;  // [1]
};

// Stash the context as a heap-allocated opaque blob hung off `ready_`.
// We don't expose it in the header to keep the public surface clean; access
// happens via this file-local helper.
namespace {
DecoderCudaContext *g_ctx_for(SpeechTokenizerDecoderCudaEngine *) ;
}  // namespace

// Shared singleton-ish ctx pointer hidden in a static map keyed off `this`.
// Using a single member would force the ctx into the header; for Phase 2.7a
// scaffold cleanliness we keep the header minimal and stash the ctx in a
// thread-local lookup. (One engine per process for now; multi-engine is a
// 2.7b concern.) -- Replaced with a dedicated owning member to avoid that
// indirection. See `SpeechTokenizerDecoderCudaEngine::ctx_` below; but as
// the header is final, we tunnel via a side table.
namespace {
struct CtxRegistry {
    SpeechTokenizerDecoderCudaEngine *eng;
    DecoderCudaContext *ctx;
};
static std::vector<CtxRegistry> &registry() {
    static std::vector<CtxRegistry> r;
    return r;
}
DecoderCudaContext *g_ctx_for(SpeechTokenizerDecoderCudaEngine *e) {
    for (auto &r : registry()) {
        if (r.eng == e) return r.ctx;
    }
    return nullptr;
}
void g_ctx_set(SpeechTokenizerDecoderCudaEngine *e, DecoderCudaContext *ctx) {
    for (auto &r : registry()) {
        if (r.eng == e) { r.ctx = ctx; return; }
    }
    registry().push_back({e, ctx});
}
}  // namespace

SpeechTokenizerDecoderCudaEngine::~SpeechTokenizerDecoderCudaEngine() {
    DecoderCudaContext *ctx = g_ctx_for(this);
    if (ctx) {
        if (ctx->first_sum_dev)    cudaFree(ctx->first_sum_dev);
        if (ctx->rest_sum_dev)     cudaFree(ctx->rest_sum_dev);
        if (ctx->first_proj_dev)   cudaFree(ctx->first_proj_dev);
        if (ctx->rest_proj_dev)    cudaFree(ctx->rest_proj_dev);
        if (ctx->first_proj_w_dev) cudaFree(ctx->first_proj_w_dev);
        if (ctx->rest_proj_w_dev)  cudaFree(ctx->rest_proj_w_dev);
        if (ctx->pre_conv_w) cudaFree(ctx->pre_conv_w);
        if (ctx->pre_conv_b) cudaFree(ctx->pre_conv_b);
        for (int i = 0; i < 2; ++i) {
            auto &u = ctx->up[i];
            if (u.up_w) cudaFree(u.up_w);
            if (u.up_b) cudaFree(u.up_b);
            if (u.dw_w) cudaFree(u.dw_w);
            if (u.dw_b) cudaFree(u.dw_b);
            if (u.norm_w) cudaFree(u.norm_w);
            if (u.norm_b) cudaFree(u.norm_b);
            if (u.pw1_w) cudaFree(u.pw1_w);
            if (u.pw1_b) cudaFree(u.pw1_b);
            if (u.pw2_w) cudaFree(u.pw2_w);
            if (u.pw2_b) cudaFree(u.pw2_b);
            if (u.gamma) cudaFree(u.gamma);
        }
        if (ctx->buf_a) cudaFree(ctx->buf_a);
        if (ctx->buf_b) cudaFree(ctx->buf_b);
        if (ctx->im2col_buf) cudaFree(ctx->im2col_buf);
        // Vocoder weights (2.7c).
        if (ctx->voc_init_w) cudaFree(ctx->voc_init_w);
        if (ctx->voc_init_b) cudaFree(ctx->voc_init_b);
        for (int i = 0; i < 4; ++i) {
            auto &vb = ctx->voc_blocks[i];
            if (vb.snake_alpha) cudaFree(vb.snake_alpha);
            if (vb.snake_beta)  cudaFree(vb.snake_beta);
            if (vb.up_w) cudaFree(vb.up_w);
            if (vb.up_b) cudaFree(vb.up_b);
            for (int j = 0; j < 3; ++j) {
                auto &ru = vb.res[j];
                if (ru.act1_alpha) cudaFree(ru.act1_alpha);
                if (ru.act1_beta)  cudaFree(ru.act1_beta);
                if (ru.conv1_w) cudaFree(ru.conv1_w);
                if (ru.conv1_b) cudaFree(ru.conv1_b);
                if (ru.act2_alpha) cudaFree(ru.act2_alpha);
                if (ru.act2_beta)  cudaFree(ru.act2_beta);
                if (ru.conv2_w) cudaFree(ru.conv2_w);
                if (ru.conv2_b) cudaFree(ru.conv2_b);
            }
        }
        if (ctx->voc_final_alpha) cudaFree(ctx->voc_final_alpha);
        if (ctx->voc_final_beta)  cudaFree(ctx->voc_final_beta);
        if (ctx->voc_final_w) cudaFree(ctx->voc_final_w);
        if (ctx->voc_final_b) cudaFree(ctx->voc_final_b);

        if (ctx->cublas) cublasDestroy(ctx->cublas);
        if (ctx->stream) cudaStreamDestroy(ctx->stream);
        delete ctx;
        g_ctx_set(this, nullptr);
    }
}

bool SpeechTokenizerDecoderCudaEngine::init_from_gguf(
    const std::string &gguf_path, int device) {
    device_ = device;
    OMX_CUDA_CHECK(cudaSetDevice(device_));

    // ---- Open GGUF ----
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[stdec_cuda] failed to load GGUF: %s\n",
                gguf_path.c_str());
        return false;
    }

    // ---- Read hparams (mirror Ascend load_hparams) ----
    gguf_read_u32(gguf_ctx, "codebook_size",         config_.codebook_size);
    gguf_read_u32(gguf_ctx, "hidden_size",           config_.hidden_size);
    gguf_read_u32(gguf_ctx, "latent_dim",            config_.latent_dim);
    gguf_read_u32(gguf_ctx, "num_hidden_layers",     config_.num_hidden_layers);
    gguf_read_u32(gguf_ctx, "num_attention_heads",   config_.num_attention_heads);
    gguf_read_u32(gguf_ctx, "num_key_value_heads",   config_.num_key_value_heads);
    gguf_read_u32(gguf_ctx, "intermediate_size",     config_.intermediate_size);
    gguf_read_u32(gguf_ctx, "num_quantizers",        config_.num_quantizers);
    gguf_read_u32(gguf_ctx, "decoder_dim",           config_.decoder_dim);
    gguf_read_u32(gguf_ctx, "sliding_window",        config_.sliding_window);
    gguf_read_u32(gguf_ctx, "output_sample_rate",    config_.output_sample_rate);
    gguf_read_u32(gguf_ctx, "decode_upsample_rate",  config_.decode_upsample_rate);
    gguf_read_f32(gguf_ctx, "rope_theta",            config_.rope_theta);
    gguf_read_f32(gguf_ctx, "rms_norm_eps",          config_.rms_norm_eps);

    // upsample_rates / upsampling_ratios (read but not used in 2.7a; logged for
    // verification of the GGUF pipe).
    int upsample_rates[4]   = {0, 0, 0, 0};
    int upsampling_ratios[2] = {0, 0};
    int n_ur = 0, n_ur2 = 0;
    gguf_read_arr_i32(gguf_ctx, "upsample_rates", upsample_rates, 4, n_ur);
    gguf_read_arr_i32(gguf_ctx, "upsampling_ratios", upsampling_ratios, 2, n_ur2);

    // RVQ output dim is fixed by the architecture (codebook_dim → 512). Derive
    // it from the output_proj weight shape we'll load below.
    config_.codebook_dim = 256;  // fixed in Qwen3-TTS

    printf("[stdec_cuda] config: codebook_size=%d codebook_dim=%d "
           "num_quantizers=%d hidden=%d latent=%d layers=%d window=%d\n",
           config_.codebook_size, config_.codebook_dim, config_.num_quantizers,
           config_.hidden_size, config_.latent_dim, config_.num_hidden_layers,
           config_.sliding_window);
    printf("[stdec_cuda] upsample_rates=[%d,%d,%d,%d] ratios=[%d,%d] "
           "output_sr=%d decode_upsample=%d\n",
           upsample_rates[0], upsample_rates[1], upsample_rates[2], upsample_rates[3],
           upsampling_ratios[0], upsampling_ratios[1],
           config_.output_sample_rate, config_.decode_upsample_rate);

    // ---- Load RVQ tensors ----
    // rvq_first: 1 codebook (semantic).  rvq_rest: num_quantizers - 1 = 15.
    auto load_group = [&](RVQGroupHost &grp, const std::string &prefix,
                           int n_layers) -> bool {
        // Output proj weight: GGUF stores [1, codebook_dim, rvq_out_dim] —
        // that's a Conv1d(in=codebook_dim, out=rvq_out_dim, k=1). The trailing
        // singleton dim is the kernel; after collapsing it the matmul shape
        // is [rvq_out_dim, codebook_dim] (row-major, rvq_out_dim fastest in
        // GGUF row-major convention, matching cublasGemmEx column-major
        // expectations once we feed it through cublas).
        ggml_tensor *opw = ggml_get_tensor(ggml_ctx,
                                            (prefix + ".output_proj.weight").c_str());
        if (!opw) {
            fprintf(stderr, "[stdec_cuda] missing %s.output_proj.weight\n",
                    prefix.c_str());
            return false;
        }
        // Expected shape: ne[0]=1 (k), ne[1]=codebook_dim, ne[2]=rvq_out_dim.
        if (opw->ne[0] != 1 || opw->ne[1] != config_.codebook_dim) {
            fprintf(stderr, "[stdec_cuda] %s.output_proj.weight bad shape: "
                            "[%lld, %lld, %lld]\n",
                    prefix.c_str(),
                    (long long)opw->ne[0], (long long)opw->ne[1],
                    (long long)opw->ne[2]);
            return false;
        }
        config_.rvq_out_dim = (int)opw->ne[2];
        size_t n_ow = (size_t)opw->ne[2] * opw->ne[1] * opw->ne[0];
        if (!load_tensor_f32(ggml_ctx,
                              (prefix + ".output_proj.weight").c_str(),
                              n_ow, grp.output_proj_w)) return false;

        grp.codebooks.resize(n_layers);
        for (int i = 0; i < n_layers; ++i) {
            std::string lp = prefix + ".vq.layers." + std::to_string(i) +
                             "._codebook.";
            if (!load_codebook(ggml_ctx, lp,
                                config_.codebook_dim, config_.codebook_size,
                                grp.codebooks[i])) return false;
        }
        return true;
    };
    if (!load_group(rvq_first_, "quantizer.rvq_first", 1)) {
        gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
    }
    if (!load_group(rvq_rest_,  "quantizer.rvq_rest",
                     config_.num_quantizers - 1)) {
        gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
    }

    // ---- Precompute normalized codebooks (embedding_sum / cluster_usage) ----
    rvq_first_norm_.resize(rvq_first_.codebooks.size());
    for (size_t i = 0; i < rvq_first_.codebooks.size(); ++i) {
        precompute_codebook_norm(rvq_first_.codebooks[i],
                                  config_.codebook_dim,
                                  config_.codebook_size,
                                  rvq_first_norm_[i]);
    }
    rvq_rest_norm_.resize(rvq_rest_.codebooks.size());
    for (size_t i = 0; i < rvq_rest_.codebooks.size(); ++i) {
        precompute_codebook_norm(rvq_rest_.codebooks[i],
                                  config_.codebook_dim,
                                  config_.codebook_size,
                                  rvq_rest_norm_[i]);
    }

    printf("[stdec_cuda] RVQ loaded: first.codebooks=%zu rest.codebooks=%zu "
           "rvq_out_dim=%d\n",
           rvq_first_.codebooks.size(), rvq_rest_.codebooks.size(),
           config_.rvq_out_dim);

    // ---- Allocate CUDA context (cuBLAS + stream + projection weights) ----
    DecoderCudaContext *ctx = new DecoderCudaContext();
    OMX_CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    OMX_CUBLAS_CHECK(cublasCreate(&ctx->cublas));
    OMX_CUBLAS_CHECK(cublasSetStream(ctx->cublas, ctx->stream));

    size_t proj_bytes = (size_t)config_.rvq_out_dim *
                         config_.codebook_dim * sizeof(float);
    OMX_CUDA_CHECK(cudaMalloc(&ctx->first_proj_w_dev, proj_bytes));
    OMX_CUDA_CHECK(cudaMalloc(&ctx->rest_proj_w_dev, proj_bytes));
    OMX_CUDA_CHECK(cudaMemcpy(ctx->first_proj_w_dev,
                               rvq_first_.output_proj_w.data(),
                               proj_bytes, cudaMemcpyHostToDevice));
    OMX_CUDA_CHECK(cudaMemcpy(ctx->rest_proj_w_dev,
                               rvq_rest_.output_proj_w.data(),
                               proj_bytes, cudaMemcpyHostToDevice));

    g_ctx_set(this, ctx);

    // ------------------------------------------------------------------
    // Phase 2.7b: load pre_conv + 2x upsample block weights to device.
    // ------------------------------------------------------------------
    auto upload_f32 = [&](const std::vector<float> &h, float **dev,
                          const char *what) -> bool {
        size_t bytes = h.size() * sizeof(float);
        cudaError_t e = cudaMalloc(dev, bytes);
        if (e != cudaSuccess) {
            fprintf(stderr, "[stdec_cuda] cudaMalloc %s (%zu B): %s\n",
                    what, bytes, cudaGetErrorString(e));
            return false;
        }
        e = cudaMemcpy(*dev, h.data(), bytes, cudaMemcpyHostToDevice);
        if (e != cudaSuccess) {
            fprintf(stderr, "[stdec_cuda] cudaMemcpy %s: %s\n",
                    what, cudaGetErrorString(e));
            return false;
        }
        return true;
    };

    // ---- pre_conv ----
    // GGUF weight shape [K=3, C_in=512, C_out=1024], axis-0 fastest. Bytes
    // are PyTorch row-major [C_out, C_in, K] (k fastest).
    // We rearrange into [K*C_in, C_out] row-major (oc fastest within slot,
    // outer index = k*C_in + ic). cuBLAS Sgemm(N, N, C_out, T, K*C_in, ...)
    // then matches the im2col output [T, K*C_in] row-major produced by
    // launch_causal_conv1d_im2col_f32.
    {
        const int K = 3, C_in = 512, C_out = config_.latent_dim;  // 1024
        std::vector<float> w_pt;  // [C_out, C_in, K] PyTorch row-major
        if (!load_tensor_f32(ggml_ctx, "pre_conv.conv.weight",
                              (size_t)K * C_in * C_out, w_pt)) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
        std::vector<float> w_flat((size_t)K * C_in * C_out);
        for (int k = 0; k < K; ++k) {
            for (int ic = 0; ic < C_in; ++ic) {
                for (int oc = 0; oc < C_out; ++oc) {
                    w_flat[((size_t)k * C_in + ic) * C_out + oc] =
                        w_pt[((size_t)oc * C_in + ic) * K + k];
                }
            }
        }
        if (!upload_f32(w_flat, &ctx->pre_conv_w, "pre_conv.weight")) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
        std::vector<float> b;
        if (!load_tensor_f32(ggml_ctx, "pre_conv.conv.bias",
                              (size_t)C_out, b)) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
        if (!upload_f32(b, &ctx->pre_conv_b, "pre_conv.bias")) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
    }

    // ---- 2 upsample blocks ----
    // ConvTranspose1d k=2 s=2: GGUF weight shape [K=2, 1024, 1024], axis-0
    // fastest -> bytes are PyTorch row-major [in_ch, out_ch, K] (k fastest).
    // launch_conv_transpose1d_k2s2_f32 expects [K, C_out, C_in] row-major
    // (C_in fastest). Rearrange.
    //
    // dwconv k=7: GGUF [K=7, 1, 1024]. Bytes: row-major [C, 1, K] (k fastest)
    // i.e. for each channel c the K kernel taps live contiguously. The
    // launch_depthwise_conv1d_causal_f32 kernel reads `w[k*C + c]`, so we
    // rearrange to [K, C] row-major (k outer, c fastest).
    //
    // pwconv1 [4096, 1024] / pwconv2 [1024, 4096]: stored as PyTorch
    // [out, in] row-major bytes — uploaded as-is, used with cublasSgemm(T, N).
    const int C = config_.latent_dim;        // 1024
    const int FFN = 4 * C;                    // 4096
    for (int b_idx = 0; b_idx < 2; ++b_idx) {
        const std::string bp = "upsample." + std::to_string(b_idx) + ".";
        auto &u = ctx->up[b_idx];

        // up.weight + bias  (ConvTranspose1d k=2 s=2)
        {
            const int K = 2;
            std::vector<float> w_pt;  // [in_ch=C, out_ch=C, K] row-major
            if (!load_tensor_f32(ggml_ctx, (bp + "0.conv.weight").c_str(),
                                  (size_t)K * C * C, w_pt)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            std::vector<float> w_flat((size_t)K * C * C);
            for (int k = 0; k < K; ++k) {
                for (int oc = 0; oc < C; ++oc) {
                    for (int ic = 0; ic < C; ++ic) {
                        w_flat[((size_t)k * C + oc) * C + ic] =
                            w_pt[((size_t)ic * C + oc) * K + k];
                    }
                }
            }
            if (!upload_f32(w_flat, &u.up_w, "up.weight")) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            std::vector<float> bv;
            if (!load_tensor_f32(ggml_ctx, (bp + "0.conv.bias").c_str(),
                                  (size_t)C, bv)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            if (!upload_f32(bv, &u.up_b, "up.bias")) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
        }

        // dwconv.weight + bias  (depthwise k=7)
        {
            const int K = 7;
            std::vector<float> w_pt;  // [C, 1, K] row-major (c outer, k fastest)
            if (!load_tensor_f32(ggml_ctx,
                                  (bp + "1.dwconv.conv.weight").c_str(),
                                  (size_t)K * C, w_pt)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            std::vector<float> w_flat((size_t)K * C);
            for (int k = 0; k < K; ++k) {
                for (int c = 0; c < C; ++c) {
                    w_flat[(size_t)k * C + c] = w_pt[(size_t)c * K + k];
                }
            }
            if (!upload_f32(w_flat, &u.dw_w, "dwconv.weight")) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            std::vector<float> bv;
            if (!load_tensor_f32(ggml_ctx,
                                  (bp + "1.dwconv.conv.bias").c_str(),
                                  (size_t)C, bv)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            if (!upload_f32(bv, &u.dw_b, "dwconv.bias")) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
        }

        // norm.weight, norm.bias (LayerNorm gamma/beta)
        {
            std::vector<float> g, b;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.norm.weight").c_str(),
                                  (size_t)C, g)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            if (!load_tensor_f32(ggml_ctx, (bp + "1.norm.bias").c_str(),
                                  (size_t)C, b)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
            }
            if (!upload_f32(g, &u.norm_w, "norm.weight")) return false;
            if (!upload_f32(b, &u.norm_b, "norm.bias")) return false;
        }

        // pwconv1: [4096, 1024] PyTorch -- store as-is.
        {
            std::vector<float> w;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.pwconv1.weight").c_str(),
                                  (size_t)FFN * C, w)) return false;
            if (!upload_f32(w, &u.pw1_w, "pwconv1.weight")) return false;
            std::vector<float> bv;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.pwconv1.bias").c_str(),
                                  (size_t)FFN, bv)) return false;
            if (!upload_f32(bv, &u.pw1_b, "pwconv1.bias")) return false;
        }

        // pwconv2: [1024, 4096] PyTorch -- store as-is.
        {
            std::vector<float> w;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.pwconv2.weight").c_str(),
                                  (size_t)C * FFN, w)) return false;
            if (!upload_f32(w, &u.pw2_w, "pwconv2.weight")) return false;
            std::vector<float> bv;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.pwconv2.bias").c_str(),
                                  (size_t)C, bv)) return false;
            if (!upload_f32(bv, &u.pw2_b, "pwconv2.bias")) return false;
        }

        // gamma: [C]
        {
            std::vector<float> g;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.gamma").c_str(),
                                  (size_t)C, g)) return false;
            if (!upload_f32(g, &u.gamma, "gamma")) return false;
        }
    }

    printf("[stdec_cuda] pre_conv + 2x upsample weights uploaded "
           "(C=%d, FFN=%d)\n", C, FFN);

    // ------------------------------------------------------------------
    // Phase 2.7c: load vocoder weights (initial conv + 4 blocks + final).
    // ------------------------------------------------------------------
    // Helper: rearrange a "K outer, C_in fastest" GGUF causal-conv1d weight
    // [K, C_in, C_out] (axis 0 fastest in GGUF semantics, byte layout
    // PyTorch row-major [C_out, C_in, K]) into [K*C_in, C_out] row-major
    // (oc fastest globally) so cuBLAS Sgemm(N, N, C_out, T, K*C_in, ...)
    // works against an im2col [T, K*C_in] row-major buffer.
    auto rearrange_kCC = [](const std::vector<float> &w_pt,
                              int K, int C_in, int C_out,
                              std::vector<float> &w_flat) {
        w_flat.assign((size_t)K * C_in * C_out, 0.0f);
        for (int k = 0; k < K; ++k) {
            for (int ic = 0; ic < C_in; ++ic) {
                for (int oc = 0; oc < C_out; ++oc) {
                    w_flat[((size_t)k * C_in + ic) * C_out + oc] =
                        w_pt[((size_t)oc * C_in + ic) * K + k];
                }
            }
        }
    };

    // Helper: rearrange a transconv [K, C_out, C_in] GGUF weight (bytes are
    // PyTorch row-major [C_in, C_out, K]) into [K, C_out, C_in] row-major
    // (C_in fastest) used by launch_causal_conv_transpose1d_f32.
    auto rearrange_transconv = [](const std::vector<float> &w_pt,
                                    int K, int C_in, int C_out,
                                    std::vector<float> &w_flat) {
        w_flat.assign((size_t)K * C_out * C_in, 0.0f);
        for (int k = 0; k < K; ++k) {
            for (int oc = 0; oc < C_out; ++oc) {
                for (int ic = 0; ic < C_in; ++ic) {
                    w_flat[((size_t)k * C_out + oc) * C_in + ic] =
                        w_pt[((size_t)ic * C_out + oc) * K + k];
                }
            }
        }
    };

    // ---- decoder.0.conv: 1024 -> 1536, k=7 (initial vocoder conv) ----
    {
        const int K_ = 7, C_in_ = 1024, C_out_ = 1536;
        std::vector<float> w_pt;
        if (!load_tensor_f32(ggml_ctx, "decoder.0.conv.weight",
                              (size_t)K_ * C_in_ * C_out_, w_pt)) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
        std::vector<float> w_flat;
        rearrange_kCC(w_pt, K_, C_in_, C_out_, w_flat);
        if (!upload_f32(w_flat, &ctx->voc_init_w, "decoder.0.weight")) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
        std::vector<float> b;
        if (!load_tensor_f32(ggml_ctx, "decoder.0.conv.bias",
                              (size_t)C_out_, b)) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
        if (!upload_f32(b, &ctx->voc_init_b, "decoder.0.bias")) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx); return false;
        }
    }

    // ---- decoder.{1..4}: 4 vocoder blocks ----
    // Channel widths: 1536 (in) -> 768 -> 384 -> 192 -> 96 (final out)
    // Strides: [8, 5, 4, 3]. Kernel = 2*stride.
    int voc_C[5]      = {1536, 768, 384, 192, 96};
    int voc_strides[4] = {8, 5, 4, 3};
    for (int bi = 0; bi < 4; ++bi) {
        auto &vb = ctx->voc_blocks[bi];
        vb.C_in   = voc_C[bi];
        vb.C_out  = voc_C[bi + 1];
        vb.stride = voc_strides[bi];
        vb.kernel = 2 * vb.stride;
        const std::string bp = "decoder." + std::to_string(bi + 1) + ".block.";
        const int Cin  = vb.C_in;
        const int Cout = vb.C_out;
        const int Ks   = vb.kernel;

        // SnakeBeta alpha/beta on input channels (C_in)
        {
            std::vector<float> a, b;
            if (!load_tensor_f32(ggml_ctx, (bp + "0.alpha").c_str(),
                                  (size_t)Cin, a)) return false;
            if (!load_tensor_f32(ggml_ctx, (bp + "0.beta").c_str(),
                                  (size_t)Cin, b)) return false;
            if (!upload_f32(a, &vb.snake_alpha, "voc.snake.alpha")) return false;
            if (!upload_f32(b, &vb.snake_beta,  "voc.snake.beta"))  return false;
        }

        // ConvTranspose1d (block.1.conv): k=2*S, C_in -> C_out
        {
            std::vector<float> w_pt;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.conv.weight").c_str(),
                                  (size_t)Ks * Cout * Cin, w_pt)) return false;
            std::vector<float> w_flat;
            rearrange_transconv(w_pt, Ks, Cin, Cout, w_flat);
            if (!upload_f32(w_flat, &vb.up_w, "voc.up.weight")) return false;
            std::vector<float> bv;
            if (!load_tensor_f32(ggml_ctx, (bp + "1.conv.bias").c_str(),
                                  (size_t)Cout, bv)) return false;
            if (!upload_f32(bv, &vb.up_b, "voc.up.bias")) return false;
        }

        // 3 residual units (block.2, block.3, block.4 — at channel C_out)
        for (int j = 0; j < 3; ++j) {
            auto &ru = vb.res[j];
            const std::string rp = bp + std::to_string(j + 2) + ".";
            const int Cr = Cout;

            std::vector<float> a, bv;
            if (!load_tensor_f32(ggml_ctx, (rp + "act1.alpha").c_str(),
                                  (size_t)Cr, a)) return false;
            if (!load_tensor_f32(ggml_ctx, (rp + "act1.beta").c_str(),
                                  (size_t)Cr, bv)) return false;
            if (!upload_f32(a,  &ru.act1_alpha, "ru.act1.alpha")) return false;
            if (!upload_f32(bv, &ru.act1_beta,  "ru.act1.beta"))  return false;

            // conv1: causal Conv1d k=7, dilation D[j], C_r -> C_r
            {
                const int K7 = 7;
                std::vector<float> w_pt;
                if (!load_tensor_f32(ggml_ctx, (rp + "conv1.conv.weight").c_str(),
                                      (size_t)K7 * Cr * Cr, w_pt)) return false;
                std::vector<float> w_flat;
                rearrange_kCC(w_pt, K7, Cr, Cr, w_flat);
                if (!upload_f32(w_flat, &ru.conv1_w, "ru.conv1.weight")) return false;
                std::vector<float> bv2;
                if (!load_tensor_f32(ggml_ctx, (rp + "conv1.conv.bias").c_str(),
                                      (size_t)Cr, bv2)) return false;
                if (!upload_f32(bv2, &ru.conv1_b, "ru.conv1.bias")) return false;
            }

            std::vector<float> a2, bv2;
            if (!load_tensor_f32(ggml_ctx, (rp + "act2.alpha").c_str(),
                                  (size_t)Cr, a2)) return false;
            if (!load_tensor_f32(ggml_ctx, (rp + "act2.beta").c_str(),
                                  (size_t)Cr, bv2)) return false;
            if (!upload_f32(a2,  &ru.act2_alpha, "ru.act2.alpha")) return false;
            if (!upload_f32(bv2, &ru.act2_beta,  "ru.act2.beta"))  return false;

            // conv2: Conv1d k=1, C_r -> C_r — i.e. a Linear/pointwise.
            // GGUF shape [K=1, C_in=Cr, C_out=Cr], bytes PyTorch [Cr, Cr, 1].
            // We just store the [C_out, C_in] = [Cr, Cr] row-major buffer
            // (== col-major [Cr, Cr]) and use cublasSgemm(opA=T) against it.
            {
                std::vector<float> w_pt;
                if (!load_tensor_f32(ggml_ctx, (rp + "conv2.conv.weight").c_str(),
                                      (size_t)Cr * Cr, w_pt)) return false;
                if (!upload_f32(w_pt, &ru.conv2_w, "ru.conv2.weight")) return false;
                std::vector<float> bv3;
                if (!load_tensor_f32(ggml_ctx, (rp + "conv2.conv.bias").c_str(),
                                      (size_t)Cr, bv3)) return false;
                if (!upload_f32(bv3, &ru.conv2_b, "ru.conv2.bias")) return false;
            }
        }
    }

    // ---- decoder.5: final SnakeBeta (alpha, beta), C=96 ----
    {
        std::vector<float> a, b;
        if (!load_tensor_f32(ggml_ctx, "decoder.5.alpha", 96, a)) return false;
        if (!load_tensor_f32(ggml_ctx, "decoder.5.beta",  96, b)) return false;
        if (!upload_f32(a, &ctx->voc_final_alpha, "voc.final.alpha")) return false;
        if (!upload_f32(b, &ctx->voc_final_beta,  "voc.final.beta"))  return false;
    }

    // ---- decoder.6.conv: 96 -> 1, k=7 (final mono audio conv) ----
    {
        const int K_ = 7, C_in_ = 96, C_out_ = 1;
        std::vector<float> w_pt;
        if (!load_tensor_f32(ggml_ctx, "decoder.6.conv.weight",
                              (size_t)K_ * C_in_ * C_out_, w_pt)) return false;
        std::vector<float> w_flat;
        rearrange_kCC(w_pt, K_, C_in_, C_out_, w_flat);
        if (!upload_f32(w_flat, &ctx->voc_final_w, "decoder.6.weight")) return false;
        std::vector<float> bv;
        if (!load_tensor_f32(ggml_ctx, "decoder.6.conv.bias",
                              (size_t)C_out_, bv)) return false;
        if (!upload_f32(bv, &ctx->voc_final_b, "decoder.6.bias")) return false;
    }

    printf("[stdec_cuda] vocoder weights uploaded "
           "(init=1024->1536, blocks=[%d,%d,%d,%d]@strides=[8,5,4,3], "
           "final=96->1)\n",
           voc_C[1], voc_C[2], voc_C[3], voc_C[4]);

    // We keep ggml_ctx alive only long enough to copy weights — done. Free.
    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    ready_ = true;
    return true;
}

bool SpeechTokenizerDecoderCudaEngine::rvq_decode(
    const int *codes, int n_codebooks, int T, std::vector<float> &out) {

    if (!ready_) {
        fprintf(stderr, "[stdec_cuda] rvq_decode called before init\n");
        return false;
    }
    if (n_codebooks != config_.num_quantizers) {
        fprintf(stderr, "[stdec_cuda] rvq_decode: expected %d codebooks, got %d\n",
                config_.num_quantizers, n_codebooks);
        return false;
    }
    if (T <= 0) {
        fprintf(stderr, "[stdec_cuda] rvq_decode: T must be > 0 (got %d)\n", T);
        return false;
    }

    DecoderCudaContext *ctx = g_ctx_for(this);
    if (!ctx) {
        fprintf(stderr, "[stdec_cuda] no CUDA context\n");
        return false;
    }

    const int cd = config_.codebook_dim;     // 256
    const int od = config_.rvq_out_dim;      // 512
    const int K  = config_.codebook_size;    // 2048

    // ---- Step 1: host gather + accumulate per group ----
    // Layout: sum_first[t * cd + c]; sum_rest[t * cd + c]. Both row-major,
    // codebook_dim fastest. This matches the Ascend `ggml_get_rows` output
    // layout `[codebook_dim, T]` after the implicit transpose.
    std::vector<float> sum_first((size_t)T * cd, 0.0f);
    std::vector<float> sum_rest ((size_t)T * cd, 0.0f);

    auto accumulate_group = [&](const std::vector<std::vector<float>> &norm,
                                 int q_start, int q_end,
                                 std::vector<float> &dst) {
        for (int qi = q_start; qi < q_end; ++qi) {
            int local = qi - q_start;
            const float *table = norm[local].data();   // [K, cd]
            for (int t = 0; t < T; ++t) {
                int code = codes[(size_t)qi * T + t];
                if (code < 0 || code >= K) {
                    fprintf(stderr, "[stdec_cuda] code OOB: q=%d t=%d code=%d\n",
                            qi, t, code);
                    return false;
                }
                const float *row = table + (size_t)code * cd;
                float *out_row = dst.data() + (size_t)t * cd;
                for (int c = 0; c < cd; ++c) out_row[c] += row[c];
            }
        }
        return true;
    };
    if (!accumulate_group(rvq_first_norm_, 0, 1, sum_first)) return false;
    if (!accumulate_group(rvq_rest_norm_, 1, config_.num_quantizers, sum_rest))
        return false;

    // ---- Step 2: GEMM each group sum through its output_proj ----
    // We compute  proj = W @ sum    where
    //   W    : [od, cd]  row-major  → cublas reads it column-major as [cd, od]
    //   sum  : [cd, T]   column-major (matches our row-major [T, cd]) once we
    //                    treat the host buffer as col-major [cd, T] (since
    //                    codebook_dim is the fastest axis).
    //   proj : [od, T]   column-major  ⇔ row-major [T, od] when we read it back
    //                    -- BUT the Ascend reference downstream expects
    //                    [out_dim, T] with out_dim fastest (rvq_out_dim fastest).
    //
    // To avoid layout confusion we use the standard pattern:
    //   cublasSgemm(N, N, od, T, cd, 1, W_dev, od, sum_dev, cd, 0, out_dev, od);
    // i.e. "W is [od, cd] col-major; sum is [cd, T] col-major; out is [od, T]
    //       col-major", which when interpreted row-major means
    //   W : row-major [cd, od]   sum : row-major [T, cd]   out : row-major [T, od]
    // That's NOT what we have on the host (we have W stored as ggml row-major
    // [rvq_out_dim, codebook_dim, 1] = row-major [codebook_dim_in_gguf_axis_0,
    // codebook_dim, rvq_out_dim] -- actually since GGUF axis 0 is fastest, the
    // raw bytes are [k=1][cd][od]_outermost  ⇒  [od, cd, 1] row-major in numpy
    // convention. After collapsing the singleton k, the bytes are an
    // [od, cd] row-major buffer  ⇔  [cd, od] col-major.
    //
    // And sum_first is host row-major [T, cd]  ⇔  col-major [cd, T].
    //
    // So the cublasSgemm call:
    //   op(A) = N  (W as [cd, od] col-major)  → we want op(A) = T to get [od, cd]
    //   op(B) = N  (sum as [cd, T] col-major) → keep
    //   result is [od, T] col-major  ⇔  row-major [T, od]
    //
    // This delivers `out_buf[t * od + d]`, but the Ascend downstream consumer
    // wants `[od, T]` with od fastest, i.e. `out_buf[t * od + d]` indexes
    // (t, d) in (time, channel) ordering. That's the row-major-with-od-fastest
    // shape which is exactly what we produce.

    // Allocate / resize device scratch as needed.
    if (ctx->scratch_T < T) {
        if (ctx->first_sum_dev) { cudaFree(ctx->first_sum_dev); ctx->first_sum_dev = nullptr; }
        if (ctx->rest_sum_dev)  { cudaFree(ctx->rest_sum_dev);  ctx->rest_sum_dev  = nullptr; }
        if (ctx->first_proj_dev){ cudaFree(ctx->first_proj_dev); ctx->first_proj_dev = nullptr; }
        if (ctx->rest_proj_dev) { cudaFree(ctx->rest_proj_dev);  ctx->rest_proj_dev  = nullptr; }
        size_t sum_bytes  = (size_t)cd * T * sizeof(float);
        size_t proj_bytes = (size_t)od * T * sizeof(float);
        OMX_CUDA_CHECK(cudaMalloc(&ctx->first_sum_dev,  sum_bytes));
        OMX_CUDA_CHECK(cudaMalloc(&ctx->rest_sum_dev,   sum_bytes));
        OMX_CUDA_CHECK(cudaMalloc(&ctx->first_proj_dev, proj_bytes));
        OMX_CUDA_CHECK(cudaMalloc(&ctx->rest_proj_dev,  proj_bytes));
        ctx->scratch_T = T;
    }

    OMX_CUDA_CHECK(cudaMemcpyAsync(ctx->first_sum_dev, sum_first.data(),
                                    (size_t)cd * T * sizeof(float),
                                    cudaMemcpyHostToDevice, ctx->stream));
    OMX_CUDA_CHECK(cudaMemcpyAsync(ctx->rest_sum_dev,  sum_rest.data(),
                                    (size_t)cd * T * sizeof(float),
                                    cudaMemcpyHostToDevice, ctx->stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    OMX_CUBLAS_CHECK(cublasSgemm(ctx->cublas,
                                  CUBLAS_OP_T,   // W is [cd, od] col-major → op_T → [od, cd]
                                  CUBLAS_OP_N,
                                  od, T, cd,
                                  &alpha,
                                  (const float *)ctx->first_proj_w_dev, cd,
                                  (const float *)ctx->first_sum_dev,    cd,
                                  &beta,
                                  (float *)ctx->first_proj_dev, od));
    OMX_CUBLAS_CHECK(cublasSgemm(ctx->cublas,
                                  CUBLAS_OP_T,
                                  CUBLAS_OP_N,
                                  od, T, cd,
                                  &alpha,
                                  (const float *)ctx->rest_proj_w_dev, cd,
                                  (const float *)ctx->rest_sum_dev,    cd,
                                  &beta,
                                  (float *)ctx->rest_proj_dev, od));

    // ---- Step 3: D2H copy of both projections, sum on host ----
    std::vector<float> proj_first((size_t)od * T);
    std::vector<float> proj_rest ((size_t)od * T);
    OMX_CUDA_CHECK(cudaMemcpyAsync(proj_first.data(), ctx->first_proj_dev,
                                    (size_t)od * T * sizeof(float),
                                    cudaMemcpyDeviceToHost, ctx->stream));
    OMX_CUDA_CHECK(cudaMemcpyAsync(proj_rest.data(),  ctx->rest_proj_dev,
                                    (size_t)od * T * sizeof(float),
                                    cudaMemcpyDeviceToHost, ctx->stream));
    OMX_CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    out.resize((size_t)od * T);
    for (size_t i = 0; i < (size_t)od * T; ++i) {
        out[i] = proj_first[i] + proj_rest[i];
    }
    return true;
}

// ---------------------------------------------------------------------------
// Phase 2.7b forward path helpers.
//
// All operate on row-major [T, C] device buffers with C fastest. cuBLAS is
// addressed using the standard "cublas col-major == row-major with axes
// transposed" trick — see init_from_gguf for the per-weight layout.
// ---------------------------------------------------------------------------
namespace {

// Ensure the scratch buffers buf_a / buf_b / im2col_buf are sized for the
// peak intermediate footprint (4*T_in × FFN=4096 floats).
bool ensure_scratch(DecoderCudaContext *ctx, int T_in,
                     int C, int FFN) {
    int T_max = 4 * T_in;
    if (ctx->scratch_T_fwd >= T_in && ctx->buf_a && ctx->buf_b &&
        ctx->im2col_buf) {
        return true;
    }
    if (ctx->buf_a)      cudaFree(ctx->buf_a);
    if (ctx->buf_b)      cudaFree(ctx->buf_b);
    if (ctx->im2col_buf) cudaFree(ctx->im2col_buf);
    ctx->buf_a = ctx->buf_b = ctx->im2col_buf = nullptr;

    size_t buf_bytes = (size_t)T_max * FFN * sizeof(float);
    cudaError_t e;
    e = cudaMalloc((void**)&ctx->buf_a, buf_bytes);
    if (e != cudaSuccess) { fprintf(stderr,"[stdec_cuda] scratch buf_a: %s\n",
                                     cudaGetErrorString(e)); return false; }
    e = cudaMalloc((void**)&ctx->buf_b, buf_bytes);
    if (e != cudaSuccess) { fprintf(stderr,"[stdec_cuda] scratch buf_b: %s\n",
                                     cudaGetErrorString(e)); return false; }
    // im2col only needed for pre_conv: [T, K=3 * C_in=512] = T * 1536 floats
    size_t im2_bytes = (size_t)T_in * 3 * 512 * sizeof(float);
    e = cudaMalloc((void**)&ctx->im2col_buf, im2_bytes);
    if (e != cudaSuccess) { fprintf(stderr,"[stdec_cuda] scratch im2col: %s\n",
                                     cudaGetErrorString(e)); return false; }
    ctx->scratch_T_fwd = T_in;
    (void)C;
    return true;
}

// Print min/max/mean/std + nan/inf for a device buffer of `n` F32 elems.
// Returns 0 on clean, 1 if NaN/Inf detected.
int dev_stats(const char *tag, const float *dev, size_t n,
               cudaStream_t stream) {
    std::vector<float> h(n);
    cudaMemcpyAsync(h.data(), dev, n * sizeof(float),
                     cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    int n_nan = 0, n_inf = 0;
    double sum = 0, sum2 = 0;
    float vmin = +1e30f, vmax = -1e30f;
    for (size_t i = 0; i < n; ++i) {
        float v = h[i];
        if (std::isnan(v)) ++n_nan;
        if (std::isinf(v)) ++n_inf;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
        sum  += v;
        sum2 += (double)v * v;
    }
    double mean = sum / (double)n;
    double var  = std::max(0.0, sum2 / (double)n - mean * mean);
    double std  = std::sqrt(var);
    fprintf(stderr,
            "[stdec_cuda] %s: n=%zu nan=%d inf=%d min=%.4f max=%.4f "
            "mean=%.4f std=%.4f\n",
            tag, n, n_nan, n_inf, vmin, vmax, mean, std);
    return (n_nan || n_inf) ? 1 : 0;
}

}  // namespace

std::vector<float> SpeechTokenizerDecoderCudaEngine::decode(
    const int *codes, int n_codebooks, int T) {

    if (!ready_) {
        fprintf(stderr, "[stdec_cuda] decode() called before init\n");
        return {};
    }
    DecoderCudaContext *ctx = g_ctx_for(this);
    if (!ctx) {
        fprintf(stderr, "[stdec_cuda] decode(): no CUDA context\n");
        return {};
    }
    const int C   = config_.latent_dim;       // 1024
    const int FFN = 4 * C;                     // 4096
    const int C_rvq = config_.rvq_out_dim;     // 512
    const int K_pre = 3;
    const bool dbg = (std::getenv("QWEN_TTS_DEC_DEBUG") != nullptr);

    auto t_total0 = std::chrono::steady_clock::now();

    // --- 1. RVQ decode (host gather + cuBLAS GEMM) → [T, 512] host F32 ----
    std::vector<float> rvq_out;
    if (!rvq_decode(codes, n_codebooks, T, rvq_out)) {
        fprintf(stderr, "[stdec_cuda] decode: rvq_decode failed\n");
        return {};
    }
    // rvq_out is row-major [T, 512] (rvq_out_dim fastest, T outer). Confirmed
    // by the layout comments in rvq_decode().
    if (dbg) {
        fprintf(stderr,
                "[stdec_cuda] rvq_decode -> [T=%d, %d] (host)\n", T, C_rvq);
    }

    // --- Scratch sizing -------------------------------------------------
    if (!ensure_scratch(ctx, T, C, FFN)) return {};

    // Upload rvq_out to device (buf_a).
    cudaMemcpyAsync(ctx->buf_a, rvq_out.data(),
                     (size_t)T * C_rvq * sizeof(float),
                     cudaMemcpyHostToDevice, ctx->stream);

    // --- 2. pre_conv: causal Conv1d 512 -> 1024, k=3 --------------------
    auto t_pre0 = std::chrono::steady_clock::now();
    {
        // im2col(buf_a [T, 512]) -> im2col_buf [T, 3*512=1536]
        launch_causal_conv1d_im2col_f32(ctx->buf_a, ctx->im2col_buf,
                                          T, C_rvq, K_pre, ctx->stream);
        // gemm: out[T, C] (row-major) = im2col[T, K*C_in] @ W[K*C_in, C]
        // cuBLAS view (col-major):  C[C, T] = W_colmaj[C, K*C_in] · im2col_colmaj[K*C_in, T]
        // W stored row-major [K*C_in, C] => col-major [C, K*C_in]. opA = N.
        // im2col stored row-major [T, K*C_in] => col-major [K*C_in, T]. opB = N.
        const float alpha = 1.0f, beta = 0.0f;
        int M = C, N = T, Kk = K_pre * C_rvq;
        cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, Kk,
                     &alpha,
                     ctx->pre_conv_w, M,
                     ctx->im2col_buf, Kk,
                     &beta,
                     ctx->buf_b, M);
        // bias add: buf_b += pre_conv_b broadcast over [T, C]
        launch_bias_add_f32(ctx->buf_b, ctx->pre_conv_b, ctx->buf_b,
                              T, C, ctx->stream);
    }
    cudaStreamSynchronize(ctx->stream);
    auto t_pre1 = std::chrono::steady_clock::now();
    if (dbg) {
        dev_stats("pre_conv", ctx->buf_b, (size_t)T * C, ctx->stream);
    }

    // buf_b now holds [T, C=1024]. Upsample blocks ping-pong.
    float *cur = ctx->buf_b;
    float *nxt = ctx->buf_a;
    int T_cur = T;

    auto upsample_block = [&](int idx) -> bool {
        auto &u = ctx->up[idx];
        // ---- ConvTranspose1d k=2 s=2: cur [T_cur, C] -> nxt [2*T_cur, C] ----
        launch_conv_transpose1d_k2s2_f32(cur, u.up_w, u.up_b, nxt,
                                            T_cur, C, C, ctx->stream);
        T_cur *= 2;
        std::swap(cur, nxt);   // cur = post-transconv [2*T, C], nxt scratch
        // We need a stable residual = post-up tensor. Stash to nxt via copy
        // (we'll overwrite cur in the ConvNeXt subgraph).
        cudaMemcpyAsync(nxt, cur, (size_t)T_cur * C * sizeof(float),
                         cudaMemcpyDeviceToDevice, ctx->stream);
        float *residual = nxt;   // [T_cur, C]
        // We need a third buffer for the ConvNeXt expansion (FFN=4096).
        // Reuse the im2col scratch for the [T_cur, C] dwconv intermediate
        // since pre_conv's im2col (T*1536) is already smaller than 2T*1024
        // for T>=2; we'll re-alloc if too small.
        // Worst-case T_cur*FFN > buf size -> ensure_scratch sized us at 4*T*FFN.
        // Use a local: route through buf_a/buf_b ping-pong when residual is
        // pinned to nxt — but since we swapped, buf_a and buf_b are 'cur'
        // (currently holds post-up) and 'nxt' (residual). We need a third
        // for the FFN expansion. Allocate a one-shot scratch.
        // Reuse im2col_buf (large enough — it was sized for T*K*C_in=T*1536
        // floats; for T_cur=2T at idx=0 that's 4*T*1536 which exceeds the
        // 2*T*FFN=8*T*1024 we need? 4*T*1536 = 6144*T vs 8192*T -- not
        // enough). Use a transient cudaMalloc; T*FFN at T_cur=4T is 4T*4096
        // = 16K*T floats which is only ~512KB at T=8. Cheap.
        float *ffn_buf = nullptr;
        size_t ffn_bytes = (size_t)T_cur * FFN * sizeof(float);
        cudaError_t e = cudaMalloc((void**)&ffn_buf, ffn_bytes);
        if (e != cudaSuccess) {
            fprintf(stderr,"[stdec_cuda] up%d ffn malloc: %s\n",
                    idx, cudaGetErrorString(e));
            return false;
        }
        // ---- dwconv k=7 (causal) on `cur` -> overwrite `cur` ----
        // Order: dw -> norm -> pw1 -> gelu -> pw2 -> *gamma -> + residual
        launch_depthwise_conv1d_causal_f32(cur, u.dw_w, u.dw_b, cur,
                                              T_cur, C, 7, ctx->stream);
        // ---- LayerNorm (affine) over channel axis ----
        launch_layernorm_f32(cur, u.norm_w, u.norm_b, cur,
                              T_cur, C, 1e-6f, ctx->stream);
        // ---- pwconv1: cur [T_cur, C] -> ffn_buf [T_cur, FFN] ----
        // Linear weight stored as PyTorch [FFN, C] row-major bytes
        // (== col-major [C, FFN]). We want col-major C[FFN, T] = W^T_colmaj * X.
        //   opA=T, A=u.pw1_w, lda=C  -> A becomes [FFN, C] col-major.
        //   opB=N, B=cur, ldb=C       -> B becomes [C, T] col-major.
        //   ldc=FFN.
        {
            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         FFN, T_cur, C,
                         &alpha,
                         u.pw1_w, C,
                         cur, C,
                         &beta,
                         ffn_buf, FFN);
            launch_bias_add_f32(ffn_buf, u.pw1_b, ffn_buf,
                                  T_cur, FFN, ctx->stream);
        }
        // ---- GELU (in-place over [T_cur*FFN]) ----
        launch_gelu_erf_f32(ffn_buf, ffn_buf, T_cur * FFN, ctx->stream);
        // ---- pwconv2: ffn_buf [T_cur, FFN] -> cur [T_cur, C] ----
        // Linear weight [C, FFN] PyTorch row-major bytes (== col-major [FFN, C]).
        //   opA=T, A=u.pw2_w, lda=FFN -> [C, FFN] col-major.
        //   opB=N, B=ffn_buf, ldb=FFN
        //   ldc=C
        {
            const float alpha = 1.0f, beta = 0.0f;
            cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                         C, T_cur, FFN,
                         &alpha,
                         u.pw2_w, FFN,
                         ffn_buf, FFN,
                         &beta,
                         cur, C);
            launch_bias_add_f32(cur, u.pw2_b, cur,
                                  T_cur, C, ctx->stream);
        }
        // ---- gamma scaling ----
        launch_channel_scale_f32(cur, u.gamma, cur, T_cur, C, ctx->stream);
        // ---- residual add: cur += residual (= nxt) ----
        launch_add_f32(cur, residual, cur, T_cur * C, ctx->stream);
        cudaFree(ffn_buf);
        return true;
    };

    // --- 3. upsample[0] (T -> 2T) -----------------------------------------
    auto t_up0_0 = std::chrono::steady_clock::now();
    if (!upsample_block(0)) return {};
    cudaStreamSynchronize(ctx->stream);
    auto t_up0_1 = std::chrono::steady_clock::now();
    if (dbg) dev_stats("upsample[0]", cur, (size_t)T_cur * C, ctx->stream);

    // --- 4. upsample[1] (2T -> 4T) ----------------------------------------
    auto t_up1_0 = std::chrono::steady_clock::now();
    if (!upsample_block(1)) return {};
    cudaStreamSynchronize(ctx->stream);
    auto t_up1_1 = std::chrono::steady_clock::now();
    if (dbg) dev_stats("upsample[1]", cur, (size_t)T_cur * C, ctx->stream);

    // --- 5. D2H output [4T, C] --------------------------------------------
    std::vector<float> out((size_t)T_cur * C);
    cudaMemcpyAsync(out.data(), cur, out.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    auto t_total1 = std::chrono::steady_clock::now();
    if (dbg) {
        auto ms = [](auto a, auto b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        fprintf(stderr,
                "[stdec_cuda] timings: pre_conv=%.2f ms up0=%.2f ms "
                "up1=%.2f ms total=%.2f ms (T_in=%d T_out=%d)\n",
                ms(t_pre0, t_pre1), ms(t_up0_0, t_up0_1),
                ms(t_up1_0, t_up1_1), ms(t_total0, t_total1),
                T, T_cur);
    }
    return out;
}

// ===========================================================================
// Phase 2.7c: decode_audio() — full pipeline ending in audio waveform.
//
// Layout: feed `decode()` result [4T, 1024] back into the vocoder forward:
//   y = vocoder_conv(x)                       # [4T, 1536], k=7 causal
//   for blk in 0..3:
//       y = snake_beta(y, blk.snake)
//       y = causal_conv_transpose1d(y, ...)    # upsample by stride S
//       for j in 0..2:
//           res = y
//           y = snake_beta(y, ru.act1)
//           y = dilated_causal_conv1d(y, ru.conv1, dilation=[1,3,9][j])
//           y = snake_beta(y, ru.act2)
//           y = causal_conv1d(y, ru.conv2, k=1)
//           y = y + res
//   y = snake_beta(y, voc_final_snake)
//   y = causal_conv1d(y, voc_final, k=7)       # [T_audio, 1]
//   y = tanh(y)
//
// Channel widths progress 1024 -> 1536 -> 768 -> 384 -> 192 -> 96 -> 1.
// Time progresses 4T -> 4T*8 -> 4T*40 -> 4T*160 -> 4T*480, total 1920*T samples.
// ===========================================================================

std::vector<float> SpeechTokenizerDecoderCudaEngine::decode_audio(
    const int *codes, int n_codebooks, int T) {

    if (!ready_) {
        fprintf(stderr, "[stdec_cuda] decode_audio() called before init\n");
        return {};
    }
    DecoderCudaContext *ctx = g_ctx_for(this);
    if (!ctx) {
        fprintf(stderr, "[stdec_cuda] decode_audio: no CUDA context\n");
        return {};
    }
    const bool dbg = (std::getenv("QWEN_TTS_DEC_DEBUG") != nullptr);

    auto t_start = std::chrono::steady_clock::now();

    // Step 1: run decode() to get [4T, 1024] (RVQ -> pre_conv -> upsample).
    std::vector<float> upsampled = decode(codes, n_codebooks, T);
    if (upsampled.empty()) {
        fprintf(stderr, "[stdec_cuda] decode_audio: decode() failed\n");
        return {};
    }

    auto t_voc0 = std::chrono::steady_clock::now();

    const int C_in_voc = config_.latent_dim;   // 1024
    int T_voc = 4 * T;                          // input T to vocoder
    const int C_init_out = 1536;
    const int K_init = 7;

    // Allocate two ping-pong device buffers sized for the *final* audio
    // length. T grows by 1920× through the vocoder (4*T -> 4*T*1920 = 4T*8*5*4*3).
    // Worst-case memory: T_audio * C_max where C_max = 1536 (right after initial conv).
    // For T=32 -> T_audio = 32*1920 = 61440, * 1536 floats = ~360 MB. Big but OK on GB10.
    // We can do better: peak channels happen at smaller T. Let's compute per-stage
    // and use the peak (T_stage * C_stage).
    //
    // Stages (T_after_stage, C_after_stage):
    //   init:  (4T,        1536)         -> 4T*1536
    //   blk0:  (4T*8,      768)          -> 32T*768  = 24576*T
    //   blk1:  (4T*40,     384)          -> 160T*384 = 61440*T
    //   blk2:  (4T*160,    192)          -> 640T*192 = 122880*T
    //   blk3:  (4T*480,    96)           -> 1920T*96 = 184320*T
    //   final: (4T*480,    1)            -> 1920T
    //
    // Peak is blk3 output: 1920T * 96 = 184320*T floats. Plus the residual
    // stash inside each block needs another buffer of equal size to the
    // current stage. So we need 3 buffers (cur, nxt, residual) sized to peak.
    size_t peak_floats = (size_t)T_voc * 8 * 5 * 4 * 3 * 96;
    // The transconv input buffer also needs to be at least cur*C_in. Since
    // C_in is at most 1536 (right before block 0), peak there is 4T*1536.
    size_t peak_pre_blk0 = (size_t)T_voc * 1536;
    if (peak_pre_blk0 > peak_floats) peak_floats = peak_pre_blk0;

    size_t buf_bytes = peak_floats * sizeof(float);
    float *bA = nullptr, *bB = nullptr, *bRes = nullptr;
    cudaMalloc((void**)&bA,   buf_bytes);
    cudaMalloc((void**)&bB,   buf_bytes);
    cudaMalloc((void**)&bRes, buf_bytes);
    if (!bA || !bB || !bRes) {
        fprintf(stderr, "[stdec_cuda] decode_audio: scratch malloc failed\n");
        if (bA) cudaFree(bA);
        if (bB) cudaFree(bB);
        if (bRes) cudaFree(bRes);
        return {};
    }

    // Upload upsampled [4T, 1024] -> bA.
    cudaMemcpyAsync(bA, upsampled.data(),
                     upsampled.size() * sizeof(float),
                     cudaMemcpyHostToDevice, ctx->stream);

    // im2col scratch for the largest causal_conv1d step. Worst K*C is at
    // res-unit conv1 inside block 0: K=7 * C=768 = 5376 floats per row,
    // times T_after_blk0_up = T_voc*8 rows. Peak = 4T*8 * 7 * 768 = 172032*T
    // floats. Worst case across all blocks happens at block 3 output:
    // 4T*480 * 7 * 96 = 1290240*T (yes, larger because T grew most).
    // Pre-allocate that.
    size_t im2col_peak = (size_t)4 * T * 480 * 7 * 96;
    // Also account for initial conv im2col (4T * 7 * 1024).
    size_t im2col_init = (size_t)T_voc * 7 * 1024;
    if (im2col_init > im2col_peak) im2col_peak = im2col_init;
    float *im2col_dev = nullptr;
    cudaMalloc((void**)&im2col_dev, im2col_peak * sizeof(float));
    if (!im2col_dev) {
        fprintf(stderr, "[stdec_cuda] decode_audio: im2col malloc failed\n");
        cudaFree(bA); cudaFree(bB); cudaFree(bRes);
        return {};
    }

    float *cur = bA;     // input buffer
    float *nxt = bB;     // output buffer

    // ---- Vocoder initial Conv1d: 1024 -> 1536, k=7 ----
    {
        const int K = K_init, Cin = C_in_voc, Cout = C_init_out;
        // im2col [T_voc, K*Cin]
        launch_causal_conv1d_im2col_f32(cur, im2col_dev,
                                          T_voc, Cin, K, ctx->stream);
        // GEMM: nxt[T_voc, Cout] = im2col[T_voc, K*Cin] @ W[K*Cin, Cout]
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     Cout, T_voc, K * Cin,
                     &alpha,
                     ctx->voc_init_w, Cout,
                     im2col_dev, K * Cin,
                     &beta,
                     nxt, Cout);
        launch_bias_add_f32(nxt, ctx->voc_init_b, nxt,
                              T_voc, Cout, ctx->stream);
        std::swap(cur, nxt);
    }
    int C_cur = C_init_out;  // 1536
    if (dbg) {
        cudaStreamSynchronize(ctx->stream);
        dev_stats("voc_init", cur, (size_t)T_voc * C_cur, ctx->stream);
    }

    int dilations[3] = {1, 3, 9};

    // ---- 4 vocoder blocks ----
    for (int bi = 0; bi < 4; ++bi) {
        auto &vb = ctx->voc_blocks[bi];
        // SnakeBeta on cur (C_cur channels)
        launch_snake_beta_f32(cur, vb.snake_alpha, vb.snake_beta, cur,
                                T_voc, C_cur, ctx->stream);

        // ConvTranspose1d: cur [T_voc, C_in] -> nxt [T_voc * stride, C_out]
        int T_up = T_voc * vb.stride;
        launch_causal_conv_transpose1d_f32(cur, vb.up_w, vb.up_b, nxt,
                                              T_voc, vb.C_in, vb.C_out,
                                              vb.kernel, vb.stride,
                                              ctx->stream);
        std::swap(cur, nxt);
        T_voc = T_up;
        C_cur = vb.C_out;

        // 3 residual units
        for (int j = 0; j < 3; ++j) {
            auto &ru = vb.res[j];
            // Save residual: copy cur -> bRes
            cudaMemcpyAsync(bRes, cur,
                             (size_t)T_voc * C_cur * sizeof(float),
                             cudaMemcpyDeviceToDevice, ctx->stream);

            // SnakeBeta act1 (in-place on cur)
            launch_snake_beta_f32(cur, ru.act1_alpha, ru.act1_beta, cur,
                                    T_voc, C_cur, ctx->stream);

            // conv1: dilated causal Conv1d k=7, dilation=dilations[j]
            {
                const int K = 7, D = dilations[j];
                launch_dilated_causal_conv1d_im2col_f32(
                    cur, im2col_dev, T_voc, C_cur, K, D, ctx->stream);
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                             C_cur, T_voc, K * C_cur,
                             &alpha,
                             ru.conv1_w, C_cur,
                             im2col_dev, K * C_cur,
                             &beta,
                             nxt, C_cur);
                launch_bias_add_f32(nxt, ru.conv1_b, nxt,
                                      T_voc, C_cur, ctx->stream);
                std::swap(cur, nxt);
            }

            // SnakeBeta act2
            launch_snake_beta_f32(cur, ru.act2_alpha, ru.act2_beta, cur,
                                    T_voc, C_cur, ctx->stream);

            // conv2: pointwise Conv1d k=1 (== Linear). Weight stored as
            // [C_out, C_in] PyTorch row-major; cublasSgemm(opA=T) treats
            // it as [C_in, C_out] col-major and gives us [T, C_out] row-major.
            {
                const float alpha = 1.0f, beta = 0.0f;
                cublasSgemm(ctx->cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                             C_cur, T_voc, C_cur,
                             &alpha,
                             ru.conv2_w, C_cur,
                             cur, C_cur,
                             &beta,
                             nxt, C_cur);
                launch_bias_add_f32(nxt, ru.conv2_b, nxt,
                                      T_voc, C_cur, ctx->stream);
                std::swap(cur, nxt);
            }

            // Residual: cur += bRes
            launch_add_f32(cur, bRes, cur, T_voc * C_cur, ctx->stream);
        }
        if (dbg) {
            cudaStreamSynchronize(ctx->stream);
            char tag[32];
            snprintf(tag, sizeof(tag), "voc_blk%d", bi);
            dev_stats(tag, cur, (size_t)T_voc * C_cur, ctx->stream);
        }
    }

    // ---- Final SnakeBeta + Conv1d (96 -> 1, k=7) + tanh ----
    // After blocks: T_voc = 4T * 1920, C_cur = 96. Pipeline to [T_audio, 1].
    launch_snake_beta_f32(cur, ctx->voc_final_alpha, ctx->voc_final_beta, cur,
                            T_voc, C_cur, ctx->stream);
    {
        const int K = 7, Cin = 96, Cout = 1;
        launch_causal_conv1d_im2col_f32(cur, im2col_dev,
                                          T_voc, Cin, K, ctx->stream);
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(ctx->cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                     Cout, T_voc, K * Cin,
                     &alpha,
                     ctx->voc_final_w, Cout,
                     im2col_dev, K * Cin,
                     &beta,
                     nxt, Cout);
        launch_bias_add_f32(nxt, ctx->voc_final_b, nxt,
                              T_voc, Cout, ctx->stream);
        std::swap(cur, nxt);
    }
    // Tanh in-place
    launch_tanh_f32(cur, cur, T_voc /* * 1 */, ctx->stream);

    cudaStreamSynchronize(ctx->stream);
    auto t_voc1 = std::chrono::steady_clock::now();

    // D2H copy [T_audio] = T_voc floats
    std::vector<float> audio((size_t)T_voc);
    cudaMemcpyAsync(audio.data(), cur, audio.size() * sizeof(float),
                     cudaMemcpyDeviceToHost, ctx->stream);
    cudaStreamSynchronize(ctx->stream);

    cudaFree(bA);
    cudaFree(bB);
    cudaFree(bRes);
    cudaFree(im2col_dev);

    auto t_end = std::chrono::steady_clock::now();
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    fprintf(stderr,
            "[stdec_cuda] decode_audio: T_in=%d T_audio=%d "
            "voc_wall=%.2f ms total=%.2f ms\n",
            T, T_voc, ms(t_voc0, t_voc1), ms(t_start, t_end));
    return audio;
}

}  // namespace ominix_cuda
