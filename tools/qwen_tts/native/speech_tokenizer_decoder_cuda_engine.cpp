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

std::vector<float> SpeechTokenizerDecoderCudaEngine::decode(
    const int *codes, int n_codebooks, int T) {
    (void)codes;
    (void)n_codebooks;
    (void)T;
    fprintf(stderr,
            "[stdec_cuda] decode() not implemented in Phase 2.7a "
            "(scaffold + RVQ only). pre_conv=2.7b, vocoder=2.7c. ABORT.\n");
    std::abort();
}

}  // namespace ominix_cuda
