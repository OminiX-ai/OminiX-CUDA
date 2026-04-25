#pragma once
// ============================================================================
// Talker CUDA Engine: Direct cuBLAS/cuDNN-based Qwen3-TTS Talker backbone on
// NVIDIA Blackwell GPUs (GB10, sm_121).
//
// Direct port of TalkerCannEngine (Ascend native; 1->32 fps on 910B). The
// architecture is identical: 28-layer Qwen3 transformer decoder, GQA 16/8,
// SwiGLU FFN, per-channel quant, NEOX RoPE theta=1e6, KV cache. The only
// thing that changes is the kernel dispatch layer:
//
//   aclnnMm                              -> cublasGemmEx
//   aclnnRmsNorm                         -> custom CUDA kernel (rms_norm.cu)
//   aclnnApplyRotaryPosEmbV2             -> custom CUDA kernel (rope.cu)
//   aclnnFusedInferAttentionScoreV2      -> cuDNN FMHA (FlashAttention-3) or
//                                           CUTLASS FMHA fallback
//   aclnnWeightQuantBatchMatmul (A16W8)  -> cublasGemmEx INT8/FP8 (Phase 2.6)
//   aclmdlRI (ACL Graph)                 -> cudaGraph_t + cudaGraphExec_t
//                                           (Phase 2.5)
//
// Caller supplies float embeddings directly (token embedding lookup happens
// upstream in talker.cpp). The engine performs:
//   input_embeds (F32 host) -> F16 transformer compute -> F32 hidden out.
//
// Precision scheme (matches the Ascend reference exactly so weight files
// and parity tests transfer 1:1):
//   - F32: I/O staging at engine boundary; RmsNorm gammas (all norm weights).
//   - F16: matmuls, residual adds, RoPE, attention, FFN, KV cache.
//   - Phase 2.6 will switch matmuls to FP8 E4M3 / INT8 with F16 scales.
//
// All intermediate buffers allocated once at init via cudaMalloc. The hot
// path issues only kernel launches (cuBLAS / cuDNN handles + custom CUDA
// kernels). No allocation per-token after warm-up.
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstdint>

#ifdef OMINIX_CUDA_USE_CUDNN
#include <cudnn.h>
#endif

struct TalkerConfig;  // defined in tools/qwen_tts/talker.h

namespace ominix_cuda {

// Forward decls; populated in Phase 2.5 (graph capture).
struct PerPosGraphCache;

class TalkerCudaEngine {
public:
    TalkerCudaEngine() = default;
    ~TalkerCudaEngine();

    // Phase 2.1 API surface (mirrors TalkerCannEngine):
    //   - init_from_gguf : open GGUF, parse hparams, upload F16 weights, alloc
    //     KV cache + scratch, build RoPE tables. Stub returns true today; the
    //     real upload lands in Phase 2.2.
    //   - forward_decode : single-token autoregressive step. Stub asserts in
    //     Phase 2.1.
    //   - forward_prefill: batched seq>1 ingestion. Stub asserts in Phase 2.1.
    //   - reset_kv_cache : clear KV pos counter (no buffer wipe; positions are
    //     overwritten in place).

    bool init_from_gguf(const std::string &gguf_path, const TalkerConfig &cfg,
                        int device = 0);

    void forward_decode(const float *input_embed, int pos, float *hidden_out);

    void forward_prefill(const float *input_embeds, int seq_len, int start_pos,
                         float *last_hidden_out);

    void reset_kv_cache();

    // RoPE position-speed factor (EOS-steering parity with Ascend / MLX).
    void set_rope_speed_factor(float factor);

    // Sector-aware MRoPE for xvec/customvoice (mirrors Ascend B6.1).
    void set_use_mrope_xvec_layout(bool enable);
    bool use_mrope_xvec_layout() const { return use_mrope_xvec_layout_; }

    // ---- Phase 2.6 quant toggles (defaults OFF; F16 lane stays the
    //      bit-identical reference) ----------------------------------
    // INT8 per-channel symmetric (A16W8). Mirrors Ascend Stretch S1.
    void set_use_int8_weights(bool enable) { use_int8_weights_ = enable; }
    bool use_int8_weights() const { return use_int8_weights_; }
    bool int8_applied()    const { return int8_applied_; }

    // Native FP8 E4M3 on Blackwell. cuBLAS supports CUDA_R_8F_E4M3 directly.
    void set_use_fp8_weights(bool enable) { use_fp8_weights_ = enable; }
    bool use_fp8_weights() const { return use_fp8_weights_; }
    bool fp8_applied()    const { return fp8_applied_; }

    // ---- Phase 2.5 CUDA Graphs ----------------------------------------
    // Captures one cudaGraph_t per `pos` slot. Replays at second visit.
    // Equivalent of the Ascend TALKER_CP_ACLGRAPH=1 lever.
    void set_use_cuda_graphs(bool enable) { use_cuda_graphs_ = enable; }
    bool use_cuda_graphs() const { return use_cuda_graphs_; }

    // ---- Stream accessors (Phase 2.3 multi-stream pipelining) ---------
    cudaStream_t get_stream()         const { return stream_; }
    cudaStream_t get_stream_b()       const { return stream_b_; }
    cudaStream_t get_primary_stream() const { return primary_stream_; }
    void set_stream(cudaStream_t s) {
        stream_ = (s != nullptr) ? s : primary_stream_;
    }

    bool is_ready() const { return ready_; }

private:
    bool ready_ = false;
    int  device_ = 0;

    // CUDA resources owned by this engine.
    cublasHandle_t cublas_ = nullptr;
#ifdef OMINIX_CUDA_USE_CUDNN
    cudnnHandle_t  cudnn_  = nullptr;
#endif

    cudaStream_t stream_         = nullptr;  // alias of primary_stream_ by default
    cudaStream_t primary_stream_ = nullptr;  // owned
    cudaStream_t stream_b_       = nullptr;  // owned (multi-stream pipelining)

    // Recorded after the final cast in forward_decode_launch (Phase 2.3).
    cudaEvent_t  decode_done_event_ = nullptr;

    // ---- Cached model dims --------------------------------------------------
    int n_embd_   = 0;   // 2048
    int n_heads_  = 0;   // 16
    int n_kv_     = 0;   // 8
    int head_dim_ = 0;   // 128
    int q_dim_    = 0;
    int kv_dim_   = 0;
    int inter_    = 0;   // 6144
    int n_layers_ = 0;   // 28
    float eps_        = 0.0f;
    float rope_theta_ = 0.0f;
    float rope_speed_factor_ = 1.0f;

    // MRoPE sector-aware layout (B6.1 parity).
    int  mrope_temporal_section_ = 0;
    bool use_mrope_xvec_layout_  = false;

    static constexpr int MAX_SEQ     = 4096;
    static constexpr int MAX_PREFILL = 512;

    // ---- Per-layer device weight buffers (uploaded once) -------------------
    struct LayerWeights {
        // F16 path (always populated in Phase 2.2).
        void *q_proj_w    = nullptr;  // F16 [q_dim,  n_embd]
        void *k_proj_w    = nullptr;  // F16 [kv_dim, n_embd]
        void *v_proj_w    = nullptr;  // F16 [kv_dim, n_embd]
        void *o_proj_w    = nullptr;  // F16 [n_embd, q_dim]
        void *q_norm_w    = nullptr;  // F32 [head_dim]
        void *k_norm_w    = nullptr;  // F32 [head_dim]
        void *gate_proj_w = nullptr;  // F16 [inter,  n_embd]
        void *up_proj_w   = nullptr;  // F16 [inter,  n_embd]
        void *down_proj_w = nullptr;  // F16 [n_embd, inter]
        void *input_ln_w  = nullptr;  // F32 [n_embd]
        void *post_ln_w   = nullptr;  // F32 [n_embd]

        // INT8 lane (Phase 2.6, A16W8). Populated only when int8_applied_.
        void *q_proj_w_i8    = nullptr;  // INT8 [q_dim,  n_embd]
        void *k_proj_w_i8    = nullptr;  // INT8 [kv_dim, n_embd]
        void *v_proj_w_i8    = nullptr;  // INT8 [kv_dim, n_embd]
        void *o_proj_w_i8    = nullptr;  // INT8 [n_embd, q_dim]
        void *gate_proj_w_i8 = nullptr;  // INT8 [inter,  n_embd]
        void *up_proj_w_i8   = nullptr;  // INT8 [inter,  n_embd]
        void *down_proj_w_i8 = nullptr;  // INT8 [n_embd, inter]
        void *q_proj_scale    = nullptr; // F16 [q_dim]
        void *k_proj_scale    = nullptr; // F16 [kv_dim]
        void *v_proj_scale    = nullptr; // F16 [kv_dim]
        void *o_proj_scale    = nullptr; // F16 [n_embd]
        void *gate_proj_scale = nullptr; // F16 [inter]
        void *up_proj_scale   = nullptr; // F16 [inter]
        void *down_proj_scale = nullptr; // F16 [n_embd]

        // FP8 lane (Phase 2.6 E4M3). Same shapes as INT8; scales reused.
        void *q_proj_w_fp8    = nullptr;
        void *k_proj_w_fp8    = nullptr;
        void *v_proj_w_fp8    = nullptr;
        void *o_proj_w_fp8    = nullptr;
        void *gate_proj_w_fp8 = nullptr;
        void *up_proj_w_fp8   = nullptr;
        void *down_proj_w_fp8 = nullptr;
    };
    std::vector<LayerWeights> layer_w_;
    void *final_norm_w_dev_ = nullptr;  // F32 [n_embd]

    // ---- Decode-path (S=1) intermediate buffers ----------------------------
    void *cur_dev_      = nullptr;  // F16 [n_embd]
    void *residual_dev_ = nullptr;  // F16 [n_embd]
    void *normed_dev_   = nullptr;  // F16 [n_embd]
    void *q_dev_        = nullptr;  // F16 [q_dim]
    void *k_dev_        = nullptr;  // F16 [kv_dim]
    void *v_dev_        = nullptr;  // F16 [kv_dim]
    void *attn_out_dev_ = nullptr;  // F16 [q_dim]
    void *o_out_dev_    = nullptr;  // F16 [n_embd]
    void *gate_dev_     = nullptr;  // F16 [inter]
    void *up_dev_       = nullptr;  // F16 [inter]
    void *ffn_out_dev_  = nullptr;  // F16 [n_embd]

    // ---- Prefill (S>1) staging (sized for MAX_PREFILL) ----------------------
    void *cur_batch_dev_      = nullptr;
    void *residual_batch_dev_ = nullptr;
    void *normed_batch_dev_   = nullptr;
    void *q_batch_dev_        = nullptr;
    void *k_batch_dev_        = nullptr;
    void *v_batch_dev_        = nullptr;
    void *attn_out_batch_dev_ = nullptr;
    void *o_out_batch_dev_    = nullptr;
    void *gate_batch_dev_     = nullptr;
    void *up_batch_dev_       = nullptr;
    void *ffn_out_batch_dev_  = nullptr;
    void *causal_mask_dev_    = nullptr;  // F16 [MAX_PREFILL * MAX_SEQ]

    // ---- RoPE tables (precomputed, NEOX layout: halves duplicated) ---------
    void *rope_cos_dev_ = nullptr;  // F16 [MAX_SEQ, head_dim]
    void *rope_sin_dev_ = nullptr;  // F16 [MAX_SEQ, head_dim]
    std::vector<float> cos_host_;   // [MAX_SEQ * head_dim]
    std::vector<float> sin_host_;

    // ---- RmsNorm rstd scratch -----------------------------------------------
    void *rstd_dev_ = nullptr;  // F32

    // ---- KV cache (per-layer F16 contiguous) -------------------------------
    std::vector<void *> k_cache_dev_;  // [n_layers] each F16 [MAX_SEQ, kv_dim]
    std::vector<void *> v_cache_dev_;
    int kv_cache_len_ = 0;

    // ---- Boundary staging --------------------------------------------------
    void *input_stage_f32_dev_  = nullptr;  // F32 [MAX_PREFILL * n_embd]
    void *output_stage_f32_dev_ = nullptr;  // F32 [n_embd]

    // ---- Phase 2.6 quant flags ---------------------------------------------
    bool use_int8_weights_ = false;
    bool int8_applied_     = false;
    bool use_fp8_weights_  = false;
    bool fp8_applied_      = false;

    // ---- Phase 2.5 CUDA Graphs ---------------------------------------------
    bool use_cuda_graphs_ = false;
    // Per-pos cache; one cudaGraphExec_t per pos slot. Lazy-populated:
    // first call at pos=p captures; subsequent calls replay.
    std::vector<cudaGraphExec_t> decode_graph_execs_;

    // ---- Internal helpers (defined in .cpp / sibling .cu files) ------------
    void alloc_dev_(void **ptr, size_t bytes);
    void build_rope_tables_();
    void build_causal_mask_();

    // Phase 2.2 — per-token forward kernel sequence (eager).
    void run_decode_ops_(int pos);

    // Phase 2.6 — calibrate one [out, in] F32 weight to per-channel symmetric
    // INT8 + F16 scale; allocates new device buffers.
    bool int8_calibrate_weight_(const float *host_w, int64_t rows, int64_t cols,
                                void *&weight_i8_dev, void *&scale_dev);
};

}  // namespace ominix_cuda
