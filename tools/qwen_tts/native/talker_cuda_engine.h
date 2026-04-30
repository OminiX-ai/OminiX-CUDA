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

    // ---- Phase 2.9 device LM-head -------------------------------------
    // Upload an LM-head weight matrix [vocab, n_embd] (row-major F32 host)
    // onto the device as F16 once. After this, callers can use
    // forward_decode_with_logits() to skip the per-token D2H of the hidden
    // state (4 KB) and replace the host matvec with a cuBLAS GEMM that
    // emits F32 logits directly (only `vocab` floats D2H per step).
    bool upload_lm_head_weights(const float *lm_head_w_f32, int vocab);
    bool has_lm_head_uploaded() const { return lm_head_w_f16_dev_ != nullptr; }
    int  lm_head_vocab() const { return lm_head_vocab_; }

    // ---- Phase 2.6 LM-head FP8 lane (Blackwell sm_121a) ---------------
    // Optional FP8 E4M3 LM-head. Calibrates per-tensor amax over the F32
    // host weight, computes a single F32 scale_w = amax / E4M3_MAX, packs
    // weights as FP8 on device. Coexists with the F16 lane — set
    // set_use_fp8_lm_head(true) to switch the inference path; set false
    // to fall back to F16. Only kicks in for forward_decode_with_logits.
    // Returns false if cublasLt is unavailable or the heuristic doesn't
    // accept the layout (caller falls back to F16 silently).
    bool upload_lm_head_weights_fp8(const float *lm_head_w_f32, int vocab);
    bool has_lm_head_uploaded_fp8() const { return lm_head_w_fp8_dev_ != nullptr; }
    void set_use_fp8_lm_head(bool enable) { use_fp8_lm_head_ = enable; }
    bool use_fp8_lm_head() const { return use_fp8_lm_head_; }

    // Same body as forward_decode(), but instead of D2H'ing the hidden
    // state and forcing the caller to compute LM-head on host, this runs
    // the LM-head GEMM on device and D2Hs only the F32 logits[vocab].
    // Requires upload_lm_head_weights() to have been called first.
    void forward_decode_with_logits(const float *input_embed, int pos,
                                    float *logits_out_f32);

    // ---- P2 (April 2026) on-device sampling autoregressive loop -------
    // Eliminates the per-step D2H of logits + host sampling + H2D of next
    // embedding. Pre-uploads the token-embedding table once, then runs a
    // tight loop on the engine's CUDA stream:
    //
    //   1. (step 0) H2D first input embedding into input_stage_f32_dev_.
    //      (subsequent steps the embedding-lookup kernel writes directly.)
    //   2. forward_decode body + LM-head GEMM (existing captured graph).
    //   3. Optional repetition-penalty kernel (rolls a recent-window over
    //      `next_token_dev_`).
    //   4. Top-K sampler kernel writes next_token_dev_[step].
    //   5. Embedding lookup kernel writes input_stage_f32_dev_ for next step.
    //   6. Every `eos_check_period` steps, host sync + D2H of the partial
    //      tokens to check EOS. On EOS: trim and return.
    //
    // Returns the count of tokens written to `out_tokens_host` (which must
    // have capacity >= max_steps).
    struct OnDevSamplingConfig {
        int      top_k              = 50;
        float    temperature        = 0.9f;
        int      do_sample          = 1;          // 0=greedy, 1=stochastic
        uint64_t seed               = 42;
        int      sample_lo          = 0;
        int      sample_hi          = 0;          // exclusive
        float    repetition_penalty = 1.0f;
        int      recent_window      = 0;
        int      recent_offset      = 0;          // added to recent ids before clamp
        int      eos_token          = -1;         // absolute; -1 disables
        int      eos_check_period   = 16;         // host sync cadence
    };

    // Pre-upload an F32 token-embedding table to device. Must match
    // [embed_vocab, n_embd_]. Returns false on failure. Memory is owned by
    // the engine (freed in dtor). Caller can re-bind by calling again with
    // a different table pointer.
    bool upload_embedding_table_f32(const float *table_f32, int embed_vocab);

    // Upload LM-head weights AND keep an F32 copy on device. The F32 copy is
    // used by decode_loop_topk_ondev() so the on-device GEMM matches the
    // host's matvec_f32 reference bit-for-bit (modulo cuBLAS GEMM ordering
    // noise). Use this for the Talker, which prior code sampled in F32 on
    // host and so requires F32 parity. Predictor stays on the F16 lane.
    bool upload_lm_head_weights_with_f32_copy(const float *lm_head_w_f32,
                                                int vocab);

    // Allocate the on-device sampler scratch buffer (next_token_dev_ size N).
    // No-op if already sized >= N. Called automatically by decode_loop_topk_ondev.
    void ensure_sampler_scratch_(int n_pending);

    int decode_loop_topk_ondev(const float *first_input_emb_f32,
                                int start_pos,
                                int max_steps,
                                const OnDevSamplingConfig &cfg,
                                int *out_tokens_host);

    // ---- P3 (April 2026) on-device predictor frame decode ------------------
    // Predictor variant of decode_loop_topk_ondev. The predictor pattern
    // differs in three ways:
    //   1) KV-cache resets at the start of every frame (15 group steps per
    //      frame; pos_dev_ resets to -1 so the first chain-graph increment
    //      yields *pos_dev_ == 0).
    //   2) Each group step samples in a per-group sub-vocab range
    //      [g*group_size, (g+1)*group_size). Caller passes group_size and
    //      n_groups; the loop derives per-step lo/hi internally.
    //   3) Repetition-penalty history is per-group, accumulated across PRIOR
    //      frames (not the current frame). The engine maintains a device-
    //      resident `[n_groups, max_frames]` rep-history buffer; the sampler
    //      writes the zero-based id (sampled_abs - lo) into
    //      rep_history_dev_[g, frame_t] at the end of each group step. The
    //      caller passes frame_t (current frame index, monotonic across the
    //      lifetime of the request) and recent_window — the kernel reads
    //      rep_history_dev_[g, max(0, frame_t - recent_window) .. frame_t).
    //
    // Embedding lookup: the next group's input embedding is the embedding at
    // ABSOLUTE token id `next_token_dev_[g]` (which is in [g*group_size,
    // (g+1)*group_size)). The engine's embedding table must therefore be the
    // predictor's full token_embd table (vocab = n_groups * group_size).
    //
    // Race-safe per P2 pattern:
    //   - pos_dev_ is initialized to -1 by a 1-thread set kernel on stream_
    //     (no H2D-from-pinned).
    //   - Inside the captured chain graph (when enabled), the FIRST node is
    //     launch_increment_int(pos_dev_).
    //   - The host never writes to *pos_host_pin_ inside this method.
    //
    // Returns 0 on failure (env not initialised, missing uploads), or
    // n_groups on success. On success, out_tokens_host[g] for g in [0,
    // n_groups) holds the sampled absolute token id (caller subtracts
    // g*group_size to get the zero-based codec id).
    struct OnDevPredictorConfig {
        int      n_groups            = 15;        // 1..MAX_PREDICTOR_GROUPS
        int      group_size          = 2048;      // per-group sub-vocab width
        int      top_k               = 50;
        float    temperature         = 0.9f;
        int      do_sample           = 1;
        uint64_t seed                = 42;
        float    repetition_penalty  = 1.0f;
        int      recent_window       = 0;
        int      max_frames          = 0;         // upper bound for rep-history
                                                  // buffer alloc; if <= frame_t
                                                  // engine grows on demand.
    };

    static constexpr int MAX_PREDICTOR_GROUPS = 16;

    int decode_predictor_frame_topk_ondev(const float *first_input_emb_f32,
                                           int frame_t,
                                           const OnDevPredictorConfig &cfg,
                                           int *out_tokens_host);

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

    // ---- Phase 2.9 device LM-head ------------------------------------------
    // Optional. Populated by upload_lm_head_weights().
    void *lm_head_w_f16_dev_     = nullptr;  // F16 [vocab, n_embd] row-major
    void *lm_head_hidden_f16_dev_ = nullptr; // F16 [n_embd] (cast staging)
    void *lm_head_logits_f32_dev_ = nullptr; // F32 [vocab]
    // P2: optional F32 LM-head weight buffer. When populated (via
    // upload_lm_head_weights_with_f32_copy()), the on-device sampling loop
    // uses an F32×F32 GEMM for parity with the host F32 matvec_f32 path
    // (avoids the F16 LM-head precision divergence that flips greedy argmax
    // for some tokens with close logits). Only set for engines whose
    // sampling MUST match host F32 reference; predictor leaves this null
    // and continues on the F16 lane.
    void *lm_head_w_f32_dev_     = nullptr;  // F32 [vocab, n_embd]
    int   lm_head_vocab_         = 0;

    // ---- Phase 2.6 FP8 LM-head lane ----------------------------------------
    // Populated by upload_lm_head_weights_fp8(); used when use_fp8_lm_head_=1
    // inside forward_decode_with_logits(). All raw void* (FP8 byte storage)
    // so the .h doesn't need to pull cuda_fp8.h. cublasLt handle is also
    // an opaque pointer.
    void *lm_head_w_fp8_dev_     = nullptr;   // E4M3 [vocab, n_embd] row-major
    void *lm_head_x_fp8_dev_     = nullptr;   // E4M3 [n_embd] (cast staging)
    void *lm_head_logits_f16_dev_ = nullptr;  // F16  [vocab] (cublasLt D)
    void *lm_head_scale_a_dev_    = nullptr;  // F32 (weight scale)
    void *lm_head_scale_b_dev_    = nullptr;  // F32 (input scale, fixed=1)
    float lm_head_scale_a_host_   = 0.0f;     // weight scale (=amax_W / E4M3_MAX)
    void *lm_head_lt_handle_      = nullptr;  // cublasLtHandle_t
    void *lm_head_lt_desc_        = nullptr;  // cublasLtMatmulDesc_t
    void *lm_head_lt_layout_a_    = nullptr;  // cublasLtMatrixLayout_t
    void *lm_head_lt_layout_b_    = nullptr;  // cublasLtMatrixLayout_t
    void *lm_head_lt_layout_d_    = nullptr;  // cublasLtMatrixLayout_t
    void *lm_head_lt_pref_        = nullptr;  // cublasLtMatmulPreference_t
    void *lm_head_lt_workspace_   = nullptr;
    size_t lm_head_lt_ws_bytes_   = 0;
    // Cached heuristic algo storage (cublasLtMatmulHeuristicResult_t — kept
    // as raw bytes to avoid leaking cublasLt types into the public header).
    void *lm_head_lt_algo_blob_   = nullptr;  // cublasLtMatmulHeuristicResult_t (heap)
    bool  use_fp8_lm_head_        = false;

    // ---- Phase 2.6 quant flags ---------------------------------------------
    bool use_int8_weights_ = false;
    bool int8_applied_     = false;
    bool use_fp8_weights_  = false;
    bool fp8_applied_      = false;

    // ---- Phase 2.5 CUDA Graphs ---------------------------------------------
    bool use_cuda_graphs_ = false;
    // Per-pos cache; one cudaGraphExec_t per pos slot. Lazy-populated:
    // first call at pos=p captures; subsequent calls replay.
    // (Legacy Phase 2.5 path — used only when OMNX_TTS_DECODE_GRAPH_ONCE=0.)
    std::vector<cudaGraphExec_t> decode_graph_execs_;

    // ---- P0 capture-once decode graph (April 2026) -------------------------
    // Capture-once / replay-many path. Topology is identical across all
    // positions for the 28-layer decoder; only per-pos pointer offsets
    // (rope_cos_row, k_slot, v_slot) and the seq_len_total scalar change.
    // We capture into a transient cudaGraph_t each call, then on the first
    // call instantiate it; on subsequent calls call cudaGraphExecUpdate to
    // patch the existing exec in-place (~µs vs ~ms for instantiate).
    //
    // Two execs because forward_decode and forward_decode_with_logits have
    // slightly different captured bodies (the latter ends with the LM-head
    // GEMM and an F16->F32 cast). Sharing is unsafe.
    bool decode_graph_once_     = false;     // env-gated by OMNX_TTS_DECODE_GRAPH_ONCE
    cudaGraphExec_t graph_once_decode_         = nullptr;  // forward_decode
    cudaGraphExec_t graph_once_decode_logits_  = nullptr;  // forward_decode_with_logits

    // ---- P1 (April 2026) device-resident pos --------------------------------
    // When decode_graph_once_ is enabled, the captured graph reads `pos` from
    // a device-side int* at kernel runtime (RoPE, attention, V-write all use
    // the *_dev launchers). The host updates pos_host_ before each replay,
    // and the captured H2D memcpy node copies it into pos_dev_ as the first
    // node of the graph. This makes the graph topology truly static across
    // positions: single instantiate, single cudaGraphLaunch per step, zero
    // cudaGraphExecUpdate calls.
    int *pos_dev_       = nullptr;   // device int [1]
    int *pos_host_pin_  = nullptr;   // pinned host int [1]; capture-safe H2D src

    // ---- P2 (April 2026) on-device sampling state --------------------------
    // Embedding-table pointer (F32 [embed_vocab, n_embd_]) used by
    // launch_embedding_lookup_f32. Owned by the engine; freed in dtor.
    void *embed_table_f32_dev_ = nullptr;
    int   embed_vocab_         = 0;

    // Output token buffer used by the on-device sampler. Size grows on demand
    // via ensure_sampler_scratch_; capacity tracked separately.
    int  *next_token_dev_      = nullptr;
    int   next_token_capacity_ = 0;

    // Chain-mode captured graph: forward_decode body + LM-head GEMM, but
    // WITHOUT the input H2D (input_stage_f32_dev_ is already populated by the
    // prior step's embedding-lookup kernel). Captured lazily on first use.
    cudaGraphExec_t graph_once_decode_chain_ = nullptr;

    // ---- P3 (April 2026) predictor on-device state -------------------------
    // Rep-history buffer for the predictor's per-group cross-frame repetition
    // penalty. Layout: [n_groups, max_frames] row-major. Each frame g writes
    // rep_history_dev_[g * rep_history_max_frames_ + frame_t] = sampled_id
    // (zero-based, in [0, group_size)). Reads are
    // rep_history_dev_[g, max(0, frame_t - recent_window) .. frame_t).
    //
    // Allocated lazily on first decode_predictor_frame_topk_ondev() call,
    // grown via a fresh cudaMalloc + cudaFree if max_frames exceeds capacity.
    int *rep_history_dev_         = nullptr;
    int  rep_history_n_groups_    = 0;
    int  rep_history_max_frames_  = 0;

    // ---- Internal helpers (defined in .cpp / sibling .cu files) ------------
    void alloc_dev_(void **ptr, size_t bytes);
    void build_rope_tables_();
    void build_causal_mask_();

    // Phase 2.2 — per-token forward kernel sequence (eager). When use_dev_pos
    // is true, routes RoPE / attention / V-write through the *_dev launchers
    // that read pos from pos_dev_ instead of using host-side `pos`.
    void run_decode_ops_(int pos);
    void run_decode_ops_dev_();   // P1: device-pos variant for capture-once

    // Phase 2.6 — calibrate one [out, in] F32 weight to per-channel symmetric
    // INT8 + F16 scale; allocates new device buffers.
    bool int8_calibrate_weight_(const float *host_w, int64_t rows, int64_t cols,
                                void *&weight_i8_dev, void *&scale_dev);

    // Phase 2.6 FP8 LM-head — frees cublasLt resources and FP8 buffers.
    // Defined in talker_cuda_engine.cpp; safe to call multiple times.
    void teardown_fp8_lm_head_();
};

}  // namespace ominix_cuda
