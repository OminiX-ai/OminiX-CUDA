#pragma once
// ============================================================================
// Talker CANN Engine: Direct ACL-based Qwen3-TTS Talker backbone on Ascend NPU
//
// Mirrors CpCannEngine but targets the 28-layer Talker backbone instead of
// the 5-layer Code Predictor. Loads weights from the standard llama-style
// GGUF produced by `export_talker_llama.py`.
//
// Caller supplies float embeddings directly (embedding lookup / text_projection
// happens upstream). The engine performs:
//   input_embeds (F32 on host) -> F16 transformer compute -> F32 hidden out.
//
// Precision scheme (matches ggml-cann's Qwen3 convention):
//   - F32: I/O staging at engine boundary; RmsNorm gammas (all norm weights).
//   - F16: matmuls, residual adds, RoPE, attention, FFN, KV cache.
//   - Attention: aclnnFusedInferAttentionScoreV2 (the op ggml-cann uses) —
//     layout=BSND, numKeyValueHeads for GQA, innerPrecise=0 for decode
//     (S=1) and innerPrecise=2 with a causal mask for prefill (S>1).
//
// All intermediate buffers allocated once at init. Per-forward path issues
// only aclnn kernel launches — no allocation on the hot path.
// ============================================================================

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <string>
#include <vector>
#include <cstdint>

struct TalkerConfig;

class TalkerCannEngine {
public:
    TalkerCannEngine() = default;
    ~TalkerCannEngine();

    // Load the Talker backbone from a llama-style GGUF (F16 matmul weights,
    // F32 norm gammas). `device` is the ACL device ID (usually 0).
    bool init_from_gguf(const std::string &gguf_path, const TalkerConfig &cfg,
                        int device = 0);

    // Single-token decode path (S=1). Uses KV cache at `pos`.
    //   input_embed: [n_embd] F32 on HOST
    //   hidden_out:  [n_embd] F32 on HOST (post final RmsNorm)
    void forward_decode(const float *input_embed, int pos, float *hidden_out);

    // Batched prefill (S>1). Appends `seq_len` tokens to the KV cache
    // starting at `start_pos`. Only the LAST position's hidden state is
    // returned (TalkerLLM only needs that for next-token sampling).
    //   input_embeds:    [seq_len, n_embd] F32 on HOST
    //   last_hidden_out: [n_embd] F32 on HOST
    void forward_prefill(const float *input_embeds, int seq_len, int start_pos,
                          float *last_hidden_out);

    // Reset KV cache position (caller should do this between utterances).
    void reset_kv_cache();

    // Set RoPE position speed factor (EOS steering — MLX parity). >1.0 makes
    // the model's internal clock run faster. Rebuilds cos/sin tables on
    // change; cheap on subsequent forwards with unchanged factor.
    void set_rope_speed_factor(float factor);

    bool is_ready() const { return ready_; }

private:
    bool ready_ = false;
    int device_ = 0;
    aclrtStream stream_ = nullptr;

    // Model dimensions (cached from config)
    int n_embd_   = 0;   // 2048
    int n_heads_  = 0;   // 16
    int n_kv_     = 0;   // 8
    int head_dim_ = 0;   // 128
    int q_dim_    = 0;   // n_heads * head_dim = 2048
    int kv_dim_   = 0;   // n_kv * head_dim    = 1024
    int inter_    = 0;   // 6144
    int n_layers_ = 0;   // 28
    float eps_        = 0.0f;
    float rope_theta_ = 0.0f;
    float rope_speed_factor_ = 1.0f;

    // Talker sequence budget — Talker may prefill up to ~100 text tokens plus
    // generate a few thousand codec frames. 4096 is a conservative cap that
    // fits 28 * 4096 * 1024 * 2 bytes * 2 (K+V) ≈ 460 MB of KV cache.
    static constexpr int MAX_SEQ = 4096;
    // Max prefill batch size handled by the preallocated staging buffers.
    // If a caller prefills more than this in one shot, we fall back to
    // chunked prefill (each chunk ≤ MAX_PREFILL).
    static constexpr int MAX_PREFILL = 512;

    // ---- NPU weight buffers (persistent, uploaded once) ----
    struct LayerWeights {
        void *q_proj_w = nullptr;    // F16 [q_dim, n_embd]
        void *k_proj_w = nullptr;    // F16 [kv_dim, n_embd]
        void *v_proj_w = nullptr;    // F16 [kv_dim, n_embd]
        void *o_proj_w = nullptr;    // F16 [n_embd, q_dim]
        void *q_norm_w = nullptr;    // F32 [head_dim]
        void *k_norm_w = nullptr;    // F32 [head_dim]
        void *gate_proj_w = nullptr; // F16 [inter, n_embd]
        void *up_proj_w   = nullptr; // F16 [inter, n_embd]
        void *down_proj_w = nullptr; // F16 [n_embd, inter]
        void *input_ln_w  = nullptr; // F32 [n_embd] (attn_norm)
        void *post_ln_w   = nullptr; // F32 [n_embd] (ffn_norm)
    };
    std::vector<LayerWeights> layer_w_;
    void *final_norm_w_dev_ = nullptr;  // F32 [n_embd] (output_norm.weight)

    // ---- NPU intermediate buffers (single-token path) ----
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

    // Prefill staging (batched: seq_len rows of each). Sized for MAX_PREFILL.
    void *cur_batch_dev_      = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *residual_batch_dev_ = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *normed_batch_dev_   = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *q_batch_dev_        = nullptr;  // F16 [MAX_PREFILL, q_dim]
    void *k_batch_dev_        = nullptr;  // F16 [MAX_PREFILL, kv_dim]
    void *v_batch_dev_        = nullptr;  // F16 [MAX_PREFILL, kv_dim]
    void *attn_out_batch_dev_ = nullptr;  // F16 [MAX_PREFILL, q_dim]
    void *o_out_batch_dev_    = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *gate_batch_dev_     = nullptr;  // F16 [MAX_PREFILL, inter]
    void *up_batch_dev_       = nullptr;  // F16 [MAX_PREFILL, inter]
    void *ffn_out_batch_dev_  = nullptr;  // F16 [MAX_PREFILL, n_embd]

    // Causal attention mask for prefill: we build a contiguous F16
    // [seq_len, seq_len_total] buffer per prefill call and upload it to
    // this scratch (sized for the worst case). FIAS's pseShift path rejects
    // strided/padded masks for some tiling keys, so we stick to a packed
    // [1, n_heads, S_q, S_kv] layout with stride[heads]=0 broadcast.
    void *causal_mask_dev_ = nullptr;  // F16 [MAX_PREFILL * MAX_SEQ]

    // RoPE cos/sin tables: precomputed per position [MAX_SEQ, head_dim] with
    // halves duplicated so NEOX-mode RotaryPositionEmbedding maps to the
    // HuggingFace/MLX rotate_half formula.
    void *rope_cos_dev_ = nullptr;  // F16
    void *rope_sin_dev_ = nullptr;  // F16

    // RmsNorm rstd scratch: sized for the largest case (prefill QK-norm over
    // MAX_PREFILL rows × n_heads heads).
    void *rstd_dev_ = nullptr;  // F32

    // ---- KV cache on NPU [n_layers][MAX_SEQ * kv_dim] (F16) ----
    std::vector<void *> k_cache_dev_;
    std::vector<void *> v_cache_dev_;
    int kv_cache_len_ = 0;

    // Boundary staging buffers (F32).
    void *input_stage_f32_dev_  = nullptr;  // F32 [MAX_PREFILL * n_embd]
    void *output_stage_f32_dev_ = nullptr;  // F32 [n_embd]

    // Scalars
    aclScalar *one_scalar_f16_ = nullptr;   // F16 1.0 (for F16 Add alpha)

    // aclnn workspace (grows on demand)
    void *workspace_dev_  = nullptr;
    size_t workspace_size_ = 0;

    // Host-side copies of the full cos/sin tables so set_rope_speed_factor
    // can rebuild the device tables without recomputing trig from scratch.
    std::vector<float> cos_host_;  // [MAX_SEQ * head_dim]
    std::vector<float> sin_host_;

    // ---- Internal helpers ----
    void alloc_dev_(void **ptr, size_t bytes);
    void ensure_workspace_(size_t needed);
    void upload_(void *dev, const void *host, size_t bytes);
    void build_rope_tables_();   // populate cos/sin and upload as F16.
    void build_causal_mask_();   // fill causal_mask_dev_ once at init.
};
