#pragma once
// ============================================================================
// Image Diffusion CUDA Engine: Native Qwen-Image-Edit (QIE) DiT backbone for
// NVIDIA Blackwell GPUs (GB10, sm_121).
//
// Mirror of Ascend's `ImageDiffusionEngine` (CANN path) tracked under
// `~/work/OminiX-Ascend/tools/qwen_image_edit/native/image_diffusion_engine.*`.
// The architecture is a 60-block double-stream (img + txt) DiT with shared
// modulation, GQA cross-attention, F32 norm gammas, and Q8_0 / Q4_0 / BF16
// matmul weights coming straight out of GGUF. The only thing that changes
// vs the Ascend reference is the kernel dispatch layer:
//
//   aclnnMm                              -> cublasGemmEx
//   aclnnRmsNorm / LayerNorm             -> custom CUDA kernel
//   aclnnApplyRotaryPosEmbV2 (multiaxis) -> custom CUDA kernel (joint RoPE)
//   aclnnFusedInferAttentionScoreV2      -> cuDNN FMHA (FlashAttention) or
//                                           CUTLASS FMHA fallback
//   aclnnWeightQuantBatchMatmul (A8W8)   -> cublasGemmEx INT8 or Q8_0 dequant
//                                           into F16 + cublasGemmEx
//   aclmdlRI (ACL Graph)                 -> cudaGraph_t / cudaGraphExec_t
//
// QIE-Edit-2511 dims (parsed from GGUF schema, frozen for Phase 3):
//   - blocks            = 60   (transformer_blocks.{0..59})
//   - hidden            = 3072 (img/joint stream)
//   - n_heads           = 24
//   - head_dim          = 128
//   - mlp_inter         = 12288   (img_mlp / txt_mlp net.0.proj out features)
//   - mod_dim           = 18432   (= 6 * hidden — img_mod / txt_mod)
//   - text_hidden       = 3584   (input txt_in/txt_norm width)
//   - timestep_inner    = 256    (time_text_embed.timestep_embedder.linear_1.in)
//   - patch_in          = 64     (img_in.weight cols; 2x2 patch * 16 ch latents)
//
// Phase 3.1 (THIS) lands the engine class skeleton + GGUF parse + weight
// upload (all 60 blocks × ~6 weight tensors each + their F32 biases + the
// global head/tail blocks). The Q8_0/Q4_0/BF16 source -> F16 device
// staging mirrors the Phase 2.2 `load_gguf_tensor_f32` pattern verbatim.
//
// Phase 3.2 will fill `forward_block` (mod1 → LN1 → modulate → QKV →
// RMSNorm → joint-RoPE → cross-attn → output → residual → LN2 → FFN →
// residual). Phase 3.3 wires the full loop + lm_head/proj_out + VAE
// decode hand-off.
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

namespace ominix_cuda {

// QIE-Edit transformer config (frozen for Phase 3). All values come from the
// GGUF schema dump (see header comment); no runtime overrides yet.
struct ImageDiffusionConfig {
    int n_blocks      = 60;
    int hidden        = 3072;
    int n_heads       = 24;
    int head_dim      = 128;
    int mlp_inter     = 12288;
    int mod_dim       = 18432;     // 6 * hidden
    int text_hidden   = 3584;
    int patch_in      = 64;        // 2x2 patch * 16 latent channels
    int timestep_inner = 256;
    float ln_eps       = 1e-6f;    // LayerNorm in DiT (affine=false in some)
    float rms_norm_eps = 1e-6f;
    float rope_theta   = 10000.0f;

    // ---- Phase 3.3a multi-axis RoPE config ----
    // Qwen-Image axial assignment: temporal=16 + h=56 + w=56 = 128 = head_dim.
    int rope_axes_temporal = 16;
    int rope_axes_h        = 56;
    int rope_axes_w        = 56;
    int max_txt_seq        = 256;   // pe_table txt slot count
    int max_img_seq        = 4096;  // pe_table img slot count (64*64 patch grid)
};

// Per-block weight pointers. Names mirror the diffusers / GGUF tensor schema:
//   transformer_blocks.<L>.attn.{add_k_proj,add_q_proj,add_v_proj}.{weight,bias}
//   transformer_blocks.<L>.attn.{to_k,to_q,to_v,to_out.0,to_add_out}.{weight,bias}
//   transformer_blocks.<L>.attn.{norm_added_k,norm_added_q,norm_k,norm_q}.weight
//   transformer_blocks.<L>.{img_mlp,txt_mlp}.net.0.proj.{weight,bias}
//   transformer_blocks.<L>.{img_mlp,txt_mlp}.net.2.{weight,bias}
//   transformer_blocks.<L>.{img_mod,txt_mod}.1.{weight,bias}
//
// All matmul weights -> F16 device buffers (rows-major [out, in]).
// All bias / norm-gamma -> F32 device buffers.
struct DiTBlockWeights {
    // ---- Image-stream attention projections ----
    void *to_q_w = nullptr;          // F16 [hidden, hidden]
    void *to_k_w = nullptr;
    void *to_v_w = nullptr;
    void *to_out_0_w = nullptr;      // F16 [hidden, hidden]
    void *to_q_b = nullptr;          // F32 [hidden]
    void *to_k_b = nullptr;
    void *to_v_b = nullptr;
    void *to_out_0_b = nullptr;

    // ---- Text-stream attention projections (add_*) ----
    void *add_q_w = nullptr;
    void *add_k_w = nullptr;
    void *add_v_w = nullptr;
    void *to_add_out_w = nullptr;
    void *add_q_b = nullptr;
    void *add_k_b = nullptr;
    void *add_v_b = nullptr;
    void *to_add_out_b = nullptr;

    // ---- QK-norm gammas (RMSNorm, F32 [head_dim]) ----
    void *norm_q_w = nullptr;
    void *norm_k_w = nullptr;
    void *norm_added_q_w = nullptr;
    void *norm_added_k_w = nullptr;

    // ---- Image MLP (SwiGLU-free; net.0.proj -> GELU -> net.2) ----
    void *img_mlp_0_w = nullptr;     // F16 [mlp_inter, hidden]
    void *img_mlp_2_w = nullptr;     // F16 [hidden, mlp_inter]
    void *img_mlp_0_b = nullptr;     // F32 [mlp_inter]
    void *img_mlp_2_b = nullptr;     // F32 [hidden]

    // ---- Text MLP ----
    void *txt_mlp_0_w = nullptr;
    void *txt_mlp_2_w = nullptr;
    void *txt_mlp_0_b = nullptr;
    void *txt_mlp_2_b = nullptr;

    // ---- AdaLN modulation projections (mod_dim = 6 * hidden) ----
    void *img_mod_w = nullptr;       // F16 [mod_dim, hidden]
    void *img_mod_b = nullptr;       // F32 [mod_dim]
    void *txt_mod_w = nullptr;
    void *txt_mod_b = nullptr;
};

class ImageDiffusionCudaEngine {
public:
    ImageDiffusionCudaEngine() = default;
    ~ImageDiffusionCudaEngine();

    // Phase 3.1 — open DiT GGUF, parse hparams, upload all 60 blocks of
    // weights + global head/tail tensors to device. LLM / vision / VAE GGUFs
    // are accepted (paths can be empty in 3.1) and will be wired in 3.2/3.3.
    bool init_from_gguf(const std::string &dit_path,
                        const std::string &llm_path = "",
                        const std::string &llm_vision_path = "",
                        const std::string &vae_path = "",
                        int device = 0);

    // Phase 3.3a — single-block joint forward.
    //
    // CHANGED FROM PHASE 3.2 (commit 9d9ded58): the caller no longer
    // synthesizes the per-block 12×hidden mod_vec. Instead the engine derives
    // it internally from `timestep` via the t_emb chain (sinusoidal[256] →
    // time_lin1 → silu → time_lin2 → silu → linear-into-mod_vec). The
    // multi-axis RoPE pe-table is precomputed at init.
    //
    //   img_in / img_out : [img_seq_len, hidden]  F32
    //   txt_in / txt_out : [txt_seq_len, hidden]  F32
    //   timestep         : float, in the canonical sigma_s * 1000 range
    //   block_idx        : 0..n_blocks-1
    //
    // Returns true on success.
    bool forward_block(int block_idx,
                       const float *img_in, int img_seq_len,
                       const float *txt_in, int txt_seq_len,
                       float timestep,
                       float *img_out, float *txt_out);

    // Phase 3.3 — proj_out / norm_out post-final-block.
    void final_proj(const float *img_in, int seq_len, float *img_out);

    bool is_ready() const { return ready_; }
    int  n_blocks() const { return cfg_.n_blocks; }
    const ImageDiffusionConfig &config() const { return cfg_; }

    // Diagnostics (Phase 3.1 smoke). Returns aggregate F16 host->device
    // weight byte count after a successful init.
    size_t total_weight_bytes() const { return total_weight_bytes_; }
    // Returns count of weights that failed a finite() sanity check on host
    // post-dequant. Should be zero for a clean GGUF.
    size_t nonfinite_weight_count() const { return nonfinite_weight_count_; }

private:
    bool ready_ = false;
    int  device_ = 0;

    cublasHandle_t cublas_ = nullptr;
#ifdef OMINIX_CUDA_USE_CUDNN
    cudnnHandle_t  cudnn_  = nullptr;
#endif
    cudaStream_t stream_         = nullptr;
    cudaStream_t primary_stream_ = nullptr;

    ImageDiffusionConfig cfg_;
    std::vector<DiTBlockWeights> blocks_;

    // ---- Global head/tail (outside transformer_blocks.*) -------------------
    void *img_in_w_   = nullptr;     // F16 [hidden, patch_in]
    void *img_in_b_   = nullptr;     // F32 [hidden]
    void *txt_in_w_   = nullptr;     // F16 [hidden, text_hidden]
    void *txt_in_b_   = nullptr;     // F32 [hidden]
    void *txt_norm_w_ = nullptr;     // F32 [text_hidden]
    void *time_lin1_w_ = nullptr;    // F16 [hidden, timestep_inner]
    void *time_lin1_b_ = nullptr;    // F32 [hidden]
    void *time_lin2_w_ = nullptr;    // F16 [hidden, hidden]
    void *time_lin2_b_ = nullptr;    // F32 [hidden]
    void *norm_out_w_ = nullptr;     // F16 [mod_dim, hidden]   (norm_out.linear)
    void *norm_out_b_ = nullptr;     // F32 [mod_dim]
    void *proj_out_w_ = nullptr;     // F16 [patch_in, hidden]
    void *proj_out_b_ = nullptr;     // F32 [patch_in]

    // Diagnostics
    size_t total_weight_bytes_ = 0;
    size_t nonfinite_weight_count_ = 0;

    // ---- Phase 3.2 forward-block scratch (lazy-allocated on first call) ----
    // All sized for the smoke target: img_seq_max=4096, txt_seq_max=256.
    int   scratch_img_seq_ = 0;
    int   scratch_txt_seq_ = 0;
    void *scratch_img_f16_     = nullptr;  // F16 [img_seq, H]
    void *scratch_txt_f16_     = nullptr;  // F16 [txt_seq, H]
    void *scratch_img_norm_    = nullptr;  // F16 [img_seq, H]
    void *scratch_txt_norm_    = nullptr;  // F16 [txt_seq, H]
    void *scratch_q_full_      = nullptr;  // F16 [seq_total, H]
    void *scratch_k_full_      = nullptr;  // F16 [seq_total, H]
    void *scratch_v_full_      = nullptr;  // F16 [seq_total, H]
    void *scratch_attn_full_   = nullptr;  // F16 [seq_total, H]
    void *scratch_img_mlp_     = nullptr;  // F16 [img_seq, mlp_inter]
    void *scratch_txt_mlp_     = nullptr;  // F16 [txt_seq, mlp_inter]
    void *scratch_img_proj_    = nullptr;  // F16 [img_seq, H] (post out-proj)
    void *scratch_txt_proj_    = nullptr;  // F16 [txt_seq, H]
    void *scratch_mod_vec_f16_ = nullptr;  // F16 [12 * H]

    // ---- Phase 3.3a multi-axis RoPE pe-table (init-time, persistent) -------
    // Layout:  cos/sin shape = [pe_total_pos, head_dim/2]
    //   pe row 0..max_txt_seq          → txt positions (diagonal t=h=w)
    //   pe row max_txt_seq..total      → img positions ((t=0, h=row, w=col))
    void *pe_cos_dev_   = nullptr;         // F16 [pe_total_pos, head_dim/2]
    void *pe_sin_dev_   = nullptr;         // F16 [pe_total_pos, head_dim/2]
    int   pe_total_pos_ = 0;

    // ---- Phase 3.3a t_emb scratch (init-time, persistent) ------------------
    void *t_emb_in_f16_   = nullptr;       // F16 [256]    (sinusoidal in)
    void *t_emb_mid_f16_  = nullptr;       // F16 [hidden] (post time_lin1)
    void *silu_t_emb_f16_ = nullptr;       // F16 [hidden] (post time_lin2 + silu)

    bool ensure_scratch_(int img_seq_len, int txt_seq_len);
    bool build_pe_table_();
    bool compute_t_emb_(float timestep);

    // Helpers
    void alloc_dev_(void **ptr, size_t bytes);
};

}  // namespace ominix_cuda
