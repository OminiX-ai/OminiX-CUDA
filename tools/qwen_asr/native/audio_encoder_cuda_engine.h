#pragma once
// ============================================================================
// AudioEncoderCudaEngine — Phase 4.2 forward.
//
// Mirror of `AudioEncoder` from the Ascend reference
// (OminiX-Ascend/tools/qwen_asr/audio_encoder.h). The Ascend implementation
// uses ggml on CPU/NPU; this CUDA version targets cuBLAS GemmEx + custom
// CUDA kernels and allocates all weight buffers via cudaMalloc once at init.
//
// Architecture (Qwen3-ASR audio encoder):
//   - 3× Conv2d (downsample, hidden=480, stride=2, kernel=3, pad=1) → GELU-erf
//   - Linear projection conv_out (7680 -> 1024)
//   - Sinusoidal positional embedding (computed at init, uploaded to device)
//   - 24× Transformer encoder block (Pre-LN), d_model=1024, heads=16,
//     ffn=4096; full bidirectional attention (no mask, matches Python
//     reference using SDPA with attention_mask=None / is_causal=False).
//   - Output MLP: LayerNorm + Linear(1024,1024) + GELU-erf + Linear(1024,2048)
//
// Output: (num_frames, output_dim=2048) F32 audio embeds for split prefill
// into the Qwen3 28L text decoder.
//
// Phase 4.2 numeric strategy: F32 throughout (encoder activation footprint is
// small enough that F32 storage is < 100MB at typical 122-frame sequences,
// and F32 matches the Ascend ggml CPU reference exactly without dynamic-range
// concerns).
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstdint>

namespace ominix_cuda {

// Per-layer transformer weight handles (device pointers, all F32).
struct AudioEncoderLayerCuda {
    // Self-attention.
    void *self_attn_layer_norm_w = nullptr;
    void *self_attn_layer_norm_b = nullptr;
    void *q_proj_w = nullptr;
    void *q_proj_b = nullptr;
    void *k_proj_w = nullptr;
    void *k_proj_b = nullptr;
    void *v_proj_w = nullptr;
    void *v_proj_b = nullptr;
    void *out_proj_w = nullptr;
    void *out_proj_b = nullptr;
    // FFN.
    void *final_layer_norm_w = nullptr;
    void *final_layer_norm_b = nullptr;
    void *fc1_w = nullptr;
    void *fc1_b = nullptr;
    void *fc2_w = nullptr;
    void *fc2_b = nullptr;
};

class AudioEncoderCudaEngine {
public:
    AudioEncoderCudaEngine() = default;
    ~AudioEncoderCudaEngine();

    // Init: open GGUF, parse hparams, upload all weights to device, prepare
    // sinusoidal positional embeddings.
    bool init_from_gguf(const std::string &gguf_path, int device = 0);

    // Forward (Phase 4.2):
    //   mel:        F32 host buffer, layout (n_mels, mel_T) row-major
    //   mel_T:      number of mel time frames (at hop=160, sr=16000)
    //   output:     F32 host buffer, capacity >= num_frames * output_dim
    //   num_frames: out — encoder output frame count
    //
    // Implementation mirrors Ascend AudioEncoder::encode:
    //   1. Chunk mel along time into n_window*2-sized chunks (last chunk
    //      gets the remainder).
    //   2. Pad chunks to max_chunk_len → padded_mel batch.
    //   3. Conv2d×3 stack with GELU-erf → conv_out_w_ projection
    //      → (chunk_num, frames_per_chunk, d_model).
    //   4. Compute valid frame counts via get_feat_extract_output_lengths,
    //      add positional embedding, concat valid frames.
    //   5. 24L pre-LN transformer (no mask) + final LayerNorm + proj1 →
    //      GELU-erf → proj2.
    bool encode(const float *mel, int n_mels, int mel_T,
                float *output, int &num_frames);

    bool is_ready()        const { return ready_; }

    // Hyperparameter accessors (populated by init_from_gguf).
    int d_model()          const { return d_model_; }
    int encoder_layers()   const { return encoder_layers_; }
    int encoder_heads()    const { return encoder_attention_heads_; }
    int encoder_ffn_dim()  const { return encoder_ffn_dim_; }
    int num_mel_bins()     const { return num_mel_bins_; }
    int output_dim()       const { return output_dim_; }
    int max_source_pos()   const { return max_source_positions_; }

private:
    bool ready_  = false;
    int  device_ = 0;

    // CUDA resources.
    cublasHandle_t cublas_   = nullptr;
    cudaStream_t   stream_   = nullptr;

    // Hyperparameters (defaults match Qwen3-ASR-1.7B audio encoder).
    int d_model_                  = 1024;
    int encoder_layers_           = 24;
    int encoder_attention_heads_  = 16;
    int encoder_ffn_dim_          = 4096;
    int num_mel_bins_             = 128;
    int downsample_hidden_size_   = 480;
    int output_dim_               = 2048;
    int max_source_positions_     = 1500;
    int n_window_                 = 50;
    int n_window_infer_           = 800;
    int mel_reduced_              = 16;
    int conv_out_dim_             = 7680;

    // Conv2d weight device pointers (F32).
    void *conv2d1_w_  = nullptr, *conv2d1_b_  = nullptr;
    void *conv2d2_w_  = nullptr, *conv2d2_b_  = nullptr;
    void *conv2d3_w_  = nullptr, *conv2d3_b_  = nullptr;
    void *conv_out_w_ = nullptr;

    std::vector<AudioEncoderLayerCuda> layers_;

    // Output MLP (F32).
    void *ln_post_w_ = nullptr, *ln_post_b_ = nullptr;
    void *proj1_w_   = nullptr, *proj1_b_   = nullptr;
    void *proj2_w_   = nullptr, *proj2_b_   = nullptr;

    // Sinusoidal positional embeddings (uploaded to device, F32).
    void *pos_emb_dev_ = nullptr;

    // Forward scratch (resized lazily by ensure_scratch_).
    void *scratch_mel_       = nullptr;  // [chunk_num, 1, num_mel_bins, max_chunk_len]
    void *scratch_im2col_    = nullptr;  // im2col workspace (largest of 3 convs)
    void *scratch_conv1_     = nullptr;  // [chunk_num, h1, w1, downsample_hidden]
    void *scratch_conv2_     = nullptr;  // [chunk_num, h2, w2, downsample_hidden]
    void *scratch_conv3_     = nullptr;  // [chunk_num, h3, w3, downsample_hidden]
    void *scratch_conv_proj_ = nullptr;  // [chunk_num, frames_per_chunk, d_model]
    void *scratch_concat_    = nullptr;  // [total_valid_frames, d_model]
    void *scratch_resid_     = nullptr;  // [total_valid_frames, d_model] (transformer residual)
    void *scratch_norm_      = nullptr;  // [total_valid_frames, d_model] (norm output)
    void *scratch_q_         = nullptr;  // [total_valid_frames, d_model]
    void *scratch_k_         = nullptr;
    void *scratch_v_         = nullptr;
    void *scratch_attn_out_  = nullptr;  // [total_valid_frames, d_model]
    void *scratch_ffn_inter_ = nullptr;  // [total_valid_frames, encoder_ffn_dim]
    void *scratch_proj1_out_ = nullptr;  // [total_valid_frames, d_model]
    void *scratch_output_    = nullptr;  // [total_valid_frames, output_dim] (final)

    // Capacity planned for scratch (so realloc only when needed).
    int  scratch_chunk_num_      = 0;
    int  scratch_max_chunk_len_  = 0;
    int  scratch_total_frames_   = 0;

    bool ensure_scratch_(int chunk_num, int max_chunk_len, int total_frames,
                          int frames_per_chunk);
    bool free_scratch_();

    // Counters reported by smoke harness.
    int n_tensors_seen_   = 0;
    int n_weights_bound_  = 0;
};

}  // namespace ominix_cuda
