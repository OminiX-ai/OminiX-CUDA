#pragma once
// ============================================================================
// AudioEncoderCudaEngine — Phase 4.1 scaffold (init + GGUF parse only).
//
// Mirror of `AudioEncoder` from the Ascend reference
// (OminiX-Ascend/tools/qwen_asr/audio_encoder.h). The Ascend implementation
// uses ggml on the CPU/NPU; the CUDA version targets cuBLAS + custom CUDA
// kernels and allocates all weight buffers via cudaMalloc once at init.
//
// Architecture (Qwen3-ASR audio encoder):
//   - 3× Conv2d (downsample, hidden=480, stride=2)
//   - Linear projection conv_out (7680 -> 1024)
//   - Sinusoidal positional embedding (computed at init, not stored)
//   - 24× Transformer encoder block (d_model=1024, heads=16, ffn=4096)
//   - Output MLP: LayerNorm + Linear(1024,1024) + GELU + Linear(1024,2048)
//
// Output: (num_frames=122 typical, output_dim=2048) F32 audio embeds
// for split prefill into the Qwen3 28L text decoder.
//
// Phase 4.1 scope: scaffold + GGUF parse + smoke. No forward yet.
// Phase 4.2 (next dispatch) will land the actual forward kernels.
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstdint>

namespace ominix_cuda {

// Per-layer transformer weight handles (device pointers). Phase 4.1 records
// these pointers from the GGUF parse pass; Phase 4.2 will consume them.
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

    // Phase 4.1 API surface (mirrors Ascend AudioEncoder::load).
    //   - init_from_gguf: open GGUF, parse hparams, count layers, record
    //     weight handles. Phase 4.1 records counts + dimensions and exits;
    //     Phase 4.2 will additionally upload F16 weights to device memory.
    bool init_from_gguf(const std::string &gguf_path, int device = 0);

    // Phase 4.2 forward path (stub returns false in Phase 4.1).
    //   mel:        F32 host buffer, layout (n_mels, mel_T) row-major
    //   mel_T:      number of mel time frames (at hop=160, sr=16000)
    //   output:     F32 host buffer, capacity >= num_frames * 2048
    //   num_frames: out — encoder output frame count (typically 122)
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

    // CUDA resources (allocated in init_from_gguf in Phase 4.2).
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

    // Conv2d weight device pointers.
    void *conv2d1_w_ = nullptr, *conv2d1_b_ = nullptr;
    void *conv2d2_w_ = nullptr, *conv2d2_b_ = nullptr;
    void *conv2d3_w_ = nullptr, *conv2d3_b_ = nullptr;
    void *conv_out_w_ = nullptr;

    std::vector<AudioEncoderLayerCuda> layers_;

    // Output MLP.
    void *ln_post_w_ = nullptr, *ln_post_b_ = nullptr;
    void *proj1_w_   = nullptr, *proj1_b_   = nullptr;
    void *proj2_w_   = nullptr, *proj2_b_   = nullptr;

    // Sinusoidal positional embeddings (host buffer; uploaded in 4.2).
    std::vector<float> pos_emb_;

    // Counters reported by smoke harness.
    int n_tensors_seen_   = 0;
    int n_weights_bound_  = 0;
};

} // namespace ominix_cuda
