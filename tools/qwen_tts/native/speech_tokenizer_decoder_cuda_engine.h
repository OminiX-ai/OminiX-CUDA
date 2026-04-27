#pragma once
// ============================================================================
// SpeechTokenizerDecoderCudaEngine — Phase 2.7a scaffold + RVQ decode.
//
// Direct port of the Ascend `SpeechTokenizerDecoderModel`
// (tools/qwen_tts/speech_tokenizer_decoder.{h,cpp} on OminiX-Ascend) targeting
// NVIDIA Blackwell GB10 (sm_121). The Ascend reference builds a ggml graph
// dispatched over aclnn ops; the CUDA port runs a hand-rolled forward driven
// directly by cuBLAS / custom CUDA kernels (no ggml-backend round-trip).
//
// Architecture (16 RVQ codebooks → 24kHz audio @ 1920× upsample):
//
//   codes[16, T]
//     RVQ decode: 1 semantic codebook + 15 acoustic → [512, T]      <-- 2.7a
//     pre_conv (Conv1d 512→1024, k=3, causal)                       <-- 2.7b
//     pre_transformer (8-layer sliding-window attn, w=72)            <-- 2.7b
//     upsample (2× ConvTranspose1d + ConvNeXt, ratios 2,2)            <-- 2.7b
//     vocoder Conv1d (1024→1536) + 4 SnakeBeta blocks                 <-- 2.7c
//     final Conv1d (96→1) → audio @ 24kHz
//
// Phase 2.7a deliverable: GGUF parse + RVQ decode only. The other stages
// abort with a clear message until 2.7b / 2.7c land.
// ============================================================================

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <string>
#include <vector>
#include <cstdint>

namespace ominix_cuda {

struct DecoderConfig {
    int codebook_size      = 2048;
    int codebook_dim       = 256;
    int hidden_size        = 1024;   // pre_transformer hidden dim
    int latent_dim         = 1024;   // pre_conv output / upsample dim
    int num_hidden_layers  = 8;
    int num_attention_heads = 16;
    int num_key_value_heads = 16;
    int head_dim           = 64;
    int intermediate_size  = 3072;
    int num_quantizers     = 16;
    int decoder_dim        = 1536;
    int sliding_window     = 72;
    float rope_theta       = 10000.0f;
    float rms_norm_eps     = 1e-5f;
    int output_sample_rate  = 24000;
    int decode_upsample_rate = 1920;

    // RVQ output dim — sum of first.output_proj and rest.output_proj output
    // channels. For Qwen3-TTS this is 512 (Conv1d codebook_dim → 512 with k=1).
    int rvq_out_dim = 512;
};

// One RVQ codebook layer. Stored on host (F32) for Phase 2.7a — RVQ decode
// is dispatched as a host loop with cuBLAS GEMM handling the output_proj.
struct RVQCodebookHost {
    std::vector<float> embedding_sum;   // [codebook_dim, codebook_size]  (col-major: codebook_dim is fastest)
    std::vector<float> cluster_usage;   // [codebook_size]
};

struct RVQGroupHost {
    // input_proj is unused in decode (same as Ascend reference).
    std::vector<float> output_proj_w;   // [rvq_out_dim, codebook_dim]
    std::vector<RVQCodebookHost> codebooks;
};

class SpeechTokenizerDecoderCudaEngine {
public:
    SpeechTokenizerDecoderCudaEngine() = default;
    ~SpeechTokenizerDecoderCudaEngine();

    // Load GGUF + populate config + parse RVQ tensors. Phase 2.7a does NOT
    // upload the pre_conv / pre_transformer / upsample / vocoder weights to
    // device; those stages are scaffolded but skipped until 2.7b lands.
    bool init_from_gguf(const std::string &gguf_path, int device = 0);

    // RVQ decode only. codes is row-major int[n_codebooks * T]; n_codebooks
    // must equal config.num_quantizers (16). Output: [rvq_out_dim, T] flat
    // F32 row-major (rvq_out_dim is fastest axis to match the Ascend ggml
    // shape convention used downstream by build_pre_conv).
    bool rvq_decode(const int *codes, int n_codebooks, int T,
                    std::vector<float> &out);

    // Phase 2.7a stub. Aborts with std::abort() and a clear log line.
    std::vector<float> decode(const int *codes, int n_codebooks, int T);

    const DecoderConfig &config() const { return config_; }
    bool is_ready() const { return ready_; }

private:
    bool ready_ = false;
    int  device_ = 0;

    DecoderConfig config_;

    // Host-side RVQ tables (used by rvq_decode for Phase 2.7a — no device
    // upload yet; we'll move to a fused CUDA kernel + cuBLAS GEMM in 2.7b).
    RVQGroupHost rvq_first_;   // 1 codebook (semantic)
    RVQGroupHost rvq_rest_;    // 15 codebooks (acoustic)

    // Cached normalized embeddings (embedding_sum / cluster_usage) per codebook.
    // Computed once at init to avoid the divide on every rvq_decode call.
    // Layout: codebooks_norm[group_idx][codebook_idx] = [codebook_dim, codebook_size]
    // row-major with codebook_dim fastest (matches Ascend ggml layout).
    std::vector<std::vector<float>> rvq_first_norm_;
    std::vector<std::vector<float>> rvq_rest_norm_;
};

}  // namespace ominix_cuda
