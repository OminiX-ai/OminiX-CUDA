#pragma once

#include "build_graph.h"
#include "ctx_manager.h"
#include "ggml.h"
#include "model_defs.h"
#include "model_loader.h"
#include "infer_session.hpp"
#include <memory>
#include <string>
#include <vector>

// Speech Tokenizer Encoder for Qwen3-TTS
// Converts 24kHz audio waveform to codec tokens (16 quantizers × T)
//
// Architecture:
//   Conv encoder: Conv1d(1→64,k=7) → 4× [ResNet + ELU + DownsampleConv] → ELU + Conv1d(1024→512,k=3)
//     downsample strides: [4, 5, 6, 8] → ÷960
//   Encoder transformer: 8 layers (LayerNorm, MHA 8 heads, GELU MLP, LayerScale, sliding_window=250)
//   Downsample: Conv1d(512→512, k=4, s=2) → ÷2
//   Total downsample: 960 × 2 = 1920 (→ 12.5 Hz frame rate)
//   RVQ quantize: 1 semantic + 15 acoustic codebooks (codebook_size=2048, codebook_dim=256)

struct EncoderConfig {
    int num_filters = 64;
    int hidden_size = 512;          // transformer hidden dim & encoder output channels
    int num_hidden_layers = 8;      // transformer layers
    int num_attention_heads = 8;
    int num_key_value_heads = 8;
    int head_dim = 64;              // hidden_size / num_attention_heads
    int intermediate_size = 2048;   // MLP hidden dim
    int codebook_size = 2048;
    int codebook_dim = 256;
    int num_quantizers = 32;        // total in GGUF (semantic + acoustic)
    int valid_num_quantizers = 16;  // actually used: 1 semantic + 15 acoustic
    int sliding_window = 250;
    float rope_theta = 10000.0f;
    float norm_eps = 1e-5f;
    float layer_scale_init = 0.01f;
    int input_sample_rate = 24000;
    int encode_downsample_rate = 1920;
    // Conv encoder downsample strides (reversed from decoder upsampling_ratios)
    int downsample_strides[4] = {4, 5, 6, 8};
};

// ResNet block: ELU → Conv1d(ch→ch/2, k=3, dilation) → ELU → Conv1d(ch/2→ch, k=1) + skip
struct EncoderResNetBlock {
    ggml_tensor *conv1_w = nullptr;  // [3, ch, ch/2] dilated
    ggml_tensor *conv1_b = nullptr;
    ggml_tensor *conv2_w = nullptr;  // [1, ch/2, ch]
    ggml_tensor *conv2_b = nullptr;
};

// Downsample block: ResNet + ELU + DownsampleConv
struct EncoderDownBlock {
    EncoderResNetBlock resnet;
    // ELU has no parameters
    ggml_tensor *ds_conv_w = nullptr;  // [kernel, ch_in, ch_out]
    ggml_tensor *ds_conv_b = nullptr;
};

// Transformer layer: LayerNorm + MHA + LayerScale + LayerNorm + GELU MLP + LayerScale
struct EncoderTransformerLayer {
    ggml_tensor *q_proj_w = nullptr;   // [512, 512]
    ggml_tensor *k_proj_w = nullptr;
    ggml_tensor *v_proj_w = nullptr;
    ggml_tensor *o_proj_w = nullptr;
    ggml_tensor *input_layernorm_w = nullptr;   // [512]
    ggml_tensor *input_layernorm_b = nullptr;
    ggml_tensor *post_attn_layernorm_w = nullptr;
    ggml_tensor *post_attn_layernorm_b = nullptr;
    ggml_tensor *self_attn_layer_scale = nullptr;  // [512]
    ggml_tensor *mlp_layer_scale = nullptr;        // [512]
    ggml_tensor *fc1_w = nullptr;   // [512, 2048]
    ggml_tensor *fc2_w = nullptr;   // [2048, 512]
};

// RVQ codebook for one quantizer layer
struct EncoderRVQCodebook {
    ggml_tensor *embed_sum = nullptr;      // [256, 2048] (need to divide by cluster_usage)
    ggml_tensor *cluster_usage = nullptr;  // [2048]
};

// RVQ group (semantic or acoustic)
struct EncoderRVQGroup {
    ggml_tensor *input_proj_w = nullptr;   // [1, 512, 256] Conv1d
    ggml_tensor *output_proj_w = nullptr;  // [1, 256, 512] Conv1d
    std::vector<EncoderRVQCodebook> codebooks;
};

class SpeechTokenizerEncoderModel : public BaseModel {
public:
    EncoderConfig config;

    bool load_hparams(const ModelLoader &loader) override;
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    void reset_input_shape() override;

    // RVQ quantizer (public for weight download in high-level class)
    EncoderRVQGroup rvq_semantic_;   // 1 codebook
    EncoderRVQGroup rvq_acoustic_;   // up to 31 codebooks (use first 15)

private:
    // Conv encoder
    ggml_tensor *init_conv_w_ = nullptr;  // [7, 1, 64]
    ggml_tensor *init_conv_b_ = nullptr;  // [64]
    EncoderDownBlock down_blocks_[4];     // strides: 4, 5, 6, 8
    ggml_tensor *final_conv_w_ = nullptr; // [3, 1024, 512]
    ggml_tensor *final_conv_b_ = nullptr; // [512]

    // Encoder transformer (8 layers)
    std::vector<EncoderTransformerLayer> tf_layers_;

    // Post-transformer downsample: Conv1d(512→512, k=4, s=2)
    ggml_tensor *downsample_conv_w_ = nullptr;  // [4, 512, 512]

    // Graph building helpers
    ggml_tensor *build_causal_conv1d(ggml_context *ctx0, ggml_tensor *x,
                                      ggml_tensor *w, ggml_tensor *b,
                                      int stride = 1, int dilation = 1);
    ggml_tensor *build_resnet_block(ggml_context *ctx0, ggml_tensor *x,
                                     const EncoderResNetBlock &blk, int dilation);
    ggml_tensor *build_conv_encoder(ggml_context *ctx0, ggml_tensor *x);
    ggml_tensor *build_transformer(ggml_context *ctx0, ggml_tensor *x);
};

// High-level Speech Tokenizer Encoder interface
class SpeechTokenizerEncoder {
public:
    SpeechTokenizerEncoder() = default;
    ~SpeechTokenizerEncoder() = default;

    bool load(const std::string &model_path, const ContextParams &params);

    // Encode audio waveform to codec tokens
    // audio: mono 24kHz float samples
    // codes: output [num_quantizers][time_steps]
    // hidden_out: optional, if non-null receives hidden states [hidden, T] before RVQ
    bool encode(const std::vector<float> &audio,
                std::vector<std::vector<int>> &codes,
                std::vector<float> *hidden_out = nullptr);

    const EncoderConfig &get_config() const { return config_; }

private:
    EncoderConfig config_;
    std::unique_ptr<InferenceSession<SpeechTokenizerEncoderModel>> session_;

    // Pre-downloaded RVQ weights for CPU-side quantization
    std::vector<float> sem_input_proj_;   // semantic input projection
    std::vector<float> sem_output_proj_;  // semantic output projection
    std::vector<float> acou_input_proj_;  // acoustic input projection
    std::vector<float> acou_output_proj_; // acoustic output projection
    std::vector<std::vector<float>> codebooks_;  // [n_q][cb_size * cb_dim]

    void prepare_codebooks();
};
