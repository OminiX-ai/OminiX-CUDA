#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"
#include <string>
#include <vector>
#include <memory>

class ModelLoader;
#include "ctx_manager.h"

struct AudioEncoderLayer {
    // Self-attention
    ggml_tensor *self_attn_layer_norm_w = nullptr;
    ggml_tensor *self_attn_layer_norm_b = nullptr;
    ggml_tensor *q_proj_w = nullptr;
    ggml_tensor *q_proj_b = nullptr;
    ggml_tensor *k_proj_w = nullptr;
    ggml_tensor *k_proj_b = nullptr;
    ggml_tensor *v_proj_w = nullptr;
    ggml_tensor *v_proj_b = nullptr;
    ggml_tensor *out_proj_w = nullptr;
    ggml_tensor *out_proj_b = nullptr;
    // FFN
    ggml_tensor *final_layer_norm_w = nullptr;
    ggml_tensor *final_layer_norm_b = nullptr;
    ggml_tensor *fc1_w = nullptr;
    ggml_tensor *fc1_b = nullptr;
    ggml_tensor *fc2_w = nullptr;
    ggml_tensor *fc2_b = nullptr;
};

class AudioEncoder {
public:
    bool load(const std::string &gguf_path, const std::string &device = "CPU", int n_threads = 4);

    // Encode mel spectrogram -> audio features
    // mel: (num_mel_bins, T) in row-major
    // output: (num_frames, output_dim) features
    bool encode(const std::vector<float> &mel, int mel_T,
                std::vector<float> &output, int &num_frames);

private:
    // Hyperparameters
    int d_model_ = 1024;
    int encoder_layers_ = 24;
    int encoder_attention_heads_ = 16;
    int encoder_ffn_dim_ = 4096;
    int num_mel_bins_ = 128;
    int downsample_hidden_size_ = 480;
    int output_dim_ = 2048;
    int max_source_positions_ = 1500;
    int n_window_ = 50;
    int n_window_infer_ = 800;
    int mel_reduced_ = 16;
    int conv_out_dim_ = 7680;

    // Conv2d weights
    ggml_tensor *conv2d1_w_ = nullptr, *conv2d1_b_ = nullptr;
    ggml_tensor *conv2d2_w_ = nullptr, *conv2d2_b_ = nullptr;
    ggml_tensor *conv2d3_w_ = nullptr, *conv2d3_b_ = nullptr;
    ggml_tensor *conv_out_w_ = nullptr;

    // Transformer layers
    std::vector<AudioEncoderLayer> layers_;

    // Output MLP
    ggml_tensor *ln_post_w_ = nullptr, *ln_post_b_ = nullptr;
    ggml_tensor *proj1_w_ = nullptr, *proj1_b_ = nullptr;
    ggml_tensor *proj2_w_ = nullptr, *proj2_b_ = nullptr;

    // Precomputed sinusoidal positional embeddings
    std::vector<float> pos_emb_; // (max_source_positions, d_model)

    // Context management
    std::unique_ptr<ContextManager> ctx_mgr_;
    std::string device_name_;
    int n_threads_ = 4;

    // Helper methods
    void compute_sinusoidal_pos_emb();
    int get_feat_extract_output_lengths(int input_lengths);

    // Run Conv2d on a single padded chunk batch
public:
    bool run_conv2d(const std::vector<float> &padded_mel,
                    int batch_size, int mel_h, int mel_w,
                    std::vector<float> &output);

    // Run Transformer encoder on concatenated frames
private:
    bool run_transformer(const std::vector<float> &hidden_states,
                         int total_frames,
                         const std::vector<int> &cu_seqlens,
                         std::vector<float> &output);
};
