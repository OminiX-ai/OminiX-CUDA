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

// ECAPA-TDNN Speaker Encoder for Qwen3-TTS
// Extracts 2048-dim speaker embedding from 24kHz audio
//
// Architecture:
//   blocks[0]: TDNN Conv1d(128, 512, kernel=5) + ReLU
//   blocks[1-3]: SE-Res2Net blocks (512 channels, scale=8)
//   mfa: Conv1d(1536, 1536, kernel=1) + ReLU
//   asp: AttentiveStatisticsPooling
//   fc: Conv1d(3072, 2048, kernel=1)

struct SpeakerEncoderConfig {
    int mel_dim = 128;
    int enc_dim = 2048;
    int channels = 512;
    int res2net_scale = 8;  // 8 chunks, 7 conv blocks
    int se_channels = 128;
    int attn_channels = 128;
    int sample_rate = 24000;
    // Mel spectrogram params
    int n_fft = 1024;
    int hop_length = 256;
    int win_length = 1024;
    int n_mels = 128;
    float fmin = 0.0f;
    float fmax = 12000.0f;
};

class SpeakerEncoderModel : public BaseModel {
public:
    SpeakerEncoderConfig config;

    bool load_hparams(const ModelLoader &loader) override;
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    void reset_input_shape() override;

private:
    // blocks[0]: initial TDNN
    ggml_tensor *blocks_0_conv_w = nullptr;
    ggml_tensor *blocks_0_conv_b = nullptr;

    // blocks[1-3]: SE-Res2Net blocks
    struct SERes2NetBlock {
        ggml_tensor *tdnn1_conv_w = nullptr;
        ggml_tensor *tdnn1_conv_b = nullptr;
        // res2net_block: 7 conv blocks (scale-1)
        ggml_tensor *res2net_conv_w[7] = {};
        ggml_tensor *res2net_conv_b[7] = {};
        ggml_tensor *tdnn2_conv_w = nullptr;
        ggml_tensor *tdnn2_conv_b = nullptr;
        // SE block
        ggml_tensor *se_conv1_w = nullptr;
        ggml_tensor *se_conv1_b = nullptr;
        ggml_tensor *se_conv2_w = nullptr;
        ggml_tensor *se_conv2_b = nullptr;
    };
    SERes2NetBlock se_blocks_[3];  // blocks[1], blocks[2], blocks[3]

    // mfa
    ggml_tensor *mfa_conv_w = nullptr;
    ggml_tensor *mfa_conv_b = nullptr;

    // asp (AttentiveStatisticsPooling)
    ggml_tensor *asp_tdnn_conv_w = nullptr;
    ggml_tensor *asp_tdnn_conv_b = nullptr;
    ggml_tensor *asp_conv_w = nullptr;
    ggml_tensor *asp_conv_b = nullptr;

    // fc
    ggml_tensor *fc_w = nullptr;
    ggml_tensor *fc_b = nullptr;

    // Graph building helpers
    ggml_tensor *build_tdnn(ggml_context *ctx0, ggml_tensor *x,
                            ggml_tensor *w, ggml_tensor *b, int dilation = 1);
    ggml_tensor *build_se_res2net(ggml_context *ctx0, ggml_tensor *x,
                                  const SERes2NetBlock &blk, int dilation);
    ggml_tensor *build_asp(ggml_context *ctx0, ggml_tensor *x);
};

// High-level Speaker Encoder interface
class SpeakerEncoder {
public:
    SpeakerEncoder() = default;
    ~SpeakerEncoder() = default;

    bool load(const std::string &model_path, const ContextParams &params);
    bool extract(const std::vector<float> &audio, int sample_rate,
                 std::vector<float> &embedding);

    // Compute mel spectrogram (public for testing)
    bool compute_mel(const std::vector<float> &audio,
                     std::vector<float> &mel_spec, int &n_frames);

private:
    SpeakerEncoderConfig config_;
    std::unique_ptr<InferenceSession<SpeakerEncoderModel>> session_;
};
