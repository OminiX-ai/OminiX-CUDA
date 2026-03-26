#include "speaker_encoder.h"
#include "stft.h"
#include "utils.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>

// ============================================================================
// Mel spectrogram computation
// ============================================================================

// Librosa slaney mel scale (htk=False, which is the default)
static float hz_to_mel(float hz) {
    const float f_sp = 200.0f / 3.0f;           // 66.667
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp; // 15.0
    const float logstep = std::log(6.4f) / 27.0f;

    if (hz >= min_log_hz) {
        return min_log_mel + std::log(hz / min_log_hz) / logstep;
    }
    return hz / f_sp;
}

static float mel_to_hz(float mel) {
    const float f_sp = 200.0f / 3.0f;
    const float min_log_hz = 1000.0f;
    const float min_log_mel = min_log_hz / f_sp;
    const float logstep = std::log(6.4f) / 27.0f;

    if (mel >= min_log_mel) {
        return min_log_hz * std::exp(logstep * (mel - min_log_mel));
    }
    return f_sp * mel;
}

// Create mel filterbank (slaney normalization)
static std::vector<float> create_mel_filterbank(
    int n_fft, int n_mels, float sr, float fmin, float fmax) {

    int n_freqs = n_fft / 2 + 1;
    std::vector<float> fb(n_mels * n_freqs, 0.0f);

    float mel_min = hz_to_mel(fmin);
    float mel_max = hz_to_mel(fmax);

    std::vector<float> mel_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        mel_pts[i] = mel_min + (mel_max - mel_min) * i / (n_mels + 1);
    }

    std::vector<float> hz_pts(n_mels + 2);
    for (int i = 0; i < n_mels + 2; i++) {
        hz_pts[i] = mel_to_hz(mel_pts[i]);
    }

    // Frequency bins
    std::vector<float> fft_freqs(n_freqs);
    for (int i = 0; i < n_freqs; i++) {
        fft_freqs[i] = sr * i / n_fft;
    }

    for (int m = 0; m < n_mels; m++) {
        float left = hz_pts[m];
        float center = hz_pts[m + 1];
        float right = hz_pts[m + 2];

        // Slaney normalization
        float enorm = 2.0f / (hz_pts[m + 2] - hz_pts[m]);

        for (int f = 0; f < n_freqs; f++) {
            float freq = fft_freqs[f];
            if (freq >= left && freq <= center && center > left) {
                fb[m * n_freqs + f] = enorm * (freq - left) / (center - left);
            } else if (freq > center && freq <= right && right > center) {
                fb[m * n_freqs + f] = enorm * (right - freq) / (right - center);
            }
        }
    }
    return fb;
}

bool SpeakerEncoder::compute_mel(const std::vector<float> &audio,
                                  std::vector<float> &mel_spec, int &n_frames) {
    const auto &c = config_;

    // compute_stft already adds reflect padding internally
    // (like Python: manual pad + STFT center=False)
    stft::STFTParams sp;
    sp.n_fft = c.n_fft;
    sp.hop_length = c.hop_length;
    sp.win_length = c.win_length;
    sp.sampling_rate = c.sample_rate;

    std::vector<float> magnitude;
    if (!stft::compute_stft(audio, sp, magnitude, n_frames)) {
        fprintf(stderr, "STFT failed\n");
        return false;
    }

    int n_freqs = c.n_fft / 2 + 1;

    // Create mel filterbank
    auto fb = create_mel_filterbank(c.n_fft, c.n_mels, (float)c.sample_rate,
                                     c.fmin, c.fmax);

    // Apply filterbank: mel = fb @ magnitude
    // magnitude: (n_freqs, n_frames) row-major
    // fb: (n_mels, n_freqs) row-major
    // output: (n_mels, n_frames) row-major
    mel_spec.resize(c.n_mels * n_frames);
    for (int m = 0; m < c.n_mels; m++) {
        for (int t = 0; t < n_frames; t++) {
            float val = 0.0f;
            for (int f = 0; f < n_freqs; f++) {
                val += fb[m * n_freqs + f] * magnitude[f * n_frames + t];
            }
            // Log compression: log(clamp(x, min=1e-5))
            mel_spec[m * n_frames + t] = std::log(std::max(val, 1e-5f));
        }
    }
    return true;
}

// ============================================================================
// SpeakerEncoderModel: load_hparams
// ============================================================================

bool SpeakerEncoderModel::load_hparams(const ModelLoader &loader) {
    loader.get_u32("enc_dim", config.enc_dim, false);
    loader.get_u32("sample_rate", config.sample_rate, false);
    return true;
}

// ============================================================================
// SpeakerEncoderModel: get_tensors_to_load
// ============================================================================

std::vector<ggml_tensor *>
SpeakerEncoderModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> t;

    // blocks[0]: initial TDNN
    blocks_0_conv_w = get_tensor(ctx, "blocks.0.conv.weight", t);
    blocks_0_conv_b = get_tensor(ctx, "blocks.0.conv.bias", t);

    // blocks[1-3]: SE-Res2Net
    for (int i = 0; i < 3; i++) {
        auto &blk = se_blocks_[i];
        std::string pfx = "blocks." + std::to_string(i + 1) + ".";

        blk.tdnn1_conv_w = get_tensor(ctx, pfx + "tdnn1.conv.weight", t);
        blk.tdnn1_conv_b = get_tensor(ctx, pfx + "tdnn1.conv.bias", t);

        for (int j = 0; j < 7; j++) {
            std::string rp = pfx + "res2net_block.blocks." + std::to_string(j) + ".conv.";
            blk.res2net_conv_w[j] = get_tensor(ctx, rp + "weight", t);
            blk.res2net_conv_b[j] = get_tensor(ctx, rp + "bias", t);
        }

        blk.tdnn2_conv_w = get_tensor(ctx, pfx + "tdnn2.conv.weight", t);
        blk.tdnn2_conv_b = get_tensor(ctx, pfx + "tdnn2.conv.bias", t);

        blk.se_conv1_w = get_tensor(ctx, pfx + "se_block.conv1.weight", t);
        blk.se_conv1_b = get_tensor(ctx, pfx + "se_block.conv1.bias", t);
        blk.se_conv2_w = get_tensor(ctx, pfx + "se_block.conv2.weight", t);
        blk.se_conv2_b = get_tensor(ctx, pfx + "se_block.conv2.bias", t);
    }

    // mfa
    mfa_conv_w = get_tensor(ctx, "mfa.conv.weight", t);
    mfa_conv_b = get_tensor(ctx, "mfa.conv.bias", t);

    // asp
    asp_tdnn_conv_w = get_tensor(ctx, "asp.tdnn.conv.weight", t);
    asp_tdnn_conv_b = get_tensor(ctx, "asp.tdnn.conv.bias", t);
    asp_conv_w = get_tensor(ctx, "asp.conv.weight", t);
    asp_conv_b = get_tensor(ctx, "asp.conv.bias", t);

    // fc
    fc_w = get_tensor(ctx, "fc.weight", t);
    fc_b = get_tensor(ctx, "fc.bias", t);

    return t;
}

void SpeakerEncoderModel::reset_input_shape() {
    input_shapes_ = {{"mel", {1, 128}}};  // (T, n_mels) - T is dynamic
}

// ============================================================================
// Graph building helpers
// ============================================================================

// TDNN: Conv1d with "same" padding + ReLU
ggml_tensor *SpeakerEncoderModel::build_tdnn(
    ggml_context *ctx0, ggml_tensor *x,
    ggml_tensor *w, ggml_tensor *b, int dilation) {
    // Conv1d with same padding
    int kernel_size = (int)w->ne[0];
    int pad = (kernel_size - 1) * dilation / 2;
    ggml_tensor *out = build_conv1d(ctx0, x, w, b, /*stride=*/1, pad, dilation);
    out = ggml_relu(ctx0, out);
    return out;
}

// SE-Res2Net block
ggml_tensor *SpeakerEncoderModel::build_se_res2net(
    ggml_context *ctx0, ggml_tensor *x,
    const SERes2NetBlock &blk, int dilation) {

    ggml_tensor *residual = x;

    // tdnn1: Conv1d(in, out, kernel=1) + ReLU
    ggml_tensor *cur = build_conv1d(ctx0, x, blk.tdnn1_conv_w, blk.tdnn1_conv_b);
    cur = ggml_relu(ctx0, cur);

    // Res2Net: split into 8 chunks along channel dim
    // cur shape: (T, 512) → split into 8 chunks of (T, 64)
    int T = (int)cur->ne[0];
    int C = (int)cur->ne[1];
    int chunk_size = C / config.res2net_scale;  // 64

    // Process chunks with residual connections
    // chunk 0: pass through
    // chunks 1-7: conv + add previous
    std::vector<ggml_tensor *> chunks;
    for (int s = 0; s < config.res2net_scale; s++) {
        // Extract chunk: view of cur[:, s*chunk_size:(s+1)*chunk_size]
        ggml_tensor *chunk = ggml_view_2d(ctx0, cur,
            T, chunk_size,
            cur->nb[1],  // stride for dim 1
            s * chunk_size * cur->nb[1]);  // offset

        if (s == 0) {
            chunks.push_back(chunk);
        } else {
            ggml_tensor *inp = chunk;
            if (s > 1) {
                inp = ggml_add(ctx0, chunk, chunks.back());
            }
            // Conv1d with same padding + ReLU
            int kernel = (int)blk.res2net_conv_w[s - 1]->ne[0];
            int pad = (kernel - 1) * dilation / 2;
            ggml_tensor *conv_out = build_conv1d(ctx0, inp,
                blk.res2net_conv_w[s - 1], blk.res2net_conv_b[s - 1],
                /*stride=*/1, pad, dilation);
            conv_out = ggml_relu(ctx0, conv_out);
            chunks.push_back(conv_out);
        }
    }

    // Concatenate along channel dim
    cur = chunks[0];
    for (size_t i = 1; i < chunks.size(); i++) {
        cur = ggml_concat(ctx0, cur, chunks[i], 1);
    }

    // tdnn2: Conv1d(out, out, kernel=1) + ReLU
    cur = build_conv1d(ctx0, cur, blk.tdnn2_conv_w, blk.tdnn2_conv_b);
    cur = ggml_relu(ctx0, cur);

    // Squeeze-Excitation
    // Global average pooling over time: (T, C) → (1, C)
    ggml_tensor *se = ggml_pool_1d(ctx0, cur, GGML_OP_POOL_AVG, T, T, 0);
    // Conv1d(C, 128, 1) + ReLU
    se = build_conv1d(ctx0, se, blk.se_conv1_w, blk.se_conv1_b);
    se = ggml_relu(ctx0, se);
    // Conv1d(128, C, 1) + Sigmoid
    se = build_conv1d(ctx0, se, blk.se_conv2_w, blk.se_conv2_b);
    se = ggml_sigmoid(ctx0, se);
    // Broadcast multiply: (T, C) * (1, C)
    se = ggml_repeat(ctx0, se, cur);
    cur = ggml_mul(ctx0, cur, se);

    // Residual connection
    cur = ggml_add(ctx0, cur, residual);
    return cur;
}

// Attentive Statistics Pooling
ggml_tensor *SpeakerEncoderModel::build_asp(
    ggml_context *ctx0, ggml_tensor *x) {
    // x: (T, C=1536)
    int T = (int)x->ne[0];
    int C = (int)x->ne[1];

    // Compute mean over time: (T, C) → (1, C)
    ggml_tensor *mean = ggml_pool_1d(ctx0, x, GGML_OP_POOL_AVG, T, T, 0);

    // Compute variance: E[x^2] - E[x]^2
    ggml_tensor *x_sq = ggml_mul(ctx0, x, x);
    ggml_tensor *mean_sq = ggml_pool_1d(ctx0, x_sq, GGML_OP_POOL_AVG, T, T, 0);
    ggml_tensor *var = ggml_sub(ctx0, mean_sq, ggml_mul(ctx0, mean, mean));
    // Clamp and sqrt for std
    // std = sqrt(max(var, 1e-12))
    // Use ggml_sqrt after adding small epsilon
    ggml_tensor *eps = ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, 1);
    ggml_set_name(eps, "eps");
    ggml_set_input(eps);
    ggml_tensor *std_val = ggml_sqrt(ctx0, ggml_add(ctx0, var, ggml_repeat(ctx0, eps, var)));

    // Expand mean and std to (T, C)
    ggml_tensor *mean_exp = ggml_repeat(ctx0, mean, x);
    ggml_tensor *std_exp = ggml_repeat(ctx0, std_val, x);

    // Concatenate [x, mean_exp, std_exp] along channel dim → (T, 3*C=4608)
    ggml_tensor *cat = ggml_concat(ctx0, x, mean_exp, 1);
    cat = ggml_concat(ctx0, cat, std_exp, 1);

    // TDNN: Conv1d(4608, 128, 1) + Tanh
    ggml_tensor *attn = build_conv1d(ctx0, cat, asp_tdnn_conv_w, asp_tdnn_conv_b);
    attn = ggml_tanh(ctx0, attn);

    // Conv1d(128, 1536, 1)
    attn = build_conv1d(ctx0, attn, asp_conv_w, asp_conv_b);

    // Softmax over time dimension
    // attn: (T, C=1536) in ggml → ne[0]=T, ne[1]=C
    // ggml_soft_max normalizes over ne[0] (T) for each channel → correct!
    attn = ggml_soft_max(ctx0, attn);

    // Weighted mean: sum(attn * x, dim=0) → (1, C)
    ggml_tensor *weighted = ggml_mul(ctx0, attn, x);
    // Sum over time by using pool with kernel=T
    ggml_tensor *w_mean = ggml_pool_1d(ctx0, weighted, GGML_OP_POOL_AVG, T, T, 0);
    // Multiply by T to get sum (pool_avg divides by T)
    w_mean = ggml_scale(ctx0, w_mean, (float)T);

    // Weighted std: sqrt(sum(attn * (x - w_mean)^2, dim=0))
    ggml_tensor *w_mean_exp = ggml_repeat(ctx0, w_mean, x);
    ggml_tensor *diff = ggml_sub(ctx0, x, w_mean_exp);
    ggml_tensor *diff_sq = ggml_mul(ctx0, diff, diff);
    ggml_tensor *w_var = ggml_mul(ctx0, attn, diff_sq);
    ggml_tensor *w_var_sum = ggml_pool_1d(ctx0, w_var, GGML_OP_POOL_AVG, T, T, 0);
    w_var_sum = ggml_scale(ctx0, w_var_sum, (float)T);
    ggml_tensor *w_std = ggml_sqrt(ctx0,
        ggml_add(ctx0, w_var_sum, ggml_repeat(ctx0, eps, w_var_sum)));

    // Concatenate mean and std: (1, 2*C=3072)
    ggml_tensor *pooled = ggml_concat(ctx0, w_mean, w_std, 1);
    return pooled;
}

// ============================================================================
// SpeakerEncoderModel: build_graph
// ============================================================================

std::vector<ggml_tensor *>
SpeakerEncoderModel::build_graph(ggml_context *ctx0) {
    auto shape = input_shapes_["mel"];
    int T = shape[0];
    int n_mels = shape[1];

    // Input: mel spectrogram (T, n_mels=128)
    ggml_tensor *mel = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, T, n_mels);
    ggml_set_name(mel, "mel");
    ggml_set_input(mel);

    // blocks[0]: TDNN Conv1d(128, 512, kernel=5) + ReLU
    ggml_tensor *cur = build_tdnn(ctx0, mel, blocks_0_conv_w, blocks_0_conv_b);
    ggml_set_name(cur, "block0");

    // blocks[1-3]: SE-Res2Net
    // Collect outputs for MFA concatenation
    std::vector<ggml_tensor *> block_outs;
    int dilations[] = {2, 3, 4};
    for (int i = 0; i < 3; i++) {
        cur = build_se_res2net(ctx0, cur, se_blocks_[i], dilations[i]);
        block_outs.push_back(cur);
    }

    // MFA: concatenate block outputs along channel dim → (T, 1536)
    ggml_tensor *mfa_in = block_outs[0];
    for (size_t i = 1; i < block_outs.size(); i++) {
        mfa_in = ggml_concat(ctx0, mfa_in, block_outs[i], 1);
    }
    // Conv1d(1536, 1536, kernel=1) + ReLU
    cur = build_conv1d(ctx0, mfa_in, mfa_conv_w, mfa_conv_b);
    cur = ggml_relu(ctx0, cur);

    ggml_set_name(cur, "after_mfa");

    // ASP: Attentive Statistics Pooling → (1, 3072)
    cur = build_asp(ctx0, cur);

    ggml_set_name(cur, "after_asp");

    // FC: Conv1d(3072, 2048, kernel=1)
    cur = build_conv1d(ctx0, cur, fc_w, fc_b);

    // Squeeze time dim: (1, 2048) → (2048)
    cur = ggml_reshape_1d(ctx0, cur, config.enc_dim);

    ggml_set_name(cur, "embedding");
    ggml_set_output(cur);
    return {cur};
}

// ============================================================================
// SpeakerEncoder: high-level interface
// ============================================================================

bool SpeakerEncoder::load(const std::string &model_path,
                           const ContextParams &params) {
    session_ = std::make_unique<InferenceSession<SpeakerEncoderModel>>(
        model_path, params);
    config_ = session_->get_model().config;
    printf("Speaker Encoder loaded (enc_dim=%d, sr=%d)\n",
           config_.enc_dim, config_.sample_rate);
    return true;
}

bool SpeakerEncoder::extract(const std::vector<float> &audio, int sample_rate,
                              std::vector<float> &embedding) {
    if (sample_rate != config_.sample_rate) {
        fprintf(stderr, "Sample rate mismatch: expected %d, got %d\n",
                config_.sample_rate, sample_rate);
        return false;
    }

    // Compute mel spectrogram
    std::vector<float> mel_spec;
    int n_frames = 0;
    if (!compute_mel(audio, mel_spec, n_frames)) {
        return false;
    }

    printf("Mel spectrogram: %d frames x %d mels\n", n_frames, config_.n_mels);

    // Set input shape (internally rebuilds graph)
    session_->set_input_shape({{"mel", {n_frames, config_.n_mels}}});

    // Set input data
    session_->set_input("mel", mel_spec);

    // Run inference
    if (!session_->run(embedding)) {
        fprintf(stderr, "Speaker encoder inference failed\n");
        return false;
    }

    printf("Speaker embedding: %zu dims\n", embedding.size());
    return true;
}