#include "audio_encoder.h"
#include "../qwen_tts/build_graph.h"
#include "../qwen_tts/ctx_manager.h"
#include "../qwen_tts/model_loader.h"
#include "../qwen_tts/utils.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <vector>

// ============================================================================
// Sinusoidal positional embedding (matching SinusoidsPositionEmbedding)
// ============================================================================

void AudioEncoder::compute_sinusoidal_pos_emb() {
    // Qwen3-ASR uses SinusoidsPositionEmbedding (NOT Whisper-style interleaved):
    //   log_timescale_increment = log(10000) / (channels/2 - 1)
    //   inv_timescales[i] = exp(-log_timescale_increment * i)  for i in 0..channels/2-1
    //   scaled_time[pos, i] = pos * inv_timescales[i]
    //   emb[pos, :] = [sin(scaled_time[pos, :]), cos(scaled_time[pos, :])]
    // Layout: first half = sin, second half = cos (concatenated, NOT interleaved)
    int half_dim = d_model_ / 2;
    pos_emb_.resize(max_source_positions_ * d_model_);

    // Precompute inv_timescales: exp(-log(10000) * i / (half_dim - 1))
    float log_timescale_increment = std::log(10000.0f) / (float)(half_dim - 1);
    std::vector<float> inv_timescales(half_dim);
    for (int i = 0; i < half_dim; i++) {
        inv_timescales[i] = std::exp(-log_timescale_increment * (float)i);
    }

    for (int pos = 0; pos < max_source_positions_; pos++) {
        for (int i = 0; i < half_dim; i++) {
            float scaled_time = (float)pos * inv_timescales[i];
            pos_emb_[pos * d_model_ + i]            = std::sin(scaled_time);  // first half: sin
            pos_emb_[pos * d_model_ + half_dim + i]  = std::cos(scaled_time);  // second half: cos
        }
    }
}

// ============================================================================
// Feature extraction output length formula (from Python)
// ============================================================================

// Python-style floor division (rounds toward negative infinity)
static inline int py_floordiv(int a, int b) {
    return (a / b) - (a % b != 0 && (a ^ b) < 0);
}

int AudioEncoder::get_feat_extract_output_lengths(int input_lengths) {
    // Python:
    // input_lengths_leave = input_lengths % 100
    // feat_lengths = (input_lengths_leave - 1) // 2 + 1
    // output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    int n_window_double = n_window_ * 2; // 100
    int input_lengths_leave = input_lengths % n_window_double;
    int feat_lengths = py_floordiv(input_lengths_leave - 1, 2) + 1;
    int output_lengths = py_floordiv(py_floordiv(feat_lengths - 1, 2) + 1 - 1, 2) + 1
                         + py_floordiv(input_lengths, n_window_double) * 13;
    return output_lengths;
}

// ============================================================================
// Load GGUF model
// ============================================================================

bool AudioEncoder::load(const std::string &gguf_path, const std::string &device, int n_threads) {
    device_name_ = device;
    n_threads_ = n_threads;

    // 1. Load GGUF file
    ModelLoader loader(gguf_path);

    // 2. Read hyperparameters
    loader.get_u32("d_model", d_model_);
    loader.get_u32("encoder_layers", encoder_layers_);
    loader.get_u32("encoder_attention_heads", encoder_attention_heads_);
    loader.get_u32("encoder_ffn_dim", encoder_ffn_dim_);
    loader.get_u32("num_mel_bins", num_mel_bins_);
    loader.get_u32("downsample_hidden_size", downsample_hidden_size_);
    loader.get_u32("output_dim", output_dim_);
    loader.get_u32("max_source_positions", max_source_positions_);
    loader.get_u32("n_window", n_window_);
    loader.get_u32("n_window_infer", n_window_infer_);
    loader.get_u32("mel_reduced", mel_reduced_);
    loader.get_u32("conv_out_dim", conv_out_dim_);

    printf("AudioEncoder: d_model=%d, layers=%d, heads=%d, ffn=%d, mel_bins=%d\n",
           d_model_, encoder_layers_, encoder_attention_heads_, encoder_ffn_dim_, num_mel_bins_);
    printf("AudioEncoder: output_dim=%d, n_window=%d, conv_out=%d\n",
           output_dim_, n_window_, conv_out_dim_);

    // 3. Create ContextManager
    // Use a large max_nodes because the transformer has many operations
    int max_nodes = 32768;
    ctx_mgr_ = std::make_unique<ContextManager>(device, n_threads, max_nodes);

    // 4. Transfer ctx_meta ownership and get tensor pointers
    ctx_mgr_->ctx_data_ = std::move(loader.ctx_meta_);
    ggml_context *ctx = ctx_mgr_->ctx_data_.get();

    std::vector<ggml_tensor *> tensors_to_load;

    // Conv2d weights
    conv2d1_w_ = get_tensor(ctx, "conv2d1.weight", tensors_to_load);
    conv2d1_b_ = get_tensor(ctx, "conv2d1.bias", tensors_to_load);
    conv2d2_w_ = get_tensor(ctx, "conv2d2.weight", tensors_to_load);
    conv2d2_b_ = get_tensor(ctx, "conv2d2.bias", tensors_to_load);
    conv2d3_w_ = get_tensor(ctx, "conv2d3.weight", tensors_to_load);
    conv2d3_b_ = get_tensor(ctx, "conv2d3.bias", tensors_to_load);
    conv_out_w_ = get_tensor(ctx, "conv_out.weight", tensors_to_load);

    // Transformer layers
    layers_.resize(encoder_layers_);
    for (int i = 0; i < encoder_layers_; i++) {
        std::string pfx = "layers." + std::to_string(i) + ".";
        auto &L = layers_[i];
        L.self_attn_layer_norm_w = get_tensor(ctx, pfx + "self_attn_layer_norm.weight", tensors_to_load);
        L.self_attn_layer_norm_b = get_tensor(ctx, pfx + "self_attn_layer_norm.bias", tensors_to_load);
        L.q_proj_w = get_tensor(ctx, pfx + "self_attn.q_proj.weight", tensors_to_load);
        L.q_proj_b = get_tensor(ctx, pfx + "self_attn.q_proj.bias", tensors_to_load);
        L.k_proj_w = get_tensor(ctx, pfx + "self_attn.k_proj.weight", tensors_to_load);
        L.k_proj_b = get_tensor(ctx, pfx + "self_attn.k_proj.bias", tensors_to_load);
        L.v_proj_w = get_tensor(ctx, pfx + "self_attn.v_proj.weight", tensors_to_load);
        L.v_proj_b = get_tensor(ctx, pfx + "self_attn.v_proj.bias", tensors_to_load);
        L.out_proj_w = get_tensor(ctx, pfx + "self_attn.out_proj.weight", tensors_to_load);
        L.out_proj_b = get_tensor(ctx, pfx + "self_attn.out_proj.bias", tensors_to_load);
        L.final_layer_norm_w = get_tensor(ctx, pfx + "final_layer_norm.weight", tensors_to_load);
        L.final_layer_norm_b = get_tensor(ctx, pfx + "final_layer_norm.bias", tensors_to_load);
        L.fc1_w = get_tensor(ctx, pfx + "fc1.weight", tensors_to_load);
        L.fc1_b = get_tensor(ctx, pfx + "fc1.bias", tensors_to_load);
        L.fc2_w = get_tensor(ctx, pfx + "fc2.weight", tensors_to_load);
        L.fc2_b = get_tensor(ctx, pfx + "fc2.bias", tensors_to_load);
    }

    // Output MLP
    ln_post_w_ = get_tensor(ctx, "ln_post.weight", tensors_to_load);
    ln_post_b_ = get_tensor(ctx, "ln_post.bias", tensors_to_load);
    proj1_w_ = get_tensor(ctx, "proj1.weight", tensors_to_load);
    proj1_b_ = get_tensor(ctx, "proj1.bias", tensors_to_load);
    proj2_w_ = get_tensor(ctx, "proj2.weight", tensors_to_load);
    proj2_b_ = get_tensor(ctx, "proj2.bias", tensors_to_load);

    // 5. Load tensor data from file
    {
        std::map<std::string, size_t> tensor_offset;
        gguf_context *ctx_gguf = loader.ctx_gguf_.get();

        for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
            const char *name = gguf_get_tensor_name(ctx_gguf, i);
            tensor_offset[name] = gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
        }

        ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(ctx_mgr_->backend_.get());
        ctx_mgr_->buffer_.reset(
            ggml_backend_alloc_ctx_tensors_from_buft(ctx_mgr_->ctx_data_.get(), buft));
        ggml_backend_buffer_set_usage(ctx_mgr_->buffer_.get(),
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        std::vector<uint8_t> read_buf;
        auto fin = std::ifstream(loader.fname_, std::ios::binary);
        if (!fin) {
            fprintf(stderr, "AudioEncoder: failed to open %s\n", loader.fname_.c_str());
            return false;
        }

        for (auto &cur : tensors_to_load) {
            auto it = tensor_offset.find(cur->name);
            if (it == tensor_offset.end()) {
                fprintf(stderr, "AudioEncoder: tensor %s not found in GGUF\n", cur->name);
                return false;
            }
            const size_t offset = it->second;
            fin.seekg(offset, std::ios::beg);
            size_t num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buft_is_host(buft)) {
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        fin.close();
        printf("AudioEncoder: loaded %zu tensors from %s\n",
               tensors_to_load.size(), gguf_path.c_str());
    }

    // 6. Compute sinusoidal positional embeddings
    compute_sinusoidal_pos_emb();

    printf("AudioEncoder: ready\n");
    return true;
}

// ============================================================================
// Run Conv2d stack on a batch of padded chunks
// Input padded_mel: (batch_size, 1, mel_h, mel_w) in row-major (N, C, H, W)
// Output: (batch_size, frames_per_chunk, d_model) in row-major
// ============================================================================

bool AudioEncoder::run_conv2d(const std::vector<float> &padded_mel,
                               int batch_size, int mel_h, int mel_w,
                               std::vector<float> &output) {
    // In GGML, tensor layout is (ne[0], ne[1], ne[2], ne[3])
    // For Conv2d input: we store as (W, H, C=1, N)
    // padded_mel is (N, 1, H, W) row-major => GGML tensor (W, H, 1, N)

    // After 3 conv layers with stride=2:
    // W_out = (mel_w + 2*1 - 3) / 2 + 1 for each layer
    // After 3 layers, time dim is roughly mel_w / 8
    // Freq dim after 3 layers: mel_h / 8 = 128/8 = 16

    // Estimate output sizes after each conv (stride=2, pad=1, kernel=3)
    // conv formula: out = (in + 2*pad - kernel) / stride + 1
    auto conv_out_size = [](int in_size) -> int {
        return (in_size + 2 * 1 - 3) / 2 + 1;
    };

    int h1 = conv_out_size(mel_h), w1 = conv_out_size(mel_w);
    int h2 = conv_out_size(h1),    w2 = conv_out_size(w1);
    int h3 = conv_out_size(h2),    w3 = conv_out_size(w2);

    // Time output dimension = w3
    int frames_out = w3;
    int flatten_dim = downsample_hidden_size_ * h3; // 480 * 16 = 7680

    // Graph size estimate
    // Conv2d ops + bias adds + GELUs + permute + matmul
    int max_nodes = 256;
    size_t buf_size = max_nodes * ggml_tensor_overhead() + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params params = {buf_size, buf.data(), true};
    ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "run_conv2d: failed to init ggml context\n");
        return false;
    }

    ggml_cgraph *gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // Input tensor: (W, H, 1, N)  [GGML layout]
    ggml_tensor *inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                           mel_w, mel_h, 1, batch_size);
    ggml_set_name(inp, "conv_input");
    ggml_set_input(inp);

    // Conv2d1: (480, 1, 3, 3) -> output (OW, OH, 480, N)
    // ggml_conv_2d(a=kernel, b=input, s0, s1, p0, p1, d0, d1)
    // kernel shape in GGML: (KW, KH, IC, OC)
    ggml_tensor *cur = ggml_conv_2d(ctx, conv2d1_w_, inp, 2, 2, 1, 1, 1, 1);
    // Add bias: reshape bias from (OC,) to (1, 1, OC, 1) for broadcasting
    {
        ggml_tensor *bias_4d = ggml_reshape_4d(ctx, conv2d1_b_, 1, 1, downsample_hidden_size_, 1);
        cur = ggml_add(ctx, cur, bias_4d);
    }
    cur = ggml_gelu_erf(ctx, cur);

    // Conv2d2
    cur = ggml_conv_2d(ctx, conv2d2_w_, cur, 2, 2, 1, 1, 1, 1);
    {
        ggml_tensor *bias_4d = ggml_reshape_4d(ctx, conv2d2_b_, 1, 1, downsample_hidden_size_, 1);
        cur = ggml_add(ctx, cur, bias_4d);
    }
    cur = ggml_gelu_erf(ctx, cur);

    // Conv2d3
    cur = ggml_conv_2d(ctx, conv2d3_w_, cur, 2, 2, 1, 1, 1, 1);
    {
        ggml_tensor *bias_4d = ggml_reshape_4d(ctx, conv2d3_b_, 1, 1, downsample_hidden_size_, 1);
        cur = ggml_add(ctx, cur, bias_4d);
    }
    cur = ggml_gelu_erf(ctx, cur);

    // cur shape: (w3, h3, 480, batch_size) = (OW=time, OH=freq, OC=channels, N=batch)
    // Python does: permute(0, 3, 1, 2).view(b, t, c*f)
    // i.e., (N, OC, OH, OW) -> (N, OW, OC, OH) -> view(N, OW, OC*OH)
    // In GGML, we need (OH, OC, OW, N) to flatten to (OH*OC=7680, OW, N)
    // ggml_permute(a, ax0, ax1, ax2, ax3): ne[ax_i] = a->ne[i]
    //   dim0 (OW) -> dim2: ax0 = 2
    //   dim1 (OH) -> dim0: ax1 = 0
    //   dim2 (OC) -> dim1: ax2 = 1
    //   dim3 (N)  -> dim3: ax3 = 3
    cur = ggml_permute(ctx, cur, 2, 0, 1, 3);
    cur = ggml_cont(ctx, cur);

    // Step 2: reshape to flatten freq*channels: (h3*480, w3, N) = (7680, time, batch)
    cur = ggml_reshape_3d(ctx, cur, flatten_dim, frames_out, batch_size);

    // Step 3: linear projection conv_out: (1024, 7680) @ (7680, time, batch) -> (1024, time, batch)
    cur = ggml_mul_mat(ctx, conv_out_w_, cur);

    ggml_set_name(cur, "conv_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Execute graph
    ggml_backend_sched_reset(ctx_mgr_->sched_.get());
    if (!ggml_backend_sched_alloc_graph(ctx_mgr_->sched_.get(), gf)) {
        fprintf(stderr, "run_conv2d: failed to alloc graph\n");
        ggml_free(ctx);
        return false;
    }

    // Set input data - convert from row-major (N, 1, H, W) to GGML (W, H, 1, N)
    // Actually GGML stores data with ne[0] contiguous, so our (N,1,H,W) row-major
    // maps naturally to GGML (W, H, 1, N) since W is the innermost dimension
    ggml_tensor *inp_tensor = ggml_graph_get_tensor(gf, "conv_input");
    ggml_backend_tensor_set(inp_tensor, padded_mel.data(), 0,
                            batch_size * 1 * mel_h * mel_w * sizeof(float));

    auto status = ggml_backend_sched_graph_compute(ctx_mgr_->sched_.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "run_conv2d: graph compute failed with error %d\n", status);
        ggml_free(ctx);
        return false;
    }

    // Extract output: (d_model, frames_out, batch_size) in GGML
    ggml_tensor *out_tensor = ggml_graph_get_tensor(gf, "conv_output");
    output.resize(ggml_nelements(out_tensor));
    ggml_backend_tensor_get(out_tensor, output.data(), 0, ggml_nbytes(out_tensor));

    ggml_free(ctx);
    return true;
}

// ============================================================================
// Run Transformer encoder + output MLP
// Input hidden_states: (total_frames, d_model) row-major
// cu_seqlens: cumulative sequence lengths for windowed attention
// Output: (total_frames, output_dim) row-major
// ============================================================================

bool AudioEncoder::run_transformer(const std::vector<float> &hidden_states,
                                    int total_frames,
                                    const std::vector<int> &cu_seqlens,
                                    std::vector<float> &output) {
    int n_heads = encoder_attention_heads_;
    int head_dim = d_model_ / n_heads;
    float kq_scale = 1.0f / std::sqrt((float)head_dim);

    // Build attention mask from cu_seqlens
    // Shape: (total_frames, total_frames)
    // 0.0 for positions that CAN attend, -inf for positions that CANNOT attend
    //
    // NOTE: The Python reference (Qwen3-ASR with SDPA on CPU) uses attention_mask=None
    // with is_causal=False, which means FULL bidirectional attention (no masking).
    // The cu_seqlens are only used for Flash Attention 2 varlen path.
    // To match the Python reference, we use all-zeros mask (no masking).
    // For production use with windowed attention, this should be changed to use
    // the cu_seqlens-based block-diagonal mask.
    std::vector<float> attn_mask(total_frames * total_frames, 0.0f);

    // Estimate max nodes for 24-layer transformer + output MLP
    // Each layer: ~20 ops, plus input/mask/output ops
    int max_nodes = encoder_layers_ * 40 + 128;
    size_t buf_size = max_nodes * ggml_tensor_overhead() + ggml_graph_overhead();
    std::vector<uint8_t> buf(buf_size);
    struct ggml_init_params params = {buf_size, buf.data(), true};
    ggml_context *ctx = ggml_init(params);
    if (!ctx) {
        fprintf(stderr, "run_transformer: failed to init ggml context\n");
        return false;
    }

    ggml_cgraph *gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // Input: (d_model, total_frames) in GGML
    ggml_tensor *inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                           d_model_, total_frames);
    ggml_set_name(inp, "transformer_input");
    ggml_set_input(inp);

    // Attention mask: (total_frames, total_frames) in GGML
    ggml_tensor *kq_mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32,
                                               total_frames, total_frames);
    ggml_set_name(kq_mask, "attn_mask");
    ggml_set_input(kq_mask);

    ggml_tensor *cur = inp;

    // 24-layer Transformer encoder (Pre-LN)
    for (int il = 0; il < encoder_layers_; il++) {
        auto &L = layers_[il];

        // === Self-attention block ===
        ggml_tensor *residual = cur;

        // Pre-LayerNorm
        cur = build_norm(ctx, cur, L.self_attn_layer_norm_w, L.self_attn_layer_norm_b,
                         NORM_TYPE_NORMAL, 1e-5f, 1, il);

        // Q, K, V projections
        // cur shape: (d_model, total_frames)
        ggml_tensor *q_cur = build_linear(ctx, cur, L.q_proj_w, L.q_proj_b, il);
        ggml_tensor *k_cur = build_linear(ctx, cur, L.k_proj_w, L.k_proj_b, il);
        ggml_tensor *v_cur = build_linear(ctx, cur, L.v_proj_w, L.v_proj_b, il);

        // Reshape for multi-head attention
        // (d_model, total_frames) -> (head_dim, n_heads, total_frames)
        q_cur = ggml_reshape_3d(ctx, q_cur, head_dim, n_heads, total_frames);
        k_cur = ggml_reshape_3d(ctx, k_cur, head_dim, n_heads, total_frames);
        v_cur = ggml_reshape_3d(ctx, v_cur, head_dim, n_heads, total_frames);

        // build_attn does: permute, scale, matmul, mask, softmax, matmul, out_proj
        ggml_tensor *attn_out = build_attn(ctx, L.out_proj_w, L.out_proj_b,
                                           q_cur, k_cur, v_cur, kq_mask, kq_scale, il);

        // Residual connection
        cur = ggml_add(ctx, residual, attn_out);

        // === FFN block ===
        residual = cur;

        // Pre-LayerNorm
        cur = build_norm(ctx, cur, L.final_layer_norm_w, L.final_layer_norm_b,
                         NORM_TYPE_NORMAL, 1e-5f, 1, il);

        // FFN: fc1 -> GELU(erf) -> fc2
        cur = build_linear(ctx, cur, L.fc1_w, L.fc1_b, il);
        cur = ggml_gelu_erf(ctx, cur);
        cur = build_linear(ctx, cur, L.fc2_w, L.fc2_b, il);

        // Residual connection
        cur = ggml_add(ctx, residual, cur);
    }

    // Output MLP: LayerNorm -> proj1 -> GELU(erf) -> proj2
    cur = build_norm(ctx, cur, ln_post_w_, ln_post_b_, NORM_TYPE_NORMAL, 1e-5f);
    cur = build_linear(ctx, cur, proj1_w_, proj1_b_);
    cur = ggml_gelu_erf(ctx, cur);
    cur = build_linear(ctx, cur, proj2_w_, proj2_b_);

    // cur shape: (output_dim, total_frames)
    ggml_set_name(cur, "transformer_output");
    ggml_set_output(cur);
    ggml_build_forward_expand(gf, cur);

    // Execute graph
    ggml_backend_sched_reset(ctx_mgr_->sched_.get());
    if (!ggml_backend_sched_alloc_graph(ctx_mgr_->sched_.get(), gf)) {
        fprintf(stderr, "run_transformer: failed to alloc graph\n");
        ggml_free(ctx);
        return false;
    }

    // Set inputs
    // hidden_states is (total_frames, d_model) row-major
    // GGML tensor is (d_model, total_frames) => data layout is the same
    // because in row-major, the last dim (d_model) is contiguous, which matches GGML ne[0]
    ggml_tensor *inp_t = ggml_graph_get_tensor(gf, "transformer_input");
    ggml_backend_tensor_set(inp_t, hidden_states.data(), 0,
                            total_frames * d_model_ * sizeof(float));

    ggml_tensor *mask_t = ggml_graph_get_tensor(gf, "attn_mask");
    ggml_backend_tensor_set(mask_t, attn_mask.data(), 0,
                            total_frames * total_frames * sizeof(float));

    auto status = ggml_backend_sched_graph_compute(ctx_mgr_->sched_.get(), gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "run_transformer: graph compute failed with error %d\n", status);
        ggml_free(ctx);
        return false;
    }

    // Extract output: (output_dim, total_frames) in GGML
    ggml_tensor *out_tensor = ggml_graph_get_tensor(gf, "transformer_output");
    output.resize(ggml_nelements(out_tensor));
    ggml_backend_tensor_get(out_tensor, output.data(), 0, ggml_nbytes(out_tensor));

    ggml_free(ctx);
    return true;
}

// ============================================================================
// Full encode pipeline
// mel: (num_mel_bins, T) row-major
// output: (num_frames, output_dim) row-major
// ============================================================================

bool AudioEncoder::encode(const std::vector<float> &mel, int mel_T,
                           std::vector<float> &output, int &num_frames) {
    // Validate input
    if ((int)mel.size() != num_mel_bins_ * mel_T) {
        fprintf(stderr, "AudioEncoder::encode: mel size mismatch: %zu vs %d*%d=%d\n",
                mel.size(), num_mel_bins_, mel_T, num_mel_bins_ * mel_T);
        return false;
    }

    printf("AudioEncoder::encode: mel_T=%d\n", mel_T);

    // ========================================================================
    // Step 1: Chunk the mel spectrogram
    // ========================================================================
    int n_window_double = n_window_ * 2; // 100
    int chunk_num = (mel_T + n_window_double - 1) / n_window_double; // ceil division

    // Compute chunk lengths
    std::vector<int> chunk_lengths(chunk_num);
    for (int i = 0; i < chunk_num; i++) {
        chunk_lengths[i] = n_window_double;
    }
    // Last chunk gets the remainder
    int remainder = mel_T % n_window_double;
    if (remainder != 0) {
        chunk_lengths[chunk_num - 1] = remainder;
    }

    printf("AudioEncoder::encode: chunk_num=%d, chunk_lengths=[", chunk_num);
    for (int i = 0; i < chunk_num; i++) {
        printf("%d%s", chunk_lengths[i], i < chunk_num - 1 ? ", " : "");
    }
    printf("]\n");

    // Find the max chunk length for padding
    int max_chunk_len = *std::max_element(chunk_lengths.begin(), chunk_lengths.end());

    // ========================================================================
    // Step 2: Prepare padded mel batch
    // Shape: (chunk_num, 1, num_mel_bins, max_chunk_len) in row-major
    // ========================================================================
    std::vector<float> padded_mel(chunk_num * 1 * num_mel_bins_ * max_chunk_len, 0.0f);

    // mel is (num_mel_bins, mel_T) row-major
    // We need to split along T (columns) into chunks, then pad
    int col_offset = 0;
    for (int c = 0; c < chunk_num; c++) {
        int clen = chunk_lengths[c];
        for (int m = 0; m < num_mel_bins_; m++) {
            for (int t = 0; t < clen; t++) {
                // Source: mel[m * mel_T + col_offset + t]
                // Dest: padded_mel[c * (1 * num_mel_bins_ * max_chunk_len) + 0 * (num_mel_bins_ * max_chunk_len) + m * max_chunk_len + t]
                int dst_idx = c * num_mel_bins_ * max_chunk_len + m * max_chunk_len + t;
                int src_idx = m * mel_T + col_offset + t;
                padded_mel[dst_idx] = mel[src_idx];
            }
        }
        col_offset += clen;
    }

    // ========================================================================
    // Step 3: Run Conv2d on the batch
    // ========================================================================
    std::vector<float> conv_output;
    if (!run_conv2d(padded_mel, chunk_num, num_mel_bins_, max_chunk_len, conv_output)) {
        fprintf(stderr, "AudioEncoder::encode: Conv2d failed\n");
        return false;
    }

    // conv_output is in GGML layout: (d_model, frames_per_chunk, chunk_num)
    // where frames_per_chunk = conv_out_size applied 3 times to max_chunk_len
    auto conv_out_size = [](int in_size) -> int {
        return (in_size + 2 * 1 - 3) / 2 + 1;
    };
    int frames_per_chunk = conv_out_size(conv_out_size(conv_out_size(max_chunk_len)));

    printf("AudioEncoder::encode: frames_per_chunk=%d\n", frames_per_chunk);

    // ========================================================================
    // Step 4: Compute per-chunk valid frame counts, add positional embeddings,
    //         apply mask, and concatenate valid frames
    // ========================================================================

    // Compute valid frame counts for each chunk
    std::vector<int> feature_lens_after_cnn(chunk_num);
    for (int c = 0; c < chunk_num; c++) {
        feature_lens_after_cnn[c] = get_feat_extract_output_lengths(chunk_lengths[c]);
    }

    // Total valid frames
    int total_valid_frames = 0;
    for (int c = 0; c < chunk_num; c++) {
        total_valid_frames += feature_lens_after_cnn[c];
    }

    printf("AudioEncoder::encode: total_valid_frames=%d, feature_lens_after_cnn=[", total_valid_frames);
    for (int c = 0; c < chunk_num; c++) {
        printf("%d%s", feature_lens_after_cnn[c], c < chunk_num - 1 ? ", " : "");
    }
    printf("]\n");

    // Extract valid frames, add positional embeddings, concatenate
    // conv_output layout (GGML): (d_model, frames_per_chunk, chunk_num)
    // Data access: element at (d, f, c) = conv_output[c * (frames_per_chunk * d_model) + f * d_model + d]
    std::vector<float> concat_frames(total_valid_frames * d_model_, 0.0f);

    // cu_seqlens for attention mask
    std::vector<int> cu_seqlens;
    cu_seqlens.push_back(0);

    int frame_offset = 0;

    for (int c = 0; c < chunk_num; c++) {
        int valid = feature_lens_after_cnn[c];

        for (int f = 0; f < valid; f++) {
            // Positional embedding is per-chunk (same positions 0..frames_per_chunk-1 for all chunks)
            // This matches Python: positional_embedding[:padded_embed.shape[1], :] broadcast over batch
            int pos = f;
            if (pos >= max_source_positions_) {
                pos = max_source_positions_ - 1; // clamp
            }
            for (int d = 0; d < d_model_; d++) {
                // conv_output element at (d, f, c):
                int src_idx = c * (frames_per_chunk * d_model_) + f * d_model_ + d;
                // Destination: concat_frames[(frame_offset + f) * d_model_ + d]
                int dst_idx = (frame_offset + f) * d_model_ + d;
                concat_frames[dst_idx] = conv_output[src_idx] + pos_emb_[pos * d_model_ + d];
            }
        }

        frame_offset += valid;
        cu_seqlens.push_back(frame_offset);
    }

    printf("AudioEncoder::encode: cu_seqlens=[");
    for (size_t i = 0; i < cu_seqlens.size(); i++) {
        printf("%d%s", cu_seqlens[i], i < cu_seqlens.size() - 1 ? ", " : "");
    }
    printf("]\n");

    // ========================================================================
    // Step 5: Run Transformer encoder + output MLP
    // ========================================================================
    std::vector<float> transformer_output;
    if (!run_transformer(concat_frames, total_valid_frames, cu_seqlens, transformer_output)) {
        fprintf(stderr, "AudioEncoder::encode: Transformer failed\n");
        return false;
    }

    // transformer_output is in GGML layout: (output_dim, total_valid_frames)
    // We need to return (total_valid_frames, output_dim) in row-major
    // Since GGML stores with ne[0] contiguous, and ne[0]=output_dim,
    // the data is already in (total_valid_frames, output_dim) row-major order!
    num_frames = total_valid_frames;
    output = std::move(transformer_output);

    printf("AudioEncoder::encode: output shape = (%d, %d)\n", num_frames, output_dim_);
    return true;
}
