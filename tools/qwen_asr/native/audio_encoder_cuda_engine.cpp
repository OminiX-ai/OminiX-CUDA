// ============================================================================
// AudioEncoderCudaEngine — Phase 4.2 forward.
//
// Mirrors AudioEncoder from OminiX-Ascend/tools/qwen_asr/audio_encoder.cpp.
// Numeric strategy: F32 throughout (encoder activations are small at the
// 122-frame typical seq length; F32 storage matches the CPU/Ascend reference
// exactly).
//
// Layout convention:
//   Conv tensors live in NCHW [N, C, H, W] (innermost = W). im2col reads
//   NCHW and writes [total_rows = N*H_out*W_out, patch_size = C_in*KH*KW]
//   row-major. cuBLAS GEMM with W^T (W = [C_out, patch_size] row-major)
//   produces [total_rows, C_out] row-major output. We then transpose
//   [total_rows, C_out] back into NCHW before feeding to the next conv.
//
//   After conv3, we extract per-frame slabs in ggml's (OH outer, OC inner)
//   order to match the conv_out_w_ layout exported from the Ascend ref.
// ============================================================================

#include "audio_encoder_cuda_engine.h"

#include "cuda_kernels/audio_encoder_kernels.h"
// All needed kernels (im2col, LayerNorm-affine, GELU-erf, attn_joint_naive_f32,
// add_bias_f32_f32) live in audio_encoder_kernels.h — qwen_image_edit's
// dit_kernels are not deployed on zgx-3675, so we intentionally avoid that
// dependency.

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ominix_cuda {

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr,                                                    \
                    "[audio_encoder_cuda] CUDA error %s:%d: %s\n", __FILE__,   \
                    __LINE__, cudaGetErrorString(_e));                         \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr,                                                    \
                    "[audio_encoder_cuda] cuBLAS error %s:%d: %d\n", __FILE__, \
                    __LINE__, (int)_s);                                        \
            return false;                                                      \
        }                                                                      \
    } while (0)

namespace {

// ---------------------------------------------------------------------------
// GGUF helpers.
// ---------------------------------------------------------------------------
std::vector<float> load_gguf_tensor_f32(ggml_context *ggml_ctx,
                                          const char *name,
                                          size_t expected_elems) {
    ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (!t) {
        fprintf(stderr, "[audio_encoder_cuda] missing tensor: %s\n", name);
        return {};
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr,
                "[audio_encoder_cuda] %s: expected %zu elems, got %zu\n",
                name, expected_elems, n);
        return {};
    }
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const ggml_type_traits *tt = ggml_get_type_traits(t->type);
        if (tt && tt->to_float) {
            tt->to_float(t->data, out.data(), (int64_t)n);
        } else {
            fprintf(stderr,
                    "[audio_encoder_cuda] %s: unsupported dtype %d\n",
                    name, (int)t->type);
            return {};
        }
    }
    return out;
}

bool upload_tensor_f32(ggml_context *ggml_ctx, const char *name,
                        size_t expected_elems, void *&dev) {
    std::vector<float> host = load_gguf_tensor_f32(ggml_ctx, name,
                                                    expected_elems);
    if (host.empty()) return false;
    cudaError_t err = cudaMalloc(&dev, expected_elems * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[audio_encoder_cuda] cudaMalloc(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    err = cudaMemcpy(dev, host.data(), expected_elems * sizeof(float),
                      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[audio_encoder_cuda] cudaMemcpy(%s) failed: %s\n",
                name, cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool gguf_get_u32_int(const struct gguf_context *gctx, const char *key,
                       int &n) {
    const int64_t kid = gguf_find_key(gctx, key);
    if (kid < 0) return false;
    n = (int)gguf_get_val_u32(gctx, kid);
    return true;
}

int conv_out_size(int in_size) {
    return (in_size + 2 * 1 - 3) / 2 + 1;
}

int py_floordiv(int a, int b) {
    return (a / b) - (a % b != 0 && (a ^ b) < 0);
}

int get_feat_extract_output_lengths(int input_lengths, int n_window) {
    int n_window_double = n_window * 2;
    int input_lengths_leave = input_lengths % n_window_double;
    int feat_lengths = py_floordiv(input_lengths_leave - 1, 2) + 1;
    int output_lengths = py_floordiv(py_floordiv(feat_lengths - 1, 2) + 1 - 1, 2) + 1
                         + py_floordiv(input_lengths, n_window_double) * 13;
    return output_lengths;
}

}  // namespace

// ============================================================================
// Class methods. (CUDA kernels live in cuda_kernels/audio_encoder_kernels.cu;
// launchers are declared in audio_encoder_kernels.h and consumed here.)
// ============================================================================

AudioEncoderCudaEngine::~AudioEncoderCudaEngine() {
    auto safe_free = [](void *&p) { if (p) { cudaFree(p); p = nullptr; } };

    safe_free(conv2d1_w_);  safe_free(conv2d1_b_);
    safe_free(conv2d2_w_);  safe_free(conv2d2_b_);
    safe_free(conv2d3_w_);  safe_free(conv2d3_b_);
    safe_free(conv_out_w_);

    for (auto &L : layers_) {
        safe_free(L.self_attn_layer_norm_w); safe_free(L.self_attn_layer_norm_b);
        safe_free(L.q_proj_w); safe_free(L.q_proj_b);
        safe_free(L.k_proj_w); safe_free(L.k_proj_b);
        safe_free(L.v_proj_w); safe_free(L.v_proj_b);
        safe_free(L.out_proj_w); safe_free(L.out_proj_b);
        safe_free(L.final_layer_norm_w); safe_free(L.final_layer_norm_b);
        safe_free(L.fc1_w); safe_free(L.fc1_b);
        safe_free(L.fc2_w); safe_free(L.fc2_b);
    }

    safe_free(ln_post_w_); safe_free(ln_post_b_);
    safe_free(proj1_w_);   safe_free(proj1_b_);
    safe_free(proj2_w_);   safe_free(proj2_b_);
    safe_free(pos_emb_dev_);

    free_scratch_();

    if (cublas_) { cublasDestroy(cublas_); cublas_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
}

bool AudioEncoderCudaEngine::free_scratch_() {
    auto safe_free = [](void *&p) { if (p) { cudaFree(p); p = nullptr; } };
    safe_free(scratch_mel_);
    safe_free(scratch_im2col_);
    safe_free(scratch_conv1_);
    safe_free(scratch_conv2_);
    safe_free(scratch_conv3_);
    safe_free(scratch_conv_proj_);
    safe_free(scratch_concat_);
    safe_free(scratch_resid_);
    safe_free(scratch_norm_);
    safe_free(scratch_q_);
    safe_free(scratch_k_);
    safe_free(scratch_v_);
    safe_free(scratch_attn_out_);
    safe_free(scratch_ffn_inter_);
    safe_free(scratch_proj1_out_);
    safe_free(scratch_output_);
    scratch_chunk_num_     = 0;
    scratch_max_chunk_len_ = 0;
    scratch_total_frames_  = 0;
    return true;
}

bool AudioEncoderCudaEngine::ensure_scratch_(int chunk_num, int max_chunk_len,
                                              int total_frames,
                                              int frames_per_chunk) {
    if (chunk_num <= scratch_chunk_num_ &&
        max_chunk_len <= scratch_max_chunk_len_ &&
        total_frames <= scratch_total_frames_) {
        return true;
    }
    free_scratch_();

    auto alloc = [](void **slot, size_t bytes) -> bool {
        cudaError_t err = cudaMalloc(slot, bytes);
        if (err != cudaSuccess) {
            fprintf(stderr,
                    "[audio_encoder_cuda] cudaMalloc(%zu) failed: %s\n",
                    bytes, cudaGetErrorString(err));
            return false;
        }
        return true;
    };

    int h0 = num_mel_bins_,  w0 = max_chunk_len;
    int h1 = conv_out_size(h0), w1 = conv_out_size(w0);
    int h2 = conv_out_size(h1), w2 = conv_out_size(w1);
    int h3 = conv_out_size(h2), w3 = conv_out_size(w2);
    int H = downsample_hidden_size_;

    if (!alloc(&scratch_mel_, (size_t)chunk_num * h0 * w0 * sizeof(float)))
        return false;

    // im2col workspace large enough for the largest of 3 convs.
    size_t im2col_max = 0;
    im2col_max = std::max(im2col_max,
                          (size_t)chunk_num * h1 * w1 * 1 * 9);
    im2col_max = std::max(im2col_max,
                          (size_t)chunk_num * h2 * w2 * H * 9);
    im2col_max = std::max(im2col_max,
                          (size_t)chunk_num * h3 * w3 * H * 9);
    if (!alloc(&scratch_im2col_, im2col_max * sizeof(float))) return false;

    // Conv outputs (NCHW after transpose-back). We allocate enough to hold
    // either the GEMM output (NHWC = total_rows × C_out) or the transposed
    // NCHW layout — same total byte count.
    if (!alloc(&scratch_conv1_,
               (size_t)chunk_num * H * h1 * w1 * sizeof(float))) return false;
    if (!alloc(&scratch_conv2_,
               (size_t)chunk_num * H * h2 * w2 * sizeof(float))) return false;
    if (!alloc(&scratch_conv3_,
               (size_t)chunk_num * H * h3 * w3 * sizeof(float))) return false;

    // post-conv linear projection: (chunk_num, frames_per_chunk, d_model).
    if (!alloc(&scratch_conv_proj_,
               (size_t)chunk_num * frames_per_chunk * d_model_ * sizeof(float)))
        return false;

    // concat valid frames (transformer input) and per-block scratches.
    if (!alloc(&scratch_concat_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_resid_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_norm_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_q_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_k_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_v_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_attn_out_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_ffn_inter_,
               (size_t)total_frames * encoder_ffn_dim_ * sizeof(float)))
        return false;
    if (!alloc(&scratch_proj1_out_,
               (size_t)total_frames * d_model_ * sizeof(float))) return false;
    if (!alloc(&scratch_output_,
               (size_t)total_frames * output_dim_ * sizeof(float))) return false;

    scratch_chunk_num_     = chunk_num;
    scratch_max_chunk_len_ = max_chunk_len;
    scratch_total_frames_  = total_frames;
    return true;
}

bool AudioEncoderCudaEngine::init_from_gguf(const std::string &gguf_path,
                                              int device) {
    device_ = device;
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

    struct ggml_context  *meta_ctx = nullptr;
    struct gguf_init_params iparams = {
        /*.no_alloc =*/ false,
        /*.ctx      =*/ &meta_ctx,
    };
    struct gguf_context *gctx = gguf_init_from_file(gguf_path.c_str(), iparams);
    if (!gctx) {
        fprintf(stderr,
                "[audio_encoder_cuda] gguf_init_from_file failed: %s\n",
                gguf_path.c_str());
        return false;
    }

    gguf_get_u32_int(gctx, "d_model",                  d_model_);
    gguf_get_u32_int(gctx, "encoder_layers",           encoder_layers_);
    gguf_get_u32_int(gctx, "encoder_attention_heads",  encoder_attention_heads_);
    gguf_get_u32_int(gctx, "encoder_ffn_dim",          encoder_ffn_dim_);
    gguf_get_u32_int(gctx, "num_mel_bins",             num_mel_bins_);
    gguf_get_u32_int(gctx, "output_dim",               output_dim_);
    gguf_get_u32_int(gctx, "downsample_hidden_size",   downsample_hidden_size_);
    gguf_get_u32_int(gctx, "max_source_positions",     max_source_positions_);
    gguf_get_u32_int(gctx, "n_window",                 n_window_);
    gguf_get_u32_int(gctx, "n_window_infer",           n_window_infer_);
    gguf_get_u32_int(gctx, "mel_reduced",              mel_reduced_);
    gguf_get_u32_int(gctx, "conv_out_dim",             conv_out_dim_);

    n_tensors_seen_  = (int)gguf_get_n_tensors(gctx);
    n_weights_bound_ = 0;

    layers_.assign(encoder_layers_, AudioEncoderLayerCuda{});

    auto upload = [&](void **slot, const std::string &name,
                       size_t elems) -> bool {
        if (!upload_tensor_f32(meta_ctx, name.c_str(), elems, *slot))
            return false;
        ++n_weights_bound_;
        return true;
    };

    bool ok = true;
    const int H = downsample_hidden_size_;

    ok = ok && upload(&conv2d1_w_, "conv2d1.weight", (size_t)H * 1 * 3 * 3);
    ok = ok && upload(&conv2d1_b_, "conv2d1.bias",   (size_t)H);
    ok = ok && upload(&conv2d2_w_, "conv2d2.weight", (size_t)H * H * 3 * 3);
    ok = ok && upload(&conv2d2_b_, "conv2d2.bias",   (size_t)H);
    ok = ok && upload(&conv2d3_w_, "conv2d3.weight", (size_t)H * H * 3 * 3);
    ok = ok && upload(&conv2d3_b_, "conv2d3.bias",   (size_t)H);
    ok = ok && upload(&conv_out_w_, "conv_out.weight",
                       (size_t)d_model_ * conv_out_dim_);

    for (int i = 0; i < encoder_layers_; ++i) {
        auto &L = layers_[i];
        const std::string p = "layers." + std::to_string(i) + ".";

        ok = ok && upload(&L.self_attn_layer_norm_w,
                          p + "self_attn_layer_norm.weight", (size_t)d_model_);
        ok = ok && upload(&L.self_attn_layer_norm_b,
                          p + "self_attn_layer_norm.bias",   (size_t)d_model_);
        ok = ok && upload(&L.q_proj_w, p + "self_attn.q_proj.weight",
                          (size_t)d_model_ * d_model_);
        ok = ok && upload(&L.q_proj_b, p + "self_attn.q_proj.bias",
                          (size_t)d_model_);
        ok = ok && upload(&L.k_proj_w, p + "self_attn.k_proj.weight",
                          (size_t)d_model_ * d_model_);
        ok = ok && upload(&L.k_proj_b, p + "self_attn.k_proj.bias",
                          (size_t)d_model_);
        ok = ok && upload(&L.v_proj_w, p + "self_attn.v_proj.weight",
                          (size_t)d_model_ * d_model_);
        ok = ok && upload(&L.v_proj_b, p + "self_attn.v_proj.bias",
                          (size_t)d_model_);
        ok = ok && upload(&L.out_proj_w, p + "self_attn.out_proj.weight",
                          (size_t)d_model_ * d_model_);
        ok = ok && upload(&L.out_proj_b, p + "self_attn.out_proj.bias",
                          (size_t)d_model_);
        ok = ok && upload(&L.final_layer_norm_w, p + "final_layer_norm.weight",
                          (size_t)d_model_);
        ok = ok && upload(&L.final_layer_norm_b, p + "final_layer_norm.bias",
                          (size_t)d_model_);
        ok = ok && upload(&L.fc1_w, p + "fc1.weight",
                          (size_t)encoder_ffn_dim_ * d_model_);
        ok = ok && upload(&L.fc1_b, p + "fc1.bias",
                          (size_t)encoder_ffn_dim_);
        ok = ok && upload(&L.fc2_w, p + "fc2.weight",
                          (size_t)d_model_ * encoder_ffn_dim_);
        ok = ok && upload(&L.fc2_b, p + "fc2.bias", (size_t)d_model_);
    }

    ok = ok && upload(&ln_post_w_, "ln_post.weight", (size_t)d_model_);
    ok = ok && upload(&ln_post_b_, "ln_post.bias",   (size_t)d_model_);
    ok = ok && upload(&proj1_w_,   "proj1.weight",
                      (size_t)d_model_ * d_model_);
    ok = ok && upload(&proj1_b_,   "proj1.bias",   (size_t)d_model_);
    ok = ok && upload(&proj2_w_,   "proj2.weight",
                      (size_t)output_dim_ * d_model_);
    ok = ok && upload(&proj2_b_,   "proj2.bias",   (size_t)output_dim_);

    if (!ok) {
        fprintf(stderr,
                "[audio_encoder_cuda] init_from_gguf: weight upload FAILED\n");
        gguf_free(gctx);
        if (meta_ctx) ggml_free(meta_ctx);
        return false;
    }

    // Sinusoidal positional embeddings.
    {
        int half_dim = d_model_ / 2;
        std::vector<float> pe((size_t)max_source_positions_ * d_model_, 0.0f);
        float log_timescale_increment =
            std::log(10000.0f) / (float)(half_dim - 1);
        std::vector<float> inv_timescales(half_dim);
        for (int i = 0; i < half_dim; ++i)
            inv_timescales[i] = std::exp(-log_timescale_increment * (float)i);
        for (int pos = 0; pos < max_source_positions_; ++pos) {
            for (int i = 0; i < half_dim; ++i) {
                float t = (float)pos * inv_timescales[i];
                pe[pos * d_model_ + i]            = std::sin(t);
                pe[pos * d_model_ + half_dim + i] = std::cos(t);
            }
        }
        CUDA_CHECK(cudaMalloc(&pos_emb_dev_, pe.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(pos_emb_dev_, pe.data(),
                                pe.size() * sizeof(float),
                                cudaMemcpyHostToDevice));
    }

    fprintf(stderr,
            "[audio_encoder_cuda] init_from_gguf: tensors=%d bound=%d "
            "d_model=%d layers=%d heads=%d ffn=%d mels=%d out=%d\n",
            n_tensors_seen_, n_weights_bound_, d_model_, encoder_layers_,
            encoder_attention_heads_, encoder_ffn_dim_, num_mel_bins_,
            output_dim_);

    gguf_free(gctx);
    if (meta_ctx) ggml_free(meta_ctx);

    ready_ = true;
    return true;
}

bool AudioEncoderCudaEngine::encode(const float *mel, int n_mels, int mel_T,
                                      float *output, int &num_frames) {
    if (!ready_) {
        fprintf(stderr, "[audio_encoder_cuda] encode: engine not ready\n");
        return false;
    }
    if (n_mels != num_mel_bins_) {
        fprintf(stderr,
                "[audio_encoder_cuda] encode: n_mels=%d != num_mel_bins=%d\n",
                n_mels, num_mel_bins_);
        return false;
    }
    if (mel_T <= 0) {
        fprintf(stderr, "[audio_encoder_cuda] encode: mel_T=%d <= 0\n", mel_T);
        return false;
    }

    // 1. Chunk mel along time.
    int n_window_double = n_window_ * 2;   // 100
    int chunk_num = (mel_T + n_window_double - 1) / n_window_double;
    std::vector<int> chunk_lens(chunk_num, n_window_double);
    int remainder = mel_T % n_window_double;
    if (remainder != 0) chunk_lens[chunk_num - 1] = remainder;
    int max_chunk_len =
        *std::max_element(chunk_lens.begin(), chunk_lens.end());

    // 2. Pack padded mel: [chunk_num, 1, num_mel_bins, max_chunk_len] (NCHW).
    std::vector<float> padded_mel(
        (size_t)chunk_num * num_mel_bins_ * max_chunk_len, 0.0f);
    int col_offset = 0;
    for (int c = 0; c < chunk_num; ++c) {
        int clen = chunk_lens[c];
        for (int m = 0; m < num_mel_bins_; ++m) {
            for (int t = 0; t < clen; ++t) {
                int dst = c * num_mel_bins_ * max_chunk_len + m * max_chunk_len + t;
                int src = m * mel_T + col_offset + t;
                padded_mel[dst] = mel[src];
            }
        }
        col_offset += clen;
    }

    // Conv shape progression.
    int h0 = num_mel_bins_, w0 = max_chunk_len;
    int h1 = conv_out_size(h0), w1 = conv_out_size(w0);
    int h2 = conv_out_size(h1), w2 = conv_out_size(w1);
    int h3 = conv_out_size(h2), w3 = conv_out_size(w2);
    int frames_per_chunk = w3;
    int H = downsample_hidden_size_;
    const int D = d_model_;

    // Per-chunk valid frame counts.
    std::vector<int> feat_lens(chunk_num);
    int total_frames = 0;
    for (int c = 0; c < chunk_num; ++c) {
        feat_lens[c] = get_feat_extract_output_lengths(chunk_lens[c], n_window_);
        total_frames += feat_lens[c];
    }

    if (!ensure_scratch_(chunk_num, max_chunk_len, total_frames,
                         frames_per_chunk)) {
        return false;
    }

    fprintf(stderr,
            "[audio_encoder_cuda] encode: mel_T=%d chunk_num=%d "
            "max_chunk_len=%d frames_per_chunk=%d total_frames=%d\n",
            mel_T, chunk_num, max_chunk_len, frames_per_chunk, total_frames);

    cudaStream_t s = stream_;
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start, s));

    // Upload padded mel.
    CUDA_CHECK(cudaMemcpyAsync(scratch_mel_, padded_mel.data(),
                                 padded_mel.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, s));

    // ---- Conv layer helper: input NCHW [N, C_in, H_in, W_in] → output NCHW
    //      [N, C_out, H_out, W_out] (via im2col + GEMM + bias + GELU + transpose).
    auto conv_layer = [&](const float *src_in, int C_in, int H_in, int W_in,
                            const float *w_dev, const float *b_dev,
                            int C_out, int H_out, int W_out,
                            float *dst_nchw) -> bool {
        int patch_size = C_in * 3 * 3;
        int total_rows = chunk_num * H_out * W_out;

        launch_im2col_f32(src_in, (float *)scratch_im2col_,
                           chunk_num, C_in, H_in, W_in, 3, 3, 2, 2, 1, 1,
                           H_out, W_out, s);

        // GEMM: scratch_dst[total_rows, C_out] = scratch_im2col_[total_rows,
        // patch_size] @ W^T, with W row-major [C_out, patch_size].
        // We GEMM directly into dst_nchw’s buffer (treating it as NHWC for
        // now; we’ll transpose afterwards).
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            C_out, total_rows, patch_size,
            &alpha,
            w_dev,                 CUDA_R_32F, patch_size,
            scratch_im2col_,       CUDA_R_32F, patch_size,
            &beta,
            dst_nchw,              CUDA_R_32F, C_out,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[audio_encoder_cuda] conv GEMM failed: %d\n", (int)st);
            return false;
        }

        // Bias add (broadcast over the leading total_rows dimension).
        launch_add_bias_rowmajor_f32(dst_nchw, b_dev, total_rows, C_out, s);
        // GELU-erf in place.
        launch_gelu_erf_f32(dst_nchw, total_rows * C_out, s);
        // Transpose NHWC → NCHW in-place (using scratch_im2col_ as temp).
        launch_nhwc_to_nchw_f32(dst_nchw, (float *)scratch_im2col_,
                                  chunk_num, H_out, W_out, C_out, s);
        // Copy back NCHW into dst_nchw.
        size_t bytes = (size_t)total_rows * C_out * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(dst_nchw, scratch_im2col_, bytes,
                                     cudaMemcpyDeviceToDevice, s));
        return true;
    };

    // Conv1: [N, 1, h0, w0] → [N, H, h1, w1]
    if (!conv_layer((const float *)scratch_mel_, 1, h0, w0,
                     (const float *)conv2d1_w_, (const float *)conv2d1_b_,
                     H, h1, w1, (float *)scratch_conv1_)) return false;
    // Conv2: [N, H, h1, w1] → [N, H, h2, w2]
    if (!conv_layer((const float *)scratch_conv1_, H, h1, w1,
                     (const float *)conv2d2_w_, (const float *)conv2d2_b_,
                     H, h2, w2, (float *)scratch_conv2_)) return false;
    // Conv3: [N, H, h2, w2] → [N, H, h3, w3]
    if (!conv_layer((const float *)scratch_conv2_, H, h2, w2,
                     (const float *)conv2d3_w_, (const float *)conv2d3_b_,
                     H, h3, w3, (float *)scratch_conv3_)) return false;

    // 3. Slab extraction in (H outer, C inner) order to produce
    //    [chunk_num * w3, h3 * H] = [chunk_num * frames_per_chunk, conv_out_dim_].
    //    We reuse scratch_im2col_ as the staging buffer.
    launch_nchw_to_frame_slab_hc((const float *)scratch_conv3_,
                                    (float *)scratch_im2col_,
                                    chunk_num, H, h3, w3, s);
    int slab_rows = chunk_num * w3;
    int slab_cols = h3 * H;   // == conv_out_dim_

    // 4. conv_out projection: [slab_rows, slab_cols] @ conv_out_w_^T
    //    → [slab_rows, d_model] (no bias).
    //    conv_out_w_ is row-major [d_model, conv_out_dim] in our upload.
    {
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            D, slab_rows, slab_cols,
            &alpha,
            conv_out_w_,             CUDA_R_32F, slab_cols,
            scratch_im2col_,         CUDA_R_32F, slab_cols,
            &beta,
            scratch_conv_proj_,      CUDA_R_32F, D,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[audio_encoder_cuda] conv_out GEMM failed: %d\n", (int)st);
            return false;
        }
    }

    // 5. Build chunk_offsets / chunk_valid on host and upload.
    std::vector<int> chunk_offsets(chunk_num);
    int offset = 0;
    for (int c = 0; c < chunk_num; ++c) {
        chunk_offsets[c] = offset;
        offset += feat_lens[c];
    }
    int *d_chunk_offsets = nullptr, *d_chunk_valid = nullptr;
    CUDA_CHECK(cudaMalloc(&d_chunk_offsets, chunk_num * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_chunk_valid,   chunk_num * sizeof(int)));
    CUDA_CHECK(cudaMemcpyAsync(d_chunk_offsets, chunk_offsets.data(),
                                 chunk_num * sizeof(int),
                                 cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_chunk_valid, feat_lens.data(),
                                 chunk_num * sizeof(int),
                                 cudaMemcpyHostToDevice, s));

    // 6. Gather valid frames + add positional embedding → scratch_concat_.
    launch_gather_with_pos_emb((const float *)scratch_conv_proj_,
                                  (const float *)pos_emb_dev_,
                                  d_chunk_offsets, d_chunk_valid,
                                  chunk_num, frames_per_chunk, D,
                                  max_source_positions_, total_frames,
                                  (float *)scratch_concat_, s);

    // 7. 24L Pre-LN transformer.
    //    Residual chain (F32) lives in scratch_resid_; initialize from concat.
    {
        size_t bytes = (size_t)total_frames * D * sizeof(float);
        CUDA_CHECK(cudaMemcpyAsync(scratch_resid_, scratch_concat_, bytes,
                                     cudaMemcpyDeviceToDevice, s));
    }

    int n_heads = encoder_attention_heads_;
    int head_dim = D / n_heads;
    float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);

    auto gemm_f32 = [&](const float *X, const float *W, float *Y,
                         int M, int K, int N, const float *bias) -> bool {
        const float alpha = 1.0f, beta = 0.0f;
        cublasStatus_t st = cublasGemmEx(
            cublas_,
            CUBLAS_OP_T, CUBLAS_OP_N,
            N, M, K,
            &alpha,
            W, CUDA_R_32F, K,
            X, CUDA_R_32F, K,
            &beta,
            Y, CUDA_R_32F, N,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr,
                    "[audio_encoder_cuda] GemmEx F32 failed: %d\n", (int)st);
            return false;
        }
        if (bias) {
            launch_add_bias_f32_f32(Y, bias, M, N, s);
        }
        return true;
    };

    for (int il = 0; il < encoder_layers_; ++il) {
        auto &L = layers_[il];

        // ----- Self-attn block -----
        // Pre-LayerNorm (affine).
        launch_layernorm_affine_f32((const float *)scratch_resid_,
                                       (const float *)L.self_attn_layer_norm_w,
                                       (const float *)L.self_attn_layer_norm_b,
                                       (float *)scratch_norm_,
                                       total_frames, D, 1e-5f, s);

        // Q, K, V projections.
        if (!gemm_f32((const float *)scratch_norm_,
                       (const float *)L.q_proj_w, (float *)scratch_q_,
                       total_frames, D, D, (const float *)L.q_proj_b))
            return false;
        if (!gemm_f32((const float *)scratch_norm_,
                       (const float *)L.k_proj_w, (float *)scratch_k_,
                       total_frames, D, D, (const float *)L.k_proj_b))
            return false;
        if (!gemm_f32((const float *)scratch_norm_,
                       (const float *)L.v_proj_w, (float *)scratch_v_,
                       total_frames, D, D, (const float *)L.v_proj_b))
            return false;

        // Naive joint attention. Q/K/V are laid out [total_frames, D] with
        // n_heads × head_dim packed in the inner dim. The kernel expects
        // [seq, n_heads, head_dim] which is the same memory layout, so we
        // pass through directly.
        launch_attn_joint_naive_f32((const float *)scratch_q_,
                                       (const float *)scratch_k_,
                                       (const float *)scratch_v_,
                                       (float *)scratch_attn_out_,
                                       total_frames, n_heads, head_dim,
                                       inv_sqrt_d, s);

        // out_proj.
        if (!gemm_f32((const float *)scratch_attn_out_,
                       (const float *)L.out_proj_w, (float *)scratch_norm_,
                       total_frames, D, D, (const float *)L.out_proj_b))
            return false;

        // Residual add: resid += attn_out_proj.
        launch_resid_add_f32((float *)scratch_resid_,
                               (const float *)scratch_norm_,
                               total_frames, D, s);

        // ----- FFN block -----
        // Pre-LN.
        launch_layernorm_affine_f32((const float *)scratch_resid_,
                                       (const float *)L.final_layer_norm_w,
                                       (const float *)L.final_layer_norm_b,
                                       (float *)scratch_norm_,
                                       total_frames, D, 1e-5f, s);
        // fc1: D -> ffn.
        if (!gemm_f32((const float *)scratch_norm_,
                       (const float *)L.fc1_w, (float *)scratch_ffn_inter_,
                       total_frames, D, encoder_ffn_dim_,
                       (const float *)L.fc1_b))
            return false;
        // GELU-erf.
        launch_gelu_erf_f32((float *)scratch_ffn_inter_,
                              total_frames * encoder_ffn_dim_, s);
        // fc2: ffn -> D.
        if (!gemm_f32((const float *)scratch_ffn_inter_,
                       (const float *)L.fc2_w, (float *)scratch_norm_,
                       total_frames, encoder_ffn_dim_, D,
                       (const float *)L.fc2_b))
            return false;
        // Residual add.
        launch_resid_add_f32((float *)scratch_resid_,
                               (const float *)scratch_norm_,
                               total_frames, D, s);
    }

    // 8. Output MLP: ln_post + proj1 + GELU-erf + proj2.
    launch_layernorm_affine_f32((const float *)scratch_resid_,
                                   (const float *)ln_post_w_,
                                   (const float *)ln_post_b_,
                                   (float *)scratch_norm_,
                                   total_frames, D, 1e-5f, s);
    if (!gemm_f32((const float *)scratch_norm_,
                   (const float *)proj1_w_, (float *)scratch_proj1_out_,
                   total_frames, D, D, (const float *)proj1_b_))
        return false;
    launch_gelu_erf_f32((float *)scratch_proj1_out_, total_frames * D, s);
    if (!gemm_f32((const float *)scratch_proj1_out_,
                   (const float *)proj2_w_, (float *)scratch_output_,
                   total_frames, D, output_dim_, (const float *)proj2_b_))
        return false;

    // 9. Download to host.
    size_t out_bytes = (size_t)total_frames * output_dim_ * sizeof(float);
    CUDA_CHECK(cudaMemcpyAsync(output, scratch_output_, out_bytes,
                                 cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaEventRecord(ev_end, s));
    CUDA_CHECK(cudaStreamSynchronize(s));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_end));
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_end);
    fprintf(stderr,
            "[audio_encoder_cuda] encode: device wall time = %.3f ms\n", ms);

    cudaFree(d_chunk_offsets);
    cudaFree(d_chunk_valid);

    num_frames = total_frames;
    return true;
}

}  // namespace ominix_cuda
