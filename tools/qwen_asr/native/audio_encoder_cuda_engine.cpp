// ============================================================================
// AudioEncoderCudaEngine — Phase 4.1 scaffold.
//
// Phase 4.1 deliverable: init_from_gguf opens the GGUF, sniffs hparams,
// counts transformer layers and tensor handles, and reports through the
// public accessors. No CUDA allocations happen in 4.1 beyond a stream and
// a cublas handle (those validate the device + driver are alive).
// ============================================================================

#include "audio_encoder_cuda_engine.h"

#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace ominix_cuda {

#define CUDA_CHECK(expr)                                                       \
    do {                                                                       \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr,                                                    \
                    "[audio_encoder_cuda] CUDA error %s:%d: %s\n", __FILE__,   \
                    __LINE__, cudaGetErrorString(_e));                         \
            return false;                                                      \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(expr)                                                     \
    do {                                                                       \
        cublasStatus_t _s = (expr);                                            \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            fprintf(stderr,                                                    \
                    "[audio_encoder_cuda] cuBLAS error %s:%d: %d\n", __FILE__, \
                    __LINE__, (int)_s);                                        \
            return false;                                                      \
        }                                                                     \
    } while (0)

AudioEncoderCudaEngine::~AudioEncoderCudaEngine() {
    if (cublas_) {
        cublasDestroy(cublas_);
        cublas_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

// Helper: read a uint32 GGUF key into n if present.
static bool gguf_get_u32(const struct gguf_context *gctx, const char *key,
                         int &n) {
    const int64_t kid = gguf_find_key(gctx, key);
    if (kid < 0) {
        return false;
    }
    n = (int)gguf_get_val_u32(gctx, kid);
    return true;
}

bool AudioEncoderCudaEngine::init_from_gguf(const std::string &gguf_path,
                                            int device) {
    device_ = device;
    CUDA_CHECK(cudaSetDevice(device_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

    // Open GGUF and sniff hparams.
    struct ggml_context  *meta_ctx = nullptr;
    struct gguf_init_params iparams = {
        /*.no_alloc =*/ true,
        /*.ctx      =*/ &meta_ctx,
    };
    struct gguf_context *gctx = gguf_init_from_file(gguf_path.c_str(), iparams);
    if (!gctx) {
        fprintf(stderr,
                "[audio_encoder_cuda] gguf_init_from_file failed: %s\n",
                gguf_path.c_str());
        return false;
    }

    // Hparams — Ascend exporter writes flat keys (no prefix).
    gguf_get_u32(gctx, "d_model",                  d_model_);
    gguf_get_u32(gctx, "encoder_layers",           encoder_layers_);
    gguf_get_u32(gctx, "encoder_attention_heads",  encoder_attention_heads_);
    gguf_get_u32(gctx, "encoder_ffn_dim",          encoder_ffn_dim_);
    gguf_get_u32(gctx, "num_mel_bins",             num_mel_bins_);
    gguf_get_u32(gctx, "output_dim",               output_dim_);
    gguf_get_u32(gctx, "downsample_hidden_size",   downsample_hidden_size_);
    gguf_get_u32(gctx, "max_source_positions",     max_source_positions_);
    gguf_get_u32(gctx, "n_window",                 n_window_);
    gguf_get_u32(gctx, "n_window_infer",           n_window_infer_);
    gguf_get_u32(gctx, "mel_reduced",              mel_reduced_);
    gguf_get_u32(gctx, "conv_out_dim",             conv_out_dim_);

    // Count tensors and resolve weight handles by name.
    const int64_t n_tensors = gguf_get_n_tensors(gctx);
    n_tensors_seen_  = (int)n_tensors;
    n_weights_bound_ = 0;

    layers_.assign(encoder_layers_, AudioEncoderLayerCuda{});

    auto find_tensor = [&](const std::string &name) -> ggml_tensor * {
        return ggml_get_tensor(meta_ctx, name.c_str());
    };

    auto bind = [&](void **slot, const std::string &name) {
        ggml_tensor *t = find_tensor(name);
        if (t) {
            // Phase 4.1 only records the host meta pointer so we can prove the
            // tensor exists in the GGUF; Phase 4.2 will upload F16 weights to
            // device memory and store the cudaMalloc pointer here.
            *slot = (void *)t;
            ++n_weights_bound_;
        }
    };

    // Conv2d (Ascend exporter writes flat names — no prefix).
    bind(&conv2d1_w_, "conv2d1.weight");
    bind(&conv2d1_b_, "conv2d1.bias");
    bind(&conv2d2_w_, "conv2d2.weight");
    bind(&conv2d2_b_, "conv2d2.bias");
    bind(&conv2d3_w_, "conv2d3.weight");
    bind(&conv2d3_b_, "conv2d3.bias");
    bind(&conv_out_w_, "conv_out.weight");

    // Transformer layers.
    for (int i = 0; i < encoder_layers_; ++i) {
        AudioEncoderLayerCuda &L = layers_[i];
        const std::string p = "layers." + std::to_string(i) + ".";
        bind(&L.self_attn_layer_norm_w, p + "self_attn_layer_norm.weight");
        bind(&L.self_attn_layer_norm_b, p + "self_attn_layer_norm.bias");
        bind(&L.q_proj_w, p + "self_attn.q_proj.weight");
        bind(&L.q_proj_b, p + "self_attn.q_proj.bias");
        bind(&L.k_proj_w, p + "self_attn.k_proj.weight");
        bind(&L.k_proj_b, p + "self_attn.k_proj.bias");
        bind(&L.v_proj_w, p + "self_attn.v_proj.weight");
        bind(&L.v_proj_b, p + "self_attn.v_proj.bias");
        bind(&L.out_proj_w, p + "self_attn.out_proj.weight");
        bind(&L.out_proj_b, p + "self_attn.out_proj.bias");
        bind(&L.final_layer_norm_w, p + "final_layer_norm.weight");
        bind(&L.final_layer_norm_b, p + "final_layer_norm.bias");
        bind(&L.fc1_w, p + "fc1.weight");
        bind(&L.fc1_b, p + "fc1.bias");
        bind(&L.fc2_w, p + "fc2.weight");
        bind(&L.fc2_b, p + "fc2.bias");
    }

    // Output MLP.
    bind(&ln_post_w_, "ln_post.weight");
    bind(&ln_post_b_, "ln_post.bias");
    bind(&proj1_w_,   "proj1.weight");
    bind(&proj1_b_,   "proj1.bias");
    bind(&proj2_w_,   "proj2.weight");
    bind(&proj2_b_,   "proj2.bias");

    fprintf(stderr,
            "[audio_encoder_cuda] init_from_gguf: tensors=%d bound=%d "
            "d_model=%d layers=%d heads=%d ffn=%d mels=%d out=%d\n",
            n_tensors_seen_, n_weights_bound_, d_model_, encoder_layers_,
            encoder_attention_heads_, encoder_ffn_dim_, num_mel_bins_,
            output_dim_);

    gguf_free(gctx);
    if (meta_ctx) {
        ggml_free(meta_ctx);
    }

    ready_ = true;
    return true;
}

bool AudioEncoderCudaEngine::encode(const float * /*mel*/, int /*n_mels*/,
                                    int /*mel_T*/, float * /*output*/,
                                    int &num_frames) {
    fprintf(stderr,
            "[audio_encoder_cuda] encode() is a Phase 4.2 stub — returning "
            "false in Phase 4.1\n");
    num_frames = 0;
    return false;
}

} // namespace ominix_cuda
