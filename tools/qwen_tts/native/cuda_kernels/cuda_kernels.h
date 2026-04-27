#pragma once
// ============================================================================
// Custom CUDA kernels for TalkerCudaEngine (Phase 2.2 forward decode body).
//
// Each launcher takes raw pointers + a stream and runs a single op. They are
// intentionally narrow-scoped (S=1 decode-path shapes) so the inner loop is
// just `cublasGemmEx` calls plus a handful of these. No ggml graph runtime
// is reachable from here.
//
// All buffers are device pointers. F16 = __half. F32 = float.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ominix_cuda {

// F32 -> F16 cast.
//   in  : F32 [n]
//   out : F16 [n]
void launch_cast_f32_to_f16(const float *in, __half *out, int n,
                            cudaStream_t stream);

// F16 -> F32 cast.
//   in  : F16 [n]
//   out : F32 [n]
void launch_cast_f16_to_f32(const __half *in, float *out, int n,
                            cudaStream_t stream);

// y = a + b (F16 elementwise add).
//   a, b, y : F16 [n]
void launch_add_f16(const __half *a, const __half *b, __half *y, int n,
                    cudaStream_t stream);

// RmsNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma.
//   x      : F16 [rows, cols]
//   gamma  : F32 [cols]   (Qwen3 norm gammas are F32 on disk)
//   y      : F16 [rows, cols]   (may alias x)
// Each row is normed independently. Implementation does one block per row.
void launch_rmsnorm_f16_g32(const __half *x, const float *gamma, __half *y,
                            int rows, int cols, float eps,
                            cudaStream_t stream);

// NEOX-mode RoPE on a [n_heads, head_dim] tile (single token).
//   x     : F16 [n_heads, head_dim]   (in-place permitted iff y == x)
//   cos   : F16 [head_dim]            (row of precomputed cos table at pos)
//   sin   : F16 [head_dim]            (row of precomputed sin table at pos)
//   y     : F16 [n_heads, head_dim]
//
// NEOX mode: tile head_dim into two halves [low, high], where
//   y[low ] = x[low ] * cos[low ] - x[high] * sin[low ]
//   y[high] = x[low ] * sin[high] + x[high] * cos[high]
// This matches the cos/sin table layout (halves duplicated) used in Ascend.
void launch_rope_neox_f16(const __half *x, const __half *cos,
                          const __half *sin, __half *y,
                          int n_heads, int head_dim, cudaStream_t stream);

// Fused SwiGLU: y[i] = silu(gate[i]) * up[i] = (gate[i] / (1+exp(-gate[i]))) * up[i].
//   gate, up, y : F16 [n]   (y may alias gate)
void launch_swiglu_f16(const __half *gate, const __half *up, __half *y, int n,
                       cudaStream_t stream);

// Single-token GQA attention (decode path, S=1).
//   q       : F16 [n_heads,   head_dim]
//   k_cache : F16 [seq_len,   n_kv,  head_dim]   (contiguous, MAX_SEQ stride
//                                                  in the leading dim, only
//                                                  the first seq_len rows are
//                                                  valid)
//   v_cache : F16 [seq_len,   n_kv,  head_dim]   (same layout as K)
//   y       : F16 [n_heads,   head_dim]
//
//   seq_len    : current valid KV length (== pos + 1)
//   n_heads    : Q heads (e.g. 16)
//   n_kv       : KV heads (e.g. 8); GQA group size = n_heads / n_kv.
//   head_dim   : per-head dim (e.g. 128)
//   inv_sqrt_d : 1.0f / sqrtf(head_dim)
//
// Implementation: one block per Q head. Each block:
//   1. Loads q[h] (head_dim floats) into shmem.
//   2. For each token t in [0, seq_len): compute score[t] = dot(q[h], k[t, h_kv]) * inv_sqrt_d
//      where h_kv = h / group_size.
//   3. softmax(scores) over the seq_len axis.
//   4. Reduces sum_t softmax[t] * v[t, h_kv, :] to produce y[h].
//
// Numerically stable: max-subtract before exp.
void launch_attn_decode_gqa_f16(const __half *q,
                                const __half *k_cache,
                                const __half *v_cache,
                                __half *y,
                                int seq_len, int n_heads, int n_kv,
                                int head_dim, float inv_sqrt_d,
                                cudaStream_t stream);

// ============================================================================
// SpeechTokenizerDecoder F32 ops (Phase 2.7b).
//
// All shapes are row-major [T, C] with C fastest. See decoder_ops.cu for
// per-op semantics; the comments below mirror that file's contract.
// ============================================================================

// Causal Conv1d im2col: zero-pad-left + window of size K.
//   in  : F32 [T, C_in]
//   out : F32 [T, K * C_in]   (out[t, k*C_in + c] = in_padded[t + k - (K-1), c])
void launch_causal_conv1d_im2col_f32(const float *in, float *out,
                                      int T, int C_in, int K,
                                      cudaStream_t stream);

// Depthwise causal Conv1d.
//   in  : F32 [T, C]
//   w   : F32 [K, C]   (per-channel filter; GGUF [K, 1, C] with the singleton
//                       in-channels-per-group axis collapsed.)
//   b   : F32 [C]   or nullptr
//   out : F32 [T, C]
void launch_depthwise_conv1d_causal_f32(const float *in, const float *w,
                                         const float *b, float *out,
                                         int T, int C, int K,
                                         cudaStream_t stream);

// ConvTranspose1d, kernel=2, stride=2 (matches the upsample.X.0.conv shape).
//   in  : F32 [T, C_in]
//   w   : F32 [K=2, C_out, C_in]   (GGUF row-major; C_in fastest)
//   b   : F32 [C_out]   or nullptr
//   out : F32 [2*T, C_out]
void launch_conv_transpose1d_k2s2_f32(const float *in, const float *w,
                                       const float *b, float *out,
                                       int T, int C_in, int C_out,
                                       cudaStream_t stream);

// LayerNorm (affine) over the last axis. One block per row.
//   x      : F32 [T, C]
//   gamma  : F32 [C]
//   beta   : F32 [C]   or nullptr
//   y      : F32 [T, C]   (may alias x)
void launch_layernorm_f32(const float *x, const float *gamma, const float *beta,
                           float *y, int T, int C, float eps,
                           cudaStream_t stream);

// GELU (erf form, ConvNeXt default).
//   y[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
void launch_gelu_erf_f32(const float *in, float *out, int n,
                          cudaStream_t stream);

// Bias add over [T, C]: y[t, c] = x[t, c] + bias[c]. In-place OK.
void launch_bias_add_f32(const float *x, const float *bias, float *y,
                          int T, int C, cudaStream_t stream);

// Channel scale: y[t, c] = x[t, c] * gamma[c]. ConvNeXt gamma residual scale.
void launch_channel_scale_f32(const float *x, const float *gamma, float *y,
                               int T, int C, cudaStream_t stream);

// y = a + b, flat [n] F32 (residual).
void launch_add_f32(const float *a, const float *b, float *y, int n,
                     cudaStream_t stream);

}  // namespace ominix_cuda
