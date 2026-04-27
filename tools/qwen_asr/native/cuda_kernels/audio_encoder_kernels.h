#pragma once
// ============================================================================
// Audio encoder CUDA kernels (Phase 4.2).
//
// Adds the small set of kernels that aren't already in qwen_image_edit's
// dit_kernels.cu: im2col for Conv2d, LayerNorm-with-affine, accurate GELU-erf,
// and per-channel bias adds for conv outputs in NCHW layout.
//
// Numeric strategy (matches qwen_image_edit Phase 3.3b):
//   - Activations stay in F32 throughout (encoder is small — total
//     activation footprint << 100 MB at 122-frame seq, 24 layers).
//   - All matmuls are cuBLAS GemmEx with F32 input, F32 output, F32 accum.
//   - Reuse launch_attn_joint_naive_f32 from dit_kernels.h for attention.
// ============================================================================

#include <cuda_runtime.h>
#include <cstdint>

namespace ominix_cuda {

// im2col for Conv2d (F32, NCHW input, kernel = (KH, KW), stride, padding).
//   in    : F32 [N, C_in, H_in, W_in]   (row-major; ne[0]=W_in)
//   out   : F32 [N, H_out, W_out, C_in * KH * KW]
//   N, C_in, H_in, W_in : input shape
//   KH, KW              : kernel size
//   stride_h, stride_w  : stride
//   pad_h, pad_w        : zero-padding on each side
//
// Output is laid out so that a single GEMM with a [C_out, C_in*KH*KW] weight
// matrix (treated as opB=T, K=C_in*KH*KW, N=C_out) produces a [N*H_out*W_out,
// C_out] result that matches PyTorch / Ascend conv2d output (in NHWC order;
// the engine reshapes as needed).
//
// H_out = (H_in + 2*pad_h - KH) / stride_h + 1
// W_out = (W_in + 2*pad_w - KW) / stride_w + 1
void launch_im2col_f32(const float *in, float *out,
                       int N, int C_in, int H_in, int W_in,
                       int KH, int KW,
                       int stride_h, int stride_w,
                       int pad_h, int pad_w,
                       int H_out, int W_out,
                       cudaStream_t stream);

// Per-channel bias add for conv outputs in [N, H, W, C] layout.
//   y[n, h, w, c] += bias[c]
void launch_add_bias_nhwc_f32(float *y, const float *bias,
                              int N, int H, int W, int C,
                              cudaStream_t stream);

// LayerNorm with affine: y = ((x - mean) / sqrt(var + eps)) * weight + bias.
//   x      : F32 [rows, cols]
//   weight : F32 [cols]
//   bias   : F32 [cols]
//   y      : F32 [rows, cols]
void launch_layernorm_affine_f32(const float *x, const float *weight,
                                 const float *bias, float *y,
                                 int rows, int cols, float eps,
                                 cudaStream_t stream);

// GELU-erf elementwise (in-place):
//   y = 0.5 * x * (1 + erf(x / sqrt(2)))
// More accurate than tanh approximation — matches ggml_gelu_erf used by the
// Ascend reference exactly.
void launch_gelu_erf_f32(float *x, int n, cudaStream_t stream);

// Naive joint attention (F32 in/out, F32 accum). Mirror of
// launch_attn_joint_naive_f32 from qwen_image_edit/dit_kernels.h, copied
// here so the audio encoder can build standalone (qwen_image_edit isn't
// deployed on zgx-3675).
//
//   q  : F32 [seq_total, n_heads, head_dim]
//   k  : F32 [seq_total, n_heads, head_dim]
//   v  : F32 [seq_total, n_heads, head_dim]
//   y  : F32 [seq_total, n_heads, head_dim]
//   inv_sqrt_d = 1 / sqrt(head_dim)
// Full bidirectional attention (no mask) — matches the Python reference
// (attention_mask=None, is_causal=False).
void launch_attn_joint_naive_f32(const float *q, const float *k,
                                  const float *v, float *y,
                                  int seq_total, int n_heads, int head_dim,
                                  float inv_sqrt_d, cudaStream_t stream);

// Add F32 bias (broadcast over rows) to F32 row-major [rows, cols].
//   y[r, c] += bias[c]
void launch_add_bias_f32_f32(float *y, const float *bias,
                              int rows, int cols, cudaStream_t stream);

// NHWC [N, H, W, C] → NCHW [N, C, H, W] transpose (F32).
void launch_nhwc_to_nchw_f32(const float *in, float *out,
                              int N, int H, int W, int C,
                              cudaStream_t stream);

// Add per-channel bias broadcast over (H, W) for NHWC layout — equivalent to
// launch_add_bias_f32_f32 with rows = N*H*W. Provided for clarity at conv
// call sites.
void launch_add_bias_rowmajor_f32(float *y, const float *bias,
                                    int rows, int cols, cudaStream_t stream);

// Extract per-frame slabs from NCHW [N, C, H, W] in (H outer, C inner) order:
//   out[(n*W + w) * H*C + h*C + c] = in[n*C*H*W + c*H*W + h*W + w]
// Produces a [N*W, H*C] matrix consumable by the conv_out projection.
void launch_nchw_to_frame_slab_hc(const float *in, float *out,
                                    int N, int C, int H, int W,
                                    cudaStream_t stream);

// Gather valid frames per chunk from a per-chunk conv_proj
// [chunk_num, frames_pc, d_model] tensor and add positional embedding.
//   chunk_offsets[ci] : starting concat-row for chunk ci
//   chunk_valid[ci]   : valid frame count for chunk ci
//   pos_emb           : F32 [max_source_pos, d_model]
//   concat_out        : F32 [total_frames, d_model]
void launch_gather_with_pos_emb(const float *conv_proj, const float *pos_emb,
                                  const int *chunk_offsets,
                                  const int *chunk_valid,
                                  int chunk_num, int frames_pc, int d_model,
                                  int max_source_pos, int total_frames,
                                  float *concat_out, cudaStream_t stream);

// Residual add: x[r, c] += delta[r, c].  F32 in/out.
void launch_resid_add_f32(float *x, const float *delta,
                            int rows, int cols, cudaStream_t stream);

}  // namespace ominix_cuda
