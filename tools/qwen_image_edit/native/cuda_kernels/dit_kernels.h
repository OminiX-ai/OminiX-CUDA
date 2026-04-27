#pragma once
// ============================================================================
// DiT block CUDA kernels for ImageDiffusionCudaEngine (Phase 3.2 forward).
//
// Phase 3.2 scope is a single-block joint forward at 1024² shape (img_seq =
// 4096, txt_seq = 256). The goal is a finite-output smoke verdict, NOT
// throughput. Numeric strategy:
//   - Activations stay in F16 device buffers (one upload at the entry, one
//     download at the exit). All matmuls are cuBLAS GemmEx with F32 accum.
//   - LayerNorm, RMSNorm and AdaLN modulation reductions all run in F32 to
//     match the Ascend reference (image_diffusion_engine.cpp:3380, 3460).
//   - The joint-RoPE table is built on host with multi-axis layout (Qwen-Image
//     temporal=16 + h=56 + w=56) and uploaded once per forward.
//   - Joint cross-attention is implemented as a naive O(seq²) kernel per Q
//     head with shmem-resident scores. Adequate for smoke; Phase 3.3+ will
//     swap to cuDNN FMHA / fused flash kernel.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ominix_cuda {

// AdaLN-modulate (in-place):  x = x * (1 + scale) + shift
// Layout:
//   x      : F16 [rows, cols]   (rows = seq, cols = hidden)
//   scale  : F16 [cols]         (broadcast over rows)
//   shift  : F16 [cols]         (broadcast over rows)
void launch_adaln_modulate_f16(__half *x, const __half *scale,
                               const __half *shift, int rows, int cols,
                               cudaStream_t stream);

// Gated residual add (F16 in/out, F32 internal accumulator):
//   y[r,c] = x[r,c] + delta[r,c] * gate[c]
//   x      : F16 [rows, cols]   (residual in/out — aliased y)
//   delta  : F16 [rows, cols]
//   gate   : F16 [cols]
void launch_gated_residual_add_f16(__half *x, const __half *delta,
                                   const __half *gate, int rows, int cols,
                                   cudaStream_t stream);

// LayerNorm (affine=false, no gamma/beta).
//   y[r,c] = (x[r,c] - mean(row)) / sqrt(var(row) + eps)
// All math in F32; F16 read / F16 write.
//   x : F16 [rows, cols]
//   y : F16 [rows, cols]
void launch_layernorm_noaffine_f16(const __half *x, __half *y,
                                   int rows, int cols, float eps,
                                   cudaStream_t stream);

// Head-wise RMSNorm: views x as [rows * n_heads, head_dim] and norms each
// head_dim slice with a shared F32 gamma[head_dim].
//   x      : F16 [rows, n_heads, head_dim]   (contiguous)
//   gamma  : F32 [head_dim]
//   y      : F16 [rows, n_heads, head_dim]
void launch_rmsnorm_head_f16_g32(const __half *x, const float *gamma,
                                 __half *y, int rows, int n_heads,
                                 int head_dim, float eps,
                                 cudaStream_t stream);

// NEOX-mode RoPE applied over a [rows, n_heads, head_dim] tile, using a
// per-row cos/sin table:
//   cos    : F16 [rows, head_dim/2]    (one row per token; head_dim/2 entries)
//   sin    : F16 [rows, head_dim/2]
// For row r, head h, pair (i, i+half):
//   y[r,h,i     ] = x[r,h,i     ] * cos[r,i] - x[r,h,i+half] * sin[r,i]
//   y[r,h,i+half] = x[r,h,i     ] * sin[r,i] + x[r,h,i+half] * cos[r,i]
// In-place permitted (y == x).
void launch_rope_neox_seq_f16(const __half *x, const __half *cos,
                              const __half *sin, __half *y,
                              int rows, int n_heads, int head_dim,
                              cudaStream_t stream);

// Phase 3.3a multi-axis NEOX RoPE.  Same math as launch_rope_neox_seq_f16,
// but the cos/sin table covers a global pe-table of shape
// [pe_total_rows, head_dim/2] and each output row picks its pe-row by
// `pe_off + row`. This lets the joint sequence pick disjoint pe ranges:
//   txt rows  → pe_off = 0
//   img rows  → pe_off = max_txt_seq
// (matches Ascend `compute_qwen_rope_pe_host` 3-axis layout: txt positions
//  occupy [0..max_txt_seq), img positions occupy [max_txt_seq..max_txt_seq+
//  img_tokens). The host fills cos/sin per multi-axis assignment so this
//  kernel stays generic.)
//
//   x        : F16 [rows, n_heads, head_dim]    (Q or K tile for one stream)
//   cos      : F16 [pe_total_rows, head_dim/2]  (engine pe_cos_dev_)
//   sin      : F16 [pe_total_rows, head_dim/2]  (engine pe_sin_dev_)
//   y        : F16 [rows, n_heads, head_dim]    (in-place permitted)
//   pe_off   : int — additive row offset into cos/sin
void launch_rope_neox_3axis_f16(const __half *x, const __half *cos,
                                const __half *sin, __half *y,
                                int rows, int n_heads, int head_dim,
                                int pe_off, cudaStream_t stream);

// Sigmoid-Linear-Unit (silu / swish) in place.
//   x : F16 [n]      —  y[i] = x[i] / (1 + exp(-x[i]))
void launch_silu_f16(__half *x, int n, cudaStream_t stream);

// GELU-tanh elementwise (in-place).
//   x : F16 [n]
void launch_gelu_tanh_f16(__half *x, int n, cudaStream_t stream);

// Add F32 bias to F16 row vector (broadcast over rows).
//   y[r,c] += bias[c]    (bias F32, y F16, in-place add)
void launch_add_bias_f32_f16(__half *y, const float *bias,
                             int rows, int cols, cudaStream_t stream);

// F32 -> F16 cast.
void launch_cast_f32_to_f16_dit(const float *in, __half *out, int n,
                                cudaStream_t stream);

// F16 -> F32 cast.
void launch_cast_f16_to_f32_dit(const __half *in, float *out, int n,
                                cudaStream_t stream);

// Naive joint cross-attention (smoke-grade, F16 in/out, F32 accum):
//   q  : F16 [seq_total, n_heads, head_dim]
//   k  : F16 [seq_total, n_heads, head_dim]   (no GQA — full head-count K)
//   v  : F16 [seq_total, n_heads, head_dim]
//   y  : F16 [seq_total, n_heads, head_dim]
//   inv_sqrt_d = 1 / sqrt(head_dim)
// One block per (q_row, head). Each block computes scores[seq_total] in
// dynamic shmem, softmaxes them, then weighted-sums V. Adequate for smoke
// at seq_total ≤ 8192.
void launch_attn_joint_naive_f16(const __half *q, const __half *k,
                                 const __half *v, __half *y,
                                 int seq_total, int n_heads, int head_dim,
                                 float inv_sqrt_d, cudaStream_t stream);

}  // namespace ominix_cuda
