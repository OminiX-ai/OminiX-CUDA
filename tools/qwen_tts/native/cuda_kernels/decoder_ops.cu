// ============================================================================
// SpeechTokenizerDecoder F32 ops — Phase 2.7b.
//
// All kernels operate on row-major [T, C] tensors with C fastest, matching the
// RVQ output layout produced in Phase 2.7a. Channels-fastest is also what
// cuBLAS GEMM expects when treating the host buffer as col-major [C, T].
//
// Ops landed in this file:
//   - causal_conv1d_im2col_f32 : zero-pad-left + im2col for k=3 conv. Output
//                                 shape [T, k * C_in] feeds straight into a
//                                 cuBLAS Sgemm against W2D[k*C_in, C_out].
//   - depthwise_conv1d_causal_f32 : depthwise k=7 conv with bias. One thread
//                                    per (t, c). Reads its own k taps from a
//                                    left-padded virtual input (no scratch).
//   - conv_transpose1d_k2s2_f32 : ConvTranspose1d with kernel=2 / stride=2.
//                                  Maps input[T, C_in] @ Wflat[C_in, 2*C_out]
//                                  into out[2*T, C_out] via a custom kernel
//                                  that interleaves the two strides. (We avoid
//                                  the cublas-then-shuffle path; T*C_out is
//                                  small enough that one fused kernel wins.)
//   - layernorm_f32 : per-row LayerNorm with affine gamma/beta. Channels are
//                     the normalized axis (last dim).
//   - gelu_erf_f32 : exact erf-form GELU, in-place over a flat [n] buffer.
//   - bias_add_f32 : per-channel bias broadcast over [T, C].
//   - residual_add_f32 : y = a + b, flat [n].
//
// All kernel launchers are F32 because the GGUF tensors for the speech-tokenizer
// decoder ship as F32, and Phase 2.7b stays F32 end-to-end. The vocoder stage
// (2.7c) keeps F32 too — F16 only buys us cache footprint savings, and the
// shapes here are already small (T ≤ 256, C ≤ 4096).
// ============================================================================

#include "cuda_kernels.h"

#include <cuda_runtime.h>

namespace ominix_cuda {

// ---------------------------------------------------------------------------
// Causal Conv1d im2col (kernel `K`, channel `C_in`, T time steps).
//
// in  : [T, C_in]   row-major
// out : [T, K * C_in]   row-major
//
// out[t, k*C_in + c] = (t + k - (K-1) >= 0) ? in[t + k - (K-1), c] : 0
//
// I.e., we left-pad with K-1 zeros, then for each output time step t take a
// window of K rows starting at t+0 (post-pad). This produces the "im2col"
// matrix that, when multiplied by W2D[K*C_in, C_out] (row-major [k, c_in, c_out]
// flattened over the first two axes), yields the causal conv output.
// ---------------------------------------------------------------------------
namespace {

__global__ void causal_conv1d_im2col_f32_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    int T, int C_in, int K) {
    int t = blockIdx.x;
    int kc = blockIdx.y * blockDim.x + threadIdx.x;
    int total_kc = K * C_in;
    if (t >= T || kc >= total_kc) return;

    int k = kc / C_in;
    int c = kc - k * C_in;
    int src_t = t + k - (K - 1);  // shift so that k=K-1 is "current" sample
    float v = 0.0f;
    if (src_t >= 0 && src_t < T) {
        v = in[(size_t)src_t * C_in + c];
    }
    out[(size_t)t * total_kc + kc] = v;
}

}  // namespace

void launch_causal_conv1d_im2col_f32(const float *in, float *out,
                                      int T, int C_in, int K,
                                      cudaStream_t stream) {
    int total_kc = K * C_in;
    int block = 256;
    dim3 grid(T, (total_kc + block - 1) / block);
    causal_conv1d_im2col_f32_kernel<<<grid, block, 0, stream>>>(
        in, out, T, C_in, K);
}

// ---------------------------------------------------------------------------
// Depthwise causal Conv1d, kernel K, channels C, no groups (1 filter / chan).
//
// in     : [T, C]   row-major
// w      : [K, C]   row-major  (GGUF stores [K, 1, C], the singleton "in_per_group"
//                                axis is implicit so we skip it.)
// b      : [C]      (or nullptr)
// out    : [T, C]   row-major
//
// out[t, c] = b[c] + Σ_{k=0..K-1} in_padded[t + k, c] * w[k, c]
//   where in_padded prepends K-1 zeros to the time axis (causal).
//
// One thread per (t, c). For K=7 and C=1024 this is 7K * T threads, fine. The
// inner K-loop unrolls trivially since K is a launch-time constant in practice
// (we always pass K=7 for ConvNeXt) but we keep it dynamic for future blocks.
// ---------------------------------------------------------------------------
namespace {

__global__ void depthwise_conv1d_causal_f32_kernel(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ b, float *__restrict__ out,
    int T, int C, int K) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;

    float acc = (b != nullptr) ? b[c] : 0.0f;
    // shift so that k=K-1 reads in[t, c] (current sample) — matches the
    // Ascend `build_depthwise_conv1d_causal` shift semantics.
    #pragma unroll 1
    for (int k = 0; k < K; ++k) {
        int src_t = t + k - (K - 1);
        if (src_t >= 0) {
            acc += in[(size_t)src_t * C + c] * w[(size_t)k * C + c];
        }
    }
    out[(size_t)t * C + c] = acc;
}

}  // namespace

void launch_depthwise_conv1d_causal_f32(const float *in, const float *w,
                                         const float *b, float *out,
                                         int T, int C, int K,
                                         cudaStream_t stream) {
    int block = 128;
    dim3 grid(T, (C + block - 1) / block);
    depthwise_conv1d_causal_f32_kernel<<<grid, block, 0, stream>>>(
        in, w, b, out, T, C, K);
}

// ---------------------------------------------------------------------------
// ConvTranspose1d with kernel=2, stride=2 (the upsample.X.0.conv shape).
//
// PyTorch ConvTranspose1d weight layout: [in_channels, out_channels, kernel].
// GGUF stores axes reversed → [kernel, out_channels, in_channels] row-major.
// So w[k, oc, ic] is contiguous with ic fastest.
//
// in   : [T,    C_in]   row-major
// w    : [K=2,  C_out, C_in]   row-major (K outer, C_in inner)
// b    : [C_out]   (or nullptr)
// out  : [T*2,  C_out]   row-major
//
// out[2*t + k, oc] = b[oc] + Σ_{ic} in[t, ic] * w[k, oc, ic]
//
// One block per output time step, one thread per (k, oc). Each thread sweeps
// the C_in axis. C_in = C_out = 1024 → 2*1024 = 2048 threads/block which
// exceeds the limit, so we tile oc by `OC_PER_BLOCK`.
// ---------------------------------------------------------------------------
namespace {

__global__ void conv_transpose1d_k2s2_f32_kernel(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ b, float *__restrict__ out,
    int T, int C_in, int C_out) {
    int t = blockIdx.x;          // input time index
    int k = blockIdx.y;          // 0 or 1 (kernel position)
    int oc = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T || k >= 2 || oc >= C_out) return;

    const float *in_row = in + (size_t)t * C_in;
    const float *w_row  = w  + ((size_t)k * C_out + oc) * C_in;

    float acc = (b != nullptr) ? b[oc] : 0.0f;
    #pragma unroll 4
    for (int ic = 0; ic < C_in; ++ic) {
        acc += in_row[ic] * w_row[ic];
    }
    out[(size_t)(2 * t + k) * C_out + oc] = acc;
}

}  // namespace

void launch_conv_transpose1d_k2s2_f32(const float *in, const float *w,
                                       const float *b, float *out,
                                       int T, int C_in, int C_out,
                                       cudaStream_t stream) {
    int block = 128;
    int oc_blocks = (C_out + block - 1) / block;
    dim3 grid(T, 2, oc_blocks);
    conv_transpose1d_k2s2_f32_kernel<<<grid, block, 0, stream>>>(
        in, w, b, out, T, C_in, C_out);
}

// ---------------------------------------------------------------------------
// LayerNorm with affine (gamma, beta). Normalizes over the last (channel)
// dimension. Each block handles one row.
//
// x       : [T, C]   row-major
// gamma   : [C]
// beta    : [C]   (or nullptr)
// y       : [T, C]   row-major (may alias x)
// eps     : standard 1e-5
//
// One block per row. Two-pass reduction (mean → variance → normalize).
// ---------------------------------------------------------------------------
namespace {

__global__ void layernorm_f32_kernel(const float *__restrict__ x,
                                       const float *__restrict__ gamma,
                                       const float *__restrict__ beta,
                                       float *__restrict__ y,
                                       int T, int C, float eps) {
    int t = blockIdx.x;
    if (t >= T) return;
    const float *xrow = x + (size_t)t * C;
    float *yrow = y + (size_t)t * C;

    extern __shared__ float smem[];   // 2 * blockDim.x

    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // ---- Pass 1: mean ----
    float sum = 0.0f;
    for (int i = tid; i < C; i += nthreads) sum += xrow[i];
    smem[tid] = sum;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / (float)C;
    __syncthreads();

    // ---- Pass 2: variance ----
    float sq = 0.0f;
    for (int i = tid; i < C; i += nthreads) {
        float d = xrow[i] - mean;
        sq += d * d;
    }
    smem[tid] = sq;
    __syncthreads();
    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float var = smem[0] / (float)C;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // ---- Pass 3: normalize + affine ----
    for (int i = tid; i < C; i += nthreads) {
        float v = (xrow[i] - mean) * inv_std;
        v = v * gamma[i];
        if (beta) v += beta[i];
        yrow[i] = v;
    }
}

}  // namespace

void launch_layernorm_f32(const float *x, const float *gamma, const float *beta,
                           float *y, int T, int C, float eps,
                           cudaStream_t stream) {
    int block = 256;
    size_t smem = sizeof(float) * block;
    layernorm_f32_kernel<<<T, block, smem, stream>>>(x, gamma, beta, y,
                                                       T, C, eps);
}

// ---------------------------------------------------------------------------
// GELU (erf form, the PyTorch default for ConvNeXt blocks).
//   y = 0.5 * x * (1 + erf(x / sqrt(2)))
// In-place permitted (y == x).
// ---------------------------------------------------------------------------
namespace {

__global__ void gelu_erf_f32_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float x = in[idx];
    // 1/sqrt(2) ≈ 0.70710678f
    out[idx] = 0.5f * x * (1.0f + erff(x * 0.70710678118654752440f));
}

}  // namespace

void launch_gelu_erf_f32(const float *in, float *out, int n,
                          cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    gelu_erf_f32_kernel<<<grid, block, 0, stream>>>(in, out, n);
}

// ---------------------------------------------------------------------------
// Bias add over [T, C]: y[t, c] = x[t, c] + bias[c].
// In-place permitted (y == x).
// ---------------------------------------------------------------------------
namespace {

__global__ void bias_add_f32_kernel(const float *x, const float *bias,
                                      float *y, int T, int C) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    y[(size_t)t * C + c] = x[(size_t)t * C + c] + bias[c];
}

}  // namespace

void launch_bias_add_f32(const float *x, const float *bias, float *y,
                          int T, int C, cudaStream_t stream) {
    int block = 256;
    dim3 grid(T, (C + block - 1) / block);
    bias_add_f32_kernel<<<grid, block, 0, stream>>>(x, bias, y, T, C);
}

// ---------------------------------------------------------------------------
// Channel-wise multiply (gamma scaling for ConvNeXt residual): y = x * gamma.
// gamma is [C], x/y are [T, C] row-major. In-place permitted.
// ---------------------------------------------------------------------------
namespace {

__global__ void channel_scale_f32_kernel(const float *x, const float *gamma,
                                           float *y, int T, int C) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;
    y[(size_t)t * C + c] = x[(size_t)t * C + c] * gamma[c];
}

}  // namespace

void launch_channel_scale_f32(const float *x, const float *gamma, float *y,
                               int T, int C, cudaStream_t stream) {
    int block = 256;
    dim3 grid(T, (C + block - 1) / block);
    channel_scale_f32_kernel<<<grid, block, 0, stream>>>(x, gamma, y, T, C);
}

// ---------------------------------------------------------------------------
// y = a + b, flat [n] F32. Used for the ConvNeXt residual.
// ---------------------------------------------------------------------------
namespace {

__global__ void add_f32_kernel(const float *a, const float *b, float *y,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = a[idx] + b[idx];
}

}  // namespace

void launch_add_f32(const float *a, const float *b, float *y, int n,
                     cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    add_f32_kernel<<<grid, block, 0, stream>>>(a, b, y, n);
}

// ---------------------------------------------------------------------------
// Phase 2.7c: dilated causal Conv1d im2col.
//
// Same as causal_conv1d_im2col but with dilation D. The Ascend reference
// build_causal_conv1d zero-pads on the left by (K-1)*D, then runs an
// undilated conv on the padded input (effectively reading every D-th sample
// for the K kernel taps). We mirror that here:
//
//   out[t, k*C_in + c] = in[t + (k - (K-1)) * D, c]   (zero if out of range)
//
// For the vocoder residual units this is called with K=7 and D in {1, 3, 9}.
// ---------------------------------------------------------------------------
namespace {

__global__ void dilated_causal_conv1d_im2col_f32_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    int T, int C_in, int K, int D) {
    int t = blockIdx.x;
    int kc = blockIdx.y * blockDim.x + threadIdx.x;
    int total_kc = K * C_in;
    if (t >= T || kc >= total_kc) return;

    int k = kc / C_in;
    int c = kc - k * C_in;
    int src_t = t + (k - (K - 1)) * D;  // current sample at k=K-1
    float v = 0.0f;
    if (src_t >= 0 && src_t < T) {
        v = in[(size_t)src_t * C_in + c];
    }
    out[(size_t)t * total_kc + kc] = v;
}

}  // namespace

void launch_dilated_causal_conv1d_im2col_f32(const float *in, float *out,
                                              int T, int C_in, int K, int D,
                                              cudaStream_t stream) {
    int total_kc = K * C_in;
    int block = 256;
    dim3 grid(T, (total_kc + block - 1) / block);
    dilated_causal_conv1d_im2col_f32_kernel<<<grid, block, 0, stream>>>(
        in, out, T, C_in, K, D);
}

// ---------------------------------------------------------------------------
// Phase 2.7c: generic causal ConvTranspose1d (stride S, kernel K).
//
// PyTorch semantics for ConvTranspose1d:
//   y_pre[t*S + k, oc] = sum_ic in[t, ic] * w[k, oc, ic]
// Then ggml_conv_transpose_1d returns a tensor of length (T-1)*S + K, which
// we trim by removing (K-S) samples from the RIGHT to keep the conv causal.
// The Ascend reference (build_causal_transconv1d) uses k = 2*S in practice
// (so trim = S samples), giving final length T*S.
//
// Implementation: one block per (t, k); threads sweep oc. Each thread does a
// dot over C_in. Output index validity is enforced by `trim` mask.
//
// Note: shapes for the vocoder upsample come in pairs (k, S):
//   block 0: K=16, S=8   -> trim=8, out_len = T*8
//   block 1: K=10, S=5   -> trim=5, out_len = T*5
//   block 2: K=8,  S=4   -> trim=4, out_len = T*4
//   block 3: K=6,  S=3   -> trim=3, out_len = T*3
// ---------------------------------------------------------------------------
namespace {

__global__ void causal_conv_transpose1d_f32_kernel(
    const float *__restrict__ in, const float *__restrict__ w,
    const float *__restrict__ b, float *__restrict__ out,
    int T, int C_in, int C_out, int K, int S, int out_len) {
    int t  = blockIdx.x;
    int k  = blockIdx.y;
    int oc = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T || k >= K || oc >= C_out) return;

    int dst_t = t * S + k;
    if (dst_t >= out_len) return;  // trimmed-right region

    const float *in_row = in + (size_t)t * C_in;
    const float *w_row  = w  + ((size_t)k * C_out + oc) * C_in;

    float acc = 0.0f;
    #pragma unroll 4
    for (int ic = 0; ic < C_in; ++ic) {
        acc += in_row[ic] * w_row[ic];
    }
    // ConvTranspose1d *accumulates* across overlapping (t, k) → same dst_t.
    // We use atomicAdd so multiple blocks can land on the same output cell.
    atomicAdd(&out[(size_t)dst_t * C_out + oc], acc);

    // Bias: only one block per (dst_t, oc) is responsible. Since multiple
    // (t, k) collide, pick the canonical writer: the block where k == 0 and
    // (k == 0 or t == 0) -- but that double-counts. Cleaner: bias is applied
    // by a separate kernel after this one.
}

__global__ void zero_f32_kernel(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = 0.0f;
}

__global__ void add_bias_inplace_f32_kernel(float *out, const float *b,
                                              int N, int C_out) {
    int i = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= N || c >= C_out) return;
    out[(size_t)i * C_out + c] += b[c];
}

}  // namespace

void launch_causal_conv_transpose1d_f32(const float *in, const float *w,
                                          const float *b, float *out,
                                          int T, int C_in, int C_out,
                                          int K, int S,
                                          cudaStream_t stream) {
    int out_len = T * S;  // post-trim length (Ascend convention with k = 2*S)
    // Zero the output first (atomicAdd accumulates).
    int n_out = out_len * C_out;
    {
        int block = 256;
        int grid = (n_out + block - 1) / block;
        zero_f32_kernel<<<grid, block, 0, stream>>>(out, n_out);
    }
    {
        int block = 128;
        int oc_blocks = (C_out + block - 1) / block;
        dim3 grid(T, K, oc_blocks);
        causal_conv_transpose1d_f32_kernel<<<grid, block, 0, stream>>>(
            in, w, b, out, T, C_in, C_out, K, S, out_len);
    }
    if (b != nullptr) {
        int block = 128;
        dim3 grid(out_len, (C_out + block - 1) / block);
        add_bias_inplace_f32_kernel<<<grid, block, 0, stream>>>(
            out, b, out_len, C_out);
    }
}

// ---------------------------------------------------------------------------
// Phase 2.7c: SnakeBeta activation.
//   y[t, c] = x[t, c] + exp(-beta[c]) * sin(x[t, c] * exp(alpha[c]))^2
// alpha, beta are [C] learnable per-channel.
// In-place permitted (y == x).
// ---------------------------------------------------------------------------
namespace {

__global__ void snake_beta_f32_kernel(const float *__restrict__ x,
                                        const float *__restrict__ alpha,
                                        const float *__restrict__ beta,
                                        float *__restrict__ y,
                                        int T, int C) {
    int t = blockIdx.x;
    int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T || c >= C) return;

    float xv = x[(size_t)t * C + c];
    float a  = expf(alpha[c]);
    float ib = expf(-beta[c]);
    float s  = sinf(xv * a);
    y[(size_t)t * C + c] = xv + ib * (s * s);
}

}  // namespace

void launch_snake_beta_f32(const float *x, const float *alpha,
                            const float *beta, float *y,
                            int T, int C, cudaStream_t stream) {
    int block = 128;
    dim3 grid(T, (C + block - 1) / block);
    snake_beta_f32_kernel<<<grid, block, 0, stream>>>(x, alpha, beta, y, T, C);
}

// ---------------------------------------------------------------------------
// Tanh elementwise.
// ---------------------------------------------------------------------------
namespace {

__global__ void tanh_f32_kernel(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = tanhf(x[i]);
}

}  // namespace

void launch_tanh_f32(const float *x, float *y, int n, cudaStream_t stream) {
    int block = 256;
    int grid = (n + block - 1) / block;
    tanh_f32_kernel<<<grid, block, 0, stream>>>(x, y, n);
}

}  // namespace ominix_cuda
