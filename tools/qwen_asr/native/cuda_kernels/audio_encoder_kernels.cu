// ============================================================================
// Audio encoder CUDA kernels — implementation. See audio_encoder_kernels.h.
// ============================================================================

#include "audio_encoder_kernels.h"

#include <cuda_runtime.h>
#include <math.h>

namespace ominix_cuda {

namespace {

// ---------------------------------------------------------------------------
// im2col F32: produces [N, H_out, W_out, C_in*KH*KW] from [N, C_in, H_in, W_in]
// One thread per output element (per (n, h_out, w_out, k)) where k = c_in*KH*KW
// index. We launch a 2D grid: (cols, N*H_out*W_out) with cols = C_in*KH*KW.
// ---------------------------------------------------------------------------
__global__ void im2col_f32_kernel(const float *__restrict__ in,
                                   float *__restrict__ out,
                                   int N, int C_in,
                                   int H_in, int W_in,
                                   int KH, int KW,
                                   int stride_h, int stride_w,
                                   int pad_h, int pad_w,
                                   int H_out, int W_out) {
    int col   = blockIdx.x * blockDim.x + threadIdx.x;  // [0, C_in*KH*KW)
    int row   = blockIdx.y;                              // [0, N*H_out*W_out)

    int patch_size = C_in * KH * KW;
    if (col >= patch_size) return;

    int total_rows = N * H_out * W_out;
    if (row >= total_rows) return;

    int n         = row / (H_out * W_out);
    int hw_out    = row % (H_out * W_out);
    int h_out     = hw_out / W_out;
    int w_out     = hw_out % W_out;

    int c_in     = col / (KH * KW);
    int kh_kw    = col % (KH * KW);
    int kh       = kh_kw / KW;
    int kw       = kh_kw % KW;

    int h_in = h_out * stride_h - pad_h + kh;
    int w_in = w_out * stride_w - pad_w + kw;

    float v = 0.0f;
    if (h_in >= 0 && h_in < H_in && w_in >= 0 && w_in < W_in) {
        size_t in_idx = (((size_t)n * C_in + c_in) * H_in + h_in) * W_in + w_in;
        v = in[in_idx];
    }
    size_t out_idx = (size_t)row * patch_size + col;
    out[out_idx] = v;
}

// ---------------------------------------------------------------------------
// Per-channel bias add for [N, H, W, C] layout (for conv NHWC outputs).
// ---------------------------------------------------------------------------
__global__ void add_bias_nhwc_f32_kernel(float *y, const float *bias,
                                          int total_rows, int C) {
    int c   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (c >= C || row >= total_rows) return;
    size_t idx = (size_t)row * C + c;
    y[idx] = y[idx] + bias[c];
}

// ---------------------------------------------------------------------------
// LayerNorm with affine. One block per row. Block size = 256.
// ---------------------------------------------------------------------------
constexpr int LN_AFFINE_BLOCK = 256;
__global__ void layernorm_affine_f32_kernel(const float *x, const float *weight,
                                              const float *bias, float *y,
                                              int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *xrow = x + (size_t)row * cols;
    float       *yrow = y + (size_t)row * cols;

    __shared__ float ssum[LN_AFFINE_BLOCK];
    __shared__ float ssq[LN_AFFINE_BLOCK];

    float lsum = 0.0f, lsq = 0.0f;
    for (int c = tid; c < cols; c += LN_AFFINE_BLOCK) {
        float v = xrow[c];
        lsum += v;
        lsq  += v * v;
    }
    ssum[tid] = lsum;
    ssq[tid]  = lsq;
    __syncthreads();
    for (int s = LN_AFFINE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssq[tid]  += ssq[tid + s];
        }
        __syncthreads();
    }
    float mean = ssum[0] / (float)cols;
    float var  = ssq[0] / (float)cols - mean * mean;
    if (var < 0.0f) var = 0.0f;
    float rstd = rsqrtf(var + eps);
    for (int c = tid; c < cols; c += LN_AFFINE_BLOCK) {
        float v = xrow[c];
        float w = weight[c];
        float b = bias[c];
        yrow[c] = (v - mean) * rstd * w + b;
    }
}

// ---------------------------------------------------------------------------
// GELU-erf elementwise (in-place).
//   y = 0.5 * x * (1 + erff(x / sqrt(2)))
// ---------------------------------------------------------------------------
__global__ void gelu_erf_f32_kernel(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    const float inv_sqrt2 = 0.70710678118654752440f;
    float e = erff(v * inv_sqrt2);
    x[idx] = 0.5f * v * (1.0f + e);
}

// ---------------------------------------------------------------------------
// Naive joint attention F32 — copy of qwen_image_edit/dit_kernels.cu's
// attn_joint_naive_f32_kernel, F32 in/out + F32 accum, full bidirectional
// attention (no mask, matches Python reference SDPA with attention_mask=None
// / is_causal=False).
// ---------------------------------------------------------------------------
constexpr int ATTN_BLOCK_F32 = 256;

__device__ __forceinline__ float warp_reduce_sum_f32(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_reduce_max_f32(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

__global__ void attn_joint_naive_f32_kernel(const float *q, const float *k,
                                              const float *v, float *y,
                                              int seq_total, int n_heads,
                                              int head_dim, float inv_sqrt_d) {
    extern __shared__ float scores_f32_[];
    int q_row = blockIdx.y;
    int head  = blockIdx.x;
    int tid   = threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;
    int n_warps = ATTN_BLOCK_F32 / 32;

    const float *q_vec =
        q + ((size_t)q_row * n_heads + head) * head_dim;

    __shared__ float q_sh[256];
    if (tid < head_dim) q_sh[tid] = q_vec[tid];
    __syncthreads();

    for (int t_base = 0; t_base < seq_total; t_base += n_warps) {
        int t = t_base + warp;
        if (t < seq_total) {
            const float *k_vec =
                k + ((size_t)t * n_heads + head) * head_dim;
            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += 32) {
                partial += q_sh[d] * k_vec[d];
            }
            partial = warp_reduce_sum_f32(partial);
            if (lane == 0) scores_f32_[t] = partial * inv_sqrt_d;
        }
    }
    __syncthreads();

    float lmax = -INFINITY;
    for (int t = tid; t < seq_total; t += ATTN_BLOCK_F32)
        lmax = fmaxf(lmax, scores_f32_[t]);
    lmax = warp_reduce_max_f32(lmax);
    __shared__ float warp_max_f32[ATTN_BLOCK_F32 / 32];
    if (lane == 0) warp_max_f32[warp] = lmax;
    __syncthreads();
    __shared__ float gmax_sh_f32;
    if (tid == 0) {
        float v = warp_max_f32[0];
        for (int w = 1; w < n_warps; ++w) v = fmaxf(v, warp_max_f32[w]);
        gmax_sh_f32 = v;
    }
    __syncthreads();
    float gmax = gmax_sh_f32;

    float lsum = 0.0f;
    for (int t = tid; t < seq_total; t += ATTN_BLOCK_F32) {
        float e = __expf(scores_f32_[t] - gmax);
        scores_f32_[t] = e;
        lsum += e;
    }
    lsum = warp_reduce_sum_f32(lsum);
    __shared__ float warp_sum_f32[ATTN_BLOCK_F32 / 32];
    if (lane == 0) warp_sum_f32[warp] = lsum;
    __syncthreads();
    __shared__ float gsum_sh_f32;
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < n_warps; ++w) s += warp_sum_f32[w];
        gsum_sh_f32 = s;
    }
    __syncthreads();
    float inv_denom = 1.0f / gsum_sh_f32;

    if (tid < head_dim) {
        float acc = 0.0f;
        for (int t = 0; t < seq_total; ++t) {
            const float *v_vec =
                v + ((size_t)t * n_heads + head) * head_dim;
            acc += scores_f32_[t] * inv_denom * v_vec[tid];
        }
        size_t out_idx = ((size_t)q_row * n_heads + head) * head_dim + tid;
        y[out_idx] = acc;
    }
}

// ---------------------------------------------------------------------------
// Add F32 bias broadcast over rows.
// ---------------------------------------------------------------------------
__global__ void add_bias_f32_f32_kernel(float *y, const float *bias,
                                          int rows, int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    y[idx] = y[idx] + bias[c];
}

// ---------------------------------------------------------------------------
// NHWC [N, H, W, C] → NCHW [N, C, H, W] transpose (F32).
// ---------------------------------------------------------------------------
__global__ void nhwc_to_nchw_f32_kernel(const float *in, float *out,
                                          int N, int H, int W, int C) {
    int c   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (c >= C) return;
    int total = N * H * W;
    if (row >= total) return;
    int n  = row / (H * W);
    int hw = row % (H * W);
    int h  = hw / W;
    int w  = hw % W;
    size_t in_idx  = ((((size_t)n * H + h) * W + w) * C + c);
    size_t out_idx = (((size_t)n * C + c) * H + h) * W + w;
    out[out_idx] = in[in_idx];
}

// ---------------------------------------------------------------------------
// Per-channel bias add for [rows, cols] row-major (rows = N*H*W in NHWC).
// ---------------------------------------------------------------------------
__global__ void add_bias_rowmajor_f32_kernel(float *y, const float *bias,
                                                int rows, int cols) {
    int c   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (c >= cols || row >= rows) return;
    size_t idx = (size_t)row * cols + c;
    y[idx] = y[idx] + bias[c];
}

// ---------------------------------------------------------------------------
// NCHW → per-frame slab in (H outer, C inner) ggml order.
//   out[(n*W + w) * H*C + h*C + c] = in[n*C*H*W + c*H*W + h*W + w]
// ---------------------------------------------------------------------------
__global__ void nchw_to_frame_slab_hc_kernel(const float *in, float *out,
                                                 int N, int C, int H, int W) {
    int hc  = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (hc >= H * C) return;
    int total_rows = N * W;
    if (row >= total_rows) return;
    int n = row / W;
    int w = row % W;
    int h = hc / C;
    int c = hc % C;
    size_t in_idx  = (((size_t)n * C + c) * H + h) * W + w;
    size_t out_idx = (size_t)row * H * C + hc;
    out[out_idx] = in[in_idx];
}

// ---------------------------------------------------------------------------
// Gather valid frames per chunk + add positional embedding.
// ---------------------------------------------------------------------------
__global__ void gather_with_pos_emb_kernel(
    const float *conv_proj, const float *pos_emb,
    const int *chunk_offsets, const int *chunk_valid,
    int chunk_num, int frames_pc, int d_model, int max_source_pos,
    float *concat_out) {
    int c   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (c >= d_model) return;

    int found_chunk = -1;
    int found_local = 0;
    for (int ci = 0; ci < chunk_num; ++ci) {
        int off  = chunk_offsets[ci];
        int vlen = chunk_valid[ci];
        if (row >= off && row < off + vlen) {
            found_chunk = ci;
            found_local = row - off;
            break;
        }
    }
    if (found_chunk < 0) return;

    int pos = found_local;
    if (pos >= max_source_pos) pos = max_source_pos - 1;

    size_t cp_idx =
        ((size_t)found_chunk * frames_pc + found_local) * d_model + c;
    size_t pe_idx = (size_t)pos * d_model + c;
    size_t out_idx = (size_t)row * d_model + c;
    concat_out[out_idx] = conv_proj[cp_idx] + pos_emb[pe_idx];
}

// ---------------------------------------------------------------------------
// Residual add F32: x[r, c] += delta[r, c].
// ---------------------------------------------------------------------------
__global__ void resid_add_f32_kernel(float *x, const float *delta,
                                       int rows, int cols) {
    int c   = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (c >= cols || row >= rows) return;
    size_t idx = (size_t)row * cols + c;
    x[idx] = x[idx] + delta[idx];
}

}  // namespace

// ===========================================================================
// Launchers
// ===========================================================================

void launch_im2col_f32(const float *in, float *out,
                       int N, int C_in, int H_in, int W_in,
                       int KH, int KW,
                       int stride_h, int stride_w,
                       int pad_h, int pad_w,
                       int H_out, int W_out,
                       cudaStream_t stream) {
    if (N <= 0 || C_in <= 0 || H_in <= 0 || W_in <= 0) return;
    int patch_size = C_in * KH * KW;
    int total_rows = N * H_out * W_out;
    dim3 block(256);
    dim3 grid((patch_size + 255) / 256, total_rows);
    im2col_f32_kernel<<<grid, block, 0, stream>>>(in, out, N, C_in,
                                                   H_in, W_in, KH, KW,
                                                   stride_h, stride_w,
                                                   pad_h, pad_w, H_out, W_out);
}

void launch_add_bias_nhwc_f32(float *y, const float *bias,
                               int N, int H, int W, int C,
                               cudaStream_t stream) {
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0) return;
    int total_rows = N * H * W;
    dim3 block(256);
    dim3 grid((C + 255) / 256, total_rows);
    add_bias_nhwc_f32_kernel<<<grid, block, 0, stream>>>(y, bias, total_rows, C);
}

void launch_layernorm_affine_f32(const float *x, const float *weight,
                                  const float *bias, float *y,
                                  int rows, int cols, float eps,
                                  cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    layernorm_affine_f32_kernel<<<rows, LN_AFFINE_BLOCK, 0, stream>>>(
        x, weight, bias, y, cols, eps);
}

void launch_gelu_erf_f32(float *x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    gelu_erf_f32_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

void launch_attn_joint_naive_f32(const float *q, const float *k,
                                   const float *v, float *y,
                                   int seq_total, int n_heads, int head_dim,
                                   float inv_sqrt_d, cudaStream_t stream) {
    if (seq_total <= 0 || n_heads <= 0 || head_dim <= 0) return;
    dim3 grid(n_heads, seq_total);
    dim3 block(ATTN_BLOCK_F32);
    size_t shmem = (size_t)seq_total * sizeof(float);
    attn_joint_naive_f32_kernel<<<grid, block, shmem, stream>>>(
        q, k, v, y, seq_total, n_heads, head_dim, inv_sqrt_d);
}

void launch_add_bias_f32_f32(float *y, const float *bias,
                              int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    add_bias_f32_f32_kernel<<<grid, block, 0, stream>>>(y, bias, rows, cols);
}

void launch_nhwc_to_nchw_f32(const float *in, float *out,
                              int N, int H, int W, int C,
                              cudaStream_t stream) {
    if (N <= 0 || H <= 0 || W <= 0 || C <= 0) return;
    dim3 block(256);
    dim3 grid((C + 255) / 256, N * H * W);
    nhwc_to_nchw_f32_kernel<<<grid, block, 0, stream>>>(in, out, N, H, W, C);
}

void launch_add_bias_rowmajor_f32(float *y, const float *bias,
                                    int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    add_bias_rowmajor_f32_kernel<<<grid, block, 0, stream>>>(y, bias, rows, cols);
}

void launch_nchw_to_frame_slab_hc(const float *in, float *out,
                                    int N, int C, int H, int W,
                                    cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return;
    int total_rows = N * W;
    int hc = H * C;
    dim3 block(256);
    dim3 grid((hc + 255) / 256, total_rows);
    nchw_to_frame_slab_hc_kernel<<<grid, block, 0, stream>>>(in, out, N, C, H, W);
}

void launch_gather_with_pos_emb(const float *conv_proj, const float *pos_emb,
                                  const int *chunk_offsets,
                                  const int *chunk_valid,
                                  int chunk_num, int frames_pc, int d_model,
                                  int max_source_pos, int total_frames,
                                  float *concat_out, cudaStream_t stream) {
    if (total_frames <= 0 || d_model <= 0) return;
    dim3 block(256);
    dim3 grid((d_model + 255) / 256, total_frames);
    gather_with_pos_emb_kernel<<<grid, block, 0, stream>>>(
        conv_proj, pos_emb, chunk_offsets, chunk_valid, chunk_num, frames_pc,
        d_model, max_source_pos, concat_out);
}

void launch_resid_add_f32(float *x, const float *delta,
                            int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    resid_add_f32_kernel<<<grid, block, 0, stream>>>(x, delta, rows, cols);
}

}  // namespace ominix_cuda
