// ============================================================================
// DiT block CUDA kernels — implementation. See dit_kernels.h for the API and
// numeric strategy. Phase 3.2 smoke-grade (correctness > throughput).
// ============================================================================

#include "dit_kernels.h"

namespace ominix_cuda {

namespace {

// ---------------------------------------------------------------------------
// AdaLN modulate: x = x * (1 + scale) + shift   (broadcast scale/shift over rows)
// ---------------------------------------------------------------------------
__global__ void adaln_modulate_kernel(__half *x, const __half *scale,
                                       const __half *shift, int rows, int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    float xv = __half2float(x[idx]);
    float sv = __half2float(scale[c]);
    float shv = __half2float(shift[c]);
    x[idx] = __float2half(xv * (1.0f + sv) + shv);
}

// ---------------------------------------------------------------------------
// Gated residual add: x += delta * gate   (gate broadcast over rows)
// ---------------------------------------------------------------------------
__global__ void gated_residual_add_kernel(__half *x, const __half *delta,
                                           const __half *gate, int rows,
                                           int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    float xv = __half2float(x[idx]);
    float dv = __half2float(delta[idx]);
    float gv = __half2float(gate[c]);
    x[idx] = __float2half(xv + dv * gv);
}

// ---------------------------------------------------------------------------
// LayerNorm (no affine).  One block per row. Block size = 256.
// ---------------------------------------------------------------------------
constexpr int LN_BLOCK = 256;
__global__ void layernorm_noaffine_kernel(const __half *x, __half *y,
                                           int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const __half *xrow = x + (size_t)row * cols;
    __half *yrow       = y + (size_t)row * cols;

    __shared__ float ssum[LN_BLOCK];
    __shared__ float ssq[LN_BLOCK];

    float lsum = 0.0f, lsq = 0.0f;
    for (int c = tid; c < cols; c += LN_BLOCK) {
        float v = __half2float(xrow[c]);
        lsum += v;
        lsq  += v * v;
    }
    ssum[tid] = lsum;
    ssq[tid]  = lsq;
    __syncthreads();
    for (int s = LN_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum[tid] += ssum[tid + s];
            ssq[tid]  += ssq[tid + s];
        }
        __syncthreads();
    }
    float mean = ssum[0] / (float)cols;
    float var  = ssq[0] / (float)cols - mean * mean;
    float rstd = rsqrtf(var + eps);
    for (int c = tid; c < cols; c += LN_BLOCK) {
        float v = __half2float(xrow[c]);
        yrow[c] = __float2half((v - mean) * rstd);
    }
}

// ---------------------------------------------------------------------------
// Head-wise RMSNorm. One block per (token, head). head_dim threads (≤256).
// ---------------------------------------------------------------------------
__global__ void rmsnorm_head_kernel(const __half *x, const float *gamma,
                                     __half *y, int n_heads, int head_dim,
                                     float eps) {
    // Grid: (n_heads, rows). Block: head_dim threads.
    int head = blockIdx.x;
    int row  = blockIdx.y;
    int tid  = threadIdx.x;
    if (tid >= head_dim) return;

    size_t off = ((size_t)row * n_heads + head) * head_dim;
    const __half *xv = x + off;
    __half *yv       = y + off;

    extern __shared__ float ss_buf[];
    float v = __half2float(xv[tid]);
    ss_buf[tid] = v * v;
    __syncthreads();
    // Reduce within head_dim. head_dim is ≤256 in DiT; use power-of-two reduce.
    for (int s = head_dim / 2; s > 0; s >>= 1) {
        if (tid < s) ss_buf[tid] += ss_buf[tid + s];
        __syncthreads();
    }
    float rstd = rsqrtf(ss_buf[0] / (float)head_dim + eps);
    yv[tid] = __float2half(v * rstd * gamma[tid]);
}

// ---------------------------------------------------------------------------
// NEOX-mode joint-sequence RoPE.  cos/sin tables are [rows, head_dim/2] F16.
// One block per (row, head). Threads stride over head_dim/2 pairs.
// ---------------------------------------------------------------------------
__global__ void rope_neox_seq_kernel(const __half *x, const __half *cos,
                                      const __half *sin, __half *y,
                                      int n_heads, int head_dim) {
    int row  = blockIdx.y;
    int head = blockIdx.x;
    int tid  = threadIdx.x;
    int half = head_dim / 2;

    size_t off = ((size_t)row * n_heads + head) * head_dim;
    const __half *xv = x + off;
    __half *yv       = y + off;
    const __half *cr = cos + (size_t)row * half;
    const __half *sr = sin + (size_t)row * half;

    for (int j = tid; j < half; j += blockDim.x) {
        float xl = __half2float(xv[j]);
        float xh = __half2float(xv[j + half]);
        float c  = __half2float(cr[j]);
        float s  = __half2float(sr[j]);
        yv[j]        = __float2half(xl * c - xh * s);
        yv[j + half] = __float2half(xl * s + xh * c);
    }
}

// ---------------------------------------------------------------------------
// Phase 3.3a multi-axis NEOX RoPE.  Same body as rope_neox_seq_kernel but
// indexes the pe-table with a fixed `pe_off` row offset.
// ---------------------------------------------------------------------------
__global__ void rope_neox_3axis_kernel(const __half *x, const __half *cos,
                                        const __half *sin, __half *y,
                                        int n_heads, int head_dim,
                                        int pe_off) {
    int row  = blockIdx.y;
    int head = blockIdx.x;
    int tid  = threadIdx.x;
    int half = head_dim / 2;

    size_t off = ((size_t)row * n_heads + head) * head_dim;
    const __half *xv = x + off;
    __half *yv       = y + off;
    const __half *cr = cos + (size_t)(pe_off + row) * half;
    const __half *sr = sin + (size_t)(pe_off + row) * half;

    for (int j = tid; j < half; j += blockDim.x) {
        float xl = __half2float(xv[j]);
        float xh = __half2float(xv[j + half]);
        float c  = __half2float(cr[j]);
        float s  = __half2float(sr[j]);
        yv[j]        = __float2half(xl * c - xh * s);
        yv[j + half] = __float2half(xl * s + xh * c);
    }
}

// ---------------------------------------------------------------------------
// SiLU (swish) in place: y = x * sigmoid(x).
// ---------------------------------------------------------------------------
__global__ void silu_kernel(__half *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __half2float(x[idx]);
    float y = v / (1.0f + __expf(-v));
    x[idx] = __float2half(y);
}

// ---------------------------------------------------------------------------
// GELU-tanh in place.
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------
__global__ void gelu_tanh_kernel(__half *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __half2float(x[idx]);
    const float k0 = 0.7978845608028654f;  // sqrt(2/pi)
    const float k1 = 0.044715f;
    float t = k0 * (v + k1 * v * v * v);
    float y = 0.5f * v * (1.0f + tanhf(t));
    x[idx] = __float2half(y);
}

// ---------------------------------------------------------------------------
// Bias add (F32 bias broadcast onto F16 [rows, cols]).
// ---------------------------------------------------------------------------
__global__ void add_bias_f32_f16_kernel(__half *y, const float *bias,
                                          int rows, int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    float yv = __half2float(y[idx]);
    y[idx] = __float2half(yv + bias[c]);
}

// ---------------------------------------------------------------------------
// F32 <-> F16 casts (engine-local copies; the qwen_tts cuda_kernels lib is
// not linked into this target, so we duplicate the trivial kernels here).
// ---------------------------------------------------------------------------
__global__ void cast_f32_to_f16_dit_kernel(const float *in, __half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(in[idx]);
}

__global__ void cast_f16_to_f32_dit_kernel(const __half *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __half2float(in[idx]);
}

// ---------------------------------------------------------------------------
// Naive joint cross-attention. One block per (q_row, head). Block size = 256.
// Dynamic shmem holds the [seq_total] score vector (max ~4352 floats = 17KB).
// ---------------------------------------------------------------------------
constexpr int ATTN_BLOCK = 256;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// F32-input variant — Phase 3.3b. Same body as attn_joint_naive_kernel but
// reads Q/K/V from F32 buffers. Mirrors Ascend §5.5.46 (BF16 widening of
// Q/K/V outputs to lift the F16 dynamic-range ceiling under high-magnitude
// AdaLN modulation). The kernel itself was already F32-accum / F16-store;
// this variant additionally lifts the F16 *input* ceiling by accepting F32
// inputs that the engine fills via F32 GEMMs.
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
    int n_warps = 256 / 32;

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
            partial = warp_reduce_sum(partial);
            if (lane == 0) scores_f32_[t] = partial * inv_sqrt_d;
        }
    }
    __syncthreads();

    float lmax = -INFINITY;
    for (int t = tid; t < seq_total; t += 256)
        lmax = fmaxf(lmax, scores_f32_[t]);
    lmax = warp_reduce_max(lmax);
    __shared__ float warp_max_f32[256 / 32];
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
    for (int t = tid; t < seq_total; t += 256) {
        float e = __expf(scores_f32_[t] - gmax);
        scores_f32_[t] = e;
        lsum += e;
    }
    lsum = warp_reduce_sum(lsum);
    __shared__ float warp_sum_f32[256 / 32];
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

__global__ void attn_joint_naive_kernel(const __half *q, const __half *k,
                                          const __half *v, __half *y,
                                          int seq_total, int n_heads,
                                          int head_dim, float inv_sqrt_d) {
    extern __shared__ float scores[];   // [seq_total]
    int q_row = blockIdx.y;
    int head  = blockIdx.x;
    int tid   = threadIdx.x;
    int warp  = tid >> 5;
    int lane  = tid & 31;
    int n_warps = ATTN_BLOCK / 32;

    const __half *q_vec =
        q + ((size_t)q_row * n_heads + head) * head_dim;

    // Load Q[q_row, head, :] into shmem (head_dim ≤ 256 ≤ ATTN_BLOCK).
    __shared__ float q_sh[256];
    if (tid < head_dim) q_sh[tid] = __half2float(q_vec[tid]);
    __syncthreads();

    // Pass 1: scores[t] = (q . k[t, head, :]) * inv_sqrt_d, for t in [0..seq_total).
    // We assign one warp to one t, looping over (seq_total / n_warps) chunks.
    for (int t_base = 0; t_base < seq_total; t_base += n_warps) {
        int t = t_base + warp;
        if (t < seq_total) {
            const __half *k_vec =
                k + ((size_t)t * n_heads + head) * head_dim;
            float partial = 0.0f;
            for (int d = lane; d < head_dim; d += 32) {
                partial += q_sh[d] * __half2float(k_vec[d]);
            }
            partial = warp_reduce_sum(partial);
            if (lane == 0) scores[t] = partial * inv_sqrt_d;
        }
    }
    __syncthreads();

    // Softmax: max-subtract → exp → normalize.
    float lmax = -INFINITY;
    for (int t = tid; t < seq_total; t += ATTN_BLOCK)
        lmax = fmaxf(lmax, scores[t]);
    lmax = warp_reduce_max(lmax);
    __shared__ float warp_max[ATTN_BLOCK / 32];
    if (lane == 0) warp_max[warp] = lmax;
    __syncthreads();
    __shared__ float gmax_sh;
    if (tid == 0) {
        float v = warp_max[0];
        for (int w = 1; w < n_warps; ++w) v = fmaxf(v, warp_max[w]);
        gmax_sh = v;
    }
    __syncthreads();
    float gmax = gmax_sh;

    float lsum = 0.0f;
    for (int t = tid; t < seq_total; t += ATTN_BLOCK) {
        float e = __expf(scores[t] - gmax);
        scores[t] = e;
        lsum += e;
    }
    lsum = warp_reduce_sum(lsum);
    __shared__ float warp_sum[ATTN_BLOCK / 32];
    if (lane == 0) warp_sum[warp] = lsum;
    __syncthreads();
    __shared__ float gsum_sh;
    if (tid == 0) {
        float s = 0.0f;
        for (int w = 0; w < n_warps; ++w) s += warp_sum[w];
        gsum_sh = s;
    }
    __syncthreads();
    float inv_denom = 1.0f / gsum_sh;

    // Pass 2: out[d] = sum_t softmax[t] * V[t, head, d]. Each thread owns
    // one d (head_dim ≤ 256 ≤ ATTN_BLOCK).
    if (tid < head_dim) {
        float acc = 0.0f;
        for (int t = 0; t < seq_total; ++t) {
            const __half *v_vec =
                v + ((size_t)t * n_heads + head) * head_dim;
            acc += scores[t] * inv_denom * __half2float(v_vec[tid]);
        }
        size_t out_idx = ((size_t)q_row * n_heads + head) * head_dim + tid;
        y[out_idx] = __float2half(acc);
    }
}

}  // namespace

// ===========================================================================
// Launchers
// ===========================================================================

void launch_adaln_modulate_f16(__half *x, const __half *scale,
                               const __half *shift, int rows, int cols,
                               cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    adaln_modulate_kernel<<<grid, block, 0, stream>>>(x, scale, shift, rows, cols);
}

void launch_gated_residual_add_f16(__half *x, const __half *delta,
                                   const __half *gate, int rows, int cols,
                                   cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    gated_residual_add_kernel<<<grid, block, 0, stream>>>(x, delta, gate,
                                                            rows, cols);
}

void launch_layernorm_noaffine_f16(const __half *x, __half *y,
                                   int rows, int cols, float eps,
                                   cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    layernorm_noaffine_kernel<<<rows, LN_BLOCK, 0, stream>>>(x, y, cols, eps);
}

void launch_rmsnorm_head_f16_g32(const __half *x, const float *gamma,
                                 __half *y, int rows, int n_heads,
                                 int head_dim, float eps,
                                 cudaStream_t stream) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0) return;
    dim3 grid(n_heads, rows);
    dim3 block(head_dim);
    size_t shmem = head_dim * sizeof(float);
    rmsnorm_head_kernel<<<grid, block, shmem, stream>>>(x, gamma, y,
                                                          n_heads, head_dim,
                                                          eps);
}

void launch_rope_neox_seq_f16(const __half *x, const __half *cos,
                              const __half *sin, __half *y,
                              int rows, int n_heads, int head_dim,
                              cudaStream_t stream) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0) return;
    int half = head_dim / 2;
    int threads = half < 256 ? half : 256;
    dim3 grid(n_heads, rows);
    dim3 block(threads);
    rope_neox_seq_kernel<<<grid, block, 0, stream>>>(x, cos, sin, y,
                                                      n_heads, head_dim);
}

void launch_rope_neox_3axis_f16(const __half *x, const __half *cos,
                                const __half *sin, __half *y,
                                int rows, int n_heads, int head_dim,
                                int pe_off, cudaStream_t stream) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0) return;
    int half = head_dim / 2;
    int threads = half < 256 ? half : 256;
    dim3 grid(n_heads, rows);
    dim3 block(threads);
    rope_neox_3axis_kernel<<<grid, block, 0, stream>>>(x, cos, sin, y,
                                                        n_heads, head_dim,
                                                        pe_off);
}

void launch_silu_f16(__half *x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    silu_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

void launch_gelu_tanh_f16(__half *x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    gelu_tanh_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

void launch_add_bias_f32_f16(__half *y, const float *bias,
                             int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    add_bias_f32_f16_kernel<<<grid, block, 0, stream>>>(y, bias, rows, cols);
}

void launch_cast_f32_to_f16_dit(const float *in, __half *out, int n,
                                cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    cast_f32_to_f16_dit_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_cast_f16_to_f32_dit(const __half *in, float *out, int n,
                                cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    cast_f16_to_f32_dit_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_attn_joint_naive_f16(const __half *q, const __half *k,
                                 const __half *v, __half *y,
                                 int seq_total, int n_heads, int head_dim,
                                 float inv_sqrt_d, cudaStream_t stream) {
    if (seq_total <= 0 || n_heads <= 0 || head_dim <= 0) return;
    dim3 grid(n_heads, seq_total);
    dim3 block(ATTN_BLOCK);
    size_t shmem = (size_t)seq_total * sizeof(float);
    attn_joint_naive_kernel<<<grid, block, shmem, stream>>>(
        q, k, v, y, seq_total, n_heads, head_dim, inv_sqrt_d);
}

// Phase 3.3b — F32-input attention (mirrors Ascend §5.5.46 BF16 widening).
void launch_attn_joint_naive_f32(const float *q, const float *k,
                                 const float *v, float *y,
                                 int seq_total, int n_heads, int head_dim,
                                 float inv_sqrt_d, cudaStream_t stream) {
    if (seq_total <= 0 || n_heads <= 0 || head_dim <= 0) return;
    dim3 grid(n_heads, seq_total);
    dim3 block(ATTN_BLOCK);
    size_t shmem = (size_t)seq_total * sizeof(float);
    attn_joint_naive_f32_kernel<<<grid, block, shmem, stream>>>(
        q, k, v, y, seq_total, n_heads, head_dim, inv_sqrt_d);
}

// ---------------------------------------------------------------------------
// Phase 3.3b — F32 helpers needed by the widened attention path.
// ---------------------------------------------------------------------------

namespace {

__global__ void rmsnorm_head_f32_kernel(const float *x, const float *gamma,
                                         float *y, int n_heads, int head_dim,
                                         float eps) {
    int head = blockIdx.x;
    int row  = blockIdx.y;
    int tid  = threadIdx.x;
    if (tid >= head_dim) return;
    size_t off = ((size_t)row * n_heads + head) * head_dim;
    const float *xv = x + off;
    float *yv       = y + off;
    extern __shared__ float ss_buf_f32[];
    float v = xv[tid];
    ss_buf_f32[tid] = v * v;
    __syncthreads();
    for (int s = head_dim / 2; s > 0; s >>= 1) {
        if (tid < s) ss_buf_f32[tid] += ss_buf_f32[tid + s];
        __syncthreads();
    }
    float rstd = rsqrtf(ss_buf_f32[0] / (float)head_dim + eps);
    yv[tid] = v * rstd * gamma[tid];
}

__global__ void rope_neox_3axis_f32_kernel(const float *x, const __half *cos,
                                            const __half *sin, float *y,
                                            int n_heads, int head_dim,
                                            int pe_off) {
    int row  = blockIdx.y;
    int head = blockIdx.x;
    int tid  = threadIdx.x;
    int half = head_dim / 2;
    size_t off = ((size_t)row * n_heads + head) * head_dim;
    const float *xv = x + off;
    float *yv       = y + off;
    const __half *cr = cos + (size_t)(pe_off + row) * half;
    const __half *sr = sin + (size_t)(pe_off + row) * half;
    for (int j = tid; j < half; j += blockDim.x) {
        float xl = xv[j];
        float xh = xv[j + half];
        float c  = __half2float(cr[j]);
        float s  = __half2float(sr[j]);
        yv[j]        = xl * c - xh * s;
        yv[j + half] = xl * s + xh * c;
    }
}

__global__ void cast_f32_to_f16_2d_kernel(const float *in, __half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(in[idx]);
}

__global__ void add_bias_f32_f32_kernel(float *y, const float *bias,
                                         int rows, int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    y[idx] = y[idx] + bias[c];
}

// ---------------------------------------------------------------------------
// Phase 3.3b — F32 residual-chain kernels.  The DiT residual buffer
// (img_resid / txt_resid) is stored in F32 to lift the F16 representable
// ceiling under high-magnitude AdaLN modulation. F16 storage saturates to
// Inf in the post-FFN gated residual add when AdaLN scale/shift are O(200)
// and the FFN's MLP_inter=12288 reduction further amplifies activations.
// ---------------------------------------------------------------------------

constexpr int LN_BLOCK_F32 = 256;

__global__ void layernorm_noaffine_f32_kernel(const float *x, float *y,
                                                int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float *xrow = x + (size_t)row * cols;
    float *yrow       = y + (size_t)row * cols;
    __shared__ float ssum_ln[LN_BLOCK_F32];
    __shared__ float ssq_ln[LN_BLOCK_F32];
    float lsum = 0.0f, lsq = 0.0f;
    for (int c = tid; c < cols; c += LN_BLOCK_F32) {
        float v = xrow[c];
        lsum += v;
        lsq  += v * v;
    }
    ssum_ln[tid] = lsum;
    ssq_ln[tid]  = lsq;
    __syncthreads();
    for (int s = LN_BLOCK_F32 / 2; s > 0; s >>= 1) {
        if (tid < s) {
            ssum_ln[tid] += ssum_ln[tid + s];
            ssq_ln[tid]  += ssq_ln[tid + s];
        }
        __syncthreads();
    }
    float mean = ssum_ln[0] / (float)cols;
    float var  = ssq_ln[0] / (float)cols - mean * mean;
    float rstd = rsqrtf(var + eps);
    for (int c = tid; c < cols; c += LN_BLOCK_F32) {
        float v = xrow[c];
        yrow[c] = (v - mean) * rstd;
    }
}

// AdaLN-modulate F32 with F16 scale/shift (mod_vec stays F16 for memory
// layout compat with the 12*H scratch buffer):
//   x[r,c] = x[r,c] * (1 + scale[c]) + shift[c]
__global__ void adaln_modulate_f32_kernel(float *x, const __half *scale,
                                            const __half *shift, int rows, int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    float xv = x[idx];
    float sv = __half2float(scale[c]);
    float shv = __half2float(shift[c]);
    x[idx] = xv * (1.0f + sv) + shv;
}

// Gated residual add: F32 residual += F16 delta * F16 gate (broadcast over rows).
__global__ void gated_residual_add_f32_kernel(float *x, const __half *delta,
                                                const __half *gate, int rows,
                                                int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    float xv = x[idx];
    float dv = __half2float(delta[idx]);
    float gv = __half2float(gate[c]);
    x[idx] = xv + dv * gv;
}

__global__ void cast_f16_to_f32_2d_kernel(const __half *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __half2float(in[idx]);
}

// F32 residual + F32 delta + F16 gate (broadcast over rows).
//   resid[r,c] += delta[r,c] * gate[c]
__global__ void gated_residual_add_f32_delta_kernel(float *x, const float *delta,
                                                      const __half *gate,
                                                      int rows, int cols) {
    int r = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || c >= cols) return;
    size_t idx = (size_t)r * cols + c;
    float gv = __half2float(gate[c]);
    x[idx] = x[idx] + delta[idx] * gv;
}

__global__ void gelu_tanh_f32_kernel(float *x, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = x[idx];
    const float k0 = 0.7978845608028654f;
    const float k1 = 0.044715f;
    float t = k0 * (v + k1 * v * v * v);
    x[idx] = 0.5f * v * (1.0f + tanhf(t));
}

}  // namespace

void launch_rmsnorm_head_f32_g32(const float *x, const float *gamma,
                                  float *y, int rows, int n_heads,
                                  int head_dim, float eps,
                                  cudaStream_t stream) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0) return;
    dim3 grid(n_heads, rows);
    dim3 block(head_dim);
    size_t shmem = head_dim * sizeof(float);
    rmsnorm_head_f32_kernel<<<grid, block, shmem, stream>>>(x, gamma, y,
                                                              n_heads, head_dim,
                                                              eps);
}

void launch_rope_neox_3axis_f32(const float *x, const __half *cos,
                                 const __half *sin, float *y,
                                 int rows, int n_heads, int head_dim,
                                 int pe_off, cudaStream_t stream) {
    if (rows <= 0 || n_heads <= 0 || head_dim <= 0) return;
    int half = head_dim / 2;
    int threads = half < 256 ? half : 256;
    dim3 grid(n_heads, rows);
    dim3 block(threads);
    rope_neox_3axis_f32_kernel<<<grid, block, 0, stream>>>(x, cos, sin, y,
                                                              n_heads, head_dim,
                                                              pe_off);
}

void launch_cast_f32_to_f16_2d(const float *in, __half *out, int n,
                                cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    cast_f32_to_f16_2d_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_add_bias_f32_f32(float *y, const float *bias,
                              int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    add_bias_f32_f32_kernel<<<grid, block, 0, stream>>>(y, bias, rows, cols);
}

void launch_layernorm_noaffine_f32(const float *x, float *y,
                                    int rows, int cols, float eps,
                                    cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    layernorm_noaffine_f32_kernel<<<rows, LN_BLOCK_F32, 0, stream>>>(x, y, cols, eps);
}

void launch_adaln_modulate_f32(float *x, const __half *scale,
                                const __half *shift, int rows, int cols,
                                cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    adaln_modulate_f32_kernel<<<grid, block, 0, stream>>>(x, scale, shift, rows, cols);
}

void launch_gated_residual_add_f32(float *x, const __half *delta,
                                    const __half *gate, int rows, int cols,
                                    cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    gated_residual_add_f32_kernel<<<grid, block, 0, stream>>>(x, delta, gate,
                                                                rows, cols);
}

void launch_cast_f16_to_f32_2d(const __half *in, float *out, int n,
                                cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    cast_f16_to_f32_2d_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_gated_residual_add_f32_delta(float *x, const float *delta,
                                          const __half *gate, int rows, int cols,
                                          cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    dim3 block(256);
    dim3 grid((cols + 255) / 256, rows);
    gated_residual_add_f32_delta_kernel<<<grid, block, 0, stream>>>(x, delta, gate,
                                                                       rows, cols);
}

void launch_gelu_tanh_f32(float *x, int n, cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    gelu_tanh_f32_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

}  // namespace ominix_cuda
