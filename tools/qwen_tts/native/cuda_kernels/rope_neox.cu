// ============================================================================
// NEOX-mode RoPE on a single-token tile [n_heads, head_dim].
//
// Given a row of cos/sin precomputed for a particular `pos` (each head_dim
// long, with the two halves duplicated so cos[low] == cos[high] etc.), the
// rotation pairs index `(i, i + half)` into a 2D rotation:
//
//   y[i      ] = x[i      ] * cos[i      ] - x[i + half] * sin[i      ]
//   y[i + half] = x[i      ] * sin[i + half] + x[i + half] * cos[i + half]
//
// Because cos/sin are duplicated across halves, this collapses to:
//   y[i      ] = x[i      ] * c - x[i + half] * s
//   y[i + half] = x[i      ] * s + x[i + half] * c
// with c = cos[i], s = sin[i]. (Same convention the Ascend reference uses.)
//
// One thread per pair `i`. One block per head (so blockDim.x = head_dim/2).
// Heads are independent so we use blockIdx.x.
// ============================================================================

#include "cuda_kernels.h"

namespace ominix_cuda {

namespace {

__global__ void rope_neox_f16_kernel(const __half *x, const __half *cos_row,
                                      const __half *sin_row, __half *y,
                                      int head_dim) {
    int head = blockIdx.x;
    int j    = threadIdx.x;
    const int half = head_dim / 2;
    if (j >= half) return;

    const __half *x_h = x + (size_t)head * head_dim;
    __half *y_h       = y + (size_t)head * head_dim;

    float xl = __half2float(x_h[j]);
    float xh = __half2float(x_h[j + half]);
    float c  = __half2float(cos_row[j]);
    float s  = __half2float(sin_row[j]);

    float yl = xl * c - xh * s;
    float yh = xl * s + xh * c;

    y_h[j]        = __float2half(yl);
    y_h[j + half] = __float2half(yh);
}

}  // namespace

void launch_rope_neox_f16(const __half *x, const __half *cos,
                          const __half *sin, __half *y,
                          int n_heads, int head_dim, cudaStream_t stream) {
    if (n_heads <= 0 || head_dim <= 0) return;
    const int half = head_dim / 2;
    dim3 grid(n_heads);
    dim3 block(half);
    rope_neox_f16_kernel<<<grid, block, 0, stream>>>(x, cos, sin, y, head_dim);
}

// ============================================================================
// P1: device-resident-pos RoPE.
//
// Same math as above, but cos_row / sin_row are computed inside the kernel
// from a device-side `pos`. Output y can either go to a tight [n_heads,
// head_dim] buffer (y_stride==0) or to a slot at row `pos` of a strided
// buffer (y_stride>0; used to write directly into the K-cache).
// ============================================================================
__global__ void rope_neox_f16_dev_kernel(const __half *x,
                                          const __half *cos_base,
                                          const __half *sin_base,
                                          __half *y_base,
                                          const int *pos_dev,
                                          int head_dim,
                                          int y_stride) {
    int head = blockIdx.x;
    int j    = threadIdx.x;
    const int half = head_dim / 2;
    if (j >= half) return;

    const int pos = *pos_dev;
    const __half *cos_row = cos_base + (size_t)pos * head_dim;
    const __half *sin_row = sin_base + (size_t)pos * head_dim;

    const __half *x_h = x + (size_t)head * head_dim;
    __half *y_h;
    if (y_stride > 0) {
        // Write into row `pos` of a strided buffer (e.g. KV cache).
        y_h = y_base + (size_t)pos * y_stride + (size_t)head * head_dim;
    } else {
        y_h = y_base + (size_t)head * head_dim;
    }

    float xl = __half2float(x_h[j]);
    float xh = __half2float(x_h[j + half]);
    float c  = __half2float(cos_row[j]);
    float s  = __half2float(sin_row[j]);

    float yl = xl * c - xh * s;
    float yh = xl * s + xh * c;

    y_h[j]        = __float2half(yl);
    y_h[j + half] = __float2half(yh);
}

void launch_rope_neox_f16_dev(const __half *x,
                              const __half *cos_base,
                              const __half *sin_base,
                              __half *y_base,
                              const int *pos_dev,
                              int n_heads, int head_dim,
                              int y_stride,
                              cudaStream_t stream) {
    if (n_heads <= 0 || head_dim <= 0) return;
    const int half = head_dim / 2;
    dim3 grid(n_heads);
    dim3 block(half);
    rope_neox_f16_dev_kernel<<<grid, block, 0, stream>>>(
        x, cos_base, sin_base, y_base, pos_dev, head_dim, y_stride);
}

// ============================================================================
// P1: device-resident-pos V-cache write. Replaces a host-pos cudaMemcpyAsync
// of v_dev_ into (v_cache_base + pos*kv_dim). One thread per element.
// ============================================================================
__global__ void v_write_kv_dev_kernel(const __half *src, __half *dst_base,
                                       const int *pos_dev, int kv_dim) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= kv_dim) return;
    const int pos = *pos_dev;
    __half *dst = dst_base + (size_t)pos * kv_dim;
    dst[i] = src[i];
}

void launch_v_write_kv_dev(const __half *src, __half *dst_base,
                           const int *pos_dev, int kv_dim,
                           cudaStream_t stream) {
    if (kv_dim <= 0) return;
    const int block = 256;
    const int grid  = (kv_dim + block - 1) / block;
    v_write_kv_dev_kernel<<<grid, block, 0, stream>>>(src, dst_base, pos_dev,
                                                       kv_dim);
}

}  // namespace ominix_cuda
