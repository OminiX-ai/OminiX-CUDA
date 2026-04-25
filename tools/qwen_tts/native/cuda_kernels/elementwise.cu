// ============================================================================
// Elementwise CUDA kernels: F32<->F16 casts and F16 add.
// All shapes are flat [n]; one block-stride loop per kernel.
// ============================================================================

#include "cuda_kernels.h"

namespace ominix_cuda {

namespace {

__global__ void cast_f32_to_f16_kernel(const float *in, __half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(in[idx]);
}

__global__ void cast_f16_to_f32_kernel(const __half *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __half2float(in[idx]);
}

__global__ void add_f16_kernel(const __half *a, const __half *b, __half *y,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // Sum in F32 to keep residual accumulators well-conditioned. The Ascend
    // reference uses aclnnAdd which is also F16-in / F16-out but goes
    // through an F32 accumulator under the hood — matching that precision
    // semantic on CUDA helps cossim parity later.
    float fa = __half2float(a[idx]);
    float fb = __half2float(b[idx]);
    y[idx] = __float2half(fa + fb);
}

}  // namespace

void launch_cast_f32_to_f16(const float *in, __half *out, int n,
                            cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    cast_f32_to_f16_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_cast_f16_to_f32(const __half *in, float *out, int n,
                            cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    cast_f16_to_f32_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_add_f16(const __half *a, const __half *b, __half *y, int n,
                    cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    add_f16_kernel<<<blocks, threads, 0, stream>>>(a, b, y, n);
}

}  // namespace ominix_cuda
