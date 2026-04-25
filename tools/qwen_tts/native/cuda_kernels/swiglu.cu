// ============================================================================
// Fused SwiGLU: y[i] = silu(gate[i]) * up[i]
//
// silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// All three buffers are F16. Compute happens in F32 to avoid sigmoid precision
// loss near zero. y may alias gate (matches the Ascend in-place pattern where
// the gate buffer is overwritten before the down-projection matmul).
// ============================================================================

#include "cuda_kernels.h"

namespace ominix_cuda {

namespace {

__global__ void swiglu_f16_kernel(const __half *gate, const __half *up,
                                   __half *y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __half2float(gate[idx]);
    float u = __half2float(up[idx]);
    float silu_g = g / (1.0f + expf(-g));
    y[idx] = __float2half(silu_g * u);
}

}  // namespace

void launch_swiglu_f16(const __half *gate, const __half *up, __half *y, int n,
                       cudaStream_t stream) {
    if (n <= 0) return;
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;
    swiglu_f16_kernel<<<blocks, threads, 0, stream>>>(gate, up, y, n);
}

}  // namespace ominix_cuda
