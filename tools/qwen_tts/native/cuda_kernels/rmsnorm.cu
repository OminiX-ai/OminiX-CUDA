// ============================================================================
// RmsNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma
//
// One block per row. Each block:
//   - Loads its row from F16 x into registers (warp-stride loop), sums squares
//     in F32.
//   - Block-reduces the sum-of-squares.
//   - Computes 1 / sqrt(sum / cols + eps).
//   - Writes y[row, c] = (x[row, c] * rstd) * gamma[c] in F16.
//
// gamma is F32 to match the Qwen3 reference (norm gammas are F32 on disk
// in the GGUF and stay F32 on device — we follow the Ascend convention).
// ============================================================================

#include "cuda_kernels.h"

namespace ominix_cuda {

namespace {

constexpr int RMSNORM_BLOCK = 256;

__global__ void rmsnorm_f16_g32_kernel(const __half *x, const float *gamma,
                                        __half *y, int cols, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const __half *x_row = x + (size_t)row * cols;
    __half *y_row       = y + (size_t)row * cols;

    // 1. Sum-of-squares in F32 (warp-stride loop).
    float ss = 0.0f;
    for (int c = tid; c < cols; c += RMSNORM_BLOCK) {
        float v = __half2float(x_row[c]);
        ss += v * v;
    }

    // 2. Block reduction in shmem.
    __shared__ float sdata[RMSNORM_BLOCK];
    sdata[tid] = ss;
    __syncthreads();
    for (int s = RMSNORM_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float mean_sq = sdata[0] / (float)cols;
    float rstd = rsqrtf(mean_sq + eps);

    // 3. y = x * rstd * gamma (F16 out).
    for (int c = tid; c < cols; c += RMSNORM_BLOCK) {
        float v = __half2float(x_row[c]);
        float g = gamma[c];
        y_row[c] = __float2half(v * rstd * g);
    }
}

}  // namespace

void launch_rmsnorm_f16_g32(const __half *x, const float *gamma, __half *y,
                            int rows, int cols, float eps,
                            cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return;
    rmsnorm_f16_g32_kernel<<<rows, RMSNORM_BLOCK, 0, stream>>>(x, gamma, y,
                                                                cols, eps);
}

}  // namespace ominix_cuda
