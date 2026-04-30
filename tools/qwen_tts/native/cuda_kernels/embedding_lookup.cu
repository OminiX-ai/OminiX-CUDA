// ============================================================================
// On-device embedding-table lookup (P2 — eliminates per-step host memcpy of
// the next embedding row + H2D into the engine's input staging).
//
// Reads a token id from a device int slot, copies the corresponding row of
// the embedding table into the Talker's input_stage_f32_dev_ buffer (F32).
// ============================================================================

#include "cuda_kernels.h"

#include <cuda_runtime.h>

namespace ominix_cuda {

namespace {

// out: F32 [hidden]; one block per launch, blockDim.x threads stride-loop the
// hidden axis. Falls back to writing zeros if tok is out of range.
__global__ void embedding_lookup_f32_kernel(const int *tok_dev,
                                             int slot,
                                             const float *table,  // [vocab, hidden]
                                             int vocab,
                                             int hidden,
                                             float *out) {
    int tok = tok_dev[slot];
    if (tok < 0 || tok >= vocab) {
        for (int i = threadIdx.x; i < hidden; i += blockDim.x) out[i] = 0.0f;
        return;
    }
    const float *row = table + (size_t)tok * (size_t)hidden;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) out[i] = row[i];
}

// Variant for predictor: take an explicit divisor + modulus so the on-device
// embedding lookup can mirror the host's `p_cur = nxt;  // already absolute`
// semantics. Predictor uses absolute token id directly (no mod 2048 needed).
// We keep the simple variant for now and add modulus if needed.

}  // namespace

void launch_embedding_lookup_f32(const int *tok_dev,
                                  int slot,
                                  const float *table_dev,
                                  int vocab,
                                  int hidden,
                                  float *out_dev,
                                  cudaStream_t stream) {
    int threads = 128;
    embedding_lookup_f32_kernel<<<1, threads, 0, stream>>>(
        tok_dev, slot, table_dev, vocab, hidden, out_dev);
}

namespace {
__global__ void increment_int_kernel(int *p) {
    if (threadIdx.x == 0 && blockIdx.x == 0) (*p) = (*p) + 1;
}

__global__ void set_int_value_kernel(int *p, int v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) (*p) = v;
}

__global__ void record_rep_history_kernel(const int *src, int src_index,
                                           int lo, int *slot) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        (*slot) = src[src_index] - lo;
    }
}
}

void launch_increment_int(int *p_dev, cudaStream_t stream) {
    increment_int_kernel<<<1, 1, 0, stream>>>(p_dev);
}

void launch_set_int_value(int *p_dev, int value, cudaStream_t stream) {
    set_int_value_kernel<<<1, 1, 0, stream>>>(p_dev, value);
}

void launch_record_rep_history(const int *src_token_dev, int src_index,
                                int lo, int *slot_dev,
                                cudaStream_t stream) {
    record_rep_history_kernel<<<1, 1, 0, stream>>>(
        src_token_dev, src_index, lo, slot_dev);
}

}  // namespace ominix_cuda
