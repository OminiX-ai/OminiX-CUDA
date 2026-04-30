// ============================================================================
// Single-token GQA attention (decode path, S=1).
//
// One block per Q head. Each block:
//   1. Cooperative load of q[h] (head_dim floats) into shmem.
//   2. Pass-1: per-token dot product score[t] = (q . k[t, h_kv]) * inv_sqrt_d.
//      Stream scores into a scratch buffer in shmem, tracking running max
//      and partial denominator (online softmax accumulator).
//   3. Pass-2: weighted sum. accum[d] += softmax[t] * v[t, h_kv, d].
//
// We hold the full scores in shmem (up to MAX_SEQ=4096 floats per head =
// 16 KB; 16 heads -> launch 16 independent blocks each with their own shmem,
// no cross-block contention).
//
// Block size: 128 threads. head_dim=128 so each thread owns one component
// for the q load + the v accumulation. The dot-product and softmax reductions
// are warp-level via __shfl_xor_sync.
//
// GQA: h_kv = h / (n_heads / n_kv). Multiple Q heads share one KV head.
//
// All shapes follow the engine convention:
//   q       [n_heads, head_dim]            contiguous, head_dim == HEAD_DIM
//   k_cache [MAX_SEQ, n_kv,   head_dim]    contiguous; we read first seq_len rows
//   v_cache [MAX_SEQ, n_kv,   head_dim]    same layout
//   y       [n_heads, head_dim]
// ============================================================================

#include "cuda_kernels.h"

namespace ominix_cuda {

namespace {

// Hard-coded for Qwen3 Talker: head_dim=128. Generalising would mean either
// templated dispatch (kept light here for clarity) or a runtime loop. The
// kernel ONLY runs against this 28L Talker model — the 128 tile size is the
// load-bearing assumption that everything else compiles around.
constexpr int HEAD_DIM = 128;
constexpr int BLOCK    = 128;  // == HEAD_DIM

__device__ __forceinline__ float warp_reduce_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, offset));
    }
    return v;
}

// scores buf is dynamic shmem of size seq_len * sizeof(float). We also use
// a small static shmem for cross-warp reductions.
__global__ void attn_decode_gqa_kernel(const __half *q,
                                        const __half *k_cache,
                                        const __half *v_cache,
                                        __half *y,
                                        int seq_len, int n_heads, int n_kv,
                                        float inv_sqrt_d) {
    extern __shared__ float scores[];           // [seq_len]

    __shared__ float q_sh[HEAD_DIM];            // current Q head
    __shared__ float warp_buf[BLOCK / 32];      // for cross-warp reductions

    const int h    = blockIdx.x;                // Q head index (0..n_heads-1)
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int group = n_heads / n_kv;
    const int h_kv  = h / group;

    // 1. Load q[h] into shmem (one thread per element since BLOCK == HEAD_DIM).
    q_sh[tid] = __half2float(q[(size_t)h * HEAD_DIM + tid]);
    __syncthreads();

    // ------------------------------------------------------------------
    // Pass 1: compute scores[t] = (q . k[t, h_kv]) * inv_sqrt_d for all t.
    // We launch one thread per dim component, parallelize the dot product
    // by warp-reducing in lockstep across all 128 threads.
    //
    // For each chunk of WARPS_PER_BLOCK tokens we compute their dots in
    // parallel: warp `w` handles token t = chunk_base + w. With BLOCK=128
    // we have 4 warps -> 4 tokens per pass.
    // ------------------------------------------------------------------
    const int warps_per_block = BLOCK / 32;     // 4
    for (int t_base = 0; t_base < seq_len; t_base += warps_per_block) {
        int t = t_base + warp;
        float dot = 0.0f;
        if (t < seq_len) {
            const __half *k_row = k_cache +
                ((size_t)t * n_kv + h_kv) * HEAD_DIM;
            float kv = __half2float(k_row[lane]);
            float qv = q_sh[lane];
            // Lane only contributes one of HEAD_DIM=128 elements; we have
            // 32 lanes per warp, so each lane handles 4 components via a
            // small inner loop.
            float sum = qv * kv;
            #pragma unroll
            for (int extra = 1; extra < HEAD_DIM / 32; ++extra) {
                int idx = lane + extra * 32;
                kv = __half2float(k_row[idx]);
                qv = q_sh[idx];
                sum += qv * kv;
            }
            dot = warp_reduce_sum(sum);
        }
        // Lane 0 of each warp writes the score. Tokens beyond seq_len do
        // nothing (they're skipped in pass 2 via the seq_len bound).
        if (lane == 0 && t < seq_len) {
            scores[t] = dot * inv_sqrt_d;
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Softmax over scores[0..seq_len). Numerically stable max-subtract.
    // ------------------------------------------------------------------
    // Find max (per-warp partial -> shmem -> single-thread reduce).
    float local_max = -INFINITY;
    for (int t = tid; t < seq_len; t += BLOCK) {
        local_max = fmaxf(local_max, scores[t]);
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_buf[warp] = local_max;
    __syncthreads();
    // Single thread sequentially reduces the warps_per_block (=4) partials.
    // Cheap and avoids the partial-warp __shfl_sync subtlety. Result via shmem.
    __shared__ float global_max_sh;
    if (tid == 0) {
        float v = warp_buf[0];
        for (int w = 1; w < warps_per_block; ++w) v = fmaxf(v, warp_buf[w]);
        global_max_sh = v;
    }
    __syncthreads();
    float global_max = global_max_sh;

    // Compute exp(score - max), accumulate denominator.
    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += BLOCK) {
        float e = __expf(scores[t] - global_max);
        scores[t] = e;
        local_sum += e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_buf[warp] = local_sum;
    __syncthreads();
    __shared__ float global_sum_sh;
    if (tid == 0) {
        float v = 0.0f;
        for (int w = 0; w < warps_per_block; ++w) v += warp_buf[w];
        global_sum_sh = v;
    }
    __syncthreads();
    float inv_denom = 1.0f / global_sum_sh;

    // ------------------------------------------------------------------
    // Pass 2: out[d] = sum_t softmax[t] * v[t, h_kv, d]. Each thread owns
    // one d (since BLOCK == HEAD_DIM).
    // ------------------------------------------------------------------
    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        const __half *v_row = v_cache +
            ((size_t)t * n_kv + h_kv) * HEAD_DIM;
        float p = scores[t] * inv_denom;
        acc += p * __half2float(v_row[tid]);
    }
    y[(size_t)h * HEAD_DIM + tid] = __float2half(acc);
}

// ============================================================================
// P1 (April 2026) — device-resident-pos GQA attention kernel.
//
// seq_len is read inside the kernel as (*pos_dev) + 1. Shared-memory `scores`
// buffer is allocated at MAX_SEQ * sizeof(float) on launch (constant across
// positions, so the captured CUDA Graph node is static); only the first
// seq_len entries are touched at runtime.
//
// On GB10 / sm_121a, max dynamic shmem per block defaults to 48 KB. With
// MAX_SEQ=4096 the worst-case allocation is 16 KB — well under the limit.
// ============================================================================
__global__ void attn_decode_gqa_dev_kernel(const __half *q,
                                            const __half *k_cache,
                                            const __half *v_cache,
                                            __half *y,
                                            const int *pos_dev,
                                            int n_heads, int n_kv,
                                            float inv_sqrt_d) {
    extern __shared__ float scores[];           // [<= max_seq]; only seq_len used

    __shared__ float q_sh[HEAD_DIM];
    __shared__ float warp_buf[BLOCK / 32];

    const int h    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int group = n_heads / n_kv;
    const int h_kv  = h / group;

    const int seq_len = (*pos_dev) + 1;

    q_sh[tid] = __half2float(q[(size_t)h * HEAD_DIM + tid]);
    __syncthreads();

    const int warps_per_block = BLOCK / 32;     // 4
    for (int t_base = 0; t_base < seq_len; t_base += warps_per_block) {
        int t = t_base + warp;
        float dot = 0.0f;
        if (t < seq_len) {
            const __half *k_row = k_cache +
                ((size_t)t * n_kv + h_kv) * HEAD_DIM;
            float kv = __half2float(k_row[lane]);
            float qv = q_sh[lane];
            float sum = qv * kv;
            #pragma unroll
            for (int extra = 1; extra < HEAD_DIM / 32; ++extra) {
                int idx = lane + extra * 32;
                kv = __half2float(k_row[idx]);
                qv = q_sh[idx];
                sum += qv * kv;
            }
            dot = warp_reduce_sum(sum);
        }
        if (lane == 0 && t < seq_len) {
            scores[t] = dot * inv_sqrt_d;
        }
    }
    __syncthreads();

    float local_max = -INFINITY;
    for (int t = tid; t < seq_len; t += BLOCK) {
        local_max = fmaxf(local_max, scores[t]);
    }
    local_max = warp_reduce_max(local_max);
    if (lane == 0) warp_buf[warp] = local_max;
    __syncthreads();
    __shared__ float global_max_sh;
    if (tid == 0) {
        float v = warp_buf[0];
        for (int w = 1; w < warps_per_block; ++w) v = fmaxf(v, warp_buf[w]);
        global_max_sh = v;
    }
    __syncthreads();
    float global_max = global_max_sh;

    float local_sum = 0.0f;
    for (int t = tid; t < seq_len; t += BLOCK) {
        float e = __expf(scores[t] - global_max);
        scores[t] = e;
        local_sum += e;
    }
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) warp_buf[warp] = local_sum;
    __syncthreads();
    __shared__ float global_sum_sh;
    if (tid == 0) {
        float v = 0.0f;
        for (int w = 0; w < warps_per_block; ++w) v += warp_buf[w];
        global_sum_sh = v;
    }
    __syncthreads();
    float inv_denom = 1.0f / global_sum_sh;

    float acc = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
        const __half *v_row = v_cache +
            ((size_t)t * n_kv + h_kv) * HEAD_DIM;
        float p = scores[t] * inv_denom;
        acc += p * __half2float(v_row[tid]);
    }
    y[(size_t)h * HEAD_DIM + tid] = __float2half(acc);
}

}  // namespace

void launch_attn_decode_gqa_f16(const __half *q,
                                const __half *k_cache,
                                const __half *v_cache,
                                __half *y,
                                int seq_len, int n_heads, int n_kv,
                                int head_dim, float inv_sqrt_d,
                                cudaStream_t stream) {
    if (seq_len <= 0 || n_heads <= 0 || head_dim != HEAD_DIM) return;
    size_t shmem_bytes = (size_t)seq_len * sizeof(float);
    dim3 grid(n_heads);
    dim3 block(BLOCK);
    attn_decode_gqa_kernel<<<grid, block, shmem_bytes, stream>>>(
        q, k_cache, v_cache, y, seq_len, n_heads, n_kv, inv_sqrt_d);
}

void launch_attn_decode_gqa_f16_dev(const __half *q,
                                    const __half *k_cache,
                                    const __half *v_cache,
                                    __half *y,
                                    const int *pos_dev,
                                    int max_seq,
                                    int n_heads, int n_kv,
                                    int head_dim, float inv_sqrt_d,
                                    cudaStream_t stream) {
    if (n_heads <= 0 || head_dim != HEAD_DIM || max_seq <= 0) return;
    // Allocate shmem at MAX_SEQ (constant across positions) so the captured
    // CUDA Graph node is static.
    size_t shmem_bytes = (size_t)max_seq * sizeof(float);
    dim3 grid(n_heads);
    dim3 block(BLOCK);
    attn_decode_gqa_dev_kernel<<<grid, block, shmem_bytes, stream>>>(
        q, k_cache, v_cache, y, pos_dev, n_heads, n_kv, inv_sqrt_d);
}

}  // namespace ominix_cuda
