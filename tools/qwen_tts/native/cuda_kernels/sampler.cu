// ============================================================================
// On-device token sampling kernels (P2 — eliminate per-step D2H of logits).
//
// Two kernels:
//   1) launch_apply_repetition_penalty_f32 — divides/multiplies logits at
//      `recent_tokens_dev[]` slots by `penalty`. Mirrors the host
//      apply_repetition_penalty in tts_server.cpp:178.
//
//   2) launch_sample_top_k_f32 — takes F32 logits[lo..hi], applies
//      temperature scaling, picks top-K with a per-block reduction,
//      softmaxes the top-K, then samples either greedy (temp<=0 / do_sample=0)
//      or stochastic (RNG seeded by host-supplied uint64 + step counter).
//      Writes the absolute token id to `next_token_dev_[slot]`.
//
// Two implementations live behind `launch_sample_top_k_f32`:
//
//   * sample_topk_kernel (serial — K rounds of full-vocab argmax with a
//     growing exclusion list). #193 reference; correct but ~370 us/call
//     for K=50, V~2048.
//
//   * sample_topk_kernel_bitonic (parallel — single bitonic sort of the
//     padded (logit,idx) shared-mem array, then take the prefix of K).
//     ~25-40 us/call. Default ON (env OMNX_TTS_PARALLEL_TOPK; default=1).
//
// The two paths share the same RNG (xorshift64*) and the same softmax math
// over the top-K, so for greedy / temperature=0 they emit identical tokens
// and for stochastic they emit tokens drawn from the same distribution
// (the RNG stream is identical for the same seed+step_idx).
//
// Single-block design: top_k <= 256 (default 50), V <= 4096 in our usage
// (Talker codec vocab 3072, predictor sub-vocab 2048). Both fit in a single
// block's worth of shared memory.
// ============================================================================

#include "cuda_kernels.h"

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cfloat>

namespace ominix_cuda {

namespace {

// --------------------------------------------------------------------------
// Repetition penalty: for each recent token id, divide logit by penalty if
// >0 else multiply by penalty. Matches the Ascend / host reference.
//
// IMPORTANT: must be applied SEQUENTIALLY because the recent list can contain
// duplicates (same token id appearing multiple times in the rep window).
// The host reference loops sequentially: each duplicate divides the same
// logit AGAIN. A naive parallel implementation has a read-modify-write data
// race when two threads target the same idx; in practice both reads see the
// pre-update value and both writes overwrite each other, so the logit is
// divided ONCE instead of N times. That tiny difference is enough to flip
// argmax and cascade the entire predictor output.
//
// Solution: single-thread sequential kernel. n_recent is bounded by
// recent_window (default 64), so this is a few hundred FP ops per call —
// negligible vs the GEMM that produced the logits.
// --------------------------------------------------------------------------
__global__ void rep_penalty_kernel(float *logits, int lo, int hi,
                                    const int *recent_tokens, int n_recent,
                                    int recent_offset, // adds to recent[i]
                                    float penalty) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int idx = 0; idx < n_recent; ++idx) {
        int t = recent_tokens[idx] + recent_offset;
        if (t < lo || t >= hi) continue;
        float v = logits[t];
        if (v > 0.0f) v /= penalty;
        else          v *= penalty;
        logits[t] = v;
    }
}

// --------------------------------------------------------------------------
// xorshift64* PRNG — fast, deterministic, no curand state needed.
// state must be non-zero; mixes seed * (step+1).
// --------------------------------------------------------------------------
__device__ inline uint64_t xorshift64s(uint64_t &s) {
    s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
    return s * 2685821657736338717ULL;
}

__device__ inline float u01_from_u64(uint64_t x) {
    // top 24 bits → float in [0, 1).
    return (float)(x >> 40) * (1.0f / 16777216.0f);
}

// --------------------------------------------------------------------------
// Sampling kernel — single block. SERIAL implementation (K-round argmax).
//
//   logits  : F32 [vocab]  (full table; we only sample within [lo, hi))
//   lo, hi  : sub-range to sample from (Talker uses [3, 3072), Predictor
//             uses [g*2048, (g+1)*2048))
//   top_k   : 1..MAX_TOP_K. <=0 disables top-k (treated as full range).
//   temp    : 1.0 means no scaling. <=0 means greedy (argmax).
//   do_sample : 0 -> greedy (argmax over temperature-scaled logits)
//               1 -> stochastic
//   seed    : per-request RNG seed (mixed with step_idx)
//   step_idx: 0..N — both for RNG mixing and output slot selection.
//   out_token_dev : int* [pending_slots]  (writes out_token_dev[step_idx])
// --------------------------------------------------------------------------
constexpr int MAX_TOP_K = 256;

__global__ void sample_topk_kernel(const float *logits,
                                    int lo, int hi,
                                    int top_k,
                                    float temp,
                                    int do_sample,
                                    uint64_t seed,
                                    int step_idx,
                                    int *out_token_dev) {
    // Block-stride pass 1: find max logit over [lo, hi).
    __shared__ float smax;
    __shared__ int   smax_idx;

    if (threadIdx.x == 0) {
        smax = -FLT_MAX;
        smax_idx = lo;
    }
    __syncthreads();

    // Greedy / argmax fast path. Even in stochastic mode we use the max for
    // numerical-stable softmax.
    float my_max = -FLT_MAX;
    int   my_idx = lo;
    for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        float v = logits[i];
        if (v > my_max) { my_max = v; my_idx = i; }
    }
    // Reduce per-thread max.
    // Simple shmem reduction: write to per-thread shmem.
    __shared__ float thr_max[1024];
    __shared__ int   thr_idx[1024];
    int tid = threadIdx.x;
    thr_max[tid] = my_max;
    thr_idx[tid] = my_idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (thr_max[tid + s] > thr_max[tid]) {
                thr_max[tid] = thr_max[tid + s];
                thr_idx[tid] = thr_idx[tid + s];
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        smax = thr_max[0];
        smax_idx = thr_idx[0];
    }
    __syncthreads();

    if (!do_sample || temp <= 0.0f) {
        if (tid == 0) out_token_dev[step_idx] = smax_idx;
        return;
    }

    // Stochastic: top-K reduction via repeated max-with-mask. This is O(K * N / B)
    // per pass but K is small (default 50) and N is small (vocab subrange ~2048-3072).
    // We replace the chosen logit with -INF in a shadow shmem mask after each pass.
    //
    // Simpler approach: do K rounds; each round finds the global max, records it,
    // and marks it as taken in a shmem boolean indexed by (idx - lo). The mark
    // table is up to 30720 entries (predictor full vocab) — too big. Use a
    // per-block "exclusion list" approach: keep an array of taken indices in
    // shmem, and in the next reduction skip any thread whose idx matches a taken one.
    //
    // For top_k <= MAX_TOP_K, the exclusion check is K compares per thread per round.
    // K=50 * blockDim.x=256 * K rounds = ~640k cmps for ~2048 vocab — trivial.

    int K = top_k;
    if (K <= 0 || K > MAX_TOP_K) K = MAX_TOP_K;
    if (K > (hi - lo)) K = hi - lo;

    __shared__ int   topk_idx[MAX_TOP_K];
    __shared__ float topk_val[MAX_TOP_K];

    if (tid == 0) { topk_idx[0] = smax_idx; topk_val[0] = smax; }
    __syncthreads();

    const float inv_t = 1.0f / temp;

    for (int r = 1; r < K; ++r) {
        // Per-thread max excluding already-taken ones.
        float my_m = -FLT_MAX;
        int   my_i = lo;
        for (int i = lo + tid; i < hi; i += blockDim.x) {
            // Skip if i is in topk_idx[0..r).
            bool taken = false;
            for (int q = 0; q < r; ++q) if (topk_idx[q] == i) { taken = true; break; }
            if (taken) continue;
            float v = logits[i];
            if (v > my_m) { my_m = v; my_i = i; }
        }
        thr_max[tid] = my_m;
        thr_idx[tid] = my_i;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (thr_max[tid + s] > thr_max[tid]) {
                    thr_max[tid] = thr_max[tid + s];
                    thr_idx[tid] = thr_idx[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) { topk_idx[r] = thr_idx[0]; topk_val[r] = thr_max[0]; }
        __syncthreads();
    }

    // Softmax over top-K with temperature.
    if (tid == 0) {
        float mx = topk_val[0] * inv_t;
        // (topk_val[0] is the global max, but after temp scaling we re-find
        //  for safety — same number under monotone scaling.)
        for (int r = 1; r < K; ++r) {
            float v = topk_val[r] * inv_t;
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int r = 0; r < K; ++r) {
            float v = expf(topk_val[r] * inv_t - mx);
            topk_val[r] = v;
            sum += v;
        }
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
        for (int r = 0; r < K; ++r) topk_val[r] *= inv_sum;

        // Sample.
        uint64_t rs = seed ^ (uint64_t)(step_idx + 1) * 0x9E3779B97F4A7C15ULL;
        if (rs == 0) rs = 0xDEADBEEFCAFEBABEULL;
        // Burn one to mix.
        xorshift64s(rs);
        float r = u01_from_u64(xorshift64s(rs));
        float cs = 0.0f;
        int chosen = topk_idx[K - 1];
        for (int q = 0; q < K; ++q) {
            cs += topk_val[q];
            if (r <= cs) { chosen = topk_idx[q]; break; }
        }
        out_token_dev[step_idx] = chosen;
    }
}

// --------------------------------------------------------------------------
// PARALLEL top-K sampler — bitonic sort of (logit, idx) pairs in shared mem.
//
// Replaces the K-round serial argmax (~370 us/call for K=50, V=2048) with a
// single sort pass (~25-40 us/call). Algorithm:
//
//   1. Pad N = next_pow2(hi - lo). Each of T=256 threads owns ELEMS = N/T
//      shared-memory slots. Slots inside [hi-lo) load real logits; slots
//      beyond are filled with sentinel -FLT_MAX.
//   2. Bitonic sort descending by logit value, carrying (val,idx) pairs.
//   3. The first K shared slots after the sort are the global top-K (sorted
//      descending). The single-thread softmax+sample tail is identical to
//      the serial path.
//
// Constraints baked in:
//   * N <= NPOW2_MAX (4096) — fits Talker codec vocab (3072) and predictor
//     sub-vocab (2048). ELEMS = NPOW2_MAX / T = 16. Shared mem cost is
//     4096 * (4+4) = 32 KB — well within Blackwell per-block budget.
//   * top_k <= MAX_TOP_K (256). The softmax tail re-uses topk_val/topk_idx
//     arrays the serial path declared.
//
// Numerical / sampling parity with serial:
//   * Greedy (do_sample==0 || temp<=0): writes argmax — bit-exact with
//     serial path's argmax.
//   * Stochastic: identical xorshift64* RNG state derivation
//     (`seed ^ (step_idx+1)*GR`), identical "burn one then draw" pattern,
//     identical softmax-over-top-K. Both paths sample from the same
//     distribution; the choice is deterministic given the same seed+step.
// --------------------------------------------------------------------------
constexpr int NPOW2_MAX = 4096;
constexpr int BITONIC_THREADS = 256;
constexpr int BITONIC_ELEMS_PER_THREAD = NPOW2_MAX / BITONIC_THREADS; // 16

__global__ void sample_topk_kernel_bitonic(const float *logits,
                                            int lo, int hi,
                                            int top_k,
                                            float temp,
                                            int do_sample,
                                            uint64_t seed,
                                            int step_idx,
                                            int *out_token_dev) {
    __shared__ float s_vals[NPOW2_MAX];
    __shared__ int   s_idxs[NPOW2_MAX];

    const int tid = threadIdx.x;
    const int V   = hi - lo;

    // Greedy fast path: skip the sort entirely. Single block-stride argmax
    // reduction. Bit-exact with the serial path; used when the host requests
    // either do_sample==0 or temperature<=0.
    if (!do_sample || temp <= 0.0f) {
        __shared__ float thr_max[BITONIC_THREADS];
        __shared__ int   thr_idx[BITONIC_THREADS];
        float my_max = -FLT_MAX;
        int   my_idx = lo;
        for (int i = lo + tid; i < hi; i += BITONIC_THREADS) {
            float v = logits[i];
            if (v > my_max) { my_max = v; my_idx = i; }
        }
        thr_max[tid] = my_max;
        thr_idx[tid] = my_idx;
        __syncthreads();
        for (int s = BITONIC_THREADS / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (thr_max[tid + s] > thr_max[tid]) {
                    thr_max[tid] = thr_max[tid + s];
                    thr_idx[tid] = thr_idx[tid + s];
                }
            }
            __syncthreads();
        }
        if (tid == 0) out_token_dev[step_idx] = thr_idx[0];
        return;
    }

    int N = 1;
    while (N < V) N <<= 1;
    if (N > NPOW2_MAX) N = NPOW2_MAX; // guarded by caller; clamp for safety.

    // 1) Load: each thread fills ELEMS slots; out-of-range = sentinel.
    for (int e = 0; e < BITONIC_ELEMS_PER_THREAD; ++e) {
        int slot = tid + e * BITONIC_THREADS;
        if (slot < N) {
            if (slot < V) {
                s_vals[slot] = logits[lo + slot];
                s_idxs[slot] = lo + slot;
            } else {
                s_vals[slot] = -FLT_MAX;
                s_idxs[slot] = lo;            // dummy; never selected
            }
        }
    }
    __syncthreads();

    // 2) Bitonic sort, DESCENDING. For each (k, j) stage, every pair (i, i^j)
    //    where i<i^j performs a compare-and-swap. The "direction" of the
    //    sub-sort alternates with bit (i & k); for an overall descending
    //    output we keep the higher value on the lower index when (i & k)==0,
    //    and the lower value on the lower index when (i & k)!=0.
    //
    //    We let each thread handle multiple pairs per stage, since N can be
    //    up to NPOW2_MAX=4096 and BITONIC_THREADS=256 → 8 pairs per thread
    //    per stage in the worst case (N/2 / T = 4096/2/256 = 8).
    const int half = N >> 1;  // number of pairs per stage
    for (int k = 2; k <= N; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            // Each thread handles pairs spaced by BITONIC_THREADS.
            for (int p = tid; p < half; p += BITONIC_THREADS) {
                // Encode pair index p as a position i s.t. (i ^ j) > i.
                int i = ((p / j) << 1) * j + (p % j);
                int ixj = i ^ j;
                if (ixj > i) {
                    bool ascending = ((i & k) == 0); // descending overall
                    float va = s_vals[i];
                    float vb = s_vals[ixj];
                    int   ia = s_idxs[i];
                    int   ib = s_idxs[ixj];
                    bool swap = ascending ? (va < vb) : (va > vb);
                    if (swap) {
                        s_vals[i]   = vb; s_vals[ixj] = va;
                        s_idxs[i]   = ib; s_idxs[ixj] = ia;
                    }
                }
            }
            __syncthreads();
        }
    }

    // After sort: s_vals[0..N) is sorted DESCENDING. The greedy case is
    // handled at the top of the kernel; here we always proceed to top-K
    // softmax + stochastic sampling.

    int K = top_k;
    if (K <= 0 || K > MAX_TOP_K) K = MAX_TOP_K;
    if (K > V) K = V;

    // 3) Softmax over the top-K + sample. Single-thread to mirror the
    //    serial path's RNG/draw exactly.
    __shared__ float topk_val_s[MAX_TOP_K];
    __shared__ int   topk_idx_s[MAX_TOP_K];

    // Cooperative copy of the prefix is a 1-warp job (K<=256), but a
    // single-thread copy is also fine — K is tiny vs the sort cost.
    if (tid < K) {
        topk_val_s[tid] = s_vals[tid];
        topk_idx_s[tid] = s_idxs[tid];
    }
    __syncthreads();

    if (tid == 0) {
        const float inv_t = 1.0f / temp;
        // Numerically stable softmax (max already at index 0 post-sort, but
        // after temperature scaling re-confirm).
        float mx = topk_val_s[0] * inv_t;
        for (int r = 1; r < K; ++r) {
            float v = topk_val_s[r] * inv_t;
            if (v > mx) mx = v;
        }
        float sum = 0.0f;
        for (int r = 0; r < K; ++r) {
            float v = expf(topk_val_s[r] * inv_t - mx);
            topk_val_s[r] = v;
            sum += v;
        }
        float inv_sum = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
        for (int r = 0; r < K; ++r) topk_val_s[r] *= inv_sum;

        // RNG: identical derivation to serial sample_topk_kernel.
        uint64_t rs = seed ^ (uint64_t)(step_idx + 1) * 0x9E3779B97F4A7C15ULL;
        if (rs == 0) rs = 0xDEADBEEFCAFEBABEULL;
        xorshift64s(rs); // burn one to mix
        float r = u01_from_u64(xorshift64s(rs));
        float cs = 0.0f;
        int chosen = topk_idx_s[K - 1];
        for (int q = 0; q < K; ++q) {
            cs += topk_val_s[q];
            if (r <= cs) { chosen = topk_idx_s[q]; break; }
        }
        out_token_dev[step_idx] = chosen;
    }
}

// --------------------------------------------------------------------------
// Cached env-var read for the parallel-topk gate. Reads OMNX_TTS_PARALLEL_TOPK
// once on first call; "0" disables (forces serial fallback), anything else
// enables. Default ON.
// --------------------------------------------------------------------------
inline int parallel_topk_enabled() {
    static int cached = -1;
    if (cached < 0) {
        const char *e = std::getenv("OMNX_TTS_PARALLEL_TOPK");
        if (e && e[0] == '0' && e[1] == '\0') cached = 0;
        else cached = 1;
    }
    return cached;
}

}  // namespace

// --------------------------------------------------------------------------
// Public launchers (declared in cuda_kernels.h, defined here).
// --------------------------------------------------------------------------

void launch_apply_repetition_penalty_f32(float *logits, int lo, int hi,
                                          const int *recent_tokens_dev,
                                          int n_recent, int recent_offset,
                                          float penalty,
                                          cudaStream_t stream) {
    if (n_recent <= 0 || penalty == 1.0f) return;
    // Sequential 1-thread kernel — see comment on rep_penalty_kernel for why.
    rep_penalty_kernel<<<1, 1, 0, stream>>>(
        logits, lo, hi, recent_tokens_dev, n_recent, recent_offset, penalty);
}

void launch_sample_top_k_f32(const float *logits,
                              int lo, int hi,
                              int top_k,
                              float temperature,
                              int do_sample,
                              uint64_t seed,
                              int step_idx,
                              int *out_token_dev,
                              cudaStream_t stream) {
    const int V = hi - lo;
    if (parallel_topk_enabled() && V > 0 && V <= NPOW2_MAX) {
        sample_topk_kernel_bitonic<<<1, BITONIC_THREADS, 0, stream>>>(
            logits, lo, hi, top_k, temperature, do_sample,
            seed, step_idx, out_token_dev);
        return;
    }
    int threads = 256;  // must be a power of two for the reduction.
    sample_topk_kernel<<<1, threads, 0, stream>>>(
        logits, lo, hi, top_k, temperature, do_sample,
        seed, step_idx, out_token_dev);
}

}  // namespace ominix_cuda
