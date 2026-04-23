// QIE-Q2.2 Q4_1 variant probe — test three sign/layout hypotheses in parallel.
//
// Agent: QIE-Q2.2-VARIANTS (2026-04-22)
// Predecessor: `test_qie_q4_1_probe.cpp` — RED (cos_sim = -0.034, op accepted
//              but numerics uncorrelated; docs/qie_q22_repack_smoke.md §7).
// Contract:    Q1.10 amendment (mixed-quant repack extension; `afb3919e`).
//
// The RED probe confirmed the op accepts W4 + antiquantScale +
// antiquantOffset at group size 32. That rules out a capability gap and
// implicates a sign / encoding / layout mismatch between our packing and
// what WQBMMv3 actually consumes. Three hypotheses (§7 of the smoke doc):
//
//   Variant A — OPPOSITE SIGN CONVENTION
//     Current (RED): offset = -m/d  (gives w_hat = (u - (-m/d))*d = u*d + m)
//     Try:           offset = +m/d  (gives w_hat = (u - (+m/d))*d = u*d - m).
//     If the op computes  w_hat = q*scale + offset  (additive) instead of
//     (q - offset)*scale, then our encoding for `u*d + m` is actually
//     scale=d, offset=m (pre-multiplied or as-is). The sign flip +m/d
//     is an additional probe point: any of (+m/d, -m/d stored as +m/d-ish)
//     that clears >0.99 pins the semantics.
//
//   Variant B — SIGNED NIBBLE ENCODING
//     Current (RED): unsigned nibble u ∈ [0, 15], offset = -m/d.
//     Try:           signed nibble s = u ^ 0x08 (so s ∈ [-8, 7] after
//                    reinterpretation), and shift offset by +8 to
//                    compensate: offset' = -m/d - 8.
//                    Dequant math:
//                      u*d + m = (s + 8)*d + m
//                             = s*d + (8d + m)
//                             = d * (s - (- (8d+m)/d))
//                             = d * (s - (-8 - m/d))
//                             = d * (s - (offset'))   ✓
//     Rationale: WQBMMv3 Q4_0 path requires XOR 0x08 to reinterpret
//     unsigned [0..15] as signed [-8..7]. If the op forces that same
//     reinterpretation on the W4 input path regardless of whether
//     antiquantOffset is provided, our unsigned Q4_1 nibble is being
//     shifted by -8 with no compensation.
//
//   Variant C — TRANSPOSED OFFSET SHAPE
//     Current (RED): offset shape [K/G, N], row-major, strides (N, 1).
//     Try:           offset shape [N, K/G], strides (K/G, 1); equivalent
//                    data re-arranged as if N is the outer axis and
//                    K/G is the inner.
//     Rationale: WQBMMv3's broadcast convention for antiquantOffset may
//     expect N-outer / K/G-inner (i.e. per-column groups laid out
//     contiguously along the group axis), not the [K/G, N] row-major
//     we naively matched to scale.
//
// Each variant runs with its own freshly-uploaded device buffers so
// cross-contamination is impossible. Each produces cos_sim + mae + wall,
// all printed as one table at the end. Host-side CPU reference is
// identical across variants and computed once (Q4_1 dequant math is
// independent of how we encode it for the NPU).
//
// Decision rule:
//   - First variant with cos_sim > 0.99 wins → emit its config name so
//     the dispatch can patch `/tmp/qie_q22_q4_1.patch` accordingly.
//   - If all four RED, escalate (CANN vendor clarification OR accept
//     F16 fallback per §7 option 3).
//
// Shape (same as the RED probe for direct comparison):
//   x            = [M=128, K=3072] F16
//   weight (W4)  = [K=3072, N=3072] INT4 per-group G=32 along K
//   y            = [M=128, N=3072] F16
//
// Build on ac03:
//   bash build_and_run_q4_1_variants.sh

#include <acl/acl.h>
#include <aclnnop/aclnn_weight_quant_batch_matmul_v3.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define ACL_CHECK(expr) do {                                                     \
    aclError __err = (expr);                                                     \
    if (__err != ACL_SUCCESS) {                                                  \
        fprintf(stderr, "ACL error %d at %s:%d: %s\n",                           \
                (int)__err, __FILE__, __LINE__, #expr);                          \
        std::abort();                                                            \
    }                                                                            \
} while (0)

#define ACL_CHECK_NN(expr) do {                                                  \
    aclnnStatus __st = (expr);                                                   \
    if (__st != 0) {                                                             \
        fprintf(stderr, "aclnn error %d at %s:%d: %s\n",                         \
                (int)__st, __FILE__, __LINE__, #expr);                          \
        std::abort();                                                            \
    }                                                                            \
} while (0)

// ---------- F16 <-> F32 ----------
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint16_t res = (uint16_t)(mant >> (14 - exp));
        return (uint16_t)(sign | res);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

// ---------- Shape constants ----------
static constexpr int64_t M = 128;
static constexpr int64_t K = 3072;
static constexpr int64_t N = 3072;
static constexpr int64_t G = 32;
static constexpr int64_t K_G = K / G;

// ---------- Per-group Q4_1 quantize (one canonical pass — used by all
// variants; individual variants then re-encode nibble/offset from this
// shared dense ground truth).
struct Q41Dense {
    // Per-group d and m for every (group, column).
    std::vector<float>   d;   // K_G × N
    std::vector<float>   m;   // K_G × N
    // Per-element unsigned nibble u ∈ [0, 15].
    std::vector<uint8_t> u;   // K × N
    // Dequant reference in F16 — shared CPU ref input.
    std::vector<uint16_t> dequant_f16; // K × N
};

static Q41Dense cpu_q4_1_dense(const std::vector<float>& w_dense,
                               int64_t k, int64_t n, int64_t g) {
    Q41Dense out;
    out.d.assign((size_t)(k / g) * n, 0.0f);
    out.m.assign((size_t)(k / g) * n, 0.0f);
    out.u.assign((size_t)k * n, 0);
    out.dequant_f16.assign((size_t)k * n, 0);

    for (int64_t col = 0; col < n; ++col) {
        for (int64_t grp = 0; grp < k / g; ++grp) {
            float vmin =  std::numeric_limits<float>::infinity();
            float vmax = -std::numeric_limits<float>::infinity();
            for (int64_t i = 0; i < g; ++i) {
                float v = w_dense[(grp * g + i) * n + col];
                if (v < vmin) vmin = v;
                if (v > vmax) vmax = v;
            }
            float d = (vmax - vmin) / 15.0f;
            float m = vmin;
            if (d == 0.0f) d = 1e-7f;
            out.d[(size_t)grp * n + col] = d;
            out.m[(size_t)grp * n + col] = m;
            for (int64_t i = 0; i < g; ++i) {
                int64_t k_idx = grp * g + i;
                float v = w_dense[k_idx * n + col];
                int u = (int)std::lrintf((v - m) / d);
                if (u < 0)  u = 0;
                if (u > 15) u = 15;
                out.u[(size_t)k_idx * n + col] = (uint8_t)u;
                out.dequant_f16[(size_t)k_idx * n + col] =
                    f32_to_f16((float)u * d + m);
            }
        }
    }
    return out;
}

// ---------- Pack nibbles column-major (linear = col*K + k_idx) ----------
static std::vector<uint8_t> pack_nibbles_colmajor(
        const std::vector<uint8_t>& u4,   // K*N unsigned 4-bit values
        int64_t k, int64_t n,
        bool xor_0x08)                     // if true, emit signed via ^0x08
{
    std::vector<uint8_t> packed((size_t)k * n / 2, 0);
    for (int64_t col = 0; col < n; ++col) {
        for (int64_t kk = 0; kk < k; ++kk) {
            uint8_t nib = u4[(size_t)kk * n + col] & 0x0f;
            if (xor_0x08) nib = (uint8_t)(nib ^ 0x08);
            size_t lin = (size_t)col * k + kk;
            size_t byte_i = lin / 2;
            if ((lin & 1) == 0) {
                packed[byte_i] = (packed[byte_i] & 0xf0) | nib;
            } else {
                packed[byte_i] = (packed[byte_i] & 0x0f) | (uint8_t)(nib << 4);
            }
        }
    }
    return packed;
}

// ---------- CPU F32 matmul y[M,N] = x[M,K] @ w[K,N] (F16 inputs) ----------
static void cpu_matmul_f16(const std::vector<uint16_t>& x,
                           const std::vector<uint16_t>& w,
                           std::vector<float>& y) {
    y.assign((size_t)M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t kk = 0; kk < K; ++kk) {
            float xv = f16_to_f32(x[(size_t)i * K + kk]);
            for (int64_t j = 0; j < N; ++j) {
                float wv = f16_to_f32(w[(size_t)kk * N + j]);
                y[(size_t)i * N + j] += xv * wv;
            }
        }
    }
}

static double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / std::sqrt(na * nb);
}

static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    double mm = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > mm) mm = e;
    }
    return mm;
}

// Per-variant result row for the final table.
struct VariantResult {
    std::string name;
    std::string note;      // one-line description of the encoding used.
    bool        op_ok;     // WQBMMv3 accepted the config.
    double      cos_sim;
    double      mae;
    double      wall_us;   // median over 20 iters.
};

// Single-variant runner. Uploads its own buffers, runs matmul, measures
// cos_sim vs the shared CPU reference, and tears down. Returns a row.
struct VariantConfig {
    std::string name;
    std::string note;
    bool        xor_nibble;          // if true, pack as u ^ 0x08.
    // offset_store(group, col) in the source-scale domain (F32) that
    // will be converted to F16 per element.
    // We express as `lambda` on (d, m) → stored F32 offset.
    float (*offset_fn)(float d, float m);
    // If true, stored offset layout is [N, K/G] (variant C); else [K/G, N].
    bool        offset_transposed;
};

static float off_neg_m_over_d(float d, float m) { return -m / d; }          // baseline
static float off_pos_m_over_d(float d, float m) { return  m / d; }          // Variant A
static float off_neg_m_over_d_minus8(float d, float m) {                    // Variant B
    return -m / d - 8.0f;
}
// Round-2 extensions driven by round-1 signals: B was the only variant
// with positive correlation (0.43), so signed-nibble is likely on the
// right track. We now sweep {sign of offset} × {+8, -8, raw m, -m} to
// pin which combination hits > 0.99.
static float off_neg_m_over_d_plus8(float d, float m) {                     // +8 shift
    return -m / d + 8.0f;
}
static float off_pos_m_over_d_minus8(float d, float m) {                    // sign flip + -8
    return  m / d - 8.0f;
}
static float off_pos_m_over_d_plus8(float d, float m) {                     // sign flip + +8
    return  m / d + 8.0f;
}
static float off_neg_m_raw(float d, float m) { (void)d; return -m; }        // raw -m
static float off_pos_m_raw(float d, float m) { (void)d; return  m; }        // raw +m

static VariantResult run_variant(
        const VariantConfig& cfg,
        const std::vector<uint16_t>& x_host,
        const std::vector<uint8_t>&  u4_unsigned,
        const std::vector<float>&    d_per_group,
        const std::vector<float>&    m_per_group,
        const std::vector<float>&    y_cpu_ref,
        aclrtStream stream)
{
    VariantResult r;
    r.name    = cfg.name;
    r.note    = cfg.note;
    r.op_ok   = false;
    r.cos_sim = 0.0;
    r.mae     = 0.0;
    r.wall_us = 0.0;

    printf("\n========== %s ==========\n", cfg.name.c_str());
    printf("  note: %s\n", cfg.note.c_str());

    // Build packed nibbles per the variant's XOR flag.
    std::vector<uint8_t> packed = pack_nibbles_colmajor(u4_unsigned, K, N,
                                                        cfg.xor_nibble);

    // Build scale (always [K/G, N] F16) — identical layout across variants.
    std::vector<uint16_t> scales_h((size_t)K_G * N, 0);
    for (size_t i = 0; i < scales_h.size(); ++i) {
        scales_h[i] = f32_to_f16(d_per_group[i]);
    }

    // Build offset in the stored (F16) domain. Data is keyed by (group, col);
    // layout depends on offset_transposed.
    std::vector<uint16_t> offsets_h((size_t)K_G * N, 0);
    if (!cfg.offset_transposed) {
        // [K/G, N] row-major — value at (g, c) stored at g*N + c.
        for (int64_t g = 0; g < K_G; ++g) {
            for (int64_t c = 0; c < N; ++c) {
                float d = d_per_group[(size_t)g * N + c];
                float m = m_per_group[(size_t)g * N + c];
                offsets_h[(size_t)g * N + c] = f32_to_f16(cfg.offset_fn(d, m));
            }
        }
    } else {
        // [N, K/G] row-major — value at (c, g) stored at c*K_G + g.
        for (int64_t g = 0; g < K_G; ++g) {
            for (int64_t c = 0; c < N; ++c) {
                float d = d_per_group[(size_t)g * N + c];
                float m = m_per_group[(size_t)g * N + c];
                offsets_h[(size_t)c * K_G + g] = f32_to_f16(cfg.offset_fn(d, m));
            }
        }
    }

    // ---- Device upload ----
    void *x_dev = nullptr, *w_dev = nullptr, *s_dev = nullptr,
         *o_dev = nullptr, *y_dev = nullptr;
    size_t x_bytes = x_host.size() * sizeof(uint16_t);
    size_t w_bytes = packed.size();
    size_t s_bytes = scales_h.size()  * sizeof(uint16_t);
    size_t o_bytes = offsets_h.size() * sizeof(uint16_t);
    size_t y_bytes = (size_t)M * N * sizeof(uint16_t);
    ACL_CHECK(aclrtMalloc(&x_dev, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&w_dev, w_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&s_dev, s_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&o_dev, o_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&y_dev, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(x_dev, x_bytes, x_host.data(), x_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(w_dev, w_bytes, packed.data(), w_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(s_dev, s_bytes, scales_h.data(), s_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(o_dev, o_bytes, offsets_h.data(), o_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    // ---- Build tensors ----
    int64_t x_shape[2]   = {M, K};
    int64_t x_strides[2] = {K, 1};
    int64_t x_storage[2] = {M, K};
    aclTensor* t_x = aclCreateTensor(
        x_shape, 2, ACL_FLOAT16, x_strides, 0, ACL_FORMAT_ND,
        x_storage, 2, x_dev);

    int64_t w_shape[2]   = {K, N};
    int64_t w_strides[2] = {1, K};
    int64_t w_storage    = K * N;
    aclTensor* t_w = aclCreateTensor(
        w_shape, 2, ACL_INT4, w_strides, 0, ACL_FORMAT_ND,
        &w_storage, 1, w_dev);

    int64_t s_shape[2]   = {K_G, N};
    int64_t s_strides[2] = {N, 1};
    int64_t s_storage[2] = {K_G, N};
    aclTensor* t_s = aclCreateTensor(
        s_shape, 2, ACL_FLOAT16, s_strides, 0, ACL_FORMAT_ND,
        s_storage, 2, s_dev);

    // Offset tensor: shape depends on variant.
    int64_t o_shape[2];
    int64_t o_strides[2];
    int64_t o_storage[2];
    if (!cfg.offset_transposed) {
        o_shape[0]   = K_G; o_shape[1]   = N;
        o_strides[0] = N;   o_strides[1] = 1;
        o_storage[0] = K_G; o_storage[1] = N;
    } else {
        o_shape[0]   = N;   o_shape[1]   = K_G;
        o_strides[0] = K_G; o_strides[1] = 1;
        o_storage[0] = N;   o_storage[1] = K_G;
    }
    aclTensor* t_o = aclCreateTensor(
        o_shape, 2, ACL_FLOAT16, o_strides, 0, ACL_FORMAT_ND,
        o_storage, 2, o_dev);

    int64_t y_shape[2]   = {M, N};
    int64_t y_strides[2] = {N, 1};
    int64_t y_storage[2] = {M, N};
    aclTensor* t_y = aclCreateTensor(
        y_shape, 2, ACL_FLOAT16, y_strides, 0, ACL_FORMAT_ND,
        y_storage, 2, y_dev);

    uint64_t ws_bytes = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
        t_x, t_w, t_s, t_o,
        nullptr, nullptr, nullptr,
        (int)G, 1, t_y, &ws_bytes, &exec);

    if (st != 0) {
        fprintf(stderr, "  [%s] WQBMMv3 GetWorkspaceSize REJECTED status=%d\n",
                cfg.name.c_str(), (int)st);
        aclDestroyTensor(t_x); aclDestroyTensor(t_w);
        aclDestroyTensor(t_s); aclDestroyTensor(t_o); aclDestroyTensor(t_y);
        aclrtFree(x_dev); aclrtFree(w_dev); aclrtFree(s_dev);
        aclrtFree(o_dev); aclrtFree(y_dev);
        r.op_ok = false;
        return r;
    }
    r.op_ok = true;

    void* ws_dev = nullptr;
    if (ws_bytes > 0) {
        ACL_CHECK(aclrtMalloc(&ws_dev, ws_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    const int warmup = 3;
    const int iters  = 20;
    for (int i = 0; i < warmup; ++i) {
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3(ws_dev, ws_bytes, exec, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
            t_x, t_w, t_s, t_o, nullptr, nullptr, nullptr,
            (int)G, 1, t_y, &ws_bytes, &exec));
    }
    std::vector<double> times_us;
    times_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3(ws_dev, ws_bytes, exec, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
            t_x, t_w, t_s, t_o, nullptr, nullptr, nullptr,
            (int)G, 1, t_y, &ws_bytes, &exec));
    }
    std::sort(times_us.begin(), times_us.end());
    r.wall_us = times_us[iters / 2];

    // ---- Copy back & compare ----
    std::vector<uint16_t> y_npu_h((size_t)M * N);
    ACL_CHECK(aclrtMemcpy(y_npu_h.data(), y_bytes, y_dev, y_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> y_npu_f32((size_t)M * N);
    for (size_t i = 0; i < y_npu_h.size(); ++i) y_npu_f32[i] = f16_to_f32(y_npu_h[i]);
    r.cos_sim = cosine_sim(y_cpu_ref, y_npu_f32);
    r.mae     = max_abs_err(y_cpu_ref, y_npu_f32);

    printf("  [%s] op_ok=1  cos_sim=%.6f  mae=%.6f  wall=%.1f us\n",
           cfg.name.c_str(), r.cos_sim, r.mae, r.wall_us);

    // ---- Cleanup variant buffers ----
    aclDestroyTensor(t_x); aclDestroyTensor(t_w);
    aclDestroyTensor(t_s); aclDestroyTensor(t_o); aclDestroyTensor(t_y);
    if (ws_dev) aclrtFree(ws_dev);
    aclrtFree(x_dev); aclrtFree(w_dev); aclrtFree(s_dev);
    aclrtFree(o_dev); aclrtFree(y_dev);
    return r;
}

int main() {
    printf("=== QIE-Q2.2 Q4_1 variant probe — sign/layout hypothesis sweep ===\n");
    printf("Shape: x[M=%lld, K=%lld] F16  @  w[K=%lld, N=%lld] INT4 (G=%lld, Q4_1)\n",
           (long long)M, (long long)K, (long long)K, (long long)N, (long long)G);
    printf("Variants (14): round 1 = {baseline,A,B,C}; round 2 sweeps B\n");
    printf("around {+-m/d} x {+-8 shift, signed nibble}; round 3 tests\n");
    printf("additive (w*q + offset) with raw m offset.\n\n");

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // ---------- Shared host tensors ----------
    std::mt19937_64 rng(0xC0FFEE);  // same seed as the RED probe.
    std::uniform_real_distribution<float> xdist(-0.08f, 0.08f);
    std::normal_distribution<float>       wdist(0.15f, 0.30f);

    std::vector<uint16_t> x_host((size_t)M * K);
    for (auto& v : x_host) v = f32_to_f16(xdist(rng));

    std::vector<float> w_dense((size_t)K * N);
    for (auto& v : w_dense) v = wdist(rng);

    printf("[host] Quantizing %lld × %lld weight Q4_1 per-group (G=%lld)...\n",
           (long long)K, (long long)N, (long long)G);
    Q41Dense q = cpu_q4_1_dense(w_dense, K, N, G);

    // Sanity: report the asymmetry of the stored offset (unchanged vs RED).
    {
        double mean_off = 0.0;
        for (size_t i = 0; i < q.d.size(); ++i) {
            mean_off += (double)(-q.m[i] / q.d[i]);
        }
        mean_off /= (double)q.d.size();
        printf("[host] mean(-m/d) = %.3f  (0 ⇒ symmetric; ~-7.5 ⇒ Q4_0-equiv)\n",
               mean_off);
    }

    // ---------- Shared CPU reference ----------
    printf("[cpu]  Computing F32 reference via CPU F16 matmul over dequant...\n");
    std::vector<float> y_cpu;
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul_f16(x_host, q.dequant_f16, y_cpu);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
    printf("[cpu]  Reference done in %.1f ms\n", cpu_ms);

    // ---------- Run each variant ----------
    std::vector<VariantConfig> cfgs = {
        // ---- Round 1 (original three hypotheses + baseline) ----
        {
            "baseline",
            "offset=-m/d, unsigned nibble, layout [K/G,N]",
            /*xor_nibble=*/false,
            off_neg_m_over_d,
            /*offset_transposed=*/false,
        },
        {
            "variant_A",
            "offset=+m/d, unsigned nibble, layout [K/G,N]  (sign flip)",
            /*xor_nibble=*/false,
            off_pos_m_over_d,
            /*offset_transposed=*/false,
        },
        {
            "variant_B",
            "offset=-m/d - 8, signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_neg_m_over_d_minus8,
            /*offset_transposed=*/false,
        },
        {
            "variant_C",
            "offset=-m/d, unsigned nibble, layout [N,K/G]  (transposed)",
            /*xor_nibble=*/false,
            off_neg_m_over_d,
            /*offset_transposed=*/true,
        },
        // ---- Round 2 (driven by B = 0.43 signal — sweep around signed-nibble) ----
        // B hit +0.43 while baseline was -0.03 and A was -0.58. Signed-nibble is
        // likely on the right track; sweep the sign/shift cross-product to pin.
        {
            "variant_B_ns",  // signed nibble, no ±8 compensation
            "offset=-m/d, signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_neg_m_over_d,
            /*offset_transposed=*/false,
        },
        {
            "variant_B_p8",  // signed nibble, offset = -m/d + 8
            "offset=-m/d + 8, signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_neg_m_over_d_plus8,
            /*offset_transposed=*/false,
        },
        {
            "variant_B_Ap8", // signed nibble, sign-flipped offset = +m/d + 8
            "offset=+m/d + 8, signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_pos_m_over_d_plus8,
            /*offset_transposed=*/false,
        },
        {
            "variant_B_Am8", // signed nibble, sign-flipped offset = +m/d - 8
            "offset=+m/d - 8, signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_pos_m_over_d_minus8,
            /*offset_transposed=*/false,
        },
        {
            "variant_B_A",   // signed nibble, sign-flipped offset = +m/d
            "offset=+m/d, signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_pos_m_over_d,
            /*offset_transposed=*/false,
        },
        // ---- Round 3 (additive convention: w_hat = q*scale + offset) ----
        // If op is additive rather than (q - off)*scale, the stored offset
        // should be in source-scale units (= ±m), not nibble-space (±m/d).
        {
            "variant_D_um",  // unsigned nibble, raw -m offset (additive?)
            "offset=-m (raw), unsigned nibble, layout [K/G,N]",
            /*xor_nibble=*/false,
            off_neg_m_raw,
            /*offset_transposed=*/false,
        },
        {
            "variant_D_up",  // unsigned nibble, raw +m offset
            "offset=+m (raw), unsigned nibble, layout [K/G,N]",
            /*xor_nibble=*/false,
            off_pos_m_raw,
            /*offset_transposed=*/false,
        },
        {
            "variant_D_sm",  // signed nibble, raw -m offset
            "offset=-m (raw), signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_neg_m_raw,
            /*offset_transposed=*/false,
        },
        {
            "variant_D_sp",  // signed nibble, raw +m offset
            "offset=+m (raw), signed nibble (u^0x08), layout [K/G,N]",
            /*xor_nibble=*/true,
            off_pos_m_raw,
            /*offset_transposed=*/false,
        },
    };

    std::vector<VariantResult> results;
    results.reserve(cfgs.size());
    for (const auto& cfg : cfgs) {
        results.push_back(run_variant(cfg, x_host, q.u, q.d, q.m, y_cpu, stream));
    }

    // ---------- Summary table ----------
    printf("\n\n=== Variant result table ===\n");
    printf("| %-10s | %-6s | %-10s | %-10s | %-10s |\n",
           "variant", "op_ok", "cos_sim", "mae", "wall_us");
    printf("|%-12s|%-8s|%-12s|%-12s|%-12s|\n",
           "------------", "--------", "------------",
           "------------", "------------");
    for (const auto& r : results) {
        printf("| %-10s | %-6s | %-10.6f | %-10.6f | %-10.1f |\n",
               r.name.c_str(),
               r.op_ok ? "yes" : "no",
               r.cos_sim, r.mae, r.wall_us);
    }

    // ---------- Verdict ----------
    printf("\n");
    int rc = 2;
    const VariantResult* winner = nullptr;
    for (const auto& r : results) {
        if (r.op_ok && r.cos_sim > 0.99) { winner = &r; break; }
    }
    if (winner) {
        printf("[verdict] GREEN on %s  (cos_sim = %.6f, mae = %.6f, wall = %.1f us)\n",
               winner->name.c_str(), winner->cos_sim, winner->mae, winner->wall_us);
        printf("          Apply this config to /tmp/qie_q22_q4_1.patch: %s\n",
               winner->note.c_str());
        rc = 0;
    } else {
        // Fall back to check for YELLOW (cos > 0.90).
        for (const auto& r : results) {
            if (r.op_ok && r.cos_sim > 0.90) {
                printf("[verdict] YELLOW on %s  (cos_sim = %.6f) — close but below 0.99\n",
                       r.name.c_str(), r.cos_sim);
                rc = 1;
                break;
            }
        }
        if (rc == 2) {
            printf("[verdict] RED — no variant cleared 0.99.\n");
            printf("          Escalate per smoke §7: CANN vendor clarification\n");
            printf("          OR accept Q4_1 F16 fallback (+1.27 GiB, 13 GiB gate still clears).\n");
        }
    }

    // ---------- Cleanup ----------
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return rc;
}
