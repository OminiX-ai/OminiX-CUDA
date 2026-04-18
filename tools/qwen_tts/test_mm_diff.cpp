// Diagnostic: test whether aclnnMm with strided weight (the "transposed view"
// pattern used by the batched prefill path) produces the same result as the
// contiguous W @ X_col pattern used by the decode path.
//
// Setup:
//   W: [K=2048, M=2048] row-major F16, random
//   X: [N=1 or 4, K=2048] row-major F16, random
// Compute Y = X @ W^T = [N, M] via:
//   (a) for each row i, aclnnMm(W_full[M,K], X[i,:][K,1], y_col[M,1]) then stack
//   (b) aclnnMm(X[N,K], W_T_view[K,M] strided (1, K), Y[N,M])
// Both should produce identical values. Does aclnnMm handle the strided
// view correctly?

#include "talker_cann_engine.h"  // to get TalkerConfig header even if unused
#include "cp_cann_symbols.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define ACL_CHECK(stmt) do { \
    aclError _e = (stmt); \
    if (_e != 0) { fprintf(stderr, "ACL error %d at %s:%d\n", _e, __FILE__, __LINE__); return 1; } \
} while(0)

static uint16_t f16_bits(float v) {
    __fp16 h = (__fp16)v;
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}

static float f16_to_f32(uint16_t b) {
    __fp16 h;
    memcpy(&h, &b, 2);
    return (float)h;
}

int main() {
    if (!cp_cann_load_symbols()) { fprintf(stderr, "symbol load failed\n"); return 1; }

    {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CANN");
        if (!reg) { fprintf(stderr, "CANN not registered\n"); return 1; }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { fprintf(stderr, "CANN init failed\n"); return 1; }
    }

    ACL_CHECK(g_cann.aclrtSetDevice(0));
    aclrtStream stream;
    ACL_CHECK(g_cann.aclrtCreateStream(&stream));

    // Real Talker dimensions: n_embd=2048, q_dim=2048, seq_len=127.
    const int K = 2048;
    const int M = 2048;
    const int N = 127;

    // Host data
    std::mt19937 rng(42);
    std::normal_distribution<float> gauss(0.0f, 0.2f);

    std::vector<uint16_t> W(M * K);  // [M, K] row-major
    std::vector<uint16_t> X(N * K);  // [N, K] row-major
    for (auto &w : W) w = f16_bits(gauss(rng));
    for (auto &x : X) x = f16_bits(gauss(rng));

    // Upload to device
    void *dW, *dX, *dY_a, *dY_b;
    ACL_CHECK(g_cann.aclrtMalloc(&dW, M * K * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(g_cann.aclrtMalloc(&dX, N * K * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(g_cann.aclrtMalloc(&dY_a, N * M * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(g_cann.aclrtMalloc(&dY_b, N * M * 2, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(g_cann.aclrtMemcpy(dW, M * K * 2, W.data(), M * K * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(g_cann.aclrtMemcpy(dX, N * K * 2, X.data(), N * K * 2, ACL_MEMCPY_HOST_TO_DEVICE));

    // ---------- Method A: per-row Mm(W, X_col) ----------
    auto make_tensor = [&](void *buf, int rank, const int64_t *shape, const int64_t *strides) -> aclTensor* {
        int64_t storage_len = 0;
        if (rank > 0) {
            int64_t max_off = 0;
            for (int i = 0; i < rank; ++i) {
                if (shape[i] > 0) max_off += (shape[i] - 1) * strides[i];
            }
            storage_len = max_off + 1;
        }
        return g_cann.aclCreateTensor(shape, rank, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND, &storage_len, 1, buf);
    };

    void *workspace = nullptr;
    uint64_t ws_size = 0;

    for (int i = 0; i < N; ++i) {
        uint16_t *x_row_dev = (uint16_t*)dX + i * K;
        uint16_t *y_row_dev = (uint16_t*)dY_a + i * M;

        int64_t W_shape[2] = {M, K};
        int64_t W_strides[2] = {K, 1};
        aclTensor *tW = make_tensor(dW, 2, W_shape, W_strides);

        int64_t Xc_shape[2] = {K, 1};
        int64_t Xc_strides[2] = {1, 1};  // col vector, only 1 col
        aclTensor *tXc = make_tensor(x_row_dev, 2, Xc_shape, Xc_strides);

        int64_t Yc_shape[2] = {M, 1};
        int64_t Yc_strides[2] = {1, 1};
        aclTensor *tYc = make_tensor(y_row_dev, 2, Yc_shape, Yc_strides);

        aclOpExecutor *exec = nullptr;
        uint64_t w_needed = 0;
        ACL_CHECK(g_cann.aclnnMmGetWorkspaceSize(tW, tXc, tYc, 0, &w_needed, &exec));
        if (w_needed > ws_size) {
            if (workspace) g_cann.aclrtFree(workspace);
            ACL_CHECK(g_cann.aclrtMalloc(&workspace, w_needed, ACL_MEM_MALLOC_HUGE_FIRST));
            ws_size = w_needed;
        }
        ACL_CHECK(g_cann.aclnnMm(w_needed > 0 ? workspace : nullptr, w_needed, exec, stream));
        g_cann.aclDestroyTensor(tW);
        g_cann.aclDestroyTensor(tXc);
        g_cann.aclDestroyTensor(tYc);
    }
    ACL_CHECK(g_cann.aclrtSynchronizeStream(stream));

    // ---------- Method B: Mm(X [N, K], W_T [K, M] strided, Y [N, M]) ----------
    {
        int64_t X_shape[2] = {N, K};
        int64_t X_strides[2] = {K, 1};
        aclTensor *tX = make_tensor(dX, 2, X_shape, X_strides);

        int64_t WT_shape[2] = {K, M};
        int64_t WT_strides[2] = {1, K};  // transposed stride view of W[M, K]
        aclTensor *tWT = make_tensor(dW, 2, WT_shape, WT_strides);

        int64_t Y_shape[2] = {N, M};
        int64_t Y_strides[2] = {M, 1};
        aclTensor *tY = make_tensor(dY_b, 2, Y_shape, Y_strides);

        aclOpExecutor *exec = nullptr;
        uint64_t w_needed = 0;
        ACL_CHECK(g_cann.aclnnMmGetWorkspaceSize(tX, tWT, tY, 0, &w_needed, &exec));
        if (w_needed > ws_size) {
            if (workspace) g_cann.aclrtFree(workspace);
            ACL_CHECK(g_cann.aclrtMalloc(&workspace, w_needed, ACL_MEM_MALLOC_HUGE_FIRST));
            ws_size = w_needed;
        }
        ACL_CHECK(g_cann.aclnnMm(w_needed > 0 ? workspace : nullptr, w_needed, exec, stream));
        g_cann.aclDestroyTensor(tX);
        g_cann.aclDestroyTensor(tWT);
        g_cann.aclDestroyTensor(tY);
    }
    ACL_CHECK(g_cann.aclrtSynchronizeStream(stream));

    // Download & compare
    std::vector<uint16_t> Ya_host(N * M), Yb_host(N * M);
    ACL_CHECK(g_cann.aclrtMemcpy(Ya_host.data(), N * M * 2, dY_a, N * M * 2, ACL_MEMCPY_DEVICE_TO_HOST));
    ACL_CHECK(g_cann.aclrtMemcpy(Yb_host.data(), N * M * 2, dY_b, N * M * 2, ACL_MEMCPY_DEVICE_TO_HOST));

    double max_diff = 0;
    double avg_diff = 0;
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < N * M; ++i) {
        float a = f16_to_f32(Ya_host[i]);
        float b = f16_to_f32(Yb_host[i]);
        double d = std::fabs(a - b);
        if (d > max_diff) max_diff = d;
        avg_diff += d;
        dot += (double)a * b;
        na += (double)a * a;
        nb += (double)b * b;
    }
    avg_diff /= (N * M);
    double cos = dot / (std::sqrt(na) * std::sqrt(nb));

    printf("Y[0][:8] method A: ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", f16_to_f32(Ya_host[i]));
    printf("\nY[0][:8] method B: ");
    for (int i = 0; i < 8; ++i) printf("%.4f ", f16_to_f32(Yb_host[i]));
    printf("\nmax_diff=%.6f avg_diff=%.6f cos_sim=%.6f\n", max_diff, avg_diff, cos);

    if (cos < 0.999) {
        printf("!! MISMATCH: the two aclnnMm paths produce different results\n");
    } else {
        printf("OK: methods agree within tolerance\n");
    }
    return 0;
}
