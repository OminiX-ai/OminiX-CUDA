// ============================================================================
// Phase 2.6 microbench — compare F16 GemmEx vs FP8 E4M3 cublasLt GEMV at the
// shapes that dominate the Talker/Predictor LM-head and FFN matmuls.
//
// Why a separate bench: the inner decode loop is a long sequence of K=1
// (matvec) GEMMs. cuBLAS historically routes K=1 through CUDA-core kernels
// rather than the tensor-core FP8 path, so the FP8 win can be zero or
// negative even on hardware that supports E4M3. Measure first, then decide.
//
// Shapes measured (out, in):
//   predictor LM-head : [30720, 1024]  -> 1920 calls/request
//   talker LM-head    : [3072,  2048]  -> 128  calls/request
//   talker FFN gate   : [6144,  2048]  -> 28*128 = 3584 calls/request
//   talker FFN down   : [2048,  6144]  -> 28*128 = 3584 calls/request
//   predictor FFN gate: [3072,  1024]  -> 5*1920 = 9600 calls/request
//
// Build: nvcc -O3 -arch=sm_121 bench_fp8_gemv.cu -lcublas -lcublasLt
//        -o /tmp/bench_fp8_gemv
//
// Per shape we report median microseconds over 200 calls (50 warmup) for
// both lanes.
// ============================================================================

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>

#define CHECK_CUDA(x) do { auto e=(x); if (e!=cudaSuccess){fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));std::exit(1);} } while(0)
#define CHECK_CUBLAS(x) do { auto s=(x); if (s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS %s:%d status=%d\n",__FILE__,__LINE__,(int)s);std::exit(1);} } while(0)

struct Shape { int out, in_; const char *name; };

static Shape SHAPES[] = {
    {30720, 1024, "pred_lm_head"},
    { 3072, 2048, "talker_lm_head"},
    { 6144, 2048, "talker_ffn_gate"},
    { 2048, 6144, "talker_ffn_down"},
    { 3072, 1024, "pred_ffn_gate"},
};

// Bench cublasGemmEx F16 IO, F32 compute (current production path).
double bench_f16_gemmex(cublasHandle_t cublas, int out, int in_dim,
                         int iters, int warmup) {
    size_t bytes_W = (size_t)out * in_dim * sizeof(__half);
    size_t bytes_x = (size_t)in_dim * sizeof(__half);
    size_t bytes_y = (size_t)out * sizeof(__half);

    __half *W, *x, *y;
    CHECK_CUDA(cudaMalloc(&W, bytes_W));
    CHECK_CUDA(cudaMalloc(&x, bytes_x));
    CHECK_CUDA(cudaMalloc(&y, bytes_y));
    CHECK_CUDA(cudaMemset(W, 0x3c, bytes_W));  // ~1.0 in f16
    CHECK_CUDA(cudaMemset(x, 0x3c, bytes_x));

    const float alpha = 1.0f, beta = 0.0f;

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            out, 1, in_dim,
            &alpha, W, CUDA_R_16F, in_dim,
                    x, CUDA_R_16F, in_dim,
            &beta,  y, CUDA_R_16F, out,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            out, 1, in_dim,
            &alpha, W, CUDA_R_16F, in_dim,
                    x, CUDA_R_16F, in_dim,
            &beta,  y, CUDA_R_16F, out,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    cudaFree(W); cudaFree(x); cudaFree(y);
    return (double)total_ms * 1000.0 / iters;  // us per call
}

// Bench cublasLtMatmul FP8 E4M3 with f32 scales.
// Layout: A[M,K] FP8 row-major (out=M, in=K), B[K,N=1] FP8, D[M,N=1] F16 out.
// Per cuBLAS docs: A, B layouts must be col-major-friendly. For row-major
// W[out, in], we treat as col-major [in, out] and op_A=T to compute y = W·x.
// FP8 cublasLt requires K to be a multiple of 16 and certain layout rules.
// We use the supported pattern: y_FP8 = scaleD * (scaleA * scaleB * (A^T · B))
// with all scales = 1.0 (we are measuring throughput, not numerics).
double bench_fp8_lt(cublasLtHandle_t lt, int out, int in_dim,
                     int iters, int warmup, bool *supported) {
    size_t bytes_W = (size_t)out * in_dim;          // FP8 = 1 byte/elem
    size_t bytes_x = (size_t)in_dim;
    size_t bytes_y = (size_t)out * sizeof(__half);  // F16 output

    __nv_fp8_e4m3 *W, *x;
    __half *y;
    CHECK_CUDA(cudaMalloc(&W, bytes_W));
    CHECK_CUDA(cudaMalloc(&x, bytes_x));
    CHECK_CUDA(cudaMalloc(&y, bytes_y));
    CHECK_CUDA(cudaMemset(W, 0x40, bytes_W));  // small positive
    CHECK_CUDA(cudaMemset(x, 0x40, bytes_x));

    float scale_one = 1.0f;
    float *d_scaleA, *d_scaleB, *d_scaleD;
    CHECK_CUDA(cudaMalloc(&d_scaleA, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scaleB, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_scaleD, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_scaleA, &scale_one, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scaleB, &scale_one, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scaleD, &scale_one, sizeof(float), cudaMemcpyHostToDevice));

    cublasLtMatmulDesc_t desc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t op_A = CUBLAS_OP_T;
    cublasOperation_t op_B = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_A, sizeof(op_A));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_B, sizeof(op_B));
    // Per CUDA 13.0 cuBLAS docs, FP8 GEMM requires A and B scale pointers.
    // D scale only when D is FP8 (we use F16 D so D-scale is optional).
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_scaleA, sizeof(d_scaleA));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_scaleB, sizeof(d_scaleB));
    // (Skip D scale — output is F16 not FP8.)

    // Layouts.
    // A storage: row-major [out, in]; cublasLt sees col-major [in, out] with ld=in -> op_A=T.
    cublasLtMatrixLayout_t aL, bL, dL;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&aL, CUDA_R_8F_E4M3, in_dim, out, in_dim));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&bL, CUDA_R_8F_E4M3, in_dim, 1,   in_dim));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&dL, CUDA_R_16F,     out,    1,   out));

    // Heuristic.
    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws_bytes = 16 * 1024 * 1024;  // 16 MB workspace
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes));

    cublasLtMatmulHeuristicResult_t heur[1];
    int n_results = 0;
    cublasStatus_t hst = cublasLtMatmulAlgoGetHeuristic(lt, desc,
        aL, bL, dL, dL, pref, 1, heur, &n_results);
    if (hst != CUBLAS_STATUS_SUCCESS || n_results == 0) {
        if (supported) *supported = false;
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(aL); cublasLtMatrixLayoutDestroy(bL); cublasLtMatrixLayoutDestroy(dL);
        cublasLtMatmulDescDestroy(desc);
        cudaFree(W); cudaFree(x); cudaFree(y);
        cudaFree(d_scaleA); cudaFree(d_scaleB); cudaFree(d_scaleD);
        return -1.0;
    }
    if (supported) *supported = true;

    void *workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, ws_bytes));
    const float alpha = 1.0f, beta = 0.0f;

    auto launch = [&]() {
        return cublasLtMatmul(lt, desc, &alpha,
            W, aL, x, bL, &beta,
            y, dL, y, dL,
            &heur[0].algo, workspace, ws_bytes, 0);
    };

    // Warmup.
    for (int i = 0; i < warmup; ++i) {
        cublasStatus_t st = launch();
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "FP8 launch failed status=%d (out=%d in=%d)\n",
                    (int)st, out, in_dim);
            if (supported) *supported = false;
            cudaFree(workspace);
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(aL); cublasLtMatrixLayoutDestroy(bL); cublasLtMatrixLayoutDestroy(dL);
            cublasLtMatmulDescDestroy(desc);
            cudaFree(W); cudaFree(x); cudaFree(y);
            cudaFree(d_scaleA); cudaFree(d_scaleB); cudaFree(d_scaleD);
            return -1.0;
        }
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s, e;
    cudaEventCreate(&s); cudaEventCreate(&e);
    cudaEventRecord(s);
    for (int i = 0; i < iters; ++i) launch();
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, s, e);
    cudaEventDestroy(s); cudaEventDestroy(e);

    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(aL); cublasLtMatrixLayoutDestroy(bL); cublasLtMatrixLayoutDestroy(dL);
    cublasLtMatmulDescDestroy(desc);
    cudaFree(W); cudaFree(x); cudaFree(y);
    cudaFree(d_scaleA); cudaFree(d_scaleB); cudaFree(d_scaleD);
    return (double)total_ms * 1000.0 / iters;
}

int main() {
    cudaSetDevice(0);
    cublasHandle_t cublas;
    cublasLtHandle_t lt;
    cublasCreate(&cublas);
    cublasLtCreate(&lt);

    int prop_dev = 0;
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, prop_dev);
    printf("[bench] device: %s sm=%d.%d  CUDA-toolkit=%d.%d\n",
           prop.name, prop.major, prop.minor, CUDART_VERSION/1000, (CUDART_VERSION%1000)/10);
    printf("[bench] shape | f16_us | fp8_us | speedup | supported\n");

    const int iters = 200, warmup = 50;
    for (auto &S : SHAPES) {
        double t_f16 = bench_f16_gemmex(cublas, S.out, S.in_, iters, warmup);
        bool supported = false;
        double t_fp8 = bench_fp8_lt(lt, S.out, S.in_, iters, warmup, &supported);
        if (supported && t_fp8 > 0) {
            printf("  %-18s | %7.2f | %7.2f | %5.2fx | yes\n",
                   S.name, t_f16, t_fp8, t_f16 / t_fp8);
        } else {
            printf("  %-18s | %7.2f | %7s | %5s  | NO\n",
                   S.name, t_f16, "n/a", "n/a");
        }
    }

    cublasLtDestroy(lt);
    cublasDestroy(cublas);
    return 0;
}
