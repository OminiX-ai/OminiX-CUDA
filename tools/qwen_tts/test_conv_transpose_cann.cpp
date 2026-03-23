// Minimal standalone test: ConvTranspose1D CANN vs CPU
// Build: see CMakeLists.txt test_conv_transpose target
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

struct ConvT1DResult {
    std::vector<float> output;
    int out_len;
};

static ConvT1DResult run_on_backend(
    ggml_backend_t backend, ggml_backend_t cpu_fallback,
    int L, int C_in, int C_out, int K, int stride,
    const std::vector<float> &input, const std::vector<float> &weight)
{
    size_t ctx_size = ggml_tensor_overhead() * 10 + 1024 * 1024;
    struct ggml_init_params p = { ctx_size, nullptr, true };
    ggml_context *ctx = ggml_init(p);

    ggml_tensor *w = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, K, C_out, C_in);
    ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, L, C_in);
    ggml_tensor *y = ggml_conv_transpose_1d(ctx, w, x, stride, 0, 1);
    ggml_set_name(w, "w"); ggml_set_name(x, "x"); ggml_set_name(y, "y");
    ggml_set_input(w); ggml_set_input(x); ggml_set_output(y);

    ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    std::vector<ggml_backend_t> backends = {backend};
    std::vector<ggml_backend_buffer_type_t> bufts = {ggml_backend_get_default_buffer_type(backend)};
    if (cpu_fallback && backend != cpu_fallback) {
        backends.push_back(cpu_fallback);
        bufts.push_back(ggml_backend_get_default_buffer_type(cpu_fallback));
    }

    ggml_backend_sched_t sched = ggml_backend_sched_new(
        backends.data(), bufts.data(), (int)backends.size(), 1024, false, true);
    ggml_backend_sched_alloc_graph(sched, graph);

    ggml_backend_tensor_set(w, weight.data(), 0, weight.size() * sizeof(float));
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    ggml_backend_sched_graph_compute(sched, graph);

    int out_len = (int)y->ne[0];
    int out_total = out_len * C_out;
    std::vector<float> out(out_total);
    ggml_backend_tensor_get(y, out.data(), 0, out_total * sizeof(float));

    ggml_backend_sched_free(sched);
    ggml_free(ctx);

    return {out, out_len};
}

int main() {
    ggml_backend_load_all();

    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    ggml_backend_dev_t cann_dev = ggml_backend_dev_by_name("CANN0");
    ggml_backend_t cann = cann_dev ? ggml_backend_dev_init(cann_dev, nullptr) : nullptr;

    if (!cann) {
        printf("CANN0 not available\n");
        return 1;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 0.1f);

    struct TC { int L, Ci, Co, K, S; const char *name; };
    TC cases[] = {
        {16,    64,   64,  4, 2, "small"},
        {32,    64,   64,  4, 2, "medium"},
        {64,   128,  128,  4, 2, "larger"},
        {94,  1024, 1024,  4, 2, "dec_up0"},
        {188, 1024, 1024,  4, 2, "dec_up1"},
        {328, 1536,  768, 16, 8, "voc_blk0"},
    };

    printf("ConvTranspose1D: CANN0 vs CPU\n");
    printf("%-10s %-6s %-6s %-6s %-3s %-3s │ %-12s %-12s\n",
           "name", "L", "Ci", "Co", "K", "S", "max_diff", "avg_diff");
    printf("────────────────────────────────────┼─────────────────────────\n");

    for (auto &tc : cases) {
        std::vector<float> input(tc.L * tc.Ci), weight(tc.K * tc.Co * tc.Ci);
        for (auto &v : input) v = dist(rng);
        for (auto &v : weight) v = dist(rng);

        auto cpu_res = run_on_backend(cpu, nullptr, tc.L, tc.Ci, tc.Co, tc.K, tc.S, input, weight);
        auto npu_res = run_on_backend(cann, cpu, tc.L, tc.Ci, tc.Co, tc.K, tc.S, input, weight);

        float max_d = 0, sum_d = 0;
        int n = std::min(cpu_res.output.size(), npu_res.output.size());
        int nans = 0;
        for (int i = 0; i < n; i++) {
            if (std::isnan(npu_res.output[i])) { nans++; continue; }
            float d = std::abs(npu_res.output[i] - cpu_res.output[i]);
            max_d = std::max(max_d, d);
            sum_d += d;
        }

        printf("%-10s %-6d %-6d %-6d %-3d %-3d │ %-12.6f %-12.8f",
               tc.name, tc.L, tc.Ci, tc.Co, tc.K, tc.S, max_d, n > 0 ? sum_d/n : 0.f);
        if (nans) printf(" NaN=%d", nans);
        printf("\n");
    }

    ggml_backend_free(cann);
    ggml_backend_free(cpu);
    return 0;
}
