/**
 * Minimal test: run conv2d1 alone and compare with Python reference.
 * This tests the GGML conv2d implementation directly.
 */
#include "audio_encoder.h"
#include "ctx_manager.h"
#include "model_loader.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <map>

// Simple NPY reader (float32 only)
static bool load_npy_f32(const char *path, std::vector<float> &data, std::vector<int> &shape) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[6]; fread(magic, 1, 6, f);
    uint8_t major, minor; fread(&major, 1, 1, f); fread(&minor, 1, 1, f);
    uint16_t hlen; fread(&hlen, 2, 1, f);
    std::vector<char> hdr(hlen); fread(hdr.data(), 1, hlen, f);
    std::string hs(hdr.begin(), hdr.end());
    auto s = hs.find('('), e = hs.find(')');
    std::string ss = hs.substr(s+1, e-s-1);
    shape.clear();
    size_t p = 0;
    while (p < ss.size()) {
        while (p < ss.size() && (ss[p]==' '||ss[p]==',')) p++;
        if (p >= ss.size()) break;
        int v = 0;
        while (p < ss.size() && ss[p]>='0' && ss[p]<='9') v = v*10+(ss[p++]-'0');
        shape.push_back(v);
    }
    int total = 1; for (int s : shape) total *= s;
    data.resize(total); fread(data.data(), 4, total, f);
    fclose(f); return true;
}

int main() {
    const char *gguf_path = "tools/qwen_asr/gguf/qwen_asr_audio_encoder_f32.gguf";
    const char *data_dir = "tools/qwen_asr/verify_data/";

    // Load padded mel (10, 1, 128, 100)
    std::vector<float> padded_mel;
    std::vector<int> mel_shape;
    char mel_path[256]; snprintf(mel_path, sizeof(mel_path), "%spadded_mel.npy", data_dir);
    if (!load_npy_f32(mel_path, padded_mel, mel_shape)) {
        printf("Failed to load padded_mel.npy\n"); return 1;
    }
    printf("Padded mel: [%d, %d, %d, %d]\n", mel_shape[0], mel_shape[1], mel_shape[2], mel_shape[3]);
    int batch = mel_shape[0]; // 10
    int mel_h = mel_shape[2]; // 128
    int mel_w = mel_shape[3]; // 100

    // Load GGUF
    ModelLoader loader(gguf_path);

    // Create backend
    ggml_backend *backend = ggml_backend_cpu_init();
    ggml_backend_cpu_set_n_threads(backend, 8);

    // Get weight tensors
    ggml_context *ctx_data = loader.ctx_meta_.get();
    std::vector<ggml_tensor *> tensors;
    ggml_tensor *conv2d1_w = ggml_get_tensor(ctx_data, "conv2d1.weight");
    ggml_tensor *conv2d1_b = ggml_get_tensor(ctx_data, "conv2d1.bias");
    tensors.push_back(conv2d1_w);
    tensors.push_back(conv2d1_b);

    printf("conv2d1_w: ne=[%lld,%lld,%lld,%lld], type=%d\n",
           (long long)conv2d1_w->ne[0], (long long)conv2d1_w->ne[1],
           (long long)conv2d1_w->ne[2], (long long)conv2d1_w->ne[3], conv2d1_w->type);
    printf("conv2d1_b: ne=[%lld], type=%d\n", (long long)conv2d1_b->ne[0], conv2d1_b->type);

    // Allocate buffer and load weights
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    ggml_backend_buffer *buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_data, buft);

    // Load weight data from file
    gguf_context *ctx_gguf = loader.ctx_gguf_.get();
    std::map<std::string, size_t> offsets;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf); i++) {
        offsets[gguf_get_tensor_name(ctx_gguf, i)] =
            gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
    }
    std::ifstream fin(gguf_path, std::ios::binary);
    for (auto *t : tensors) {
        auto it = offsets.find(t->name);
        if (it == offsets.end()) { printf("Tensor %s not found\n", t->name); return 1; }
        fin.seekg(it->second);
        size_t nbytes = ggml_nbytes(t);
        std::vector<uint8_t> tmp(nbytes);
        fin.read((char*)tmp.data(), nbytes);
        ggml_backend_tensor_set(t, tmp.data(), 0, nbytes);
    }
    fin.close();

    // Print first few weight values
    {
        float w_vals[9];
        ggml_backend_tensor_get(conv2d1_w, w_vals, 0, 9 * sizeof(float));
        printf("conv2d1_w first 9: ");
        for (int i = 0; i < 9; i++) printf("%.6f ", w_vals[i]);
        printf("\n");
        printf("Expected:          0.011719 -0.010132  0.015259  0.018433 -0.095215 -0.082520  0.042480  0.027954  0.046631\n");
    }

    // Build graph: just conv2d1 + bias + gelu
    int max_nodes = 64;
    size_t buf_size = max_nodes * ggml_tensor_overhead() + ggml_graph_overhead();
    std::vector<uint8_t> compute_buf(buf_size);
    struct ggml_init_params params = {buf_size, compute_buf.data(), true};
    ggml_context *ctx = ggml_init(params);
    ggml_cgraph *gf = ggml_new_graph_custom(ctx, max_nodes, false);

    // Input: (IW=100, IH=128, IC=1, N=10)
    ggml_tensor *inp = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, mel_w, mel_h, 1, batch);
    ggml_set_name(inp, "input");
    ggml_set_input(inp);

    // Conv2d
    ggml_tensor *conv = ggml_conv_2d(ctx, conv2d1_w, inp, 2, 2, 1, 1, 1, 1);
    printf("conv output ne: [%lld, %lld, %lld, %lld]\n",
           (long long)conv->ne[0], (long long)conv->ne[1],
           (long long)conv->ne[2], (long long)conv->ne[3]);

    // Add bias
    ggml_tensor *bias4d = ggml_reshape_4d(ctx, conv2d1_b, 1, 1, 480, 1);
    ggml_tensor *conv_bias = ggml_add(ctx, conv, bias4d);

    // GELU
    ggml_tensor *out = ggml_gelu_erf(ctx, conv_bias);
    ggml_set_name(out, "output");
    ggml_set_output(out);
    ggml_build_forward_expand(gf, out);

    // Allocate and run
    ggml_backend_sched *sched = ggml_backend_sched_new(&backend, NULL, 1, max_nodes, false, false);
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        printf("Failed to alloc graph\n"); return 1;
    }

    // Set input
    ggml_tensor *inp_t = ggml_graph_get_tensor(gf, "input");
    ggml_backend_tensor_set(inp_t, padded_mel.data(), 0, padded_mel.size() * sizeof(float));

    auto status = ggml_backend_sched_graph_compute(sched, gf);
    if (status != GGML_STATUS_SUCCESS) {
        printf("Graph compute failed: %d\n", status); return 1;
    }

    // Get output
    ggml_tensor *out_t = ggml_graph_get_tensor(gf, "output");
    printf("output ne: [%lld, %lld, %lld, %lld]\n",
           (long long)out_t->ne[0], (long long)out_t->ne[1],
           (long long)out_t->ne[2], (long long)out_t->ne[3]);

    int total_out = ggml_nelements(out_t);
    std::vector<float> out_data(total_out);
    ggml_backend_tensor_get(out_t, out_data.data(), 0, total_out * sizeof(float));

    // Load Python reference
    std::vector<float> ref_data;
    std::vector<int> ref_shape;
    char ref_path[256]; snprintf(ref_path, sizeof(ref_path), "%safter_conv2d1.npy", data_dir);
    if (!load_npy_f32(ref_path, ref_data, ref_shape)) {
        printf("Failed to load reference\n"); return 1;
    }
    printf("Reference shape: [%d, %d, %d, %d]\n", ref_shape[0], ref_shape[1], ref_shape[2], ref_shape[3]);
    // Python ref is (10, 480, 64, 50) row-major = (N, OC, OH, OW)

    // GGML output is (OW, OH, OC, N) in ne order
    // In memory: element at (ow, oh, oc, n) = data[n*OC*OH*OW + oc*OH*OW + oh*OW + ow]
    // Python: element at (n, oc, oh, ow) = data[n*OC*OH*OW + oc*OH*OW + oh*OW + ow]
    // These are the SAME layout!

    int OW = out_t->ne[0], OH = out_t->ne[1], OC = out_t->ne[2], N = out_t->ne[3];
    printf("GGML output: OW=%d, OH=%d, OC=%d, N=%d\n", OW, OH, OC, N);

    // Compare first few values
    printf("\nC++ out[0,0,0,:5] (n=0,oc=0,oh=0,ow=0..4): ");
    for (int ow = 0; ow < 5; ow++) {
        printf("%.8f ", out_data[0*OC*OH*OW + 0*OH*OW + 0*OW + ow]);
    }
    printf("\nPy  ref[0,0,0,:5] (n=0,oc=0,oh=0,ow=0..4): ");
    for (int ow = 0; ow < 5; ow++) {
        printf("%.8f ", ref_data[0*OC*OH*OW + 0*OH*OW + 0*OW + ow]);
    }
    printf("\n");

    // Compute stats
    float maxd = 0, sumd = 0;
    for (int i = 0; i < total_out; i++) {
        float d = std::abs(out_data[i] - ref_data[i]);
        maxd = std::max(maxd, d);
        sumd += d;
    }
    printf("\nMax abs diff: %.8f\n", maxd);
    printf("Mean abs diff: %.8f\n", sumd / total_out);

    // Correlation
    double sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    for (int i = 0; i < total_out; i++) {
        sum_a += out_data[i]; sum_b += ref_data[i];
        sum_ab += out_data[i] * ref_data[i];
        sum_a2 += out_data[i] * out_data[i]; sum_b2 += ref_data[i] * ref_data[i];
    }
    double mean_a = sum_a / total_out, mean_b = sum_b / total_out;
    double cov = sum_ab / total_out - mean_a * mean_b;
    double std_a = std::sqrt(sum_a2 / total_out - mean_a * mean_a);
    double std_b = std::sqrt(sum_b2 / total_out - mean_b * mean_b);
    printf("Correlation: %.6f\n", cov / (std_a * std_b + 1e-10));

    ggml_free(ctx);
    ggml_backend_buffer_free(buf);
    ggml_backend_sched_free(sched);
    ggml_backend_free(backend);
    return 0;
}
