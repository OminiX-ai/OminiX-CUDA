/**
 * Test the audio encoder by loading Python intermediate data and comparing step by step.
 * Reads padded_mel.npy from Python, feeds it through C++ conv2d+transformer,
 * and compares with Python reference at each stage.
 */
#include "audio_encoder.h"
#include "ctx_manager.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>

// Simple NPY reader for float32 arrays
static bool load_npy_f32(const std::string &path, std::vector<float> &data, std::vector<int> &shape) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", path.c_str());
        return false;
    }

    // Read NPY header
    char magic[6];
    fread(magic, 1, 6, f);
    uint8_t major, minor;
    fread(&major, 1, 1, f);
    fread(&minor, 1, 1, f);
    uint16_t header_len;
    fread(&header_len, 2, 1, f);

    std::vector<char> header(header_len);
    fread(header.data(), 1, header_len, f);
    std::string header_str(header.begin(), header.end());

    // Parse shape from header
    auto shape_start = header_str.find('(');
    auto shape_end = header_str.find(')');
    if (shape_start == std::string::npos || shape_end == std::string::npos) {
        fprintf(stderr, "Cannot parse shape from %s\n", path.c_str());
        fclose(f);
        return false;
    }
    std::string shape_str = header_str.substr(shape_start + 1, shape_end - shape_start - 1);

    shape.clear();
    size_t pos = 0;
    while (pos < shape_str.size()) {
        while (pos < shape_str.size() && (shape_str[pos] == ' ' || shape_str[pos] == ','))
            pos++;
        if (pos >= shape_str.size()) break;
        int val = 0;
        while (pos < shape_str.size() && shape_str[pos] >= '0' && shape_str[pos] <= '9') {
            val = val * 10 + (shape_str[pos] - '0');
            pos++;
        }
        shape.push_back(val);
    }

    // Calculate total elements
    int total = 1;
    for (int s : shape) total *= s;
    data.resize(total);
    fread(data.data(), sizeof(float), total, f);
    fclose(f);
    return true;
}

static float compute_correlation(const float *a, const float *b, int n) {
    double sum_a = 0, sum_b = 0, sum_ab = 0, sum_a2 = 0, sum_b2 = 0;
    for (int i = 0; i < n; i++) {
        sum_a += a[i];
        sum_b += b[i];
        sum_ab += a[i] * b[i];
        sum_a2 += a[i] * a[i];
        sum_b2 += b[i] * b[i];
    }
    double mean_a = sum_a / n;
    double mean_b = sum_b / n;
    double cov = sum_ab / n - mean_a * mean_b;
    double std_a = std::sqrt(sum_a2 / n - mean_a * mean_a);
    double std_b = std::sqrt(sum_b2 / n - mean_b * mean_b);
    if (std_a < 1e-10 || std_b < 1e-10) return 0.0;
    return cov / (std_a * std_b);
}

static float max_abs_diff(const float *a, const float *b, int n) {
    float maxd = 0;
    for (int i = 0; i < n; i++) {
        float d = std::abs(a[i] - b[i]);
        if (d > maxd) maxd = d;
    }
    return maxd;
}

int main() {
    const std::string data_dir = "tools/qwen_asr/verify_data/";
    const std::string gguf_path = "tools/qwen_asr/gguf/qwen_asr_audio_encoder_f32.gguf";

    printf("=== Audio Encoder Step-by-Step Test ===\n\n");

    // Load the audio encoder
    AudioEncoder encoder;
    if (!encoder.load(gguf_path, "CPU", 8)) {
        printf("Failed to load audio encoder\n");
        return 1;
    }

    // Load Python padded mel
    std::vector<float> padded_mel;
    std::vector<int> mel_shape;
    if (!load_npy_f32(data_dir + "padded_mel.npy", padded_mel, mel_shape)) {
        printf("Failed to load padded_mel.npy\n");
        return 1;
    }
    printf("Padded mel shape: [%d, %d, %d, %d]\n",
           mel_shape[0], mel_shape[1], mel_shape[2], mel_shape[3]);

    int batch_size = mel_shape[0];  // 10
    int mel_h = mel_shape[2];       // 128
    int mel_w = mel_shape[3];       // 100

    // Run Conv2d
    std::vector<float> conv_output;
    if (!encoder.run_conv2d(padded_mel, batch_size, mel_h, mel_w, conv_output)) {
        printf("Conv2d failed\n");
        return 1;
    }

    // conv_output is (d_model, frames_per_chunk, batch_size) in GGML layout
    // = (1024, 13, 10)
    int d_model = 1024;
    auto conv_out_sz = [](int in_size) -> int { return (in_size + 2 - 3) / 2 + 1; };
    int frames_per_chunk = conv_out_sz(conv_out_sz(conv_out_sz(mel_w)));
    printf("Conv output: d_model=%d, frames=%d, batch=%d (total elements: %zu)\n",
           d_model, frames_per_chunk, batch_size, conv_output.size());

    // Load Python conv_out reference
    std::vector<float> py_conv_out;
    std::vector<int> py_shape;
    if (load_npy_f32(data_dir + "after_conv_out.npy", py_conv_out, py_shape)) {
        printf("\nPython after_conv_out shape: [%d, %d, %d]\n",
               py_shape[0], py_shape[1], py_shape[2]);

        // C++ conv_output is in GGML layout: (d_model, frames_per_chunk, batch_size)
        // Python after_conv_out is: (batch_size, frames_per_chunk, d_model) row-major
        // Compare chunk 0, frame 0
        printf("\n--- Conv output comparison (chunk 0, frame 0) ---\n");
        printf("C++ [0, 0, :5]: ");
        for (int d = 0; d < 5; d++) {
            // GGML layout: element at (d, f=0, c=0) = data[0 * fpc*dm + 0 * dm + d]
            printf("%.6f ", conv_output[d]);
        }
        printf("\n");

        printf("Py  [0, 0, :5]: ");
        for (int d = 0; d < 5; d++) {
            // Python row-major: element at (c=0, f=0, d) = data[0 * fpc*dm + 0 * dm + d]
            printf("%.6f ", py_conv_out[d]);
        }
        printf("\n");

        // Full comparison: convert C++ to (batch, frames, d_model) format
        int total = batch_size * frames_per_chunk * d_model;
        std::vector<float> cpp_reordered(total);
        for (int c = 0; c < batch_size; c++) {
            for (int f = 0; f < frames_per_chunk; f++) {
                for (int d = 0; d < d_model; d++) {
                    int src_idx = c * (frames_per_chunk * d_model) + f * d_model + d;
                    int dst_idx = c * (frames_per_chunk * d_model) + f * d_model + d;
                    cpp_reordered[dst_idx] = conv_output[src_idx];
                }
            }
        }

        float corr = compute_correlation(py_conv_out.data(), cpp_reordered.data(), total);
        float maxd = max_abs_diff(py_conv_out.data(), cpp_reordered.data(), total);
        printf("Correlation: %.6f\n", corr);
        printf("Max abs diff: %.6f\n", maxd);
    }

    // Load Python hidden_states_input reference (after pos emb + mask)
    std::vector<float> py_hidden;
    std::vector<int> py_hidden_shape;
    if (load_npy_f32(data_dir + "hidden_states_input.npy", py_hidden, py_hidden_shape)) {
        printf("\n--- Hidden states comparison (after pos emb) ---\n");
        printf("Python hidden shape: [%d, %d]\n", py_hidden_shape[0], py_hidden_shape[1]);

        // Run the full encode pipeline with Python mel input
        // Load the input_features (128, 936) and feature_attention_mask
        std::vector<float> mel_input;
        std::vector<int> mel_input_shape;
        if (!load_npy_f32(data_dir + "input_features.npy", mel_input, mel_input_shape)) {
            printf("Failed to load input_features.npy\n");
        } else {
            printf("Mel input shape: [%d, %d]\n", mel_input_shape[0], mel_input_shape[1]);

            std::vector<float> encode_output;
            int num_frames = 0;
            if (encoder.encode(mel_input, mel_input_shape[1], encode_output, num_frames)) {
                printf("\nEncoder output: %d frames, %d dim\n", num_frames,
                       (int)encode_output.size() / num_frames);

                // Load Python reference audio features
                std::vector<float> py_features;
                std::vector<int> py_feat_shape;
                if (load_npy_f32(data_dir + "audio_features.npy", py_features, py_feat_shape)) {
                    printf("Python features: [%d, %d]\n", py_feat_shape[0], py_feat_shape[1]);

                    int total_elem = py_feat_shape[0] * py_feat_shape[1];
                    float corr = compute_correlation(py_features.data(), encode_output.data(), total_elem);
                    float maxd = max_abs_diff(py_features.data(), encode_output.data(), total_elem);
                    printf("\n=== Full pipeline comparison ===\n");
                    printf("Correlation: %.6f\n", corr);
                    printf("Max abs diff: %.6f\n", maxd);

                    printf("\nPy [0, :5]: ");
                    for (int d = 0; d < 5; d++) printf("%.6f ", py_features[d]);
                    printf("\nC++ [0, :5]: ");
                    for (int d = 0; d < 5; d++) printf("%.6f ", encode_output[d]);
                    printf("\n");
                }
            }
        }
    }

    return 0;
}
