/**
 * Test Code Predictor GGUF loading and inference.
 * Compares C++ output against Python reference data.
 *
 * Usage: test_code_predictor <code_predictor.gguf> <data_dir>
 */
#include "talker.h"
#include "infer_session.hpp"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <algorithm>
#include <numeric>

static bool load_bin_f32(const std::string &path, std::vector<float> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        printf("ERROR: cannot open %s\n", path.c_str());
        return false;
    }
    size_t bytes = f.tellg();
    f.seekg(0);
    out.resize(bytes / sizeof(float));
    f.read(reinterpret_cast<char *>(out.data()), bytes);
    return true;
}

static bool load_bin_i32(const std::string &path, std::vector<int> &out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        printf("ERROR: cannot open %s\n", path.c_str());
        return false;
    }
    size_t bytes = f.tellg();
    f.seekg(0);
    out.resize(bytes / sizeof(int));
    f.read(reinterpret_cast<char *>(out.data()), bytes);
    return true;
}

static float max_abs_diff(const std::vector<float> &a,
                          const std::vector<float> &b) {
    float d = 0.0f;
    size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; i++) {
        d = std::max(d, std::abs(a[i] - b[i]));
    }
    return d;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <code_predictor.gguf> <data_dir>\n", argv[0]);
        printf("  data_dir should contain ref_cp_*.bin files from verify_code_predictor.py\n");
        return 1;
    }

    std::string gguf_path = argv[1];
    std::string data_dir = argv[2];

    printf("Loading reference data...\n");
    fflush(stdout);

    // Load reference data
    std::vector<float> ref_hidden, ref_projected, ref_logits, ref_layer0;
    std::vector<int> ref_tokens;

    if (!load_bin_f32(data_dir + "/ref_cp_hidden_states.bin", ref_hidden) ||
        !load_bin_i32(data_dir + "/ref_cp_group_tokens.bin", ref_tokens) ||
        !load_bin_f32(data_dir + "/ref_cp_projected.bin", ref_projected) ||
        !load_bin_f32(data_dir + "/ref_cp_logits.bin", ref_logits)) {
        printf("Failed to load reference data\n");
        return 1;
    }
    load_bin_f32(data_dir + "/ref_cp_layer0.bin", ref_layer0);  // optional

    int seq_len = (int)ref_tokens.size();
    printf("Reference data: seq_len=%d, hidden=[2048,%d], logits=[2048,%d]\n",
           seq_len, seq_len, seq_len);

    // Load Code Predictor GGUF
    printf("Loading Code Predictor from %s...\n", gguf_path.c_str());
    fflush(stdout);
    ContextParams params;
    params.n_threads = 4;
    printf("Creating InferenceSession...\n");
    fflush(stdout);
    InferenceSession<CodePredictorModel> session(gguf_path, params);
    printf("Session created.\n");
    fflush(stdout);
    auto &model = session.get_model();
    printf("Config: hidden=%d, layers=%d, vocab=%d, talker_hidden=%d\n",
           model.config.hidden_size, model.config.num_hidden_layers,
           model.config.vocab_size, model.config.talker_hidden_size);

    // Set input shape for seq_len tokens
    session.set_input_shape({
        {"hidden_states", {model.config.talker_hidden_size, seq_len}},
        {"group_tokens", {seq_len}},
    });

    // Set inputs
    session.set_input("hidden_states", ref_hidden);
    session.set_input("group_tokens", ref_tokens);

    // Build causal mask
    std::vector<float> kq_mask(seq_len * seq_len, 0.0f);
    for (int i = 0; i < seq_len; i++) {
        for (int j = i + 1; j < seq_len; j++) {
            kq_mask[i * seq_len + j] = -INFINITY;
        }
    }
    session.set_input("kq_mask", kq_mask);

    // Position IDs [0, 1, 2, ..., seq_len-1]
    std::vector<int> pos(seq_len);
    for (int i = 0; i < seq_len; i++) pos[i] = i;
    session.set_input("pos", pos);

    // Run inference (multi-output: logits + projected)
    printf("\nRunning inference...\n");
    std::vector<std::vector<float>> outputs;
    if (!session.run(outputs)) {
        printf("Inference failed!\n");
        return 1;
    }

    printf("Got %zu outputs\n", outputs.size());
    auto &logits = outputs[0];
    auto &projected = outputs.size() > 1 ? outputs[1] : outputs[0];

    // Compare projected output first
    if (outputs.size() > 1) {
        printf("\n=== Projected output comparison ===\n");
        printf("C++ projected: %zu elements\n", projected.size());
        printf("Ref projected: %zu elements\n", ref_projected.size());
        float proj_mad = max_abs_diff(projected, ref_projected);
        printf("Projected max diff: %.6f\n", proj_mad);
    }

    // Compare layer 0 output
    if (outputs.size() > 2 && !ref_layer0.empty()) {
        auto &layer0 = outputs[2];
        printf("\n=== Layer 0 output comparison ===\n");
        printf("C++ layer0: %zu elements\n", layer0.size());
        printf("Ref layer0: %zu elements\n", ref_layer0.size());
        float l0_mad = max_abs_diff(layer0, ref_layer0);
        printf("Layer0 max diff: %.6f\n", l0_mad);
        // Per-position diff for layer0
        int cp_hidden = model.config.hidden_size;
        for (int s = 0; s < seq_len; s++) {
            float pos_mad = 0.0f;
            for (int d = 0; d < cp_hidden; d++) {
                int idx = s * cp_hidden + d;
                if (idx < (int)layer0.size() && idx < (int)ref_layer0.size())
                    pos_mad = std::max(pos_mad, std::abs(layer0[idx] - ref_layer0[idx]));
            }
            printf("  pos %d: max_diff=%.6f\n", s, pos_mad);
        }
    }

    printf("\n=== Logits comparison ===\n");

    printf("Output logits: %zu elements\n", logits.size());
    printf("Expected:      %zu elements\n", ref_logits.size());

    if (logits.size() != ref_logits.size()) {
        printf("ERROR: size mismatch!\n");
        return 1;
    }

    // Compare
    float mad = max_abs_diff(logits, ref_logits);
    printf("\nMax absolute difference: %.6f\n", mad);

    // Per-position max diff
    int vocab_size = model.config.vocab_size;
    printf("\nPer-position max diff:\n");
    for (int s = 0; s < seq_len; s++) {
        float pos_mad = 0.0f;
        for (int v = 0; v < vocab_size; v++) {
            int idx = s * vocab_size + v;
            pos_mad = std::max(pos_mad, std::abs(logits[idx] - ref_logits[idx]));
        }
        printf("  pos %d: max_diff=%.6f\n", s, pos_mad);
    }

    // Print first few values
    printf("\nFirst 5 logits (C++ vs Python):\n");
    for (int i = 0; i < 5 && i < (int)logits.size(); i++) {
        printf("  [%d] %.6f vs %.6f (diff=%.6f)\n",
               i, logits[i], ref_logits[i],
               std::abs(logits[i] - ref_logits[i]));
    }

    // Check predicted token at last position
    int offset = (seq_len - 1) * vocab_size;
    auto it_cpp = std::max_element(logits.begin() + offset,
                                    logits.begin() + offset + vocab_size);
    int pred_cpp = (int)std::distance(logits.begin() + offset, it_cpp);

    auto it_py = std::max_element(ref_logits.begin() + offset,
                                   ref_logits.begin() + offset + vocab_size);
    int pred_py = (int)std::distance(ref_logits.begin() + offset, it_py);

    printf("\nPredicted token (last pos): C++=%d, Python=%d\n", pred_cpp, pred_py);

    bool pass = (mad < 0.01f);
    printf("\n%s (threshold=0.01)\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
