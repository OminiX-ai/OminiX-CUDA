// Test Code Predictor autoregressive flow
// Loads saved talker hidden state, calls predict_code_groups, compares with Python reference
#include "talker.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <fstream>

int main() {
    const char *llama_gguf = "gguf/qwen_tts_talker_llama.gguf";
    const char *embed_gguf = "gguf/qwen_tts_talker.gguf";
    const char *cp_gguf    = "gguf/qwen_tts_code_predictor.gguf";

    printf("=== Test CP Autoregressive Flow ===\n");

    // 1. Load all models
    TalkerLLM talker;
    if (!talker.load_model(llama_gguf, embed_gguf, cp_gguf, 4)) {
        printf("FAIL: load_model failed\n");
        return 1;
    }
    printf("Models loaded.\n");

    // 2. Load saved talker hidden state from C++ prefill
    std::vector<float> hidden(2048);
    {
        std::ifstream f("../../logs/cpp_prefill_hidden.bin", std::ios::binary);
        if (!f.is_open()) {
            printf("ERROR: cannot open logs/cpp_prefill_hidden.bin\n");
            printf("Run the full pipeline first to generate this file.\n");
            return 1;
        }
        f.read(reinterpret_cast<char*>(hidden.data()), 2048 * sizeof(float));
        printf("Loaded talker hidden state, [:5]=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
               hidden[0], hidden[1], hidden[2], hidden[3], hidden[4]);
    }

    // 3. Call predict_code_groups with group0_token=302
    int group0_token = 302;
    std::vector<int> group_tokens;

    printf("\nRunning predict_code_groups(g0=%d)...\n", group0_token);
    if (!talker.predict_code_groups(hidden.data(), 1, group0_token, group_tokens)) {
        printf("FAIL: predict_code_groups failed\n");
        return 1;
    }

    // 4. Print results
    printf("\nC++ group tokens (1-15): [");
    for (int i = 0; i < (int)group_tokens.size(); i++)
        printf("%d%s", group_tokens[i], i < (int)group_tokens.size()-1 ? "," : "");
    printf("]\n");

    // 5. Compare with Python reference
    std::vector<int> py_ref = {281, 16, 655, 1269, 560, 587, 1901, 299, 310, 622, 781, 549, 359, 240, 92};
    printf("Py  group tokens (1-15): [");
    for (int i = 0; i < (int)py_ref.size(); i++)
        printf("%d%s", py_ref[i], i < (int)py_ref.size()-1 ? "," : "");
    printf("]\n");

    bool match = (group_tokens.size() == py_ref.size());
    if (match) {
        for (int i = 0; i < (int)py_ref.size(); i++) {
            if (group_tokens[i] != py_ref[i]) {
                match = false;
                printf("  MISMATCH at group %d: C++=%d, Py=%d\n",
                       i+1, group_tokens[i], py_ref[i]);
            }
        }
    }

    printf("\n%s\n", match ? "PASS: all groups match!" : "FAIL: mismatch detected");
    return match ? 0 : 1;
}
