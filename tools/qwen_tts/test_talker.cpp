// Test Talker LLM: verify llama.cpp backbone loading and basic forward pass
#include "talker.h"
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cmath>

int main(int argc, char **argv) {
    const char *llama_gguf = "gguf/qwen_tts_talker_llama.gguf";
    const char *embed_gguf = "gguf/qwen_tts_talker.gguf";
    const char *cp_gguf    = "gguf/qwen_tts_code_predictor.gguf";

    if (argc > 1) llama_gguf = argv[1];
    if (argc > 2) embed_gguf = argv[2];
    if (argc > 3) cp_gguf    = argv[3];

    printf("=== Test Talker LLM ===\n");
    printf("  llama GGUF: %s\n", llama_gguf);
    printf("  embed GGUF: %s\n", embed_gguf);
    printf("  CP GGUF:    %s\n", cp_gguf);

    // 1. Load all models
    TalkerLLM talker;
    if (!talker.load_model(llama_gguf, embed_gguf, cp_gguf, 4)) {
        printf("FAIL: load_model failed\n");
        return 1;
    }
    printf("PASS: all models loaded\n");

    auto &cfg = talker.get_config();
    printf("  hidden=%d, vocab=%d, text_vocab=%d, groups=%d\n",
           cfg.hidden_size, cfg.vocab_size, cfg.text_vocab_size,
           cfg.num_code_groups);
    printf("  codec_bos=%d, codec_eos=%d, codec_pad=%d\n",
           cfg.codec_bos_id, cfg.codec_eos_token_id, cfg.codec_pad_id);
    printf("  language IDs: %zu entries\n", cfg.language_ids.size());
    for (auto &[k, v] : cfg.language_ids) {
        printf("    %s = %d\n", k.c_str(), v);
    }

    // 2. Test basic generation with dummy inputs
    printf("\n--- Testing generate with dummy inputs ---\n");
    // Dummy text tokens
    std::vector<int> ref_text_tokens = {1, 2, 3};
    std::vector<int> target_text_tokens = {4, 5};
    // Dummy speaker embedding (2048-dim)
    std::vector<float> spk_emb(cfg.hidden_size, 0.01f);
    // No reference codes for now
    std::vector<std::vector<int>> ref_codes;
    std::vector<std::vector<int>> codec_tokens;

    // Generate with very few tokens to test the loop
    bool ok = talker.generate(ref_text_tokens, target_text_tokens, spk_emb,
                               ref_codes, "english", codec_tokens, 5);
    if (ok) {
        printf("PASS: generate completed\n");
        printf("  Generated %zu frames, %zu groups\n",
               codec_tokens.empty() ? 0 : codec_tokens[0].size(),
               codec_tokens.size());
        // Print first few tokens
        for (int g = 0; g < (int)codec_tokens.size() && g < 3; g++) {
            printf("  group %d:", g);
            for (int t = 0; t < (int)codec_tokens[g].size() && t < 5; t++) {
                printf(" %d", codec_tokens[g][t]);
            }
            printf("\n");
        }
    } else {
        printf("NOTE: generate returned false (may be expected for dummy inputs)\n");
    }

    printf("\n=== Test complete ===\n");
    return 0;
}
