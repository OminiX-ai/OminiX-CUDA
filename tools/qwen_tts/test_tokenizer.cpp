// Quick test for BPE tokenizer against Python reference
#include "bpe_tokenizer.h"
#include <cstdio>
#include <cstring>

int main() {
    const char *vocab_path = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base/vocab.json";
    const char *merges_path = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base/merges.txt";

    BpeTokenizer tok;
    if (!tok.load(vocab_path, merges_path)) {
        printf("FAIL: cannot load tokenizer\n");
        return 1;
    }
    printf("PASS: tokenizer loaded\n");

    // Test cases from Python: encode(text, add_special_tokens=False)
    struct TestCase {
        const char *text;
        std::vector<int> expected;
    };

    TestCase tests[] = {
        {"Hello, world!", {9707, 11, 1879, 0}},
        {"<|im_start|>assistant", {151644, 77091}},
        // Chinese text
        {"\xe4\xbd\xa0\xe5\xa5\xbd\xe4\xb8\x96\xe7\x95\x8c", {108386, 99489}},  // "你好世界"
        // Full TTS prompt format
        {"<|im_start|>assistant\n", {151644, 77091, 198}},
        {"<|im_end|>\n", {151645, 198}},
    };

    int pass = 0, fail = 0;
    for (auto &tc : tests) {
        auto ids = tok.encode(tc.text);
        bool match = (ids == tc.expected);
        printf("%s: \"%s\" -> [", match ? "PASS" : "FAIL", tc.text);
        for (size_t i = 0; i < ids.size(); i++) {
            if (i > 0) printf(", ");
            printf("%d", ids[i]);
        }
        printf("]");
        if (!match) {
            printf(" (expected [");
            for (size_t i = 0; i < tc.expected.size(); i++) {
                if (i > 0) printf(", ");
                printf("%d", tc.expected[i]);
            }
            printf("])");
            fail++;
        } else {
            pass++;
        }
        printf("\n");
    }

    // Test special token IDs
    printf("\nim_start=%d (expect 151644)\n", tok.im_start_id());
    printf("im_end=%d (expect 151645)\n", tok.im_end_id());

    printf("\n%d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}
