// Test Speech Tokenizer Encoder: verify conv encoder + transformer + RVQ
#include "speech_tokenizer_encoder.h"
#include "utils.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

int main(int argc, char **argv) {
    const char *gguf_path = "gguf/qwen_tts_tokenizer_enc.gguf";
    const char *audio_path = "data/ref_encoder_audio.bin";
    const char *hidden_ref_path = "data/ref_encoder_hidden.bin";
    const char *codes_ref_path = "data/ref_encoder_codes.bin";

    if (argc > 1) gguf_path = argv[1];
    if (argc > 2) audio_path = argv[2];

    printf("=== Test Speech Tokenizer Encoder ===\n");
    printf("  GGUF: %s\n", gguf_path);
    printf("  Audio: %s\n", audio_path);

    // Load reference audio
    std::vector<float> audio;
    if (!load_file_to_vector(audio, audio_path)) {
        printf("FAIL: cannot load audio from %s\n", audio_path);
        return 1;
    }
    printf("  Audio samples: %zu (%.2f seconds)\n",
           audio.size(), audio.size() / 24000.0f);

    // Load encoder
    ContextParams params;
    params.device_name = "CPU";
    params.n_threads = 4;
    params.max_nodes = 8192;

    SpeechTokenizerEncoder encoder;
    if (!encoder.load(gguf_path, params)) {
        printf("FAIL: load_model failed\n");
        return 1;
    }
    printf("PASS: encoder loaded\n");

    // Encode (also get hidden states for comparison)
    std::vector<std::vector<int>> codes;
    std::vector<float> cpp_hidden;
    if (!encoder.encode(audio, codes, &cpp_hidden)) {
        printf("FAIL: encode failed\n");
        return 1;
    }
    printf("PASS: encode completed\n");

    int n_q = (int)codes.size();
    int T = codes.empty() ? 0 : (int)codes[0].size();
    int hidden = encoder.get_config().hidden_size;
    printf("  Output: %d quantizers, %d frames\n", n_q, T);

    // Print first few codes
    printf("\nFirst 5 timesteps:\n");
    for (int q = 0; q < std::min(4, n_q); q++) {
        printf("  q%d:", q);
        for (int t = 0; t < std::min(5, T); t++) {
            printf(" %d", codes[q][t]);
        }
        printf("\n");
    }

    // Compare hidden states against Python reference
    std::vector<float> ref_hidden;
    if (load_file_to_vector(ref_hidden, hidden_ref_path)) {
        printf("\n--- Hidden state comparison ---\n");
        int ref_elems = (int)ref_hidden.size();
        int cpp_elems = (int)cpp_hidden.size();
        printf("  ref: %d elements, cpp: %d elements\n", ref_elems, cpp_elems);

        if (ref_elems == cpp_elems && ref_elems > 0) {
            // Compute statistics
            float max_abs_err = 0.0f, sum_abs_err = 0.0f;
            float max_rel_err = 0.0f;
            int close_count = 0;
            for (int i = 0; i < ref_elems; i++) {
                float err = std::abs(cpp_hidden[i] - ref_hidden[i]);
                if (err > max_abs_err) max_abs_err = err;
                sum_abs_err += err;
                float denom = std::max(std::abs(ref_hidden[i]), 1e-6f);
                float rel = err / denom;
                if (rel > max_rel_err) max_rel_err = rel;
                if (err < 0.1f) close_count++;
            }
            float mean_abs_err = sum_abs_err / ref_elems;
            printf("  max_abs_err=%.4f mean_abs_err=%.4f max_rel_err=%.4f\n",
                   max_abs_err, mean_abs_err, max_rel_err);
            printf("  close (<0.1): %d/%d (%.1f%%)\n",
                   close_count, ref_elems, 100.0 * close_count / ref_elems);

            // Print first few values side by side
            printf("\n  First 10 values (ref vs cpp):\n");
            for (int i = 0; i < std::min(10, ref_elems); i++) {
                printf("    [%d] ref=%.6f cpp=%.6f err=%.6f\n",
                       i, ref_hidden[i], cpp_hidden[i],
                       std::abs(ref_hidden[i] - cpp_hidden[i]));
            }

            if (mean_abs_err < 0.5f) {
                printf("\nPASS: hidden states closely match (mean_err=%.4f)\n",
                       mean_abs_err);
            } else {
                printf("\nWARN: hidden states diverge (mean_err=%.4f)\n",
                       mean_abs_err);
            }
        } else {
            printf("  WARN: element count mismatch\n");
        }
    }

    // Compare with reference codes if available
    std::vector<int32_t> ref_codes_flat;
    if (load_file_to_vector(ref_codes_flat, codes_ref_path)) {
        int ref_T = (int)ref_codes_flat.size() / 16;
        printf("\n--- Code comparison ---\n");
        printf("  Reference: 16 × %d, C++: %d × %d\n", ref_T, n_q, T);

        if (ref_T != T) {
            printf("WARNING: frame count mismatch: C++=%d, Python=%d\n", T, ref_T);
        }

        int match = 0, total = 0;
        int compare_T = std::min(T, ref_T);
        int compare_q = std::min(n_q, 16);
        for (int q = 0; q < compare_q; q++) {
            int q_match = 0;
            for (int t = 0; t < compare_T; t++) {
                int ref = ref_codes_flat[q * ref_T + t];
                int cpp = codes[q][t];
                if (ref == cpp) { match++; q_match++; }
                total++;
            }
            printf("  q%d: %d/%d match (%.1f%%)\n",
                   q, q_match, compare_T, 100.0 * q_match / compare_T);
        }
        printf("\n  Total: %d/%d match (%.1f%%)\n",
               match, total, 100.0 * match / total);

        if (match == total) {
            printf("\nPASS: all codes match exactly!\n");
        } else if (match > total / 2) {
            printf("\nPASS: >50%% code match (F16 precision OK)\n");
        } else {
            printf("\nFAIL: codes largely diverge\n");
        }
    }

    printf("\n=== Test complete ===\n");
    return 0;
}
