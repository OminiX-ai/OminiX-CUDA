// ============================================================================
// Phase 2.4 smoke test for the Code Predictor (codec) on CUDA.
//
// Architecture observation: the qwen3_tts_predictor.gguf shipped on
// zgx-3675 is a flat llama.cpp-style Qwen3 transformer export with the
// SAME tensor schema as the Talker GGUF — `blk.<L>.attn_q.weight`,
// `blk.<L>.attn_norm.weight`, `output_norm.weight`, `token_embd.weight`,
// `output.weight` — only the hyperparameters differ:
//
//                Talker        Predictor
//   hidden       2048          1024
//   layers       28            5
//   heads (Q/KV) 16 / 8        16 / 8
//   head_dim     128           128
//   inter (FFN)  6144          3072
//   vocab        ~157k         30720    (= 15 groups × 2048 codec vocab)
//   rope theta   1e6           1e6
//   rms eps      1e-6          1e-6
//
// That makes Phase 2.4 a straight reuse of TalkerCudaEngine: same
// kernels, same forward_decode, same RoPE / GQA / SwiGLU body. We just
// hand it a TalkerConfig populated from CodePredictorConfig values, and
// drive an autoreg loop to prove the transformer engine works end-to-end
// on the codec dims.
//
// Smoke gates (mirror the Phase 2.3 talker autoreg gates):
//   1. init_from_gguf returns true.
//   2. 256 forward_decode steps complete without NaN/inf in any hidden.
//   3. Token sequence is not constant (n_unique >= 2).
//   4. Longest run < 95% of n_steps.
//
// Out of scope (deferred to Phase 2.5/2.6):
//   - Full audio synthesis (requires SpeechTokenizerDecoder, separate model).
//   - Numerical parity vs Ascend / Python ref dumps.
//   - The 15-group iterative MTP loop (predictor's per-group LM heads).
//
// Usage:
//   test_codec_cuda <predictor.gguf> [n_steps]
// ============================================================================

#include "talker_cuda_engine.h"
#include "../talker.h"

#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

bool load_tensor_f32(ggml_context *ctx, const char *name,
                      size_t expected_elems, std::vector<float> &out) {
    ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "[codec] missing tensor: %s\n", name);
        return false;
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr, "[codec] %s: expected %zu, got %zu\n",
                name, expected_elems, n);
        return false;
    }
    out.resize(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const ggml_type_traits *tt = ggml_get_type_traits(t->type);
        if (!tt || !tt->to_float) {
            fprintf(stderr, "[codec] %s: unsupported dtype %d\n",
                    name, (int)t->type);
            return false;
        }
        tt->to_float(t->data, out.data(), (int64_t)n);
    }
    return true;
}

void matvec_f32(const float *W, const float *x, float *out,
                 int vocab_size, int n_embd) {
    for (int v = 0; v < vocab_size; ++v) {
        const float *row = W + (size_t)v * n_embd;
        float s = 0.0f;
        for (int i = 0; i < n_embd; ++i) s += row[i] * x[i];
        out[v] = s;
    }
}

int argmax_f32(const float *logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; ++i) {
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    }
    return best;
}

bool check_finite(const float *x, int n, int pos) {
    for (int i = 0; i < n; ++i) {
        if (std::isnan(x[i]) || std::isinf(x[i])) {
            fprintf(stderr, "[codec] pos=%d hidden[%d] = %f (non-finite)\n",
                    pos, i, x[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: test_codec_cuda <predictor.gguf> [n_steps]\n");
        return 2;
    }
    const std::string gguf_path = argv[1];
    int n_steps = (argc >= 3) ? std::atoi(argv[2]) : 256;
    if (n_steps <= 0) n_steps = 256;

    // ---- 1. Build a TalkerConfig populated from CodePredictorConfig ------
    // CodePredictorConfig defaults (talker.h:81-93) match the predictor.gguf
    // metadata (qwen3.embedding_length=1024, block_count=5, head_count=16,
    // head_count_kv=8, key_length=128, feed_forward_length=3072,
    // rope.freq_base=1e6, rms_eps=1e-6).
    CodePredictorConfig cp;
    TalkerConfig cfg;
    cfg.hidden_size           = cp.hidden_size;            // 1024
    cfg.num_hidden_layers     = cp.num_hidden_layers;      // 5
    cfg.num_attention_heads   = cp.num_attention_heads;    // 16
    cfg.num_key_value_heads   = cp.num_key_value_heads;    // 8
    cfg.intermediate_size     = cp.intermediate_size;      // 3072
    cfg.head_dim              = cp.head_dim;               // 128
    cfg.rope_theta            = cp.rope_theta;             // 1e6
    cfg.rms_norm_eps          = cp.rms_norm_eps;           // 1e-6
    // codec_bos_id stays at the default; predictor isn't a token-stream model
    // (its inputs are talker-hidden vectors), but for autoreg smoke we use a
    // deterministic non-zero token to seed the embedding lookup.

    printf("[codec] config: hidden=%d layers=%d heads=%d/%d head_dim=%d "
           "inter=%d rope_theta=%.0f rms_eps=%.2e\n",
           cfg.hidden_size, cfg.num_hidden_layers,
           cfg.num_attention_heads, cfg.num_key_value_heads,
           cfg.head_dim, cfg.intermediate_size,
           cfg.rope_theta, cfg.rms_norm_eps);

    // ---- 2. Init engine --------------------------------------------------
    ominix_cuda::TalkerCudaEngine eng;
    if (!eng.init_from_gguf(gguf_path, cfg, /*device=*/0)) {
        fprintf(stderr, "[codec] init_from_gguf FAILED\n");
        return 1;
    }
    eng.reset_kv_cache();
    printf("[codec] engine init OK\n");

    const int n_embd = cfg.hidden_size;

    // ---- 3. Pull token_embd + output (codec head) for host-side embed/argmax
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[codec] gguf_init failed: %s\n", gguf_path.c_str());
        return 1;
    }

    ggml_tensor *embd_t = ggml_get_tensor(ggml_ctx, "token_embd.weight");
    if (!embd_t) {
        fprintf(stderr, "[codec] token_embd.weight not in GGUF\n");
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }
    if (embd_t->ne[0] != n_embd) {
        fprintf(stderr, "[codec] token_embd ne[0]=%lld != n_embd=%d\n",
                (long long)embd_t->ne[0], n_embd);
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }
    const int vocab_size = (int)embd_t->ne[1];
    printf("[codec] vocab_size=%d  n_embd=%d  n_steps=%d\n",
           vocab_size, n_embd, n_steps);
    // For predictor.gguf vocab=30720 = 15 groups × 2048 codec entries.
    if (vocab_size % 2048 == 0) {
        printf("[codec] vocab decomposes as %d groups × 2048\n",
               vocab_size / 2048);
    }

    std::vector<float> token_embd_w;
    if (!load_tensor_f32(ggml_ctx, "token_embd.weight",
                          (size_t)vocab_size * n_embd, token_embd_w)) {
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }

    std::vector<float> lm_head_w;
    bool lm_head_tied = false;
    {
        ggml_tensor *out_t = ggml_get_tensor(ggml_ctx, "output.weight");
        if (out_t) {
            if (!load_tensor_f32(ggml_ctx, "output.weight",
                                  (size_t)vocab_size * n_embd, lm_head_w)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx);
                return 1;
            }
        } else {
            lm_head_w = token_embd_w;
            lm_head_tied = true;
        }
    }
    printf("[codec] loaded token_embd (%zu MB) + lm_head (%s, %zu MB)\n",
           token_embd_w.size() * sizeof(float) / (1024 * 1024),
           lm_head_tied ? "tied" : "explicit",
           lm_head_w.size() * sizeof(float) / (1024 * 1024));

    // ---- 4. Pick a deterministic seed token in-vocab ---------------------
    int initial_token = 1;
    if (initial_token >= vocab_size) initial_token = vocab_size - 1;
    printf("[codec] initial_token=%d\n", initial_token);

    // ---- 5. Autoregressive loop -----------------------------------------
    std::vector<float> input_emb(n_embd);
    std::vector<float> hidden(n_embd);
    std::vector<float> logits(vocab_size);
    std::vector<int>   tokens;
    tokens.reserve(n_steps);

    // Phase 2.5 — warmup pass under CUDA Graphs to populate per-pos cache.
    const bool use_graphs = eng.use_cuda_graphs();
    std::vector<int> warmup_tokens;
    if (use_graphs) {
        warmup_tokens.reserve(n_steps);
        int cur_w = initial_token;
        for (int pos = 0; pos < n_steps; ++pos) {
            std::memcpy(input_emb.data(),
                        token_embd_w.data() + (size_t)cur_w * n_embd,
                        n_embd * sizeof(float));
            eng.forward_decode(input_emb.data(), pos, hidden.data());
            if (!check_finite(hidden.data(), n_embd, pos)) {
                gguf_free(gguf_ctx); ggml_free(ggml_ctx);
                return 1;
            }
            matvec_f32(lm_head_w.data(), hidden.data(), logits.data(),
                        vocab_size, n_embd);
            int nxt = argmax_f32(logits.data(), vocab_size);
            warmup_tokens.push_back(nxt);
            cur_w = nxt;
        }
        eng.reset_kv_cache();
        printf("[codec] warmup pass complete (graphs captured) — "
               "first 16 tokens:");
        for (int i = 0; i < 16 && i < (int)warmup_tokens.size(); ++i)
            printf(" %d", warmup_tokens[i]);
        printf("\n");
    }

    int cur = initial_token;
    auto t_start = std::chrono::high_resolution_clock::now();
    double total_engine_ms = 0.0;

    for (int pos = 0; pos < n_steps; ++pos) {
        std::memcpy(input_emb.data(),
                    token_embd_w.data() + (size_t)cur * n_embd,
                    n_embd * sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        eng.forward_decode(input_emb.data(), pos, hidden.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        total_engine_ms +=
            std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (!check_finite(hidden.data(), n_embd, pos)) {
            gguf_free(gguf_ctx); ggml_free(ggml_ctx);
            return 1;
        }

        matvec_f32(lm_head_w.data(), hidden.data(), logits.data(),
                    vocab_size, n_embd);
        int nxt = argmax_f32(logits.data(), vocab_size);
        tokens.push_back(nxt);
        cur = nxt;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_wall_ms =
        std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (use_graphs && !warmup_tokens.empty()) {
        int matches = 0;
        int first_diff = -1;
        for (size_t i = 0; i < tokens.size() && i < warmup_tokens.size(); ++i) {
            if (tokens[i] == warmup_tokens[i]) ++matches;
            else if (first_diff < 0) first_diff = (int)i;
        }
        double cossim = (double)matches / (double)tokens.size();
        printf("[codec] graph parity: warm-vs-replay match %d/%d "
               "(cossim=%.6f)  first_diff=%d\n",
               matches, (int)tokens.size(), cossim, first_diff);
        if (matches != (int)tokens.size()) {
            fprintf(stderr,
                    "[codec] FAIL: graph replay tokens differ from "
                    "capture-pass tokens at index %d\n", first_diff);
            gguf_free(gguf_ctx); ggml_free(ggml_ctx);
            return 1;
        }
    }

    // ---- 6. Sanity gates -----------------------------------------------
    std::set<int> uniq(tokens.begin(), tokens.end());
    int n_unique = (int)uniq.size();
    bool sane_diverse  = (n_unique >= 2);
    int max_run = 0, run = 1;
    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == tokens[i - 1]) { ++run; if (run > max_run) max_run = run; }
        else run = 1;
    }
    if (run > max_run) max_run = run;
    bool sane_runs = (max_run < (int)(0.95 * tokens.size()));

    printf("[codec] tokens generated: %d  unique=%d  longest_run=%d\n",
           (int)tokens.size(), n_unique, max_run);

    printf("[codec] first 16 tokens:");
    for (int i = 0; i < 16 && i < (int)tokens.size(); ++i) printf(" %d", tokens[i]);
    printf("\n");
    printf("[codec] last  16 tokens:");
    int last_start = std::max(0, (int)tokens.size() - 16);
    for (int i = last_start; i < (int)tokens.size(); ++i) printf(" %d", tokens[i]);
    printf("\n");

    double avg_ms      = total_engine_ms / (double)n_steps;
    double tok_per_sec = 1000.0 * (double)n_steps / total_wall_ms;
    printf("[codec] engine total=%.2f ms  avg=%.2f ms/step  "
           "wall total=%.2f ms  TPS=%.2f\n",
           total_engine_ms, avg_ms, total_wall_ms, tok_per_sec);

    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    if (!sane_diverse) {
        fprintf(stderr,
                "[codec] FAIL: token sequence is constant (n_unique=1)\n");
        return 1;
    }
    if (!sane_runs) {
        fprintf(stderr,
                "[codec] FAIL: longest run %d covers > 95%% of sequence\n",
                max_run);
        return 1;
    }

    printf("[codec] Phase 2.4 codec smoke PASS  "
           "n_steps=%d  TPS=%.2f  avg_engine=%.2f ms/step\n",
           n_steps, tok_per_sec, avg_ms);
    return 0;
}
