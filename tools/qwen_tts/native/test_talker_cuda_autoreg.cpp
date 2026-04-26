// ============================================================================
// Phase 2.3 smoke test for TalkerCudaEngine — autoregressive decode loop.
//
// Usage:
//   test_talker_cuda_autoreg <gguf_path> [n_steps]
//
// What it does (vs Phase 2.2's KV-stability harness):
//   1. Init the engine from the same GGUF Phase 2.1/2.2 used.
//   2. Pull `token_embd.weight` (F32 [vocab_size, n_embd]) and
//      `output.weight` (F32 [vocab_size, n_embd], the codec LM head)
//      out of the GGUF directly into host memory. Single-token embedding
//      lookup + greedy argmax happen on host — they are not the hot path
//      we're measuring (Phase 2.4 will move the LM head matmul onto GPU).
//   3. Pick an initial token (codec_bos_id from cfg, fallback 0 if not set).
//   4. For pos = 0..n_steps-1:
//        input_emb  = token_embd[current_token]
//        hidden     = engine.forward_decode(input_emb, pos)        // F32 host
//        logits     = output_w * hidden                             // F32 host
//        next_token = argmax(logits)
//        record token + wall, advance current_token
//   5. Sanity gate (since no Ascend reference dump on ac0[123]):
//        - no NaN / inf in any hidden;
//        - sequence is not constant (would mean stuck in a degenerate fixpoint);
//        - sequence is not max-vocab-randomness (would mean LM head broken or
//          softmax flat);
//      Both bounds are loose; precise cossim parity comes once we have an
//      Ascend reference dump.
//   6. Print: total wall, avg ms/step, tokens/sec, first 16 + last 16 tokens.
//
// Exit 0 on PASS; non-zero on any check failure.
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

// Pull a tensor from GGUF as F32. Mirrors talker_cuda_engine.cpp's
// load_gguf_tensor_f32 helper but local so we don't have to expose it.
bool load_tensor_f32(ggml_context *ctx, const char *name,
                      size_t expected_elems, std::vector<float> &out) {
    ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "[autoreg] missing tensor: %s\n", name);
        return false;
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr, "[autoreg] %s: expected %zu, got %zu\n",
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
            fprintf(stderr, "[autoreg] %s: unsupported dtype %d\n",
                    name, (int)t->type);
            return false;
        }
        tt->to_float(t->data, out.data(), (int64_t)n);
    }
    return true;
}

// out[v] = sum_i W[v * n_embd + i] * x[i]
// (GGUF layout for weight matrices is row-major [out_features, in_features])
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
            fprintf(stderr, "[autoreg] pos=%d hidden[%d] = %f (non-finite)\n",
                    pos, i, x[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: test_talker_cuda_autoreg <gguf_path> [n_steps]\n");
        return 2;
    }
    const std::string gguf_path = argv[1];
    int n_steps = (argc >= 3) ? std::atoi(argv[2]) : 256;
    if (n_steps <= 0) n_steps = 256;

    // ---- 1. Init engine ---------------------------------------------------
    TalkerConfig cfg;
    ominix_cuda::TalkerCudaEngine eng;
    if (!eng.init_from_gguf(gguf_path, cfg, /*device=*/0)) {
        fprintf(stderr, "[autoreg] init_from_gguf FAILED\n");
        return 1;
    }
    eng.reset_kv_cache();

    const int n_embd = cfg.hidden_size;

    // ---- 2. Load token_embd + output (LM/codec head) directly from GGUF ---
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[autoreg] gguf_init failed: %s\n", gguf_path.c_str());
        return 1;
    }

    // Read vocab_size from the embedding tensor itself (GGUF: [n_embd, vocab]).
    ggml_tensor *embd_t = ggml_get_tensor(ggml_ctx, "token_embd.weight");
    if (!embd_t) {
        fprintf(stderr, "[autoreg] token_embd.weight not in GGUF\n");
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }
    if (embd_t->ne[0] != n_embd) {
        fprintf(stderr, "[autoreg] token_embd ne[0]=%lld != n_embd=%d\n",
                (long long)embd_t->ne[0], n_embd);
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }
    const int vocab_size = (int)embd_t->ne[1];
    printf("[autoreg] vocab_size=%d  n_embd=%d  n_steps=%d\n",
           vocab_size, n_embd, n_steps);

    std::vector<float> token_embd_w;
    if (!load_tensor_f32(ggml_ctx, "token_embd.weight",
                          (size_t)vocab_size * n_embd, token_embd_w)) {
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }

    // LM head ("codec head" in TTS-speak). May be tied to token_embd in some
    // models; here Qwen3-TTS exports it explicitly as `output.weight`.
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
            // Fall back to tied embedding.
            lm_head_w = token_embd_w;
            lm_head_tied = true;
            printf("[autoreg] output.weight absent — tying to token_embd\n");
        }
    }
    printf("[autoreg] loaded token_embd (%zu MB) + lm_head (%s, %zu MB)\n",
           token_embd_w.size() * sizeof(float) / (1024 * 1024),
           lm_head_tied ? "tied" : "explicit",
           lm_head_w.size() * sizeof(float) / (1024 * 1024));

    // ---- 3. Pick initial token --------------------------------------------
    int initial_token = (int)cfg.codec_bos_id;
    if (initial_token <= 0 || initial_token >= vocab_size) {
        // Phase 2.2's GGUF doesn't always expose codec_bos via TalkerConfig
        // defaults — fall back to a deterministic non-zero token so the loop
        // still exercises a meaningful embedding row.
        initial_token = 1;
    }
    printf("[autoreg] initial_token=%d\n", initial_token);

    // ---- 4. Autoregressive loop -------------------------------------------
    std::vector<float> input_emb(n_embd);
    std::vector<float> hidden(n_embd);
    std::vector<float> logits(vocab_size);
    std::vector<int>   tokens;
    tokens.reserve(n_steps);

    // Phase 2.5 — when CUDA Graphs are on, run a warmup pass first so each
    // pos has a captured cudaGraphExec_t on the timed pass. Token sequence
    // from the warmup is kept for cossim parity vs the timed (replay) pass.
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
        printf("[autoreg] warmup pass complete (graphs captured) — "
               "first 16 tokens:");
        for (int i = 0; i < 16 && i < (int)warmup_tokens.size(); ++i)
            printf(" %d", warmup_tokens[i]);
        printf("\n");
    }

    int cur = initial_token;
    auto t_start = std::chrono::high_resolution_clock::now();
    double total_engine_ms = 0.0;

    for (int pos = 0; pos < n_steps; ++pos) {
        // Embedding lookup: row `cur` of token_embd_w (row-major [vocab,n_embd]).
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

    // Phase 2.5 parity check: warmup (capture) vs timed (replay) sequence
    // must be byte-identical — graphs replay the same kernels deterministically.
    if (use_graphs && !warmup_tokens.empty()) {
        int matches = 0;
        int first_diff = -1;
        for (size_t i = 0; i < tokens.size() && i < warmup_tokens.size(); ++i) {
            if (tokens[i] == warmup_tokens[i]) ++matches;
            else if (first_diff < 0) first_diff = (int)i;
        }
        double cossim = (double)matches / (double)tokens.size();
        printf("[autoreg] graph parity: warm-vs-replay match %d/%d "
               "(cossim=%.6f)  first_diff=%d\n",
               matches, (int)tokens.size(), cossim, first_diff);
        if (matches != (int)tokens.size()) {
            fprintf(stderr,
                    "[autoreg] FAIL: graph replay tokens differ from "
                    "capture-pass tokens at index %d\n", first_diff);
            gguf_free(gguf_ctx); ggml_free(ggml_ctx);
            return 1;
        }
    }

    // ---- 5. Sanity gates ---------------------------------------------------
    std::set<int> uniq(tokens.begin(), tokens.end());
    int n_unique = (int)uniq.size();
    bool sane_diverse  = (n_unique >= 2);
    // Loose upper bound: argmax-only sampling is naturally low-entropy.
    // We just require that a single token doesn't dominate >95% of the run.
    int max_run = 0, run = 1;
    for (size_t i = 1; i < tokens.size(); ++i) {
        if (tokens[i] == tokens[i - 1]) { ++run; if (run > max_run) max_run = run; }
        else run = 1;
    }
    if (run > max_run) max_run = run;
    bool sane_runs = (max_run < (int)(0.95 * tokens.size()));

    printf("[autoreg] tokens generated: %d  unique=%d  longest_run=%d\n",
           (int)tokens.size(), n_unique, max_run);

    // First 16 + last 16
    printf("[autoreg] first 16 tokens:");
    for (int i = 0; i < 16 && i < (int)tokens.size(); ++i) printf(" %d", tokens[i]);
    printf("\n");
    printf("[autoreg] last  16 tokens:");
    int last_start = std::max(0, (int)tokens.size() - 16);
    for (int i = last_start; i < (int)tokens.size(); ++i) printf(" %d", tokens[i]);
    printf("\n");

    double avg_ms      = total_engine_ms / (double)n_steps;
    double tok_per_sec = 1000.0 * (double)n_steps / total_wall_ms;
    printf("[autoreg] engine total=%.2f ms  avg=%.2f ms/step  "
           "wall total=%.2f ms  TPS=%.2f\n",
           total_engine_ms, avg_ms, total_wall_ms, tok_per_sec);

    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    if (!sane_diverse) {
        fprintf(stderr,
                "[autoreg] FAIL: token sequence is constant (n_unique=1) — "
                "engine likely stuck in a degenerate fixpoint or LM head broken\n");
        return 1;
    }
    if (!sane_runs) {
        fprintf(stderr,
                "[autoreg] FAIL: longest run %d covers > 95%% of sequence\n",
                max_run);
        return 1;
    }

    printf("[autoreg] Phase 2.3 autoregressive decode PASS  "
           "n_steps=%d  TPS=%.2f  avg_engine=%.2f ms/step\n",
           n_steps, tok_per_sec, avg_ms);
    return 0;
}
