// ============================================================================
// Phase 2.7d E2E TTS smoke harness — text -> codec tokens -> audio waveform.
//
// Pipeline (text-driven seed -> Talker autoreg -> Predictor groups -> Vocoder):
//
//   text "Hello world..."
//     -> BPE tokenize (vocab.json + merges.txt; same Qwen2 tokenizer the
//        Ascend qwen_tts uses)
//     -> derive a deterministic seed codec token from the text-token hash,
//        so different prompts produce different audio
//     -> TalkerCudaEngine autoreg (28L Qwen3, GGUF
//        qwen3_tts_talker.gguf) for N_TALKER_STEPS, recording codebook-0
//        (semantic) tokens
//     -> CodePredictor (5L Qwen3, GGUF qwen3_tts_predictor.gguf) — Pattern B
//        sequential: feed full semantic stream, autoreg 15 acoustic codebooks
//     -> assemble [16, T] codes
//     -> SpeechTokenizerDecoderCudaEngine.decode_audio -> [T*1920] @ 24kHz
//     -> save WAV
//
// Honest scope note (Phase 2.7d):
//   The qwen3_tts_talker.gguf shipped on zgx-3675 is a flat llama.cpp-style
//   codec-only export (vocab=3072 codec tokens, no text vocab, no text_proj,
//   no spk_embedding tensors). Real text-conditioned generation requires
//   parsing qwen3_assets.gguf (text_embedding + text_projection + codec_head)
//   and porting TalkerLLM::build_input_embeddings — out of scope for 2.7d.
//
//   This harness validates the full pipeline structure end-to-end:
//     - text->BPE tokens (real)
//     - BPE tokens -> seed codec token (proxy; deterministic per-prompt)
//     - Talker autoreg producing semantic codes (real engine)
//     - Predictor producing acoustic codes (real engine)
//     - Vocoder producing audio (real engine, audible from dummy codes 2.7c)
//   The audio will NOT be intelligible as the input text — it is an
//   un-conditioned codec-token autoregression seeded by text. We expect
//   non-silent, NaN/Inf-free, plausibly speech-shaped output.
//
// GREEN gate (Phase 2.7d):
//   - Audio length matches T_total * 1920
//   - No NaN/Inf
//   - Within [-1, 1]
//   - std > 0.01 (non-silent)
//   - Wall breakdown reported per stage
//
// Usage:
//   test_qwen_tts_e2e <text> [n_talker_steps=128] [n_predictor_steps=15]
//
// Hardcoded model paths (tracks the dispatch description):
//   talker:    ~/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_talker.gguf
//   predictor: ~/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_predictor.gguf
//   decoder:   ~/qwen3_tts_cuda/gguf_decoder/qwen_tts_tokenizer_dec.gguf
//   vocab:     ~/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base/vocab.json
//   merges:    ~/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base/merges.txt
// ============================================================================

#include "talker_cuda_engine.h"
#include "speech_tokenizer_decoder_cuda_engine.h"
#include "../talker.h"
#include "../../qwen_common/bpe_tokenizer.h"
#include "../../qwen_common/audio_io.h"

#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr const char *TALKER_GGUF =
    "/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_talker.gguf";
constexpr const char *PREDICTOR_GGUF =
    "/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_predictor.gguf";
constexpr const char *DECODER_GGUF =
    "/home/user1/qwen3_tts_cuda/gguf_decoder/qwen_tts_tokenizer_dec.gguf";
constexpr const char *VOCAB_JSON =
    "/home/user1/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base/vocab.json";
constexpr const char *MERGES_TXT =
    "/home/user1/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base/merges.txt";
constexpr const char *OUT_WAV = "/tmp/qwen_tts_e2e.wav";

bool load_tensor_f32(ggml_context *ctx, const char *name,
                      size_t expected_elems, std::vector<float> &out) {
    ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\\n", name); return false; }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr, "%s: expected %zu got %zu\\n", name, expected_elems, n);
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
            fprintf(stderr, "%s: unsupported dtype %d\\n", name, (int)t->type);
            return false;
        }
        tt->to_float(t->data, out.data(), (int64_t)n);
    }
    return true;
}

void matvec_f32(const float *W, const float *x, float *out, int rows, int cols) {
    for (int v = 0; v < rows; ++v) {
        const float *row = W + (size_t)v * cols;
        float s = 0.0f;
        for (int i = 0; i < cols; ++i) s += row[i] * x[i];
        out[v] = s;
    }
}

int argmax_f32(const float *logits, int lo, int hi) {
    int best = lo; float bv = logits[lo];
    for (int i = lo + 1; i < hi; ++i) if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: test_qwen_tts_e2e <text> [n_talker_steps=128] [n_predictor_steps=15]\\n");
        return 2;
    }
    std::string text = argv[1];
    int n_talker_steps = (argc >= 3) ? std::atoi(argv[2]) : 128;
    int n_predictor_steps = (argc >= 4) ? std::atoi(argv[3]) : 15;
    if (n_talker_steps <= 0) n_talker_steps = 128;
    if (n_predictor_steps <= 0 || n_predictor_steps > 15) n_predictor_steps = 15;

    printf("[e2e] text='%s'\\n", text.c_str());
    printf("[e2e] n_talker_steps=%d  n_predictor_steps=%d\\n",
           n_talker_steps, n_predictor_steps);

    auto wall_t0 = std::chrono::high_resolution_clock::now();

    // ---- Stage A: BPE tokenize ---------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    BpeTokenizer tok;
    if (!tok.load(VOCAB_JSON, MERGES_TXT)) {
        fprintf(stderr, "[e2e] BPE load FAIL\\n");
        return 1;
    }
    auto text_tokens = tok.encode(text);
    if (text_tokens.empty()) {
        fprintf(stderr, "[e2e] tokenize empty\\n");
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_bpe = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[e2e] BPE: %zu tokens in %.2f ms; first 10:", text_tokens.size(), ms_bpe);
    for (size_t i = 0; i < std::min<size_t>(10, text_tokens.size()); ++i) printf(" %d", text_tokens[i]);
    printf("\\n");

    // ---- Stage B: Talker engine init + GGUF embeddings/head ----------------
    t0 = std::chrono::high_resolution_clock::now();
    TalkerConfig talker_cfg;
    ominix_cuda::TalkerCudaEngine talker;
    if (!talker.init_from_gguf(TALKER_GGUF, talker_cfg, /*device=*/0)) {
        fprintf(stderr, "[e2e] talker init FAIL\\n");
        return 1;
    }
    talker.reset_kv_cache();
    const int t_n_embd = talker_cfg.hidden_size;

    ggml_context *t_ctx = nullptr;
    gguf_init_params gp; gp.no_alloc = false; gp.ctx = &t_ctx;
    gguf_context *t_gguf = gguf_init_from_file(TALKER_GGUF, gp);
    if (!t_gguf || !t_ctx) { fprintf(stderr, "[e2e] talker gguf open FAIL\\n"); return 1; }
    ggml_tensor *t_embd_t = ggml_get_tensor(t_ctx, "token_embd.weight");
    const int t_vocab = (int)t_embd_t->ne[1];
    std::vector<float> t_embd_w, t_lm_head_w;
    if (!load_tensor_f32(t_ctx, "token_embd.weight", (size_t)t_vocab * t_n_embd, t_embd_w)) return 1;
    if (!load_tensor_f32(t_ctx, "output.weight", (size_t)t_vocab * t_n_embd, t_lm_head_w)) return 1;
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms_talker_init = std::chrono::duration<double, std::milli>(t2 - t0).count();
    printf("[e2e] talker init+weights: %.2f ms (vocab=%d, n_embd=%d)\\n",
           ms_talker_init, t_vocab, t_n_embd);

    // ---- Stage C: Talker autoreg (semantic codebook 0) ---------------------
    // Seed codec token: deterministic hash of text tokens, mod codec vocab,
    // skipping the first 3 control tokens. This makes different prompts
    // produce different (but un-conditioned) codec streams.
    uint64_t h = 1469598103934665603ULL;
    for (int t : text_tokens) { h ^= (uint64_t)t; h *= 1099511628211ULL; }
    int seed_token = 3 + (int)(h % (uint64_t)(t_vocab - 3));
    printf("[e2e] seed codec token (text-derived): %d\\n", seed_token);

    std::vector<int> semantic_tokens;
    semantic_tokens.reserve(n_talker_steps);
    std::vector<float> input_emb(t_n_embd), hidden(t_n_embd), logits(t_vocab);

    int cur = seed_token;
    t0 = std::chrono::high_resolution_clock::now();
    for (int pos = 0; pos < n_talker_steps; ++pos) {
        std::memcpy(input_emb.data(),
                    t_embd_w.data() + (size_t)cur * t_n_embd,
                    t_n_embd * sizeof(float));
        talker.forward_decode(input_emb.data(), pos, hidden.data());
        for (int i = 0; i < t_n_embd; ++i) {
            if (std::isnan(hidden[i]) || std::isinf(hidden[i])) {
                fprintf(stderr, "[e2e] talker hidden NaN/Inf at pos=%d\\n", pos);
                return 1;
            }
        }
        matvec_f32(t_lm_head_w.data(), hidden.data(), logits.data(),
                    t_vocab, t_n_embd);
        // Greedy argmax over codec entries (skip first 3 control tokens).
        int nxt = argmax_f32(logits.data(), 3, t_vocab);
        semantic_tokens.push_back(nxt);
        cur = nxt;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_talker_run = std::chrono::duration<double, std::milli>(t3 - t0).count();
    printf("[e2e] talker autoreg %d steps: %.2f ms (%.2f TPS)\\n",
           n_talker_steps, ms_talker_run, n_talker_steps * 1000.0 / ms_talker_run);
    printf("[e2e] semantic[0..7]:");
    for (int i = 0; i < 8 && i < (int)semantic_tokens.size(); ++i) printf(" %d", semantic_tokens[i]);
    printf("\\n");

    // ---- Stage D: Predictor (acoustic codebooks 1..15) — Pattern B sequential
    // CodePredictor is a 5L Qwen3 with vocab=30720 = 15 groups x 2048.
    // Group g (0..14) entries occupy logits[g*2048 .. (g+1)*2048].
    // For each semantic token, we run a fresh autoreg of n_predictor_steps
    // through the predictor seeded by that semantic token's value (modded
    // into codec vocab range). This is Pattern B sequential — no live
    // cross-attention with the Talker hidden state. Per dispatch fallback,
    // this preserves pipeline structure and produces audio that responds
    // to the semantic stream (which itself responds to the text).
    t0 = std::chrono::high_resolution_clock::now();
    CodePredictorConfig cp_def;
    TalkerConfig cp_cfg;
    cp_cfg.hidden_size = cp_def.hidden_size;
    cp_cfg.num_hidden_layers = cp_def.num_hidden_layers;
    cp_cfg.num_attention_heads = cp_def.num_attention_heads;
    cp_cfg.num_key_value_heads = cp_def.num_key_value_heads;
    cp_cfg.intermediate_size = cp_def.intermediate_size;
    cp_cfg.head_dim = cp_def.head_dim;
    cp_cfg.rope_theta = cp_def.rope_theta;
    cp_cfg.rms_norm_eps = cp_def.rms_norm_eps;
    ominix_cuda::TalkerCudaEngine predictor;
    if (!predictor.init_from_gguf(PREDICTOR_GGUF, cp_cfg, /*device=*/0)) {
        fprintf(stderr, "[e2e] predictor init FAIL\\n");
        return 1;
    }
    const int p_n_embd = cp_cfg.hidden_size;
    ggml_context *p_ctx = nullptr;
    gguf_init_params gp2; gp2.no_alloc = false; gp2.ctx = &p_ctx;
    gguf_context *p_gguf = gguf_init_from_file(PREDICTOR_GGUF, gp2);
    if (!p_gguf || !p_ctx) { fprintf(stderr, "[e2e] predictor gguf FAIL\\n"); return 1; }
    ggml_tensor *p_embd_t = ggml_get_tensor(p_ctx, "token_embd.weight");
    const int p_vocab = (int)p_embd_t->ne[1];
    std::vector<float> p_embd_w, p_lm_head_w;
    if (!load_tensor_f32(p_ctx, "token_embd.weight", (size_t)p_vocab * p_n_embd, p_embd_w)) return 1;
    if (!load_tensor_f32(p_ctx, "output.weight", (size_t)p_vocab * p_n_embd, p_lm_head_w)) return 1;
    auto t4 = std::chrono::high_resolution_clock::now();
    double ms_pred_init = std::chrono::duration<double, std::milli>(t4 - t0).count();
    printf("[e2e] predictor init+weights: %.2f ms (vocab=%d, n_embd=%d)\\n",
           ms_pred_init, p_vocab, p_n_embd);

    // Per-semantic-token, run 15 predictor steps for groups 1..15.
    int T = (int)semantic_tokens.size();
    std::vector<std::vector<int>> codes(16, std::vector<int>(T, 0));
    for (int t = 0; t < T; ++t) codes[0][t] = semantic_tokens[t] % 2048;

    std::vector<float> p_input_emb(p_n_embd), p_hidden(p_n_embd), p_logits(p_vocab);
    t0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < T; ++t) {
        predictor.reset_kv_cache();
        int p_cur = (semantic_tokens[t] % p_vocab);
        for (int g = 0; g < n_predictor_steps; ++g) {
            std::memcpy(p_input_emb.data(),
                        p_embd_w.data() + (size_t)p_cur * p_n_embd,
                        p_n_embd * sizeof(float));
            predictor.forward_decode(p_input_emb.data(), g, p_hidden.data());
            matvec_f32(p_lm_head_w.data(), p_hidden.data(), p_logits.data(),
                        p_vocab, p_n_embd);
            // Group-g acoustic logits: slice [g*2048, (g+1)*2048)
            int lo = g * 2048;
            int hi = lo + 2048;
            int nxt = argmax_f32(p_logits.data(), lo, hi);
            int acoustic_tok = nxt - lo;  // 0..2047
            codes[g + 1][t] = acoustic_tok;
            p_cur = nxt;  // feed back full-vocab token (with group offset)
        }
        if (n_predictor_steps < 15) {
            for (int g = n_predictor_steps; g < 15; ++g) codes[g + 1][t] = 0;
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_pred_run = std::chrono::duration<double, std::milli>(t5 - t0).count();
    int total_pred_steps = T * n_predictor_steps;
    printf("[e2e] predictor sequential %d frames x %d groups = %d steps: %.2f ms (%.2f TPS)\\n",
           T, n_predictor_steps, total_pred_steps,
           ms_pred_run, total_pred_steps * 1000.0 / ms_pred_run);

    // ---- Stage E: Vocoder ---------------------------------------------------
    t0 = std::chrono::high_resolution_clock::now();
    ominix_cuda::SpeechTokenizerDecoderCudaEngine dec;
    if (!dec.init_from_gguf(DECODER_GGUF, /*device=*/0)) {
        fprintf(stderr, "[e2e] decoder init FAIL\\n");
        return 1;
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    double ms_dec_init = std::chrono::duration<double, std::milli>(t6 - t0).count();

    // Flatten codes to [16 * T] row-major (codebook-major).
    std::vector<int> flat(16 * T);
    for (int q = 0; q < 16; ++q)
        for (int t = 0; t < T; ++t) flat[q * T + t] = codes[q][t];

    auto audio = dec.decode_audio(flat.data(), 16, T);
    auto t7 = std::chrono::high_resolution_clock::now();
    double ms_voc_run = std::chrono::duration<double, std::milli>(t7 - t6).count();
    printf("[e2e] decoder init: %.2f ms; decode_audio: %.2f ms; samples=%zu (%.2f sec @ 24kHz)\\n",
           ms_dec_init, ms_voc_run, audio.size(), audio.size() / 24000.0);

    // ---- Stage F: Audio sanity + WAV save ----------------------------------
    int n_nan = 0, n_inf = 0, n_oor = 0;
    double sum = 0, sum2 = 0;
    float vmin = 1e30f, vmax = -1e30f;
    for (float v : audio) {
        if (std::isnan(v)) ++n_nan;
        if (std::isinf(v)) ++n_inf;
        if (v < -1.001f || v > 1.001f) ++n_oor;
        sum += v; sum2 += (double)v * v;
        if (v < vmin) vmin = v; if (v > vmax) vmax = v;
    }
    double n = (double)audio.size();
    double mean = sum / n;
    double var = std::max(0.0, sum2 / n - mean * mean);
    double std_ = std::sqrt(var);
    printf("[e2e] audio stats: nan=%d inf=%d out_of_range=%d  min=%.4f max=%.4f mean=%.4f std=%.4f\\n",
           n_nan, n_inf, n_oor, vmin, vmax, mean, std_);

    t0 = std::chrono::high_resolution_clock::now();
    if (!audio_io::save_wav(OUT_WAV, audio, 24000, 1)) {
        fprintf(stderr, "[e2e] save_wav FAIL\\n");
        return 1;
    }
    auto t8 = std::chrono::high_resolution_clock::now();
    double ms_wav = std::chrono::duration<double, std::milli>(t8 - t0).count();
    printf("[e2e] saved WAV: %s (%.2f ms)\\n", OUT_WAV, ms_wav);

    auto wall_t1 = std::chrono::high_resolution_clock::now();
    double ms_wall_total = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    double audio_sec = audio.size() / 24000.0;
    double rtf = ms_wall_total / 1000.0 / std::max(audio_sec, 1e-9);
    printf("[e2e] WALL BREAKDOWN:\\n");
    printf("        bpe       %8.2f ms\\n", ms_bpe);
    printf("        talker    %8.2f ms (init %.0f + run %.0f)\\n",
           ms_talker_init + ms_talker_run, ms_talker_init, ms_talker_run);
    printf("        predictor %8.2f ms (init %.0f + run %.0f)\\n",
           ms_pred_init + ms_pred_run, ms_pred_init, ms_pred_run);
    printf("        vocoder   %8.2f ms (init %.0f + run %.0f)\\n",
           ms_dec_init + ms_voc_run, ms_dec_init, ms_voc_run);
    printf("        wav_save  %8.2f ms\\n", ms_wav);
    printf("        TOTAL     %8.2f ms\\n", ms_wall_total);
    printf("[e2e] audio: %.2f sec  RTF=%.3f (lower=better; <1 = real-time)\\n",
           audio_sec, rtf);

    bool green = (n_nan == 0 && n_inf == 0 && n_oor == 0 &&
                  std_ > 0.01 &&
                  audio.size() == (size_t)T * 1920);
    printf("[e2e] verdict: %s  (NaN-free=%d, in-range=%d, std>0.01=%d, shape-ok=%d)\\n",
           green ? "GREEN" : "YELLOW",
           (n_nan == 0 && n_inf == 0), (n_oor == 0),
           (std_ > 0.01), (audio.size() == (size_t)T * 1920));

    gguf_free(t_gguf); ggml_free(t_ctx);
    gguf_free(p_gguf); ggml_free(p_ctx);
    return green ? 0 : 1;
}
