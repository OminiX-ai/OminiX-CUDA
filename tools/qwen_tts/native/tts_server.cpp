// ============================================================================
// Phase 2.10 — TTS warm-start daemon.
//
// Loads all models (BPE tokenizer, Talker, qwen3_assets, Predictor + LM-head,
// SpeechTokenizerDecoder) once at startup, then serves synthesis requests on
// a TCP socket. Amortizes the ~11 s of init+weight loading dominating the
// cold-start path of test_qwen_tts_e2e.
//
// Protocol (line-based, one request per line, ASCII tab-separated):
//   request : <text> '\t' <out_wav_path> '\n'
//   reply   : 'OK'  '\t' <wall_ms> '\t' <rtf> '\t' <audio_sec> '\n'
//          or 'ERR' '\t' <message> '\n'
// Lines are TAB-separated and \n-terminated; the text field MUST NOT contain
// '\t' or '\n'. Single-connection, single-thread server (CUDA state is
// non-reentrant).
//
// Default port 7777 (override via env OMINIX_TTS_SERVER_PORT). Bind 127.0.0.1.
//
// This file is a refactor of test_qwen_tts_e2e.cpp — the per-request inference
// path is identical except: no per-request gguf reopen, no per-request
// engine-init, and the WAV is written to the requested path.
// ============================================================================

#include "talker_cuda_engine.h"
#include "speech_tokenizer_decoder_cuda_engine.h"
#include "../talker.h"
#include "../../qwen_common/bpe_tokenizer.h"
#include "../../qwen_common/audio_io.h"

#include "ggml.h"
#include "gguf.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <string>
#include <vector>
#include <csignal>
#include <cerrno>

namespace {

constexpr const char *TALKER_GGUF =
    "/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_talker.gguf";
constexpr const char *PREDICTOR_GGUF =
    "/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_predictor.gguf";
constexpr const char *ASSETS_GGUF =
    "/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_assets.gguf";
constexpr const char *DECODER_GGUF =
    "/home/user1/qwen3_tts_cuda/gguf_decoder/qwen_tts_tokenizer_dec.gguf";
constexpr const char *VOCAB_JSON =
    "/home/user1/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base/vocab.json";
constexpr const char *MERGES_TXT =
    "/home/user1/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base/merges.txt";

// Special token IDs (text vocab).
constexpr int IM_START      = 151644;
constexpr int ASSISTANT     = 77091;
constexpr int NEWLINE       = 198;
constexpr int TTS_PAD       = 151671;
constexpr int TTS_BOS       = 151672;
constexpr int TTS_EOS       = 151673;
constexpr int CODEC_BOS     = 2149;
constexpr int CODEC_EOS     = 2150;

// ------- Tensor loaders (clones from test_qwen_tts_e2e.cpp) -----------------
bool load_tensor_f32(ggml_context *ctx, const char *name,
                      size_t expected_elems, std::vector<float> &out) {
    ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) { fprintf(stderr, "missing tensor: %s\n", name); return false; }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr, "%s: expected %zu got %zu\n", name, expected_elems, n);
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
            fprintf(stderr, "%s: unsupported dtype %d\n", name, (int)t->type);
            return false;
        }
        tt->to_float(t->data, out.data(), (int64_t)n);
    }
    return true;
}

bool load_assets_embedding(ggml_context *ctx, const char *name,
                            int vocab, int hidden,
                            std::vector<float> &out_vh) {
    ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) { fprintf(stderr, "missing assets tensor: %s\n", name); return false; }
    if (t->ne[0] != (int64_t)vocab || t->ne[1] != (int64_t)hidden) {
        fprintf(stderr, "%s: shape mismatch ne=[%lld,%lld] expected=[%d,%d]\n",
                name, (long long)t->ne[0], (long long)t->ne[1], vocab, hidden);
        return false;
    }
    std::vector<float> raw;
    if (!load_tensor_f32(ctx, name, (size_t)vocab * hidden, raw)) return false;
    out_vh.resize((size_t)vocab * hidden);
    for (int h = 0; h < hidden; ++h) {
        const float *row = raw.data() + (size_t)h * vocab;
        for (int v = 0; v < vocab; ++v) {
            out_vh[(size_t)v * hidden + h] = row[v];
        }
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

// Sampling — env-overridable knobs (read once per request from process env).
struct SamplingConfig {
    float temperature        = 0.9f;
    int   top_k              = 50;
    float top_p              = 1.0f;
    float repetition_penalty = 1.05f;
    int   recent_window      = 64;
    bool  do_sample          = true;
    uint64_t seed            = 42;
};

static float env_f(const char *k, float def) {
    const char *v = std::getenv(k);
    if (!v || !*v) return def;
    return (float)std::atof(v);
}
static int env_i(const char *k, int def) {
    const char *v = std::getenv(k);
    if (!v || !*v) return def;
    return std::atoi(v);
}
static uint64_t env_u64(const char *k, uint64_t def) {
    const char *v = std::getenv(k);
    if (!v || !*v) return def;
    return (uint64_t)std::strtoull(v, nullptr, 10);
}

SamplingConfig load_sampling_config_from_env() {
    SamplingConfig c;
    c.temperature        = env_f("OMINIX_TTS_TEMPERATURE",  c.temperature);
    c.top_k              = env_i("OMINIX_TTS_TOP_K",        c.top_k);
    c.top_p              = env_f("OMINIX_TTS_TOP_P",        c.top_p);
    c.repetition_penalty = env_f("OMINIX_TTS_REP_PENALTY",  c.repetition_penalty);
    c.seed               = env_u64("OMINIX_TTS_SEED",       c.seed);
    int dos              = env_i("OMINIX_TTS_DO_SAMPLE",    c.do_sample ? 1 : 0);
    c.do_sample          = (dos != 0);
    return c;
}

void apply_repetition_penalty(float *logits, int vocab_size,
                              const std::vector<int> &recent, int lo, int hi,
                              float penalty) {
    (void)vocab_size;
    if (penalty == 1.0f) return;
    for (int tok : recent) {
        int t = tok + lo;
        if (t < lo || t >= hi) continue;
        if (logits[t] > 0.0f) logits[t] /= penalty;
        else                  logits[t] *= penalty;
    }
}

int sample_token(float *logits, int lo, int hi,
                 const SamplingConfig &cfg, std::mt19937 &rng) {
    if (!cfg.do_sample || cfg.temperature <= 0.0f) {
        return argmax_f32(logits, lo, hi);
    }
    const float inv_t = 1.0f / cfg.temperature;
    std::vector<std::pair<float,int>> cands;
    cands.reserve(hi - lo);
    for (int i = lo; i < hi; ++i) {
        float l = logits[i] * inv_t;
        if (l > -1e9f) cands.emplace_back(l, i);
    }
    if (cands.empty()) return argmax_f32(logits, lo, hi);
    std::sort(cands.begin(), cands.end(),
              [](const auto &a, const auto &b){ return a.first > b.first; });
    if (cfg.top_k > 0 && cfg.top_k < (int)cands.size())
        cands.resize(cfg.top_k);
    float mx = cands.front().first;
    float sum = 0.0f;
    for (auto &c : cands) { c.first = std::exp(c.first - mx); sum += c.first; }
    if (sum <= 0.0f) return cands.front().second;
    for (auto &c : cands) c.first /= sum;
    if (cfg.top_p < 1.0f) {
        float cs = 0.0f;
        int cutoff = (int)cands.size();
        for (int i = 0; i < (int)cands.size(); ++i) {
            cs += cands[i].first;
            if (cs >= cfg.top_p) { cutoff = i + 1; break; }
        }
        cands.resize(cutoff);
        sum = 0.0f;
        for (auto &c : cands) sum += c.first;
        for (auto &c : cands) c.first /= sum;
    }
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    float r = u(rng);
    float cs = 0.0f;
    for (const auto &c : cands) {
        cs += c.first;
        if (r <= cs) return c.second;
    }
    return cands.back().second;
}

// ============================================================================
// Server-side context — populated once at startup, reused for every request.
// ============================================================================
struct ServerCtx {
    BpeTokenizer tok;

    // Talker engine + weights.
    ominix_cuda::TalkerCudaEngine talker;
    int t_n_embd = 0;
    int t_vocab  = 0;             // codec vocab
    std::vector<float> t_lm_head_w;     // [t_vocab, t_n_embd] for first-codec sample
    std::vector<float> text_embd_vh;    // [TEXT_VOCAB, t_n_embd]
    std::vector<float> codec_embd0_vh;  // [t_vocab,    t_n_embd]

    // Predictor engine + weights (token embd held on host; LM-head uploaded to dev).
    ominix_cuda::TalkerCudaEngine predictor;
    int p_n_embd = 0;
    int p_vocab  = 0;             // 30720 = 15 * 2048
    std::vector<float> p_embd_w;        // [p_vocab, p_n_embd]

    // Speech-tokenizer-decoder.
    ominix_cuda::SpeechTokenizerDecoderCudaEngine decoder;
};

constexpr int TEXT_VOCAB = 151936;

bool init_server_ctx(ServerCtx &S) {
    auto t0 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[tts_server] loading BPE tokenizer...\n");
    if (!S.tok.load(VOCAB_JSON, MERGES_TXT)) {
        fprintf(stderr, "[tts_server] BPE load FAIL\n");
        return false;
    }

    // ---- Talker ---------------------------------------------------------
    fprintf(stderr, "[tts_server] init Talker engine + weights...\n");
    TalkerConfig talker_cfg;
    if (!S.talker.init_from_gguf(TALKER_GGUF, talker_cfg, /*device=*/0)) {
        fprintf(stderr, "[tts_server] talker init FAIL\n");
        return false;
    }
    S.talker.reset_kv_cache();
    S.t_n_embd = talker_cfg.hidden_size;

    {
        ggml_context *t_ctx = nullptr;
        gguf_init_params gp; gp.no_alloc = false; gp.ctx = &t_ctx;
        gguf_context *t_gguf = gguf_init_from_file(TALKER_GGUF, gp);
        if (!t_gguf || !t_ctx) { fprintf(stderr, "[tts_server] talker gguf FAIL\n"); return false; }
        ggml_tensor *t_embd_t = ggml_get_tensor(t_ctx, "token_embd.weight");
        S.t_vocab = (int)t_embd_t->ne[1];
        if (!load_tensor_f32(t_ctx, "output.weight",
                              (size_t)S.t_vocab * S.t_n_embd, S.t_lm_head_w)) {
            return false;
        }
        gguf_free(t_gguf); ggml_free(t_ctx);
    }

    // ---- qwen3_assets ---------------------------------------------------
    fprintf(stderr, "[tts_server] loading qwen3_assets (text_embd + codec_embd.0)...\n");
    {
        ggml_context *a_ctx = nullptr;
        gguf_init_params gp_a; gp_a.no_alloc = false; gp_a.ctx = &a_ctx;
        gguf_context *a_gguf = gguf_init_from_file(ASSETS_GGUF, gp_a);
        if (!a_gguf || !a_ctx) { fprintf(stderr, "[tts_server] assets gguf FAIL\n"); return false; }
        if (!load_assets_embedding(a_ctx, "text_embd",  TEXT_VOCAB, S.t_n_embd, S.text_embd_vh))   return false;
        if (!load_assets_embedding(a_ctx, "codec_embd.0", S.t_vocab, S.t_n_embd, S.codec_embd0_vh)) return false;
        gguf_free(a_gguf); ggml_free(a_ctx);
    }

    // ---- Predictor ------------------------------------------------------
    fprintf(stderr, "[tts_server] init Predictor engine + weights...\n");
    {
        CodePredictorConfig cp_def;
        TalkerConfig cp_cfg;
        cp_cfg.hidden_size         = cp_def.hidden_size;
        cp_cfg.num_hidden_layers   = cp_def.num_hidden_layers;
        cp_cfg.num_attention_heads = cp_def.num_attention_heads;
        cp_cfg.num_key_value_heads = cp_def.num_key_value_heads;
        cp_cfg.intermediate_size   = cp_def.intermediate_size;
        cp_cfg.head_dim            = cp_def.head_dim;
        cp_cfg.rope_theta          = cp_def.rope_theta;
        cp_cfg.rms_norm_eps        = cp_def.rms_norm_eps;
        if (!S.predictor.init_from_gguf(PREDICTOR_GGUF, cp_cfg, /*device=*/0)) {
            fprintf(stderr, "[tts_server] predictor init FAIL\n");
            return false;
        }
        S.p_n_embd = cp_cfg.hidden_size;

        ggml_context *p_ctx = nullptr;
        gguf_init_params gp2; gp2.no_alloc = false; gp2.ctx = &p_ctx;
        gguf_context *p_gguf = gguf_init_from_file(PREDICTOR_GGUF, gp2);
        if (!p_gguf || !p_ctx) { fprintf(stderr, "[tts_server] predictor gguf FAIL\n"); return false; }
        ggml_tensor *p_embd_t = ggml_get_tensor(p_ctx, "token_embd.weight");
        S.p_vocab = (int)p_embd_t->ne[1];
        if (!load_tensor_f32(p_ctx, "token_embd.weight",
                              (size_t)S.p_vocab * S.p_n_embd, S.p_embd_w)) {
            return false;
        }
        std::vector<float> p_lm_head_w;
        if (!load_tensor_f32(p_ctx, "output.weight",
                              (size_t)S.p_vocab * S.p_n_embd, p_lm_head_w)) {
            return false;
        }
        if (!S.predictor.upload_lm_head_weights(p_lm_head_w.data(), S.p_vocab)) {
            fprintf(stderr, "[tts_server] predictor LM-head upload FAIL\n");
            return false;
        }
        // Phase 2.6 — optional FP8 lane on the predictor LM-head. Microbench
        // shows ~2.4x at this shape; only kicks in when enabled at startup.
        if (env_i("OMINIX_TTS_USE_FP8_LMHEAD", 0)) {
            if (S.predictor.upload_lm_head_weights_fp8(p_lm_head_w.data(),
                                                        S.p_vocab)) {
                S.predictor.set_use_fp8_lm_head(true);
                fprintf(stderr,
                        "[tts_server] FP8 predictor LM-head ENABLED\n");
            } else {
                fprintf(stderr,
                        "[tts_server] FP8 predictor LM-head NOT available "
                        "(falling back to F16)\n");
            }
        }
        gguf_free(p_gguf); ggml_free(p_ctx);
    }

    // ---- Decoder --------------------------------------------------------
    fprintf(stderr, "[tts_server] init SpeechTokenizerDecoder...\n");
    if (!S.decoder.init_from_gguf(DECODER_GGUF, /*device=*/0)) {
        fprintf(stderr, "[tts_server] decoder init FAIL\n");
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fprintf(stderr,
            "[tts_server] WARM: all models loaded in %.2f ms (talker hidden=%d codec_vocab=%d, "
            "predictor hidden=%d vocab=%d)\n",
            ms, S.t_n_embd, S.t_vocab, S.p_n_embd, S.p_vocab);
    return true;
}

struct SynthResult {
    bool   ok        = false;
    double wall_ms   = 0.0;
    double audio_sec = 0.0;
    double rtf       = 0.0;
    std::string err;
};

SynthResult synthesize_one(ServerCtx &S, const std::string &text,
                           const std::string &out_wav,
                           int n_talker_steps = 128,
                           int n_predictor_steps = 15) {
    SynthResult R;
    auto wall_t0 = std::chrono::high_resolution_clock::now();

    SamplingConfig samp = load_sampling_config_from_env();
    std::mt19937 rng((uint32_t)samp.seed);

    auto text_tokens = S.tok.encode(text);
    if (text_tokens.empty()) { R.err = "tokenize empty"; return R; }

    auto lookup_text = [&](int tok_id, float *out) {
        if (tok_id < 0 || tok_id >= TEXT_VOCAB) {
            std::memset(out, 0, S.t_n_embd * sizeof(float));
            return;
        }
        std::memcpy(out, S.text_embd_vh.data() + (size_t)tok_id * S.t_n_embd,
                    S.t_n_embd * sizeof(float));
    };
    auto lookup_codec0 = [&](int tok_id, float *out) {
        if (tok_id < 0 || tok_id >= S.t_vocab) {
            std::memset(out, 0, S.t_n_embd * sizeof(float));
            return;
        }
        std::memcpy(out, S.codec_embd0_vh.data() + (size_t)tok_id * S.t_n_embd,
                    S.t_n_embd * sizeof(float));
    };

    // Reset Talker state for this request.
    S.talker.reset_kv_cache();

    // Build prefill.
    std::vector<int> prefill_kinds, prefill_ids;
    auto add_text  = [&](int id) { prefill_kinds.push_back(0); prefill_ids.push_back(id); };
    auto add_codec = [&](int id) { prefill_kinds.push_back(1); prefill_ids.push_back(id); };
    add_text(IM_START); add_text(ASSISTANT); add_text(NEWLINE);
    for (int i = 0; i < 3; ++i) add_text(TTS_PAD);
    add_text(TTS_BOS);
    for (int t : text_tokens) add_text(t);
    add_text(TTS_EOS);
    add_codec(CODEC_BOS);

    int prefill_len = (int)prefill_kinds.size();
    std::vector<float> input_emb(S.t_n_embd), hidden(S.t_n_embd), logits(S.t_vocab);

    for (int pos = 0; pos < prefill_len; ++pos) {
        if (prefill_kinds[pos] == 0) lookup_text(prefill_ids[pos], input_emb.data());
        else                          lookup_codec0(prefill_ids[pos], input_emb.data());
        S.talker.forward_decode(input_emb.data(), pos, hidden.data());
    }

    // First codec token.
    matvec_f32(S.t_lm_head_w.data(), hidden.data(), logits.data(), S.t_vocab, S.t_n_embd);
    int first_tok = sample_token(logits.data(), 3, S.t_vocab, samp, rng);

    std::vector<int> semantic_tokens;
    semantic_tokens.reserve(n_talker_steps);
    semantic_tokens.push_back(first_tok);

    int pos_cursor = prefill_len;
    int cur_tok = first_tok;
    bool hit_eos = false;
    for (int step = 1; step < n_talker_steps; ++step) {
        lookup_codec0(cur_tok, input_emb.data());
        S.talker.forward_decode(input_emb.data(), pos_cursor++, hidden.data());
        matvec_f32(S.t_lm_head_w.data(), hidden.data(), logits.data(), S.t_vocab, S.t_n_embd);
        std::vector<int> recent_abs;
        int from = std::max(0, (int)semantic_tokens.size() - samp.recent_window);
        for (int i = from; i < (int)semantic_tokens.size(); ++i)
            recent_abs.push_back(semantic_tokens[i]);
        apply_repetition_penalty(logits.data(), S.t_vocab, recent_abs, 0, S.t_vocab,
                                  samp.repetition_penalty);
        int nxt = sample_token(logits.data(), 3, S.t_vocab, samp, rng);
        semantic_tokens.push_back(nxt);
        cur_tok = nxt;
        if (nxt == CODEC_EOS) { hit_eos = true; break; }
    }
    if (hit_eos && !semantic_tokens.empty() && semantic_tokens.back() == CODEC_EOS)
        semantic_tokens.pop_back();
    int T = (int)semantic_tokens.size();
    if (T <= 0) { R.err = "no codec tokens"; return R; }

    // Predictor — 15 acoustic groups per frame.
    std::vector<std::vector<int>> codes(16, std::vector<int>(T, 0));
    for (int t = 0; t < T; ++t) codes[0][t] = semantic_tokens[t] % 2048;

    std::vector<float> p_input_emb(S.p_n_embd), p_logits(S.p_vocab);
    for (int t = 0; t < T; ++t) {
        S.predictor.reset_kv_cache();
        int p_cur = (semantic_tokens[t] % S.p_vocab);
        for (int g = 0; g < n_predictor_steps; ++g) {
            std::memcpy(p_input_emb.data(),
                        S.p_embd_w.data() + (size_t)p_cur * S.p_n_embd,
                        S.p_n_embd * sizeof(float));
            S.predictor.forward_decode_with_logits(p_input_emb.data(), g, p_logits.data());
            int lo = g * 2048;
            int hi = lo + 2048;
            std::vector<int> recent_g;
            int from = std::max(0, t - samp.recent_window);
            for (int tt = from; tt < t; ++tt) recent_g.push_back(codes[g + 1][tt]);
            apply_repetition_penalty(p_logits.data(), S.p_vocab, recent_g, lo, hi,
                                      samp.repetition_penalty);
            int nxt = sample_token(p_logits.data(), lo, hi, samp, rng);
            codes[g + 1][t] = nxt - lo;
            p_cur = nxt;
        }
        if (n_predictor_steps < 15) {
            for (int g = n_predictor_steps; g < 15; ++g) codes[g + 1][t] = 0;
        }
    }

    // Vocoder.
    std::vector<int> flat(16 * T);
    for (int q = 0; q < 16; ++q)
        for (int t = 0; t < T; ++t) flat[q * T + t] = codes[q][t];
    auto audio = S.decoder.decode_audio(flat.data(), 16, T);

    // Save WAV.
    if (!audio_io::save_wav(out_wav.c_str(), audio, 24000, 1)) {
        R.err = "save_wav failed";
        return R;
    }

    auto wall_t1 = std::chrono::high_resolution_clock::now();
    R.wall_ms   = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    R.audio_sec = audio.size() / 24000.0;
    R.rtf       = R.wall_ms / 1000.0 / std::max(R.audio_sec, 1e-9);
    R.ok        = true;
    fprintf(stderr,
            "[tts_server] req text='%s' -> %s | %.2f ms (audio %.2fs, RTF %.3f, T=%d codec frames)\n",
            text.c_str(), out_wav.c_str(), R.wall_ms, R.audio_sec, R.rtf, T);
    return R;
}

// ============================================================================
// TCP server loop.
// ============================================================================
volatile std::sig_atomic_t g_should_exit = 0;

void on_signal(int sig) {
    (void)sig;
    g_should_exit = 1;
}

bool read_line(int fd, std::string &out, size_t max_bytes = 8192) {
    out.clear();
    char buf[256];
    while (out.size() < max_bytes) {
        ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0) return false;
        for (ssize_t i = 0; i < n; ++i) {
            if (buf[i] == '\n') {
                out.append(buf, i);
                return true;
            }
        }
        out.append(buf, n);
    }
    return false;
}

bool send_all(int fd, const std::string &s) {
    const char *p = s.data();
    size_t left = s.size();
    while (left) {
        ssize_t n = ::send(fd, p, left, 0);
        if (n <= 0) return false;
        p += n; left -= (size_t)n;
    }
    return true;
}

int run_server(ServerCtx &S, int port) {
    int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0) {
        fprintf(stderr, "[tts_server] socket() FAIL: %s\n", strerror(errno));
        return 1;
    }
    int yes = 1;
    ::setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   // 127.0.0.1 only
    addr.sin_port        = htons((uint16_t)port);
    if (::bind(srv, (sockaddr *)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[tts_server] bind(127.0.0.1:%d) FAIL: %s\n",
                port, strerror(errno));
        ::close(srv);
        return 1;
    }
    if (::listen(srv, 4) < 0) {
        fprintf(stderr, "[tts_server] listen() FAIL: %s\n", strerror(errno));
        ::close(srv);
        return 1;
    }

    fprintf(stderr, "[tts_server] LISTENING on 127.0.0.1:%d (warm)\n", port);
    fflush(stderr);

    while (!g_should_exit) {
        sockaddr_in cli; socklen_t cl = sizeof(cli);
        int c = ::accept(srv, (sockaddr *)&cli, &cl);
        if (c < 0) {
            if (errno == EINTR) continue;
            fprintf(stderr, "[tts_server] accept FAIL: %s\n", strerror(errno));
            break;
        }

        std::string line;
        if (!read_line(c, line)) {
            ::close(c);
            continue;
        }
        // Parse '<text>\t<out_path>'.
        auto tab = line.find('\t');
        if (tab == std::string::npos) {
            send_all(c, "ERR\tmalformed request (need text\\tout_path)\n");
            ::close(c);
            continue;
        }
        std::string text    = line.substr(0, tab);
        std::string outpath = line.substr(tab + 1);
        if (text.empty() || outpath.empty()) {
            send_all(c, "ERR\tempty text or out_path\n");
            ::close(c);
            continue;
        }

        SynthResult R = synthesize_one(S, text, outpath);
        char reply[512];
        if (R.ok) {
            std::snprintf(reply, sizeof(reply), "OK\t%.3f\t%.4f\t%.4f\n",
                          R.wall_ms, R.rtf, R.audio_sec);
        } else {
            std::snprintf(reply, sizeof(reply), "ERR\t%s\n", R.err.c_str());
        }
        send_all(c, reply);
        ::close(c);
    }
    ::close(srv);
    fprintf(stderr, "[tts_server] shutdown\n");
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    std::signal(SIGINT,  on_signal);
    std::signal(SIGTERM, on_signal);
    std::signal(SIGPIPE, SIG_IGN);

    int port = (int)env_u64("OMINIX_TTS_SERVER_PORT", 7777);
    if (argc >= 2) {
        if (std::string(argv[1]) == "--port" && argc >= 3) {
            port = std::atoi(argv[2]);
        }
    }

    ServerCtx S;
    if (!init_server_ctx(S)) {
        fprintf(stderr, "[tts_server] init FAIL\n");
        return 1;
    }
    return run_server(S, port);
}
