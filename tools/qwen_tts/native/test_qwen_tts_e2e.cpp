// ============================================================================
// Phase 2.7e E2E TTS smoke harness — TEXT-CONDITIONED text -> codec -> audio.
//
// Phase 2.7d shipped a structurally-GREEN pipeline that fed a deterministic
// hash-derived seed token into the Talker — audio was non-silent but
// un-conditioned on the prompt. Phase 2.7e wires in qwen3_assets.gguf
// (text_embd + codec_embd.0..15 + small_to_mtp proj) so the Talker is now
// actually conditioned on the user's BPE-tokenized prompt before starting
// codec-token autoregression.
//
// qwen3_assets.gguf inventory (cgisky export):
//   text_embd                  ne=[151936, 2048]   (vocab inner-stride)
//   codec_embd.0..15           ne=[3072 or 2048, 2048]
//   proj.weight + proj.bias    ne=[1024, 2048] / [1024]   (predictor->talker)
// (Note: ne convention here is opposite from talker.gguf token_embd —
//  text_embd is stored with vocab as the *contiguous* dimension. We dequant
//  via ggml's to_float trait then transpose into [vocab][hidden] so token
//  rows are contiguous.)
//
// Pipeline:
//   text "Hello world..."
//     -> BPE tokenize (Qwen2 vocab.json + merges.txt; 151643 + a few special)
//     -> Build prefill embedding sequence:
//          [im_start=151644, assistant=77091, \n=198,
//           tts_pad×K, tts_bos=151672,
//           target_text_token_0 ... target_text_token_N-1,
//           tts_eos=151673,
//           codec_bos=2149]
//        Looked up via text_embd for text-vocab IDs and codec_embd.0 for
//        codec_bos.
//     -> Run forward_decode for each prefill position to populate KV cache.
//     -> Switch to codec autoreg: greedy argmax over LM head, feed codec_embd.0
//        of next token. Stop on codec_eos=2150 or step budget.
//     -> Predictor + SpeechTokenizerDecoder unchanged (Phase 2.7c+d).
//
// GREEN gate v0 (Phase 2.7e):
//   - No NaN/Inf
//   - Audio shape consistent (n = T*1920 samples)
//   - Within [-1, 1]
//   - std > 0.01
//   - Two different prompts -> different audio envelopes (run with second
//     prompt at end and report std + length delta).
//
// Usage:
//   test_qwen_tts_e2e <text> [n_talker_steps=128] [n_predictor_steps=15]
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
#include <algorithm>
#include <random>
#include <string>
#include <vector>

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
constexpr const char *OUT_WAV = "/tmp/qwen_tts_e2e.wav";

// Special token IDs (from Ascend qwen_tts/talker.h) — text-vocab.
constexpr int IM_START      = 151644;
constexpr int ASSISTANT     = 77091;
constexpr int NEWLINE       = 198;
constexpr int TTS_PAD       = 151671;
constexpr int TTS_BOS       = 151672;
constexpr int TTS_EOS       = 151673;
constexpr int CODEC_PAD     = 2148;
constexpr int CODEC_BOS     = 2149;
constexpr int CODEC_EOS     = 2150;

// Load a GGUF tensor (any quant) into f32 host vector.
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

// Load a 2D GGUF tensor that ggml stores with ne=[vocab_inner, hidden_outer]
// (the cgisky qwen3_assets layout — opposite of normal token_embd) and
// transpose to row-major [vocab][hidden] so token-id rows are contiguous.
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
    // raw layout: flat[h * vocab + v] (ne[0]=vocab inner, ne[1]=hidden outer)
    // Transpose to flat[v * hidden + h] (vocab outer, hidden inner).
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

// ============================================================================
// Phase 2.8 — sampling (temperature + top-k + top-p + repetition penalty).
// Mirrors Ascend tools/qwen_tts/talker.cpp::sample_token defaults
// (temp=0.9, top_k=50, top_p=1.0, rep_penalty=1.05).
// ============================================================================
struct SamplingConfig {
    float temperature        = 0.9f;
    int   top_k              = 50;
    float top_p              = 1.0f;
    float repetition_penalty = 1.05f;
    int   recent_window      = 64;
    bool  do_sample          = true;
    uint64_t seed            = 42;
};

// Read env var with fallback.
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

// Apply repetition penalty to a recent window of tokens (Ascend style: divide
// positive logits by penalty, multiply negative ones).
void apply_repetition_penalty(float *logits, int vocab_size,
                              const std::vector<int> &recent, int lo, int hi,
                              float penalty) {
    if (penalty == 1.0f) return;
    for (int tok : recent) {
        int t = tok + lo;  // map sub-vocab index to logit slot for predictor
        if (t < lo || t >= hi) continue;
        if (logits[t] > 0.0f) logits[t] /= penalty;
        else                  logits[t] *= penalty;
    }
}

// Sample one token from logits[lo..hi) with temperature/top-k/top-p.
// Returns absolute index in [lo, hi). Greedy if !do_sample or temperature<=0.
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

    // Softmax (subtract max for stability).
    float mx = cands.front().first;
    float sum = 0.0f;
    for (auto &c : cands) { c.first = std::exp(c.first - mx); sum += c.first; }
    if (sum <= 0.0f) return cands.front().second;
    for (auto &c : cands) c.first /= sum;

    // Top-p (nucleus) cutoff.
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

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr,
                "usage: test_qwen_tts_e2e <text> [n_talker_steps=128] [n_predictor_steps=15]\n");
        return 2;
    }
    std::string text = argv[1];
    int n_talker_steps = (argc >= 3) ? std::atoi(argv[2]) : 128;
    int n_predictor_steps = (argc >= 4) ? std::atoi(argv[3]) : 15;
    if (n_talker_steps <= 0) n_talker_steps = 128;
    if (n_predictor_steps <= 0 || n_predictor_steps > 15) n_predictor_steps = 15;

    printf("[e2e] text='%s'\n", text.c_str());
    printf("[e2e] n_talker_steps=%d  n_predictor_steps=%d\n",
           n_talker_steps, n_predictor_steps);

    // Phase 2.8 sampling config (env-overridable).
    SamplingConfig samp = load_sampling_config_from_env();
    printf("[e2e] sampling: do_sample=%d temp=%.2f top_k=%d top_p=%.2f "
           "rep_penalty=%.2f recent_window=%d seed=%llu\n",
           (int)samp.do_sample, samp.temperature, samp.top_k, samp.top_p,
           samp.repetition_penalty, samp.recent_window,
           (unsigned long long)samp.seed);
    std::mt19937 rng((uint32_t)samp.seed);

    auto wall_t0 = std::chrono::high_resolution_clock::now();

    // ---- Stage A: BPE tokenize ---------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    BpeTokenizer tok;
    if (!tok.load(VOCAB_JSON, MERGES_TXT)) {
        fprintf(stderr, "[e2e] BPE load FAIL\n");
        return 1;
    }
    auto text_tokens = tok.encode(text);
    if (text_tokens.empty()) {
        fprintf(stderr, "[e2e] tokenize empty\n");
        return 1;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_bpe = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[e2e] BPE: %zu tokens in %.2f ms; first 10:", text_tokens.size(), ms_bpe);
    for (size_t i = 0; i < std::min<size_t>(10, text_tokens.size()); ++i) printf(" %d", text_tokens[i]);
    printf("\n");

    // ---- Stage B: Talker engine init ---------------------------------------
    t0 = std::chrono::high_resolution_clock::now();
    TalkerConfig talker_cfg;
    ominix_cuda::TalkerCudaEngine talker;
    if (!talker.init_from_gguf(TALKER_GGUF, talker_cfg, /*device=*/0)) {
        fprintf(stderr, "[e2e] talker init FAIL\n");
        return 1;
    }
    talker.reset_kv_cache();
    const int t_n_embd = talker_cfg.hidden_size;

    // Load Talker GGUF for token_embd and lm_head (codec vocab path).
    ggml_context *t_ctx = nullptr;
    gguf_init_params gp; gp.no_alloc = false; gp.ctx = &t_ctx;
    gguf_context *t_gguf = gguf_init_from_file(TALKER_GGUF, gp);
    if (!t_gguf || !t_ctx) { fprintf(stderr, "[e2e] talker gguf open FAIL\n"); return 1; }
    ggml_tensor *t_embd_t = ggml_get_tensor(t_ctx, "token_embd.weight");
    const int t_vocab = (int)t_embd_t->ne[1];
    std::vector<float> t_embd_w, t_lm_head_w;
    if (!load_tensor_f32(t_ctx, "token_embd.weight", (size_t)t_vocab * t_n_embd, t_embd_w)) return 1;
    if (!load_tensor_f32(t_ctx, "output.weight", (size_t)t_vocab * t_n_embd, t_lm_head_w)) return 1;
    auto t2 = std::chrono::high_resolution_clock::now();
    double ms_talker_init = std::chrono::duration<double, std::milli>(t2 - t0).count();
    printf("[e2e] talker init+weights: %.2f ms (codec_vocab=%d, n_embd=%d)\n",
           ms_talker_init, t_vocab, t_n_embd);

    // ---- Stage B2: Load qwen3_assets.gguf (text_embd + codec_embd.0) -------
    t0 = std::chrono::high_resolution_clock::now();
    ggml_context *a_ctx = nullptr;
    gguf_init_params gp_a; gp_a.no_alloc = false; gp_a.ctx = &a_ctx;
    gguf_context *a_gguf = gguf_init_from_file(ASSETS_GGUF, gp_a);
    if (!a_gguf || !a_ctx) { fprintf(stderr, "[e2e] assets gguf open FAIL\n"); return 1; }

    constexpr int TEXT_VOCAB = 151936;
    std::vector<float> text_embd_vh;     // [TEXT_VOCAB, t_n_embd] row-major
    std::vector<float> codec_embd0_vh;   // [t_vocab, t_n_embd] row-major
    if (!load_assets_embedding(a_ctx, "text_embd",  TEXT_VOCAB, t_n_embd, text_embd_vh)) return 1;
    if (!load_assets_embedding(a_ctx, "codec_embd.0", t_vocab, t_n_embd, codec_embd0_vh)) return 1;
    auto t2b = std::chrono::high_resolution_clock::now();
    double ms_assets = std::chrono::duration<double, std::milli>(t2b - t0).count();
    printf("[e2e] assets loaded: %.2f ms (text_embd=%dx%d, codec_embd.0=%dx%d)\n",
           ms_assets, TEXT_VOCAB, t_n_embd, t_vocab, t_n_embd);

    auto lookup_text = [&](int tok_id, float *out) {
        if (tok_id < 0 || tok_id >= TEXT_VOCAB) {
            std::memset(out, 0, t_n_embd * sizeof(float));
            return;
        }
        std::memcpy(out, text_embd_vh.data() + (size_t)tok_id * t_n_embd,
                    t_n_embd * sizeof(float));
    };
    auto lookup_codec0 = [&](int tok_id, float *out) {
        if (tok_id < 0 || tok_id >= t_vocab) {
            std::memset(out, 0, t_n_embd * sizeof(float));
            return;
        }
        std::memcpy(out, codec_embd0_vh.data() + (size_t)tok_id * t_n_embd,
                    t_n_embd * sizeof(float));
    };

    // ---- Stage C: Text-conditioned prefill + Talker autoreg ----------------
    // Build prefill embedding sequence (Pattern B simplified — direct text
    // embeddings, no separate text_projection MLP since cgisky export has
    // text_hidden == talker_hidden == 2048).
    //
    //   [im_start, assistant, \n,
    //    tts_pad×3, tts_bos,
    //    text_token_0 ... text_token_N-1,
    //    tts_eos,
    //    codec_bos]   ← last embedding is codec-domain
    //
    // After this prefill the Talker's KV cache is text-conditioned; we
    // then take the hidden at the codec_bos position and do greedy
    // argmax over codec vocab to pick the first real codec token.
    std::vector<int>   prefill_kinds;  // 0=text, 1=codec0
    std::vector<int>   prefill_ids;
    auto add_text  = [&](int id) { prefill_kinds.push_back(0); prefill_ids.push_back(id); };
    auto add_codec = [&](int id) { prefill_kinds.push_back(1); prefill_ids.push_back(id); };

    add_text(IM_START); add_text(ASSISTANT); add_text(NEWLINE);
    for (int i = 0; i < 3; ++i) add_text(TTS_PAD);
    add_text(TTS_BOS);
    for (int t : text_tokens) add_text(t);
    add_text(TTS_EOS);
    add_codec(CODEC_BOS);

    int prefill_len = (int)prefill_kinds.size();
    printf("[e2e] prefill_len=%d (role=3 + pads=3 + tts_bos=1 + text=%zu + tts_eos=1 + codec_bos=1)\n",
           prefill_len, text_tokens.size());

    std::vector<float> input_emb(t_n_embd), hidden(t_n_embd), logits(t_vocab);

    t0 = std::chrono::high_resolution_clock::now();
    for (int pos = 0; pos < prefill_len; ++pos) {
        if (prefill_kinds[pos] == 0) lookup_text(prefill_ids[pos], input_emb.data());
        else                          lookup_codec0(prefill_ids[pos], input_emb.data());
        talker.forward_decode(input_emb.data(), pos, hidden.data());
        for (int i = 0; i < t_n_embd; ++i) {
            if (std::isnan(hidden[i]) || std::isinf(hidden[i])) {
                fprintf(stderr, "[e2e] talker prefill NaN/Inf at pos=%d kind=%d id=%d\n",
                        pos, prefill_kinds[pos], prefill_ids[pos]);
                return 1;
            }
        }
    }
    auto t2c = std::chrono::high_resolution_clock::now();
    double ms_prefill = std::chrono::duration<double, std::milli>(t2c - t0).count();
    printf("[e2e] prefill: %.2f ms (%.2f TPS)\n",
           ms_prefill, prefill_len * 1000.0 / ms_prefill);

    // First codec token: sample over codec vocab from the codec_bos hidden.
    matvec_f32(t_lm_head_w.data(), hidden.data(), logits.data(), t_vocab, t_n_embd);
    // Repetition penalty has no history yet for the first token.
    int first_tok = sample_token(logits.data(), 3, t_vocab, samp, rng);

    std::vector<int> semantic_tokens;
    semantic_tokens.reserve(n_talker_steps);
    semantic_tokens.push_back(first_tok);

    // Continue autoreg with sampling + repetition penalty over recent window.
    int pos_cursor = prefill_len;
    int cur_tok = first_tok;
    bool hit_eos = false;
    t0 = std::chrono::high_resolution_clock::now();
    for (int step = 1; step < n_talker_steps; ++step) {
        lookup_codec0(cur_tok, input_emb.data());
        talker.forward_decode(input_emb.data(), pos_cursor++, hidden.data());
        for (int i = 0; i < t_n_embd; ++i) {
            if (std::isnan(hidden[i]) || std::isinf(hidden[i])) {
                fprintf(stderr, "[e2e] talker autoreg NaN/Inf at step=%d\n", step);
                return 1;
            }
        }
        matvec_f32(t_lm_head_w.data(), hidden.data(), logits.data(), t_vocab, t_n_embd);
        // Build recent-window absolute-codec indices for rep penalty.
        std::vector<int> recent_abs;
        int from = std::max(0, (int)semantic_tokens.size() - samp.recent_window);
        for (int i = from; i < (int)semantic_tokens.size(); ++i)
            recent_abs.push_back(semantic_tokens[i]);
        // apply_repetition_penalty here uses lo=0 so token id == logit slot.
        apply_repetition_penalty(logits.data(), t_vocab, recent_abs,
                                 0, t_vocab, samp.repetition_penalty);
        int nxt = sample_token(logits.data(), 3, t_vocab, samp, rng);
        semantic_tokens.push_back(nxt);
        cur_tok = nxt;
        if (nxt == CODEC_EOS) { hit_eos = true; break; }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ms_talker_run = std::chrono::duration<double, std::milli>(t3 - t0).count();
    int produced = (int)semantic_tokens.size();
    printf("[e2e] talker autoreg %d steps: %.2f ms (%.2f TPS) eos=%d\n",
           produced, ms_talker_run, produced * 1000.0 / ms_talker_run, (int)hit_eos);
    printf("[e2e] semantic[0..7]:");
    for (int i = 0; i < 8 && i < produced; ++i) printf(" %d", semantic_tokens[i]);
    printf("\n");

    // Drop the trailing eos token if present (don't feed to vocoder).
    if (hit_eos && !semantic_tokens.empty() && semantic_tokens.back() == CODEC_EOS)
        semantic_tokens.pop_back();
    int T = (int)semantic_tokens.size();
    if (T <= 0) {
        fprintf(stderr, "[e2e] no codec tokens produced — abort\n");
        return 1;
    }

    // ---- Stage D: Predictor (acoustic codebooks 1..15) ---------------------
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
        fprintf(stderr, "[e2e] predictor init FAIL\n");
        return 1;
    }
    const int p_n_embd = cp_cfg.hidden_size;
    ggml_context *p_ctx = nullptr;
    gguf_init_params gp2; gp2.no_alloc = false; gp2.ctx = &p_ctx;
    gguf_context *p_gguf = gguf_init_from_file(PREDICTOR_GGUF, gp2);
    if (!p_gguf || !p_ctx) { fprintf(stderr, "[e2e] predictor gguf FAIL\n"); return 1; }
    ggml_tensor *p_embd_t = ggml_get_tensor(p_ctx, "token_embd.weight");
    const int p_vocab = (int)p_embd_t->ne[1];
    std::vector<float> p_embd_w, p_lm_head_w;
    if (!load_tensor_f32(p_ctx, "token_embd.weight", (size_t)p_vocab * p_n_embd, p_embd_w)) return 1;
    if (!load_tensor_f32(p_ctx, "output.weight", (size_t)p_vocab * p_n_embd, p_lm_head_w)) return 1;
    auto t4 = std::chrono::high_resolution_clock::now();
    double ms_pred_init = std::chrono::duration<double, std::milli>(t4 - t0).count();
    printf("[e2e] predictor init+weights: %.2f ms (vocab=%d, n_embd=%d)\n",
           ms_pred_init, p_vocab, p_n_embd);

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
            int lo = g * 2048;
            int hi = lo + 2048;
            // Repetition penalty within this codebook over recent frames.
            std::vector<int> recent_g;
            int from = std::max(0, t - samp.recent_window);
            for (int tt = from; tt < t; ++tt) recent_g.push_back(codes[g + 1][tt]);
            apply_repetition_penalty(p_logits.data(), p_vocab, recent_g,
                                     lo, hi, samp.repetition_penalty);
            int nxt = sample_token(p_logits.data(), lo, hi, samp, rng);
            int acoustic_tok = nxt - lo;
            codes[g + 1][t] = acoustic_tok;
            p_cur = nxt;
        }
        if (n_predictor_steps < 15) {
            for (int g = n_predictor_steps; g < 15; ++g) codes[g + 1][t] = 0;
        }
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double ms_pred_run = std::chrono::duration<double, std::milli>(t5 - t0).count();
    int total_pred_steps = T * n_predictor_steps;
    printf("[e2e] predictor sequential %d frames x %d groups = %d steps: %.2f ms (%.2f TPS)\n",
           T, n_predictor_steps, total_pred_steps,
           ms_pred_run, total_pred_steps * 1000.0 / ms_pred_run);

    // ---- Stage E: Vocoder ---------------------------------------------------
    t0 = std::chrono::high_resolution_clock::now();
    ominix_cuda::SpeechTokenizerDecoderCudaEngine dec;
    if (!dec.init_from_gguf(DECODER_GGUF, /*device=*/0)) {
        fprintf(stderr, "[e2e] decoder init FAIL\n");
        return 1;
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    double ms_dec_init = std::chrono::duration<double, std::milli>(t6 - t0).count();

    std::vector<int> flat(16 * T);
    for (int q = 0; q < 16; ++q)
        for (int t = 0; t < T; ++t) flat[q * T + t] = codes[q][t];

    auto audio = dec.decode_audio(flat.data(), 16, T);
    auto t7 = std::chrono::high_resolution_clock::now();
    double ms_voc_run = std::chrono::duration<double, std::milli>(t7 - t6).count();
    printf("[e2e] decoder init: %.2f ms; decode_audio: %.2f ms; samples=%zu (%.2f sec @ 24kHz)\n",
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
    printf("[e2e] audio stats: nan=%d inf=%d out_of_range=%d  min=%.4f max=%.4f mean=%.4f std=%.4f\n",
           n_nan, n_inf, n_oor, vmin, vmax, mean, std_);

    t0 = std::chrono::high_resolution_clock::now();
    if (!audio_io::save_wav(OUT_WAV, audio, 24000, 1)) {
        fprintf(stderr, "[e2e] save_wav FAIL\n");
        return 1;
    }
    auto t8 = std::chrono::high_resolution_clock::now();
    double ms_wav = std::chrono::duration<double, std::milli>(t8 - t0).count();
    printf("[e2e] saved WAV: %s (%.2f ms)\n", OUT_WAV, ms_wav);

    auto wall_t1 = std::chrono::high_resolution_clock::now();
    double ms_wall_total = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
    double audio_sec = audio.size() / 24000.0;
    double rtf = ms_wall_total / 1000.0 / std::max(audio_sec, 1e-9);
    printf("[e2e] WALL BREAKDOWN:\n");
    printf("        bpe       %8.2f ms\n", ms_bpe);
    printf("        talker    %8.2f ms (init %.0f + assets %.0f + prefill %.0f + run %.0f)\n",
           ms_talker_init + ms_assets + ms_prefill + ms_talker_run,
           ms_talker_init, ms_assets, ms_prefill, ms_talker_run);
    printf("        predictor %8.2f ms (init %.0f + run %.0f)\n",
           ms_pred_init + ms_pred_run, ms_pred_init, ms_pred_run);
    printf("        vocoder   %8.2f ms (init %.0f + run %.0f)\n",
           ms_dec_init + ms_voc_run, ms_dec_init, ms_voc_run);
    printf("        wav_save  %8.2f ms\n", ms_wav);
    printf("        TOTAL     %8.2f ms\n", ms_wall_total);
    printf("[e2e] audio: %.2f sec  RTF=%.3f (lower=better; <1 = real-time)\n",
           audio_sec, rtf);

    bool green = (n_nan == 0 && n_inf == 0 && n_oor == 0 &&
                  std_ > 0.01 &&
                  audio.size() == (size_t)T * 1920);
    printf("[e2e] verdict: %s  (NaN-free=%d, in-range=%d, std>0.01=%d, shape-ok=%d)\n",
           green ? "GREEN" : "YELLOW",
           (n_nan == 0 && n_inf == 0), (n_oor == 0),
           (std_ > 0.01), (audio.size() == (size_t)T * 1920));

    gguf_free(t_gguf); ggml_free(t_ctx);
    gguf_free(p_gguf); ggml_free(p_ctx);
    gguf_free(a_gguf); ggml_free(a_ctx);
    return green ? 0 : 1;
}
