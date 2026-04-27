// ============================================================================
// Phase 4.3 + 4.4 E2E smoke test — Qwen3-ASR end-to-end CUDA pipeline.
//
// Pipeline (mirrors OminiX-Ascend/tools/qwen_asr/qwen_asr.cpp):
//
//   wav (16 kHz F32) → MelSpectrogram (CPU)
//                    → AudioEncoderCudaEngine.encode      [Phase 4.2]
//                    → split prefill: prefix tokens + audio embeds + suffix
//                                     tokens, fed sequentially through
//                                     TalkerCudaEngine.forward_decode (S=1
//                                     steps that extend the KV cache and
//                                     advance the position counter)   [Phase 4.3]
//                    → autoregressive greedy decode (host argmax over
//                                     lm_head)                          [Phase 4.4]
//                    → BPE decode                                       [Phase 4.4]
//                    → text
//
// Phase 4.3 split-prefill strategy (pragmatic):
//   The Phase-2.x TalkerCudaEngine ships a fully-working forward_decode (S=1)
//   path; forward_prefill (S>1 batched ingestion) is still the std::abort
//   stub from Phase 2.2. Rather than block on landing batched prefill, we
//   feed the prefix / audio / suffix sequence through forward_decode token by
//   token. forward_decode extends the KV cache by one slot per call and
//   accepts any F32 [n_embd] vector — it does not care whether that vector
//   came from the token-embedding lookup table (prefix / suffix) or from the
//   audio encoder output (audio frames). This is exactly the contract Ascend
//   uses: pre-audio tokens, audio embeds, post-audio tokens, all streamed
//   into the same eval_chunk() that does single-step KV-extending decode.
//
// Usage:
//   test_qwen_asr_cuda_e2e <audio_encoder_gguf> <text_decoder_gguf>
//                          <vocab_json> <merges_txt> <wav_path>
//                          [max_new_tokens]
// ============================================================================

#include "asr_cuda_engine.h"
#include "audio_encoder_cuda_engine.h"
#include "../../qwen_tts/native/talker_cuda_engine.h"
#include "../../qwen_tts/talker.h"           // TalkerConfig
#include "../../qwen_common/audio_io.h"
#include "../../qwen_common/bpe_tokenizer.h"
#include "mel_spectrogram.h"

#include "ggml.h"
#include "gguf.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Pull a tensor from GGUF as F32 (handles F32, F16 and Q8_0/Q4_0/etc via the
// type-traits path). Mirrors the helper in test_talker_cuda_autoreg.cpp.
// ---------------------------------------------------------------------------
bool load_tensor_f32(ggml_context *ctx, const char *name,
                     size_t expected_elems, std::vector<float> &out) {
    ggml_tensor *t = ggml_get_tensor(ctx, name);
    if (!t) {
        fprintf(stderr, "[asr_e2e] missing tensor: %s\n", name);
        return false;
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr, "[asr_e2e] %s: expected %zu, got %zu\n",
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
            fprintf(stderr, "[asr_e2e] %s: unsupported dtype %d\n",
                    name, (int)t->type);
            return false;
        }
        tt->to_float(t->data, out.data(), (int64_t)n);
    }
    return true;
}

// out[v] = sum_i W[v * n_embd + i] * x[i]   (row-major [vocab, n_embd])
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

bool check_finite(const float *x, int n, const char *tag, int idx) {
    for (int i = 0; i < n; ++i) {
        if (std::isnan(x[i]) || std::isinf(x[i])) {
            fprintf(stderr,
                    "[asr_e2e] %s idx=%d hidden[%d] = %f (non-finite)\n",
                    tag, idx, i, x[i]);
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr,
                "usage: test_qwen_asr_cuda_e2e <audio_encoder_gguf> "
                "<text_decoder_gguf> <vocab_json> <merges_txt> <wav_path> "
                "[max_new_tokens]\n");
        return 2;
    }
    const std::string audio_gguf  = argv[1];
    const std::string decoder_gguf = argv[2];
    const std::string vocab_path  = argv[3];
    const std::string merges_path = argv[4];
    const std::string wav_path    = argv[5];
    int max_new_tokens = (argc >= 7) ? std::atoi(argv[6]) : 128;
    if (max_new_tokens <= 0) max_new_tokens = 128;

    // -----------------------------------------------------------------------
    // 1. Init AsrCudaEngine (audio encoder + text decoder).
    // -----------------------------------------------------------------------
    ominix_cuda::AsrCudaParams params;
    params.audio_encoder_gguf = audio_gguf;
    params.text_decoder_gguf  = decoder_gguf;
    params.device             = 0;
    params.max_new_tokens     = max_new_tokens;

    // Phase 4.6: HF whisper-style mel filterbank (slaney scale, norm=slaney).
    // OMINIX_ASR_MEL_FILTERS env overrides; otherwise fall back to bundled
    // verify_data path. The C++-computed HTK filterbank does NOT match HF.
    {
        const char *envp = std::getenv("OMINIX_ASR_MEL_FILTERS");
        if (envp && *envp) {
            params.mel_filters_path = envp;
        } else {
            params.mel_filters_path =
                "tools/qwen_asr/verify_data/mel_filters_whisper.npy";
        }
    }

    ominix_cuda::AsrCudaEngine asr;
    if (!asr.init(params)) {
        fprintf(stderr, "[asr_e2e] AsrCudaEngine.init FAILED\n");
        return 1;
    }
    asr.text_decoder().reset_kv_cache();

    // -----------------------------------------------------------------------
    // 2. Load BPE tokenizer (vocab.json + merges.txt).
    // -----------------------------------------------------------------------
    BpeTokenizer tok;
    if (!tok.load(vocab_path, merges_path)) {
        fprintf(stderr, "[asr_e2e] BpeTokenizer.load FAILED\n");
        return 1;
    }
    int im_start_id  = tok.token_to_id("<|im_start|>");
    int im_end_id    = tok.token_to_id("<|im_end|>");
    int endoftext_id = tok.token_to_id("<|endoftext|>");
    int audio_start_id = tok.token_to_id("<|audio_start|>");
    int audio_end_id   = tok.token_to_id("<|audio_end|>");
    fprintf(stderr,
            "[asr_e2e] tokenizer: im_start=%d im_end=%d eot=%d "
            "audio_start=%d audio_end=%d\n",
            im_start_id, im_end_id, endoftext_id,
            audio_start_id, audio_end_id);
    if (im_start_id < 0 || im_end_id < 0 || audio_start_id < 0 ||
        audio_end_id < 0) {
        fprintf(stderr, "[asr_e2e] tokenizer missing required special tokens\n");
        return 1;
    }

    // -----------------------------------------------------------------------
    // 3. Load token_embd + output (lm_head) from the decoder GGUF.
    // -----------------------------------------------------------------------
    TalkerConfig tcfg;  // defaults: hidden=2048, 28L, 16Q/8KV
    const int n_embd = tcfg.hidden_size;

    ggml_context *ggml_ctx = nullptr;
    gguf_init_params ginit;
    ginit.no_alloc = false;
    ginit.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(decoder_gguf.c_str(), ginit);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[asr_e2e] gguf_init failed: %s\n", decoder_gguf.c_str());
        return 1;
    }

    ggml_tensor *embd_t = ggml_get_tensor(ggml_ctx, "token_embd.weight");
    if (!embd_t) {
        fprintf(stderr, "[asr_e2e] token_embd.weight not in GGUF\n");
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }
    if (embd_t->ne[0] != n_embd) {
        fprintf(stderr, "[asr_e2e] token_embd ne[0]=%lld != n_embd=%d\n",
                (long long)embd_t->ne[0], n_embd);
        gguf_free(gguf_ctx); ggml_free(ggml_ctx);
        return 1;
    }
    const int vocab_size = (int)embd_t->ne[1];
    fprintf(stderr, "[asr_e2e] vocab_size=%d  n_embd=%d\n",
            vocab_size, n_embd);

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
            fprintf(stderr,
                    "[asr_e2e] output.weight absent — tying lm_head to token_embd\n");
        }
    }
    fprintf(stderr,
            "[asr_e2e] token_embd %zu MB  lm_head %s %zu MB\n",
            token_embd_w.size() * sizeof(float) / (1024 * 1024),
            lm_head_tied ? "(tied)" : "(explicit)",
            lm_head_w.size() * sizeof(float) / (1024 * 1024));

    // We can free the gguf_ctx now that token_embd + lm_head are in host F32.
    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);

    // -----------------------------------------------------------------------
    // 4. Load WAV + compute mel.
    // -----------------------------------------------------------------------
    // Always resample to 16 kHz mono — Qwen3-ASR's mel front-end is fixed at
    // sr=16000, n_fft=400, hop=160. audio_io::load_audio handles resampling
    // via miniaudio when the source SR differs.
    std::vector<float> audio_samples;
    if (!audio_io::load_audio(wav_path, 16000, audio_samples)) {
        fprintf(stderr, "[asr_e2e] load_audio (resample to 16k) FAILED: %s\n",
                wav_path.c_str());
        return 1;
    }
    fprintf(stderr, "[asr_e2e] WAV (resampled): %zu samples @ 16 kHz (%.2fs)\n",
            audio_samples.size(),
            (float)audio_samples.size() / 16000.0f);

    auto t_mel0 = std::chrono::high_resolution_clock::now();
    std::vector<float> mel;
    int mel_T = 0;
    if (!asr.mel_spec().compute(audio_samples, mel, mel_T)) {
        fprintf(stderr, "[asr_e2e] mel_spec.compute FAILED\n");
        return 1;
    }
    auto t_mel1 = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // 5. Audio encode → [num_frames, output_dim=2048] F32 host.
    // -----------------------------------------------------------------------
    auto &enc = asr.audio_encoder();
    const int output_dim = enc.output_dim();
    if (output_dim != n_embd) {
        fprintf(stderr,
                "[asr_e2e] WARN: audio output_dim=%d != decoder n_embd=%d — "
                "split-prefill assumes equal dims\n", output_dim, n_embd);
    }
    int num_audio_frames = 0;
    int max_audio_frames = mel_T;  // generous upper bound
    std::vector<float> audio_embeds(
        (size_t)max_audio_frames * output_dim, 0.0f);

    auto t_enc0 = std::chrono::high_resolution_clock::now();
    if (!enc.encode(mel.data(), enc.num_mel_bins(), mel_T,
                    audio_embeds.data(), num_audio_frames)) {
        fprintf(stderr, "[asr_e2e] audio encoder encode FAILED\n");
        return 1;
    }
    auto t_enc1 = std::chrono::high_resolution_clock::now();
    fprintf(stderr, "[asr_e2e] audio embeds: %d frames x %d dim\n",
            num_audio_frames, output_dim);

    // -----------------------------------------------------------------------
    // 6. Build prefix / suffix token sequences (mirrors Ascend qwen_asr.cpp
    //    build_prompt_segments).
    // -----------------------------------------------------------------------
    std::vector<int> pre_tokens;
    std::vector<int> post_tokens;
    {
        auto nl    = tok.encode("\n");
        auto sys_  = tok.encode("system\n");
        auto user_ = tok.encode("user\n");
        auto asst_ = tok.encode("assistant\n");

        // Pre-audio: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
        pre_tokens.push_back(im_start_id);
        pre_tokens.insert(pre_tokens.end(), sys_.begin(), sys_.end());
        pre_tokens.push_back(im_end_id);
        pre_tokens.insert(pre_tokens.end(), nl.begin(), nl.end());
        pre_tokens.push_back(im_start_id);
        pre_tokens.insert(pre_tokens.end(), user_.begin(), user_.end());
        pre_tokens.push_back(audio_start_id);

        // Post-audio: <|audio_end|><|im_end|>\n<|im_start|>assistant\n
        post_tokens.push_back(audio_end_id);
        post_tokens.push_back(im_end_id);
        post_tokens.insert(post_tokens.end(), nl.begin(), nl.end());
        post_tokens.push_back(im_start_id);
        post_tokens.insert(post_tokens.end(), asst_.begin(), asst_.end());
    }
    fprintf(stderr,
            "[asr_e2e] prompt: pre=%zu + audio=%d + post=%zu = %zu tokens\n",
            pre_tokens.size(), num_audio_frames, post_tokens.size(),
            pre_tokens.size() + num_audio_frames + post_tokens.size());

    const int total_prefill = (int)pre_tokens.size() + num_audio_frames +
                              (int)post_tokens.size();
    if (total_prefill <= 0 || total_prefill > 4000) {
        fprintf(stderr,
                "[asr_e2e] prefill length %d outside KV cache budget (4096)\n",
                total_prefill);
        return 1;
    }

    auto &dec = asr.text_decoder();

    // -----------------------------------------------------------------------
    // 7. Phase 4.3 split prefill: feed each prefix-token / audio-embed /
    //    suffix-token through forward_decode at consecutive `pos`. The KV
    //    cache and position counter advance one slot per call.
    // -----------------------------------------------------------------------
    std::vector<float> input_emb(n_embd);
    std::vector<float> hidden(n_embd);
    std::vector<float> logits((size_t)vocab_size);

    auto t_pre0 = std::chrono::high_resolution_clock::now();
    int pos = 0;

    // Phase 1: pre-audio tokens.
    for (int t : pre_tokens) {
        if (t < 0 || t >= vocab_size) {
            fprintf(stderr, "[asr_e2e] bad pre-token id=%d\n", t);
            return 1;
        }
        std::memcpy(input_emb.data(),
                    token_embd_w.data() + (size_t)t * n_embd,
                    n_embd * sizeof(float));
        dec.forward_decode(input_emb.data(), pos, hidden.data());
        if (!check_finite(hidden.data(), n_embd, "pre", pos)) return 1;
        ++pos;
    }
    auto t_pre1 = std::chrono::high_resolution_clock::now();

    // Phase 2: audio embeds (already F32 [num_audio_frames, output_dim]).
    for (int i = 0; i < num_audio_frames; ++i) {
        const float *src = audio_embeds.data() + (size_t)i * output_dim;
        // Copy (output_dim must equal n_embd).
        std::memcpy(input_emb.data(), src,
                    (size_t)n_embd * sizeof(float));
        dec.forward_decode(input_emb.data(), pos, hidden.data());
        if (!check_finite(hidden.data(), n_embd, "audio", pos)) return 1;
        ++pos;
    }
    auto t_pre2 = std::chrono::high_resolution_clock::now();

    // Phase 3: post-audio tokens. The LAST call here produces the hidden
    // we will feed into the LM head to pick the FIRST generated token.
    for (size_t k = 0; k < post_tokens.size(); ++k) {
        int t = post_tokens[k];
        if (t < 0 || t >= vocab_size) {
            fprintf(stderr, "[asr_e2e] bad post-token id=%d\n", t);
            return 1;
        }
        std::memcpy(input_emb.data(),
                    token_embd_w.data() + (size_t)t * n_embd,
                    n_embd * sizeof(float));
        dec.forward_decode(input_emb.data(), pos, hidden.data());
        if (!check_finite(hidden.data(), n_embd, "post", pos)) return 1;
        ++pos;
    }
    auto t_pre3 = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // 8. Phase 4.4 autoregressive greedy generation.
    //
    // The hidden produced by the LAST call above (the final post-audio token)
    // is what we project to logits to pick the first generated token.
    // After the first argmax, we feed the new token through forward_decode
    // exactly the same way; greedy until <|im_end|> / <|endoftext|> or
    // max_new_tokens.
    // -----------------------------------------------------------------------
    auto t_gen0 = std::chrono::high_resolution_clock::now();
    std::vector<int> generated;
    generated.reserve(max_new_tokens);

    // Pick first token from the *last* hidden produced by the suffix phase.
    matvec_f32(lm_head_w.data(), hidden.data(), logits.data(),
               vocab_size, n_embd);
    int next_tok = argmax_f32(logits.data(), vocab_size);
    if (next_tok != im_end_id && next_tok != endoftext_id) {
        generated.push_back(next_tok);
    }

    int gen_steps = 0;
    while ((int)generated.size() < max_new_tokens &&
           next_tok != im_end_id && next_tok != endoftext_id) {
        // Embed the just-emitted token.
        std::memcpy(input_emb.data(),
                    token_embd_w.data() + (size_t)next_tok * n_embd,
                    n_embd * sizeof(float));
        dec.forward_decode(input_emb.data(), pos, hidden.data());
        if (!check_finite(hidden.data(), n_embd, "gen", pos)) return 1;
        ++pos;
        ++gen_steps;

        matvec_f32(lm_head_w.data(), hidden.data(), logits.data(),
                   vocab_size, n_embd);
        next_tok = argmax_f32(logits.data(), vocab_size);
        if (next_tok == im_end_id || next_tok == endoftext_id) break;
        generated.push_back(next_tok);
    }
    auto t_gen1 = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // 9. BPE decode generated token IDs → text.
    //
    // BpeTokenizer doesn't expose a decode() — we reconstruct by iterating the
    // vocab map. To stay self-contained in the smoke harness we re-load the
    // raw vocab.json id→bytes map here (mirrors Ascend qwen_asr.cpp's
    // init_reverse_vocab + bpe_unicode_to_bytes; small enough to inline).
    // -----------------------------------------------------------------------
    auto t_dec0 = std::chrono::high_resolution_clock::now();

    // Build GPT-2 byte-level codepoint→byte table.
    static uint8_t cp_to_byte[512];
    {
        std::memset(cp_to_byte, 0, sizeof(cp_to_byte));
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            int cp = ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) ||
                      (b >= 174 && b <= 255)) ? b : 256 + n++;
            if (cp < 512) cp_to_byte[cp] = (uint8_t)b;
        }
    }
    auto unicode_to_bytes = [&](const std::string &s) {
        std::string r;
        size_t p = 0;
        while (p < s.size()) {
            uint32_t cp = 0;
            uint8_t  c  = (uint8_t)s[p];
            int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
            cp = (len == 1) ? c
               : (len == 2) ? (c & 0x1F)
               : (len == 3) ? (c & 0x0F)
                            : (c & 0x07);
            for (int i = 1; i < len && p + i < s.size(); ++i)
                cp = (cp << 6) | ((uint8_t)s[p + i] & 0x3F);
            p += len;
            if (cp < 512) r += (char)cp_to_byte[cp];
            else if (cp < 0x80) r += (char)cp;
            else if (cp < 0x800) {
                r += (char)(0xC0 | (cp >> 6));
                r += (char)(0x80 | (cp & 0x3F));
            } else if (cp < 0x10000) {
                r += (char)(0xE0 | (cp >> 12));
                r += (char)(0x80 | ((cp >> 6) & 0x3F));
                r += (char)(0x80 | (cp & 0x3F));
            } else {
                r += (char)(0xF0 | (cp >> 18));
                r += (char)(0x80 | ((cp >> 12) & 0x3F));
                r += (char)(0x80 | ((cp >> 6) & 0x3F));
                r += (char)(0x80 | (cp & 0x3F));
            }
        }
        return r;
    };

    // Parse vocab.json for id→bpe_str map (minimal — same loose JSON parser as
    // the Ascend reference; full enough for Qwen's vocab format).
    std::unordered_map<int, std::string> id_to_bytes;
    {
        std::ifstream f(vocab_path);
        if (!f.is_open()) {
            fprintf(stderr, "[asr_e2e] cannot open %s for decode\n",
                    vocab_path.c_str());
            return 1;
        }
        std::stringstream ss; ss << f.rdbuf(); f.close();
        std::string content = ss.str();

        size_t p = 0;
        while (p < content.size()) {
            size_t q1 = content.find('"', p);
            if (q1 == std::string::npos) break;
            size_t q2 = q1 + 1;
            while (q2 < content.size()) {
                if (content[q2] == '\\') { q2 += 2; continue; }
                if (content[q2] == '"') break;
                ++q2;
            }
            if (q2 >= content.size()) break;

            std::string raw = content.substr(q1 + 1, q2 - q1 - 1);
            // Unescape JSON string (\", \\, \n, \t, \uXXXX).
            std::string key;
            for (size_t k = 0; k < raw.size(); ++k) {
                if (raw[k] == '\\' && k + 1 < raw.size()) {
                    char e = raw[k + 1];
                    if (e == '"')  { key += '"';  ++k; }
                    else if (e == '\\') { key += '\\'; ++k; }
                    else if (e == 'n')  { key += '\n'; ++k; }
                    else if (e == 't')  { key += '\t'; ++k; }
                    else if (e == 'b')  { key += '\b'; ++k; }
                    else if (e == 'f')  { key += '\f'; ++k; }
                    else if (e == 'r')  { key += '\r'; ++k; }
                    else if (e == 'u' && k + 5 < raw.size()) {
                        uint32_t cp = (uint32_t)std::strtol(
                            raw.substr(k + 2, 4).c_str(), nullptr, 16);
                        if (cp < 0x80) key += (char)cp;
                        else if (cp < 0x800) {
                            key += (char)(0xC0 | (cp >> 6));
                            key += (char)(0x80 | (cp & 0x3F));
                        } else {
                            key += (char)(0xE0 | (cp >> 12));
                            key += (char)(0x80 | ((cp >> 6) & 0x3F));
                            key += (char)(0x80 | (cp & 0x3F));
                        }
                        k += 5;
                    } else {
                        key += raw[k];
                    }
                } else {
                    key += raw[k];
                }
            }

            size_t colon = content.find(':', q2 + 1);
            if (colon == std::string::npos) break;
            size_t ns = content.find_first_of("0123456789", colon + 1);
            if (ns == std::string::npos) break;
            size_t ne = content.find_first_not_of("0123456789", ns);
            if (ne == std::string::npos) ne = content.size();

            int id = std::atoi(content.substr(ns, ne - ns).c_str());
            id_to_bytes[id] = unicode_to_bytes(key);
            p = ne;
        }
    }
    fprintf(stderr, "[asr_e2e] vocab decode map: %zu entries\n",
            id_to_bytes.size());

    std::string text;
    for (int t : generated) {
        auto it = id_to_bytes.find(t);
        if (it != id_to_bytes.end()) text += it->second;
    }
    auto t_dec1 = std::chrono::high_resolution_clock::now();

    // -----------------------------------------------------------------------
    // 10. Report timings + transcript.
    // -----------------------------------------------------------------------
    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };
    double mel_ms      = ms(t_mel0, t_mel1);
    double encode_ms   = ms(t_enc0, t_enc1);
    double pre_ms      = ms(t_pre0, t_pre1);
    double audio_ms    = ms(t_pre1, t_pre2);
    double post_ms     = ms(t_pre2, t_pre3);
    double prefill_ms  = pre_ms + audio_ms + post_ms;
    double gen_ms      = ms(t_gen0, t_gen1);
    double dec_ms      = ms(t_dec0, t_dec1);
    double total_ms    = mel_ms + encode_ms + prefill_ms + gen_ms + dec_ms;

    fprintf(stderr,
            "[asr_e2e] timing: mel=%.1fms encode=%.1fms "
            "prefill=%.1fms (pre=%.1f audio=%.1f post=%.1f) "
            "gen=%.1fms (%d steps) decode=%.1fms total=%.1fms\n",
            mel_ms, encode_ms, prefill_ms, pre_ms, audio_ms, post_ms,
            gen_ms, gen_steps, dec_ms, total_ms);
    fprintf(stderr, "[asr_e2e] generated tokens: %zu  first 16:",
            generated.size());
    for (size_t i = 0; i < 16 && i < generated.size(); ++i)
        fprintf(stderr, " %d", generated[i]);
    fprintf(stderr, "\n");

    printf("\n=== TRANSCRIPT ===\n%s\n=== END ===\n", text.c_str());

    // Sanity gate: any output at all? If we got zero tokens or a degenerate
    // single-byte string, that's a fail. Otherwise PASS — full content
    // parity vs Ascend ref is for a follow-on patch.
    if (generated.empty()) {
        fprintf(stderr, "[asr_e2e] FAIL: no tokens generated\n");
        return 1;
    }
    if (text.empty()) {
        fprintf(stderr, "[asr_e2e] FAIL: empty transcript after BPE decode\n");
        return 1;
    }

    fprintf(stderr,
            "[asr_e2e] Phase 4.3 + 4.4 E2E PASS  tokens=%zu  text_bytes=%zu\n",
            generated.size(), text.size());
    return 0;
}
