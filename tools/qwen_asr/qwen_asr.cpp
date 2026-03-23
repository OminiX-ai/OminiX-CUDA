#include "qwen_asr.h"
#include "audio_io.h"
#include "bpe_tokenizer.h"
#include "ggml.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>

// ============================================================================
// eval_chunk: feed tokens OR embeddings to llama.cpp
// ============================================================================

bool QwenASR::eval_chunk(llama_token *tokens, float *embd, int n_tokens,
                         bool is_last, llama_pos pos_offset) {
    std::vector<int8_t> logits(n_tokens, 0);
    if (is_last) {
        logits[n_tokens - 1] = 1;
    }

    llama_batch batch = {
        .n_tokens   = n_tokens,
        .token      = tokens,
        .embd       = embd,
        .pos        = nullptr,       // auto-generate from KV cache
        .n_seq_id   = nullptr,       // auto: seq_id = 0
        .seq_id     = nullptr,       // auto: seq_id = 0
        .logits     = logits.data(),
        .pos_offset = pos_offset,
    };

    int ret = llama_decode(llama_ctx_, batch);
    if (ret != 0) {
        fprintf(stderr, "[qwen_asr] eval_chunk failed (ret=%d, n=%d, mode=%s)\n",
                ret, n_tokens, tokens ? "token" : "embd");
        return false;
    }
    return true;
}

QwenASR::~QwenASR() {
    if (llama_ctx_) llama_free(llama_ctx_);
    if (llama_model_) llama_model_free(llama_model_);
}

bool QwenASR::load(const QwenASRParams &params) {
    max_new_tokens_ = params.max_new_tokens;

    if (!params.mel_filters_path.empty()) {
        if (mel_spec_.load_mel_filterbank(params.mel_filters_path))
            printf("[qwen_asr] loaded mel filterbank\n");
    }

    tokenizer_ = std::make_unique<BpeTokenizer>();
    if (!tokenizer_->load(params.vocab_path, params.merges_path)) {
        printf("[qwen_asr] failed to load tokenizer\n");
        return false;
    }

    im_start_id_ = tokenizer_->token_to_id("<|im_start|>");
    im_end_id_ = tokenizer_->token_to_id("<|im_end|>");
    endoftext_id_ = tokenizer_->token_to_id("<|endoftext|>");
    audio_start_id_ = tokenizer_->token_to_id("<|audio_start|>");
    audio_end_id_ = tokenizer_->token_to_id("<|audio_end|>");
    audio_pad_id_ = tokenizer_->token_to_id("<|audio_pad|>");

    vocab_path_ = params.vocab_path;
    init_reverse_vocab(params.vocab_path);

    // Audio encoder (CANN if gpu_layers > 0)
    std::string enc_device = (params.n_gpu_layers > 0) ? "CANN0" : params.device;
    printf("[qwen_asr] loading audio encoder on %s\n", enc_device.c_str());
    if (!audio_encoder_.load(params.audio_encoder_path, enc_device, params.n_threads)) {
        if (enc_device != "CPU") {
            printf("[qwen_asr] CANN failed, falling back to CPU\n");
            if (!audio_encoder_.load(params.audio_encoder_path, "CPU", params.n_threads))
                return false;
        } else return false;
    }

    // llama.cpp decoder
    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers;

    llama_model_ = llama_model_load_from_file(params.decoder_path.c_str(), model_params);
    if (!llama_model_) {
        printf("[qwen_asr] failed to load llama model\n");
        return false;
    }

    n_embd_ = llama_model_n_embd(llama_model_);
    printf("[qwen_asr] decoder: n_embd=%d, gpu_layers=%d\n", n_embd_, params.n_gpu_layers);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads;
    ctx_params.embeddings = false;
    ctx_params.flash_attn_type = (params.n_gpu_layers > 0)
        ? LLAMA_FLASH_ATTN_TYPE_ENABLED
        : LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_ctx_ = llama_init_from_model(llama_model_, ctx_params);
    if (!llama_ctx_) return false;

    // Warmup
    llama_set_warmup(llama_ctx_, true);
    {
        llama_token warmup_tok = 0;
        eval_chunk(&warmup_tok, nullptr, 1, true);
        llama_memory_clear(llama_get_memory(llama_ctx_), true);
    }
    llama_set_warmup(llama_ctx_, false);

    printf("[qwen_asr] ready\n");
    return true;
}

// ============================================================================
// Build prompt segments
// ============================================================================

void QwenASR::build_prompt_segments(int num_audio_frames,
                                     std::vector<int> &pre, std::vector<int> &post) {
    pre.clear(); post.clear();
    auto nl = tokenizer_->encode("\n");

    // Pre-audio: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    pre.push_back(im_start_id_);
    auto sys = tokenizer_->encode("system\n");
    pre.insert(pre.end(), sys.begin(), sys.end());
    pre.push_back(im_end_id_);
    pre.insert(pre.end(), nl.begin(), nl.end());
    pre.push_back(im_start_id_);
    auto user = tokenizer_->encode("user\n");
    pre.insert(pre.end(), user.begin(), user.end());
    pre.push_back(audio_start_id_);

    // Post-audio: <|audio_end|><|im_end|>\n<|im_start|>assistant\n
    post.push_back(audio_end_id_);
    post.push_back(im_end_id_);
    post.insert(post.end(), nl.begin(), nl.end());
    post.push_back(im_start_id_);
    auto asst = tokenizer_->encode("assistant\n");
    post.insert(post.end(), asst.begin(), asst.end());
}

// ============================================================================
// Transcribe
// ============================================================================

bool QwenASR::transcribe(const std::string &audio_path, std::string &output_text) {
    std::vector<float> audio_16k;
    if (!audio_io::load_audio(audio_path, 16000, audio_16k)) return false;
    printf("[qwen_asr] audio: %zu samples (%.2fs)\n", audio_16k.size(), (float)audio_16k.size()/16000.f);
    return transcribe(audio_16k, output_text);
}

bool QwenASR::transcribe(const std::vector<float> &audio_16k, std::string &output_text) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> mel;
    int mel_T = 0;
    mel_spec_.compute(audio_16k, mel, mel_T);

    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> audio_features;
    int num_audio_frames = 0;
    audio_encoder_.encode(mel, mel_T, audio_features, num_audio_frames);
    printf("[qwen_asr] audio features: %d x %d\n", num_audio_frames, n_embd_);

    auto t2 = std::chrono::high_resolution_clock::now();

    transcribe_from_features(audio_features, num_audio_frames, output_text);

    auto t3 = std::chrono::high_resolution_clock::now();
    printf("[qwen_asr] timing: mel=%.0fms, encoder=%.0fms, decoder=%.0fms, total=%.0fms\n",
           std::chrono::duration<double,std::milli>(t1-t0).count(),
           std::chrono::duration<double,std::milli>(t2-t1).count(),
           std::chrono::duration<double,std::milli>(t3-t2).count(),
           std::chrono::duration<double,std::milli>(t3-t0).count());
    return true;
}

// ============================================================================
// Transcribe from features — split prefill: token → embd → token
// ============================================================================

bool QwenASR::transcribe_from_features(const std::vector<float> &audio_features,
                                        int num_audio_frames, std::string &output_text) {
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<int> pre_tokens, post_tokens;
    build_prompt_segments(num_audio_frames, pre_tokens, post_tokens);
    int total = (int)pre_tokens.size() + num_audio_frames + (int)post_tokens.size();
    printf("[qwen_asr] prompt: %d (pre=%zu + audio=%d + post=%zu)\n",
           total, pre_tokens.size(), num_audio_frames, post_tokens.size());

    llama_memory_clear(llama_get_memory(llama_ctx_), true);

    // Phase 1: pre-audio tokens
    if (!eval_chunk(pre_tokens.data(), nullptr, (int)pre_tokens.size(), false)) {
        printf("[qwen_asr] pre-audio prefill failed\n");
        return false;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // Phase 2: audio embeddings
    if (!eval_chunk(nullptr, (float*)audio_features.data(), num_audio_frames, false)) {
        printf("[qwen_asr] audio embd prefill failed\n");
        return false;
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // Phase 3: post-audio tokens (is_last=true → output logits)
    if (!eval_chunk(post_tokens.data(), nullptr, (int)post_tokens.size(), true)) {
        printf("[qwen_asr] post-audio prefill failed\n");
        return false;
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    // Phase 4: autoregressive generation
    const llama_vocab *vocab = llama_model_get_vocab(llama_model_);
    int vocab_size = llama_vocab_n_tokens(vocab);
    std::vector<int> gen_tokens;

    for (int step = 0; step < max_new_tokens_; step++) {
        const float *logits = llama_get_logits_ith(llama_ctx_, -1);
        if (!logits) break;

        int best = 0;
        for (int j = 1; j < vocab_size; j++)
            if (logits[j] > logits[best]) best = j;

        if (best == im_end_id_ || best == endoftext_id_) break;
        gen_tokens.push_back(best);

        llama_token tok = best;
        if (!eval_chunk(&tok, nullptr, 1, true)) break;
    }

    auto t4 = std::chrono::high_resolution_clock::now();

    output_text = decode_tokens(gen_tokens);

    printf("[qwen_asr] gen %zu tokens | pre=%.0fms audio=%.0fms post=%.0fms gen=%.0fms\n",
           gen_tokens.size(),
           std::chrono::duration<double,std::milli>(t1-t0).count(),
           std::chrono::duration<double,std::milli>(t2-t1).count(),
           std::chrono::duration<double,std::milli>(t3-t2).count(),
           std::chrono::duration<double,std::milli>(t4-t3).count());

    return true;
}

// ============================================================================
// Token decoding
// ============================================================================

void QwenASR::init_reverse_vocab(const std::string &vocab_path) {
    memset(unicode_to_byte_, 0, sizeof(unicode_to_byte_));
    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp = ((b>=33&&b<=126)||(b>=161&&b<=172)||(b>=174&&b<=255)) ? b : 256+n++;
        if (cp < 512) unicode_to_byte_[cp] = (uint8_t)b;
    }

    std::ifstream f(vocab_path);
    if (!f.is_open()) return;
    std::stringstream ss; ss << f.rdbuf(); f.close();
    std::string content = ss.str();

    size_t pos = 0;
    while (pos < content.size()) {
        size_t q1 = content.find('"', pos);
        if (q1 == std::string::npos) break;
        size_t q2 = q1 + 1;
        while (q2 < content.size()) {
            if (content[q2] == '\\') { q2 += 2; continue; }
            if (content[q2] == '"') break;
            q2++;
        }
        if (q2 >= content.size()) break;

        std::string key_raw = content.substr(q1+1, q2-q1-1);
        std::string key;
        for (size_t k = 0; k < key_raw.size(); k++) {
            if (key_raw[k] == '\\' && k+1 < key_raw.size()) {
                char e = key_raw[k+1];
                if (e=='"'){key+='"';k++;}
                else if(e=='\\'){key+='\\';k++;}
                else if(e=='n'){key+='\n';k++;}
                else if(e=='t'){key+='\t';k++;}
                else if(e=='u'&&k+5<key_raw.size()){
                    uint32_t cp=(uint32_t)strtol(key_raw.substr(k+2,4).c_str(),nullptr,16);
                    if(cp<0x80)key+=(char)cp;
                    else if(cp<0x800){key+=(char)(0xC0|(cp>>6));key+=(char)(0x80|(cp&0x3F));}
                    else{key+=(char)(0xE0|(cp>>12));key+=(char)(0x80|((cp>>6)&0x3F));key+=(char)(0x80|(cp&0x3F));}
                    k+=5;
                } else key+=key_raw[k];
            } else key+=key_raw[k];
        }

        size_t colon = content.find(':', q2+1);
        if (colon == std::string::npos) break;
        size_t ns = content.find_first_of("0123456789", colon+1);
        if (ns == std::string::npos) break;
        size_t ne = content.find_first_not_of("0123456789", ns);
        if (ne == std::string::npos) ne = content.size();

        int id = atoi(content.substr(ns, ne-ns).c_str());
        id_to_bytes_[id] = bpe_unicode_to_bytes(key);
        pos = ne;
    }
    printf("[qwen_asr] vocab: %zu entries\n", id_to_bytes_.size());
}

std::string QwenASR::bpe_unicode_to_bytes(const std::string &s) {
    std::string r;
    size_t p = 0;
    while (p < s.size()) {
        uint32_t cp = 0;
        uint8_t c = (uint8_t)s[p];
        int len = (c<0x80)?1:(c<0xE0)?2:(c<0xF0)?3:4;
        cp = (len==1)?c:(len==2)?(c&0x1F):(len==3)?(c&0x0F):(c&0x07);
        for (int i=1;i<len&&p+i<s.size();i++) cp=(cp<<6)|((uint8_t)s[p+i]&0x3F);
        p += len;
        if (cp<512) r+=(char)unicode_to_byte_[cp];
        else if(cp<0x80) r+=(char)cp;
        else if(cp<0x800){r+=(char)(0xC0|(cp>>6));r+=(char)(0x80|(cp&0x3F));}
        else if(cp<0x10000){r+=(char)(0xE0|(cp>>12));r+=(char)(0x80|((cp>>6)&0x3F));r+=(char)(0x80|(cp&0x3F));}
        else{r+=(char)(0xF0|(cp>>18));r+=(char)(0x80|((cp>>12)&0x3F));r+=(char)(0x80|((cp>>6)&0x3F));r+=(char)(0x80|(cp&0x3F));}
    }
    return r;
}

std::string QwenASR::decode_tokens(const std::vector<int> &ids) {
    std::string r;
    for (int id : ids) { auto it=id_to_bytes_.find(id); if(it!=id_to_bytes_.end()) r+=it->second; }
    return r;
}
