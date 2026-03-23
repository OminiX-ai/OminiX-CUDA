#include "gptsovits.h"
#include "audio_io.h"
#include "audio_postprocess.h"
#include "llama.h"
#include "stft.h"
#include "utils.h"
#include <cstddef>
#include <numeric>
#include <string>

bool Text2SemanticDecoder::infer_panel_naive(
    const std::vector<llama_token> &all_phones,
    const std::vector<float> &all_bert_features,
    const std::vector<llama_token> &ref_semantic_tokens,
    std::vector<llama_token> &generated_tokens) {
    int num_tokens = all_phones.size();
    // // TODO: use all_bert_features
    // std::vector<float> bert_feats(1024 * num_tokens, 0.f);
    std::vector<llama_pos> pos(num_tokens);
    std::iota(pos.begin(), pos.end(), 0);
    Timer timer;
    timer.start();
    std::vector<float> x_pos;
    ar_text_infer_->run(all_phones, pos, all_bert_features, x_pos);
    // ar_text_infer_->synchronize();
    // print_vector(x_pos, 20);
    t2s_transformer_.generate(x_pos.data(), x_pos.size() / 512,
                              ref_semantic_tokens, generated_tokens, false);
    // print_vector(generated_tokens, generated_tokens.size());
    return true;
}

bool SynthesizerTrn::decode(
    const std::vector<llama_token> &pred_semantic_tokens,
    const std::vector<llama_token> &phones,
    const std::vector<float> &refer_audio_spec,
    std::vector<int16_t> &out_audio_fragment) {
    /*
        refer_audio_spec: 1x1025xseq_len -> cut 1x704xseq_len
    */
    int n_tokens = refer_audio_spec.size() / 1025;
    std::vector<float> refer_audio_spec_704(
        refer_audio_spec.begin(), refer_audio_spec.begin() + 704 * n_tokens);
    std::vector<float> ge;
    // ge: 1x512
    if (!ref_enc_->run(refer_audio_spec_704, ge)) {
        fprintf(stderr, "MelStyleEncoder run failed\n");
        return false;
    }
    // print_vector(ge, 20);

    std::vector<float> upsampled_quantized;
    if (!codebook_->run(pred_semantic_tokens, upsampled_quantized)) {
        fprintf(stderr, "EuclideanCodebook run failed\n");
        return false;
    }
    // print_vector(upsampled_quantized, 20);

    // TextEncoder
    std::vector<float> means, exp_logs;
    enc_p_->run(upsampled_quantized, phones, ge, means, exp_logs);
    // print_vector(means, 20);
    // print_vector(exp_logs, 20);

    // // z_p = m_p + torch.randn_like(m_p) * exp_logs * noise_scale
    // z_p: 1x192xn
    std::vector<float> z_p;
    float noise_scale = 0.5f;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (size_t i = 0; i < means.size(); ++i) {
        float rand_value = normal(rng_);
        float z_p_value = means[i] + exp_logs[i] * noise_scale * rand_value;
        z_p.push_back(z_p_value);
    }

    // flow: z = self.flow(z_p, y_mask, g=ge, reverse=True)
    // return 1x192xn
    std::vector<float> z;
    flow_->run(z_p, ge, z);
    // print_vector(z, 20);
    // Generator ->
    //  out_audio_fragment o = self.dec((z * y_mask)[:, :, :], g=ge) input: z, ge
    std::vector<float> raw_audio;
    dec_->run(z, ge, raw_audio);
    // print_vector(raw_audio, 20);
    // save_vector_to_file(raw_audio,
    //                     "/home/ubuntu/data_1/wjr/weights/gptsovits/data/res.raw");

    // audio post process
    std::vector<std::vector<std::vector<float>>> audio_batch = {{raw_audio}};
    std::vector<std::vector<int>> batch_index_list = {{0}};
    // sampling_rate 32000, speed_factor 1.0, split_bucket false, interval 0.3
    auto result = audio_post::AudioPostProcessor::process(
        audio_batch, 32000, batch_index_list, 1.0f, true, 0.3f);
    out_audio_fragment = result.second;
    return true;
}

bool SynthesizerTrn::decode_with_profile(
    const std::vector<llama_token> &pred_semantic_tokens,
    const std::vector<llama_token> &phones,
    const std::vector<float> &refer_audio_spec,
    std::vector<int16_t> &out_audio_fragment, VITSProfile &profile) {
    Timer timer;

    int n_tokens = refer_audio_spec.size() / 1025;
    std::vector<float> refer_audio_spec_704(
        refer_audio_spec.begin(), refer_audio_spec.begin() + 704 * n_tokens);

    // MelStyleEncoder
    ref_enc_->synchronize(); // Sync before timing
    timer.start();
    std::vector<float> ge;
    if (!ref_enc_->run(refer_audio_spec_704, ge)) {
        fprintf(stderr, "MelStyleEncoder run failed\n");
        return false;
    }
    ref_enc_->synchronize(); // Sync after GPU operation
    profile.mel_style_enc_ms = timer.elapsed<Timer::ms>();

    // Codebook
    codebook_->synchronize();
    timer.start();
    std::vector<float> upsampled_quantized;
    if (!codebook_->run(pred_semantic_tokens, upsampled_quantized)) {
        fprintf(stderr, "EuclideanCodebook run failed\n");
        return false;
    }
    codebook_->synchronize();
    profile.codebook_ms = timer.elapsed<Timer::ms>();

    // TextEncoder
    enc_p_->synchronize();
    timer.start();
    std::vector<float> means, exp_logs;
    enc_p_->run(upsampled_quantized, phones, ge, means, exp_logs);
    enc_p_->synchronize();
    profile.text_encoder_ms = timer.elapsed<Timer::ms>();

    // z_p = m_p + torch.randn_like(m_p) * exp_logs * noise_scale
    std::vector<float> z_p;
    float noise_scale = 0.5f;
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (size_t i = 0; i < means.size(); ++i) {
        float rand_value = normal(rng_);
        float z_p_value = means[i] + exp_logs[i] * noise_scale * rand_value;
        z_p.push_back(z_p_value);
    }

    // Flow
    flow_->synchronize();
    timer.start();
    std::vector<float> z;
    flow_->run(z_p, ge, z);
    flow_->synchronize();
    profile.flow_ms = timer.elapsed<Timer::ms>();

    // Generator
    dec_->synchronize();
    timer.start();
    std::vector<float> raw_audio;
    dec_->run(z, ge, raw_audio);
    dec_->synchronize();
    profile.generator_ms = timer.elapsed<Timer::ms>();

    // Audio post process (CPU only, no sync needed)
    timer.start();
    std::vector<std::vector<std::vector<float>>> audio_batch = {{raw_audio}};
    std::vector<std::vector<int>> batch_index_list = {{0}};
    auto result = audio_post::AudioPostProcessor::process(
        audio_batch, 32000, batch_index_list, 1.0f, true, 0.3f);
    out_audio_fragment = result.second;
    profile.audio_post_ms = timer.elapsed<Timer::ms>();

    return true;
}

// TTS::set_ref_spec - Load 32kHz audio and compute STFT spectrogram
bool TTS::set_ref_spec(const std::string &ref_audio_path) {
    // Load audio at 32kHz
    std::vector<float> audio_32k;
    if (!audio_io::load_audio(ref_audio_path, 32000, audio_32k)) {
        fprintf(stderr, "set_ref_spec: failed to load audio from %s\n",
                ref_audio_path.c_str());
        return false;
    }

    // Normalize volume if max > 1
    audio_io::normalize_volume(audio_32k);

    // Compute STFT spectrogram
    int n_frames = 0;
    if (!stft::spectrogram_torch(audio_32k, prompt_cache_.refer_audio_spec,
                                 n_frames)) {
        fprintf(stderr, "set_ref_spec: STFT computation failed\n");
        return false;
    }

    prompt_cache_.spec_n_frames = n_frames;
    prompt_cache_.ref_audio_path = ref_audio_path;

    return true;
}

// TTS::set_prompt_semantic - Load 16kHz audio and extract semantic tokens
bool TTS::set_prompt_semantic(const std::string &ref_audio_path) {
    printf("ref_audio_path: %s\n", ref_audio_path.c_str());
    if (!cnhubert_ || !rvq_) {
        fprintf(stderr, "set_prompt_semantic: CNHubert not initialized. Call "
                        "init_cnhubert() first.\n");
        return false;
    }

    // Load audio at 16kHz
    std::vector<float> audio_16k;
    if (!audio_io::load_audio(ref_audio_path, 16000, audio_16k)) {
        fprintf(stderr, "set_prompt_semantic: failed to load audio from %s\n",
                ref_audio_path.c_str());
        return false;
    }

    // Validate duration (3-10 seconds as per Python implementation)
    if (!audio_io::validate_duration(audio_16k, 16000, 3.0f, 10.0f)) {
        fprintf(stderr,
                "set_prompt_semantic: audio duration should be between 3-10 "
                "seconds\n");
        // Continue anyway, just warn
    }

    // Add 0.3 second padding (4800 samples at 16kHz)
    audio_io::pad_zeros(audio_16k, 0.3 * 32000);

    // Extract features using CNHubert
    std::vector<float> hubert_features;
    if (!cnhubert_->run(audio_16k, hubert_features)) {
        fprintf(stderr, "set_prompt_semantic: CNHubert inference failed\n");
        return false;
    }
    // print_vector(hubert_features, 20);

    // Project through ssl_proj and quantize to semantic tokens (done in GGML graph)
    // load_file_to_vector(hubert_features, "/home/wjr/mount/weights/gptsovits/data/hubert_features_py.raw");
    if (!rvq_->run(hubert_features, prompt_cache_.prompt_semantic)) {
        fprintf(stderr, "set_prompt_semantic: ssl_proj + quantize inference failed\n");
        return false;
    }
    printf("prompt_semantic size: %zu\n", prompt_cache_.prompt_semantic.size());
    // print_vector(prompt_cache_.prompt_semantic, 20);
    return true;
}

// TTS::set_ref_audio - Combined set_ref_spec and set_prompt_semantic
bool TTS::set_ref_audio(const std::string &ref_audio_path) {
    prompt_cache_.clear();

    if (!set_prompt_semantic(ref_audio_path)) {
        return false;
    }

    if (!set_ref_spec(ref_audio_path)) {
        return false;
    }

    prompt_cache_.is_valid = true;
    return true;
}

// TTS::run_with_cache - Run TTS using cached reference audio
bool TTS::run(const std::string &ref_audio_path,
              const std::vector<llama_token> &ref_phones,
              const std::vector<llama_token> &text_phones,
              const std::vector<float> &all_bert_features,
              const std::vector<llama_token> &ref_semantic_tokens,
              std::vector<int16_t> &out_audio_fragment) {
    // set_ref_audio
    if (!set_ref_audio(ref_audio_path)) {
        return false;
    }
    if (!prompt_cache_.is_valid) {
        fprintf(
            stderr,
            "run_with_cache: cache is not valid. Call set_ref_audio() first.\n");
        return false;
    }

    if (!prompt_cache_.has_ref_spec()) {
        fprintf(stderr, "run_with_cache: reference spectrogram not cached\n");
        return false;
    }

    if (!prompt_cache_.has_prompt_semantic()) {
        fprintf(stderr, "run_with_cache: prompt semantic tokens not cached\n");
        return false;
    }

    return run(ref_phones, text_phones, all_bert_features, ref_semantic_tokens,
               prompt_cache_.refer_audio_spec,
               //  prompt_cache_.prompt_semantic, prompt_cache_.refer_audio_spec,
               out_audio_fragment);
}
// TTS::run - High-level pipeline with text preprocessing
bool TTS::run(const std::string &ref_audio_path, const std::string &ref_text,
              const std::string &ref_lang, const std::string &target_text,
              const std::string &target_lang,
              std::vector<int16_t> &out_audio_fragment) {
    if (!text_preprocessor_) {
        fprintf(stderr, "TTS::run: TextPreprocessor not initialized\n");
        return false;
    }

    // 1. Set reference audio (features and prompt cache)
    if (!set_ref_audio(ref_audio_path)) {
        fprintf(stderr, "TTS::run: failed to set reference audio\n");
        return false;
    }

    // 2. Preprocess reference text
    auto ref_res = text_preprocessor_->preprocess(ref_text, ref_lang);
    if (ref_res.phones.empty()) {
        fprintf(stderr, "TTS::run: failed to preprocess reference text\n");
        return false;
    }

    // 3. Preprocess target text
    auto target_res = text_preprocessor_->preprocess(target_text, target_lang);
    if (target_res.phones.empty()) {
        fprintf(stderr, "TTS::run: failed to preprocess target text\n");
        return false;
    }

    // 4. Combine BERT features
    // Concatenate ref_bert and target_bert
    std::vector<float> all_bert = ref_res.bert_features;
    all_bert.insert(all_bert.end(), target_res.bert_features.begin(),
                    target_res.bert_features.end());

    printf("TTS::run pipeline: Ref phones: %zu, Target phones: %zu, Combined "
           "BERT size: %zu\n",
           ref_res.phones.size(), target_res.phones.size(), all_bert.size());

    // 5. Run inference with cached prompt data
    return run(ref_res.phones, target_res.phones, all_bert,
               prompt_cache_.prompt_semantic, prompt_cache_.refer_audio_spec,
               out_audio_fragment);
}

// TTS::run_with_profiling - Run with timing profiling after warmup
bool TTS::run_with_profiling(const std::string &ref_audio_path,
                             const std::string &ref_text,
                             const std::string &ref_lang,
                             const std::string &target_text,
                             const std::string &target_lang,
                             std::vector<int16_t> &out_audio_fragment,
                             TTSProfile &profile, int warmup_runs) {
    if (!text_preprocessor_) {
        fprintf(stderr, "TTS::run_with_profiling: TextPreprocessor not initialized\n");
        return false;
    }

    // Warmup runs
    printf("Running %d warmup run(s)...\n", warmup_runs);
    for (int i = 0; i < warmup_runs; ++i) {
        std::vector<int16_t> dummy_out;
        if (!run(ref_audio_path, ref_text, ref_lang, target_text, target_lang, dummy_out)) {
            fprintf(stderr, "TTS::run_with_profiling: warmup run %d failed\n", i + 1);
            return false;
        }
    }
    printf("Warmup complete. Starting profiled run...\n");

    Timer timer;
    Timer total_timer;
    total_timer.start();

    // Clear cache to measure full pipeline
    prompt_cache_.clear();

    // 1. STFT - set_ref_spec
    timer.start();
    std::vector<float> audio_32k;
    if (!audio_io::load_audio(ref_audio_path, 32000, audio_32k)) {
        fprintf(stderr, "run_with_profiling: failed to load audio\n");
        return false;
    }
    audio_io::normalize_volume(audio_32k);
    int n_frames = 0;
    if (!stft::spectrogram_torch(audio_32k, prompt_cache_.refer_audio_spec, n_frames)) {
        fprintf(stderr, "run_with_profiling: STFT failed\n");
        return false;
    }
    prompt_cache_.spec_n_frames = n_frames;
    profile.stft_ms = timer.stop<Timer::ms>();

    // 2. CNHubert
    std::vector<float> audio_16k;
    if (!audio_io::load_audio(ref_audio_path, 16000, audio_16k)) {
        fprintf(stderr, "run_with_profiling: failed to load 16k audio\n");
        return false;
    }
    audio_io::pad_zeros(audio_16k, 0.3 * 32000);
    // cnhubert_->synchronize(); // Sync before timing
    // timer.start();
    std::vector<float> hubert_features;
    if (!cnhubert_->run(audio_16k, hubert_features)) {
        fprintf(stderr, "run_with_profiling: CNHubert failed\n");
        return false;
    }
    cnhubert_->synchronize(); // Sync after GPU operation
    profile.cnhubert_ms = timer.stop<Timer::ms>();

    // 3. SSL Proj + Quantize (combined in GGML graph)
    // rvq_->synchronize();
    // timer.start();
    if (!rvq_->run(hubert_features, prompt_cache_.prompt_semantic)) {
        fprintf(stderr, "run_with_profiling: SSL proj + quantize failed\n");
        return false;
    }
    rvq_->synchronize();
    profile.ssl_proj_ms = timer.stop<Timer::ms>();
    profile.quantize_ms = 0; // Quantize is now part of ssl_proj
    prompt_cache_.is_valid = true;

    // 5. Text preprocessing (ref + target)
    // timer.start();
    auto ref_res = text_preprocessor_->preprocess(ref_text, ref_lang);
    auto target_res = text_preprocessor_->preprocess(target_text, target_lang);
    if (ref_res.phones.empty() || target_res.phones.empty()) {
        fprintf(stderr, "run_with_profiling: text preprocessing failed\n");
        return false;
    }
    profile.text_preprocess_ms = timer.stop<Timer::ms>();

    // Combine phones and BERT features
    std::vector<llama_token> all_phones;
    all_phones.insert(all_phones.end(), ref_res.phones.begin(), ref_res.phones.end());
    all_phones.insert(all_phones.end(), target_res.phones.begin(), target_res.phones.end());
    std::vector<float> all_bert = ref_res.bert_features;
    all_bert.insert(all_bert.end(), target_res.bert_features.begin(),
                    target_res.bert_features.end());

    // 6. Text2Semantic
    // timer.start();
    std::vector<llama_token> all_pred_semantic;
    if (!t2s_model_->infer_panel_naive(all_phones, all_bert,
                                       prompt_cache_.prompt_semantic,
                                       all_pred_semantic)) {
        fprintf(stderr, "run_with_profiling: Text2Semantic failed\n");
        return false;
    }
    all_pred_semantic.erase(all_pred_semantic.begin());
    profile.t2s_ms = timer.stop<Timer::ms>();

    // 7. VITS decode with profiling
    if (!vits_model_->decode_with_profile(all_pred_semantic, target_res.phones,
                                          prompt_cache_.refer_audio_spec,
                                          out_audio_fragment, profile.vits)) {
        fprintf(stderr, "run_with_profiling: VITS decode failed\n");
        return false;
    }

    profile.total_ms = total_timer.stop<Timer::ms>();
    return true;
}
