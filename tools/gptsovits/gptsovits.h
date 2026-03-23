#pragma once
#include "llama.h"
#include "llm.h"
#include "model_defs.h"
#include "prompt_cache.h"
#include "text_preprocessor.h"
#include "timer.hpp"
#include "utils.h"
#include "vits.hpp"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <numeric>
#include <random>
#include <string>

// Profiling structure for VITS synthesis breakdown
struct VITSProfile {
    double mel_style_enc_ms = 0;
    double codebook_ms = 0;
    double text_encoder_ms = 0;
    double flow_ms = 0;
    double generator_ms = 0;
    double audio_post_ms = 0;

    double total() const {
        return mel_style_enc_ms + codebook_ms + text_encoder_ms + flow_ms +
               generator_ms + audio_post_ms;
    }
};

// Profiling structure for full TTS pipeline
struct TTSProfile {
    double stft_ms = 0;
    double cnhubert_ms = 0;
    double ssl_proj_ms = 0;
    double quantize_ms = 0;
    double text_preprocess_ms = 0;
    double t2s_ms = 0;
    VITSProfile vits;
    double total_ms = 0;

    void print() const {
        printf("\n=== GPTSoVITS Profiling Results ===\n");
        printf("---------------------------------\n");
        printf("Reference Audio Processing:\n");
        printf("  STFT:              %8.2f ms\n", stft_ms);
        printf("  CNHubert:          %8.2f ms\n", cnhubert_ms);
        printf("  SSL Proj:          %8.2f ms\n", ssl_proj_ms);
        printf("  Quantize:          %8.2f ms\n", quantize_ms);
        printf("Text Processing:\n");
        printf("  Text Preprocess:   %8.2f ms\n", text_preprocess_ms);
        printf("Inference:\n");
        printf("  Text2Semantic:     %8.2f ms\n", t2s_ms);
        printf("VITS Synthesis:\n");
        printf("  MelStyleEncoder:   %8.2f ms\n", vits.mel_style_enc_ms);
        printf("  Codebook:          %8.2f ms\n", vits.codebook_ms);
        printf("  TextEncoder:       %8.2f ms\n", vits.text_encoder_ms);
        printf("  Flow:              %8.2f ms\n", vits.flow_ms);
        printf("  Generator:         %8.2f ms\n", vits.generator_ms);
        printf("  AudioPostProcess:  %8.2f ms\n", vits.audio_post_ms);
        printf("---------------------------------\n");
        printf("Total:               %8.2f ms\n", total_ms);
        printf("=================================\n\n");
    }
};

class Text2SemanticDecoder {
public:
  Text2SemanticDecoder(const std::string &ar_text_model_path,
                       const std::string &t2s_transformer_path,
                       LlmParam llm_params) {
    t2s_transformer_.load_model(t2s_transformer_path, llm_params);
    ContextParams params = {.device_name = "CUDA0",
                            .n_threads = 4,
                            .max_nodes = 2048,
                            .verbosity = GGML_LOG_LEVEL_DEBUG};
    ar_text_infer_ = std::make_unique<ArTextInfer>(ar_text_model_path, params);
  }

  bool infer_panel_naive(const std::vector<llama_token> &all_phones,
                         const std::vector<float> &all_bert_features,
                         const std::vector<llama_token> &ref_semantic_tokens,
                         std::vector<llama_token> &generated_tokens);

private:
  T2STrasformer t2s_transformer_;
  std::unique_ptr<ArTextInfer> ar_text_infer_ = nullptr;
};

// class MelStyleEncoder {
// public:
//   MelStyleEncoder() = default;

//   bool run(const std::vector<float> &refer,
//            const std::vector<float> &refer_mask, std::vector<float> &out_ge);
// };

class SynthesizerTrn {
public:
  SynthesizerTrn(const std::string &ref_enc_model_path,
                 const std::string &codebook_model_path,
                 const std::string &text_encoder_model_path,
                 const std::string &flow_model_path,
                 const std::string &generator_model_path,
                 ContextParams params) {
    ref_enc_ = std::make_unique<MelStyleEncoder>(ref_enc_model_path, params);
    codebook_ =
        std::make_unique<EuclideanCodebook>(codebook_model_path, params);
    enc_p_ = std::make_unique<TextEncoder>(text_encoder_model_path, params);
    flow_ = std::make_unique<FlowBlock>(flow_model_path, params);
    dec_ = std::make_unique<Generator>(generator_model_path, params);
  }

  bool decode(const std::vector<llama_token> &pred_semantic_tokens,
              const std::vector<llama_token> &phones,
              const std::vector<float> &refer_audio_spec,
              std::vector<int16_t> &out_audio_fragment);

  // Decode with profiling - returns timing breakdown in VITSProfile
  bool decode_with_profile(const std::vector<llama_token> &pred_semantic_tokens,
                           const std::vector<llama_token> &phones,
                           const std::vector<float> &refer_audio_spec,
                           std::vector<int16_t> &out_audio_fragment,
                           VITSProfile &profile);

private:
  std::unique_ptr<MelStyleEncoder> ref_enc_ = nullptr;
  std::unique_ptr<EuclideanCodebook> codebook_ = nullptr;
  std::unique_ptr<TextEncoder> enc_p_ = nullptr;
  std::unique_ptr<FlowBlock> flow_ = nullptr;
  std::unique_ptr<Generator> dec_ = nullptr;
  std::mt19937 rng_ = std::mt19937(LLAMA_DEFAULT_SEED);
};

class TTS {
public:
  // Constructor without CNHubert (for backward compatibility)
  TTS(const std::string &ar_text_model_path,
      const std::string &t2s_transformer_path, LlmParam llm_params,
      const std::string &ref_enc_model_path,
      const std::string &codebook_model_path,
      const std::string &text_encoder_model_path,
      const std::string &flow_model_path,
      const std::string &generator_model_path, ContextParams params) {
    t2s_model_ = std::make_unique<Text2SemanticDecoder>(
        ar_text_model_path, t2s_transformer_path, llm_params);
    vits_model_ = std::make_unique<SynthesizerTrn>(
        ref_enc_model_path, codebook_model_path, text_encoder_model_path,
        flow_model_path, generator_model_path, params);
  }

  // Constructor with CNHubert initialization
  TTS(const std::string &ar_text_model_path,
      const std::string &t2s_transformer_path, LlmParam llm_params,
      const std::string &ref_enc_model_path,
      const std::string &codebook_model_path,
      const std::string &text_encoder_model_path,
      const std::string &flow_model_path,
      const std::string &generator_model_path,
      const std::string &cnhubert_model_path,
      const std::string &rvq_path,
      const std::string &bert_model_path, const std::string &tokenizer_path,
      ContextParams params) {
    t2s_model_ = std::make_unique<Text2SemanticDecoder>(
        ar_text_model_path, t2s_transformer_path, llm_params);
    vits_model_ = std::make_unique<SynthesizerTrn>(
        ref_enc_model_path, codebook_model_path, text_encoder_model_path,
        flow_model_path, generator_model_path, params);
    // Initialize CNHubert and ResidualVectorQuantizer
    if (!init_cnhubert(cnhubert_model_path, rvq_path, params)) {
      fprintf(stderr, "TTS: Failed to initialize CNHubert\n");
    }
    // Initialize TextPreprocessor
    text_preprocessor_ = std::make_unique<TextPreprocessor>(
        bert_model_path, tokenizer_path, "", llm_params);
  }

  // Initialize CNHubert and ResidualVectorQuantizer for prompt semantic extraction
  bool init_cnhubert(const std::string &cnhubert_model_path,
                     const std::string &rvq_path,
                     ContextParams params) {
    cnhubert_ = std::make_unique<CNHubertInfer>(cnhubert_model_path, params);
    rvq_ = std::make_unique<ResidualVectorQuantizerInfer>(
        rvq_path, params);
    if (!rvq_->load_codebook()) {
      fprintf(stderr, "Failed to load ssl_proj quantizer codebook\n");
      return false;
    }
    return true;
  }

  // Set reference audio spectrogram (32kHz audio -> STFT)
  // Mirrors Python's _set_ref_spec
  bool set_ref_spec(const std::string &ref_audio_path);

  // Set prompt semantic tokens (16kHz audio -> CNHubert -> quantize)
  // Mirrors Python's _set_prompt_semantic
  bool set_prompt_semantic(const std::string &ref_audio_path);

  // Combined: set both ref_spec and prompt_semantic from same audio file
  bool set_ref_audio(const std::string &ref_audio_path);

  // Run TTS with cached reference audio
  bool run(const std::string &ref_audio_path,
           const std::vector<llama_token> &ref_phones,
           const std::vector<llama_token> &text_phones,
           const std::vector<float> &all_bert_features,
           const std::vector<llama_token> &ref_semantic_tokens,
           std::vector<int16_t> &out_audio_fragment);

  // New High-level run method taking raw text
  bool run(const std::string &ref_audio_path, const std::string &ref_text,
           const std::string &ref_lang, const std::string &target_text,
           const std::string &target_lang,
           std::vector<int16_t> &out_audio_fragment);

  // Get the current prompt cache (read-only)
  const PromptCache &get_prompt_cache() const { return prompt_cache_; }

  // Clear the prompt cache
  void clear_cache() { prompt_cache_.clear(); }

  bool run(const std::vector<llama_token> &ref_phones,
           const std::vector<llama_token> &text_phones,
           const std::vector<float> &all_bert_features,
           const std::vector<llama_token> &ref_semantic_tokens,
           const std::vector<float> &refer_audio_spec,
           std::vector<int16_t> &out_audio_fragment) {

    std::vector<llama_token> all_pred_semantic;
    std::vector<llama_token> all_phones;
    all_phones.insert(all_phones.end(), ref_phones.begin(), ref_phones.end());
    all_phones.insert(all_phones.end(), text_phones.begin(), text_phones.end());
    // int num_tokens = all_phones.size();
    // TODO: use all_bert_features
    // std::vector<float> bert_feats(1024 * num_tokens, 0.f);
    if (!t2s_model_->infer_panel_naive(all_phones, all_bert_features,
                                       ref_semantic_tokens,
                                       all_pred_semantic)) {
      fprintf(stderr, "Text2SemanticDecoder infer failed\n");
      return false;
    }
    // TODO: remove
    all_pred_semantic.erase(all_pred_semantic.begin());
    // print_vector(all_pred_semantic, all_pred_semantic.size());
    if (!vits_model_->decode(all_pred_semantic, text_phones, refer_audio_spec,
                             out_audio_fragment)) {
      fprintf(stderr, "SynthesizerTrn decode failed\n");
      return false;
    }
    return true;
    // std::vector<int> upsample_rates = {10, 8, 2, 2, 2};
    // int upsample_rate =
    //     std::accumulate(upsample_rates.begin(), upsample_rates.end(), 1,
    //                     std::multiplies<int>());
    // int audio_frag_idx = semantic_tokens.size() * 2 * upsample_rate;
  }

  // Run with profiling - includes warmup runs to reduce variance
  bool run_with_profiling(const std::string &ref_audio_path,
                          const std::string &ref_text,
                          const std::string &ref_lang,
                          const std::string &target_text,
                          const std::string &target_lang,
                          std::vector<int16_t> &out_audio_fragment,
                          TTSProfile &profile, int warmup_runs = 1);

private:
  std::unique_ptr<Text2SemanticDecoder> t2s_model_ = nullptr;
  std::unique_ptr<SynthesizerTrn> vits_model_ = nullptr;
  std::unique_ptr<CNHubertInfer> cnhubert_ = nullptr;
  std::unique_ptr<ResidualVectorQuantizerInfer> rvq_ = nullptr;
  std::unique_ptr<TextPreprocessor> text_preprocessor_ = nullptr;
  PromptCache prompt_cache_;
};
class GptSovits {

public:
};