#pragma once

#include "llama.h"
#include <string>
#include <vector>

// Cache structure for reference audio processing results
// Mirrors Python's TTS.prompt_cache structure
struct PromptCache {
  // Reference audio file path (used as cache key)
  std::string ref_audio_path;

  // Output from _set_ref_spec
  // Shape: 1025 x n_frames (row-major, frequency bins x time frames)
  std::vector<float> refer_audio_spec;
  int spec_n_frames = 0;

  // Output from _set_prompt_semantic
  // Semantic tokens sequence
  std::vector<llama_token> prompt_semantic;

  // Optional: cached intermediate results
  // MelStyleEncoder output (512-dim style vector)
  std::vector<float> ge;

  // Prompt text and related features (optional, for future use)
  std::string prompt_text;
  std::string prompt_lang;
  std::vector<llama_token> prompt_phones;
  std::vector<float> prompt_bert_features;

  // Cache validity flag
  bool is_valid = false;

  // Clear all cached data
  void clear() {
    ref_audio_path.clear();
    refer_audio_spec.clear();
    spec_n_frames = 0;
    prompt_semantic.clear();
    ge.clear();
    prompt_text.clear();
    prompt_lang.clear();
    prompt_phones.clear();
    prompt_bert_features.clear();
    is_valid = false;
  }

  // Check if reference spec is cached
  bool has_ref_spec() const {
    return !refer_audio_spec.empty() && spec_n_frames > 0;
  }

  // Check if prompt semantic is cached
  bool has_prompt_semantic() const { return !prompt_semantic.empty(); }

  // Check if style embedding is cached
  bool has_ge() const { return !ge.empty(); }
};
