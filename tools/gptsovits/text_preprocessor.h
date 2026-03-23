#pragma once

#include "llm.h"
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <tuple>

// Forward declarations for optional third-party libraries
#ifdef USE_CPP_PINYIN
namespace Pinyin {
class Pinyin;
}
#endif

#ifdef USE_ESPEAK
struct EspeakHandle;
#endif

class TextPreprocessor {
public:
  struct Result {
    std::vector<llama_token> phones;
    std::vector<float> bert_features;
    std::string norm_text;
  };

  struct Config {
    std::string bert_model_path;
    std::string pinyin_dict_path;      // Path to cpp-pinyin dict folder
    std::string opencpop_map_path;     // Path to opencpop-strict.txt
    std::string cmu_dict_path;         // Path to CMU pronunciation dictionary
    bool use_espeak = false;           // Use espeak-ng for English G2P
    bool normalize_numbers = true;     // Convert numbers to text
    bool v_to_u = true;                // Convert 'v' to 'ü' in pinyin
    int pinyin_tone_style = 3;         // 3 = TONE3 (tone digits after vowel)
  };

  // Legacy constructor for backward compatibility
  TextPreprocessor(
      const std::string &bert_model_path, const std::string &tokenizer_path,
      const std::string &symbols_map_path = "",
      LlmParam llm_params = {});

  // Constructor with config (recommended for new code)
  explicit TextPreprocessor(const Config &config, LlmParam llm_params = {});

  ~TextPreprocessor();

  // Main interface: get phones and BERT features
  // Currently supports: "en" (English), "zh" (Chinese)
  Result preprocess(const std::string &text, const std::string &lang = "zh");

  // Core function matching Python's get_phones_and_bert
  // Returns: (phones, bert_features, norm_text)
  std::tuple<std::vector<llama_token>, std::vector<float>, std::string>
  get_phones_and_bert(const std::string &text, const std::string &language,
                      const std::string &version = "v2", bool final = false);

private:
  Config config_;
  std::unique_ptr<Llm> bert_model_;
  std::map<std::string, llama_token> symbol_to_id_;
  std::vector<std::string> symbols_;

#ifdef USE_CPP_PINYIN
  std::unique_ptr<Pinyin::Pinyin> pinyin_engine_;
  // Pinyin to phone converter (consonant, vowel)
  std::map<std::string, std::pair<std::string, std::string>> pinyin_to_phones_;
#endif

#ifdef USE_ESPEAK
  EspeakHandle *espeak_handle_;
#endif

  // English pronunciation dictionary (word -> phonemes)
  std::map<std::string, std::vector<std::string>> en_dict_;

  // G2P functions
  void g2p(const std::string &text, const std::string &lang,
           std::vector<std::string> &out_phones, std::vector<int> &out_word2ph);

  bool g2p_chinese(const std::string &text,
                   std::vector<std::string> &out_phones,
                   std::vector<int> &out_word2ph);

  bool g2p_english(const std::string &text,
                   std::vector<std::string> &out_phones);

  // Text normalization
  std::string normalize_text(const std::string &text, const std::string &lang);
  std::string normalize_chinese_text(const std::string &text);
  std::string chinese_num_to_text(const std::string &text);
  std::string replace_punctuation(const std::string &text);

  // BERT extraction
  bool get_bert_feature(const std::string &norm_text,
                        const std::vector<int> &word2ph,
                        std::vector<float> &out_features);

  std::vector<float> get_bert_inf(const std::vector<llama_token> &phones,
                                  const std::vector<int> &word2ph,
                                  const std::string &norm_text,
                                  const std::string &language);

  // clean_text_inf: Returns (phones_ids, word2ph, norm_text)
  std::tuple<std::vector<llama_token>, std::vector<int>, std::string>
  clean_text_inf(const std::string &text, const std::string &language,
                 const std::string &version = "v2");

  // Symbol management
  void load_symbols(const std::string &version);
#ifdef USE_CPP_PINYIN
  void load_pinyin_to_phones_map();
#endif
  void load_en_dict();  // Load English pronunciation dictionary
  bool load_cmu_dict(const std::string &path);  // Load CMU dict from file
  std::string ipa_to_arpa(const std::string &ipa);  // Convert IPA to ARPA phonemes
  llama_token get_symbol_id(const std::string &symbol);

  std::vector<llama_token>
  cleaned_text_to_sequence(const std::vector<std::string> &cleaned_text,
                           const std::string &version = "v2");

  // Utility functions
  bool is_chinese(const std::string &text);
  bool is_english(const std::string &text);

#ifdef USE_ESPEAK
  bool init_espeak();
  void cleanup_espeak();
#endif
};
