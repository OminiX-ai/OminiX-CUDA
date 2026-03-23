#include "text_preprocessor.h"
#include "utils.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

// Include cpp-pinyin
#ifdef USE_CPP_PINYIN
#include <cpp-pinyin/Pinyin.h>
#include <cpp-pinyin/PinyinRes.h>
#endif

// Include espeak-ng if available
#ifdef USE_ESPEAK
extern "C" {
#include <espeak-ng/speak_lib.h>
}
#endif

struct EspeakHandle {
  bool initialized = false;
};

// Complete symbols from symbols.py (GPT-SoVITS)
// IMPORTANT: Must match Python's sorted order exactly!
// Python uses: symbols = sorted(set([pad] + c + v + ja_symbols + pu_symbols + list(arpa)))
static const std::vector<std::string> SYMBOLS_V2 = {
    "!", ",", "-", ".", "?", "AA", "AA0", "AA1", "AA2", "AE0",
    "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2", "AW0", "AW1",
    "AW2", "AY0", "AY1", "AY2", "B", "CH", "D", "DH", "E1", "E2",
    "E3", "E4", "E5", "EE", "EH0", "EH1", "EH2", "ER", "ER0", "ER1",
    "ER2", "EY0", "EY1", "EY2", "En1", "En2", "En3", "En4", "En5", "F",
    "G", "HH", "I", "IH", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2",
    "JH", "K", "L", "M", "N", "NG", "OO", "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2", "P", "R", "S", "SH", "SP", "SP2", "SP3",
    "T", "TH", "U", "UH0", "UH1", "UH2", "UNK", "UW0", "UW1", "UW2",
    "V", "W", "Y", "Z", "ZH", "_", "a", "a1", "a2", "a3",
    "a4", "a5", "ai1", "ai2", "ai3", "ai4", "ai5", "an1", "an2", "an3",
    "an4", "an5", "ang1", "ang2", "ang3", "ang4", "ang5", "ao1", "ao2", "ao3",
    "ao4", "ao5", "b", "by", "c", "ch", "cl", "d", "dy", "e",
    "e1", "e2", "e3", "e4", "e5", "ei1", "ei2", "ei3", "ei4", "ei5",
    "en1", "en2", "en3", "en4", "en5", "eng1", "eng2", "eng3", "eng4", "eng5",
    "er1", "er2", "er3", "er4", "er5", "f", "g", "gy", "h", "hy",
    "i", "i01", "i02", "i03", "i04", "i05", "i1", "i2", "i3", "i4",
    "i5", "ia1", "ia2", "ia3", "ia4", "ia5", "ian1", "ian2", "ian3", "ian4",
    "ian5", "iang1", "iang2", "iang3", "iang4", "iang5", "iao1", "iao2", "iao3", "iao4",
    "iao5", "ie1", "ie2", "ie3", "ie4", "ie5", "in1", "in2", "in3", "in4",
    "in5", "ing1", "ing2", "ing3", "ing4", "ing5", "iong1", "iong2", "iong3", "iong4",
    "iong5", "ir1", "ir2", "ir3", "ir4", "ir5", "iu1", "iu2", "iu3", "iu4",
    "iu5", "j", "k", "ky", "l", "m", "my", "n", "ny", "o",
    "o1", "o2", "o3", "o4", "o5", "ong1", "ong2", "ong3", "ong4", "ong5",
    "ou1", "ou2", "ou3", "ou4", "ou5", "p", "py", "q", "r", "ry",
    "s", "sh", "t", "ts", "u", "u1", "u2", "u3", "u4", "u5",
    "ua1", "ua2", "ua3", "ua4", "ua5", "uai1", "uai2", "uai3", "uai4", "uai5",
    "uan1", "uan2", "uan3", "uan4", "uan5", "uang1", "uang2", "uang3", "uang4", "uang5",
    "ui1", "ui2", "ui3", "ui4", "ui5", "un1", "un2", "un3", "un4", "un5",
    "uo1", "uo2", "uo3", "uo4", "uo5", "v", "v1", "v2", "v3", "v4",
    "v5", "van1", "van2", "van3", "van4", "van5", "ve1", "ve2", "ve3", "ve4",
    "ve5", "vn1", "vn2", "vn3", "vn4", "vn5", "w", "x", "y", "z",
    "zh", "…"
}; // Total: 322 symbols (must match Python exactly)

// Legacy constructor for backward compatibility
TextPreprocessor::TextPreprocessor(
    const std::string &bert_model_path, const std::string &tokenizer_path,
    const std::string &symbols_map_path, LlmParam llm_params) {

  // Initialize config with legacy parameters
  config_.bert_model_path = bert_model_path;
  // tokenizer_path and symbols_map_path are not used in new implementation
  (void)tokenizer_path;
  (void)symbols_map_path;

  // Set default config values
  config_.use_espeak = false;
  config_.normalize_numbers = true;
  config_.v_to_u = true;
  config_.pinyin_tone_style = 3;

#ifdef USE_ESPEAK
  espeak_handle_ = nullptr;
#endif

  // Load symbols
  load_symbols("v2");

#ifdef USE_CPP_PINYIN
  // Load pinyin to phones mapping
  load_pinyin_to_phones_map();
#endif

  // Load English pronunciation dictionary
  load_en_dict();

  // Initialize cpp-pinyin
#ifdef USE_CPP_PINYIN
  try {
    pinyin_engine_ = std::make_unique<Pinyin::Pinyin>();
    std::cout << "cpp-pinyin initialized successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize cpp-pinyin: " << e.what() << std::endl;
    pinyin_engine_ = nullptr;
  }
#endif

  // Load BERT model
  if (!config_.bert_model_path.empty()) {
    bert_model_ = std::make_unique<Llm>();
    if (!bert_model_->load_model(config_.bert_model_path, llm_params)) {
      std::cerr << "Failed to load BERT model from " << config_.bert_model_path
                << std::endl;
      bert_model_ = nullptr;
    }
  }
}

TextPreprocessor::TextPreprocessor(const Config &config,
                                   LlmParam llm_params)
    : config_(config)
#ifdef USE_ESPEAK
    , espeak_handle_(nullptr)
#endif
{

  // Load symbols
  load_symbols("v2");

#ifdef USE_CPP_PINYIN
  // Load pinyin to phones mapping
  load_pinyin_to_phones_map();
#endif

  // Load English pronunciation dictionary
  load_en_dict();

  // Initialize cpp-pinyin
#ifdef USE_CPP_PINYIN
  try {
    // Note: cpp-pinyin doesn't have setDictionaryPath, it uses default dict
    // if (!config_.pinyin_dict_path.empty()) {
    //   Pinyin::setDictionaryPath(config_.pinyin_dict_path);
    // }
    pinyin_engine_ = std::make_unique<Pinyin::Pinyin>();
    std::cout << "cpp-pinyin initialized successfully" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Failed to initialize cpp-pinyin: " << e.what() << std::endl;
    pinyin_engine_ = nullptr;
  }
#endif

  // Initialize espeak-ng for English
#ifdef USE_ESPEAK
  if (config_.use_espeak) {
    espeak_handle_ = new EspeakHandle();
    if (!init_espeak()) {
      std::cerr << "Failed to initialize espeak-ng, falling back to mock"
                << std::endl;
      config_.use_espeak = false;
    }
  }
#endif

  // Load BERT model
  if (!config_.bert_model_path.empty()) {
    bert_model_ = std::make_unique<Llm>();
    if (!bert_model_->load_model(config_.bert_model_path, llm_params)) {
      std::cerr << "Failed to load BERT model from " << config_.bert_model_path
                << std::endl;
      bert_model_ = nullptr;
    }
  }
}

TextPreprocessor::~TextPreprocessor() {
#ifdef USE_ESPEAK
  cleanup_espeak();
  if (espeak_handle_) {
    delete espeak_handle_;
  }
#endif
}

void TextPreprocessor::load_symbols(const std::string &version) {
  (void)version;
  symbols_ = SYMBOLS_V2;
  for (size_t i = 0; i < symbols_.size(); ++i) {
    symbol_to_id_[symbols_[i]] = static_cast<llama_token>(i);
  }
}

#ifdef USE_CPP_PINYIN
void TextPreprocessor::load_pinyin_to_phones_map() {
  // Load mapping from opencpop-strict.txt file used by GPT-SoVITS
  // Format: pinyin<tab>phones separated by spaces
  // Example: ba<tab>b a

  std::string map_path = config_.opencpop_map_path;

  // If no path specified, try default location
  if (map_path.empty()) {
    map_path = "tools/gptsovits/data/opencpop-strict.txt";
  }

  std::ifstream file(map_path);
  if (!file.is_open()) {
    std::cerr << "Warning: Could not open opencpop-strict.txt at " << map_path
              << ", using fallback mappings" << std::endl;

    // Fallback: minimal set of hardcoded mappings
    pinyin_to_phones_["ni"] = {"n", "i"};
    pinyin_to_phones_["hao"] = {"h", "ao"};
    pinyin_to_phones_["shi"] = {"sh", "i0"};
    pinyin_to_phones_["jie"] = {"j", "ie"};
    pinyin_to_phones_["da"] = {"d", "a"};
    pinyin_to_phones_["jia"] = {"j", "ia"};
    pinyin_to_phones_["huan"] = {"h", "uan"};
    pinyin_to_phones_["ying"] = {"_", "ing"};
    pinyin_to_phones_["wo"] = {"_", "uo"};
    pinyin_to_phones_["men"] = {"m", "en"};

    std::cout << "Loaded " << pinyin_to_phones_.size()
              << " fallback pinyin-to-phones mappings" << std::endl;
    return;
  }

  // Load mappings from file
  std::string line;
  int count = 0;
  while (std::getline(file, line)) {
    if (line.empty()) continue;

    // Split by tab
    size_t tab_pos = line.find('\t');
    if (tab_pos == std::string::npos) continue;

    std::string pinyin = line.substr(0, tab_pos);
    std::string phones_str = line.substr(tab_pos + 1);

    // Split phones by space - expecting exactly 2 parts: consonant and vowel
    std::istringstream iss(phones_str);
    std::string consonant, vowel;
    if (iss >> consonant >> vowel) {
      pinyin_to_phones_[pinyin] = {consonant, vowel};
      count++;
    }
  }

  file.close();
  std::cout << "Loaded " << count << " pinyin-to-phones mappings from "
            << map_path << std::endl;
}
#endif

#ifdef USE_ESPEAK
bool TextPreprocessor::init_espeak() {
  int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, nullptr, 0);
  if (result < 0) {
    return false;
  }
  espeak_handle_->initialized = true;
  espeak_SetVoiceByName("en");
  return true;
}

void TextPreprocessor::cleanup_espeak() {
  if (espeak_handle_ && espeak_handle_->initialized) {
    espeak_Terminate();
    espeak_handle_->initialized = false;
  }
}
#endif

void TextPreprocessor::load_en_dict() {
  // Try to load from CMU dict file
  std::string dict_path = config_.cmu_dict_path;

  // If no path specified, try default location
  if (dict_path.empty()) {
    dict_path = "tools/gptsovits/data/cmudict.rep";
  }

  if (load_cmu_dict(dict_path)) {
    std::cout << "Loaded CMU dictionary from " << dict_path
              << " with " << en_dict_.size() << " entries" << std::endl;
    return;
  }

  std::cerr << "Warning: Could not load CMU dictionary from " << dict_path
            << ", using fallback built-in dictionary" << std::endl;

  // Fallback: Load a comprehensive built-in dictionary
  // This is a production-ready subset of common English words
  // Based on CMU Pronouncing Dictionary format
  en_dict_ = {
      // Articles
      {"a", {"AH0"}},
      {"an", {"AE1", "N"}},
      {"the", {"DH", "AH0"}},

      // Common verbs
      {"is", {"IH1", "Z"}},
      {"are", {"AA1", "R"}},
      {"was", {"W", "AA1", "Z"}},
      {"were", {"W", "ER1"}},
      {"be", {"B", "IY1"}},
      {"been", {"B", "IH1", "N"}},
      {"have", {"HH", "AE1", "V"}},
      {"has", {"HH", "AE1", "Z"}},
      {"had", {"HH", "AE1", "D"}},
      {"do", {"D", "UW1"}},
      {"does", {"D", "AH1", "Z"}},
      {"did", {"D", "IH1", "D"}},
      {"will", {"W", "IH1", "L"}},
      {"would", {"W", "UH1", "D"}},
      {"can", {"K", "AE1", "N"}},
      {"could", {"K", "UH1", "D"}},
      {"should", {"SH", "UH1", "D"}},
      {"may", {"M", "EY1"}},
      {"might", {"M", "AY1", "T"}},
      {"must", {"M", "AH1", "S", "T"}},
      {"go", {"G", "OW1"}},
      {"went", {"W", "EH1", "N", "T"}},
      {"gone", {"G", "AO1", "N"}},
      {"get", {"G", "EH1", "T"}},
      {"got", {"G", "AA1", "T"}},
      {"make", {"M", "EY1", "K"}},
      {"made", {"M", "EY1", "D"}},
      {"see", {"S", "IY1"}},
      {"saw", {"S", "AO1"}},
      {"seen", {"S", "IY1", "N"}},
      {"say", {"S", "EY1"}},
      {"said", {"S", "EH1", "D"}},
      {"come", {"K", "AH1", "M"}},
      {"came", {"K", "EY1", "M"}},
      {"take", {"T", "EY1", "K"}},
      {"took", {"T", "UH1", "K"}},
      {"give", {"G", "IH1", "V"}},
      {"gave", {"G", "EY1", "V"}},
      {"find", {"F", "AY1", "N", "D"}},
      {"found", {"F", "AW1", "N", "D"}},
      {"know", {"N", "OW1"}},
      {"knew", {"N", "UW1"}},
      {"think", {"TH", "IH1", "NG", "K"}},
      {"thought", {"TH", "AO1", "T"}},
      {"tell", {"T", "EH1", "L"}},
      {"told", {"T", "OW1", "L", "D"}},
      {"become", {"B", "IH0", "K", "AH1", "M"}},
      {"leave", {"L", "IY1", "V"}},
      {"left", {"L", "EH1", "F", "T"}},
      {"feel", {"F", "IY1", "L"}},
      {"felt", {"F", "EH1", "L", "T"}},
      {"bring", {"B", "R", "IH1", "NG"}},
      {"brought", {"B", "R", "AO1", "T"}},
      {"begin", {"B", "IH0", "G", "IH1", "N"}},
      {"began", {"B", "IH0", "G", "AE1", "N"}},
      {"keep", {"K", "IY1", "P"}},
      {"kept", {"K", "EH1", "P", "T"}},
      {"hold", {"HH", "OW1", "L", "D"}},
      {"held", {"HH", "EH1", "L", "D"}},
      {"write", {"R", "AY1", "T"}},
      {"wrote", {"R", "OW1", "T"}},
      {"stand", {"S", "T", "AE1", "N", "D"}},
      {"stood", {"S", "T", "UH1", "D"}},
      {"hear", {"HH", "IH1", "R"}},
      {"heard", {"HH", "ER1", "D"}},
      {"let", {"L", "EH1", "T"}},
      {"mean", {"M", "IY1", "N"}},
      {"meant", {"M", "EH1", "N", "T"}},
      {"set", {"S", "EH1", "T"}},
      {"meet", {"M", "IY1", "T"}},
      {"met", {"M", "EH1", "T"}},
      {"run", {"R", "AH1", "N"}},
      {"ran", {"R", "AE1", "N"}},
      {"move", {"M", "UW1", "V"}},
      {"live", {"L", "IH1", "V"}},
      {"believe", {"B", "IH0", "L", "IY1", "V"}},
      {"happen", {"HH", "AE1", "P", "AH0", "N"}},
      {"seem", {"S", "IY1", "M"}},
      {"try", {"T", "R", "AY1"}},
      {"ask", {"AE1", "S", "K"}},
      {"need", {"N", "IY1", "D"}},
      {"want", {"W", "AA1", "N", "T"}},
      {"use", {"Y", "UW1", "Z"}},
      {"show", {"SH", "OW1"}},
      {"work", {"W", "ER1", "K"}},
      {"call", {"K", "AO1", "L"}},
      {"follow", {"F", "AA1", "L", "OW0"}},
      {"turn", {"T", "ER1", "N"}},
      {"start", {"S", "T", "AA1", "R", "T"}},
      {"help", {"HH", "EH1", "L", "P"}},
      {"talk", {"T", "AO1", "K"}},
      {"put", {"P", "UH1", "T"}},
      {"read", {"R", "IY1", "D"}},
      {"allow", {"AH0", "L", "AW1"}},
      {"add", {"AE1", "D"}},
      {"spend", {"S", "P", "EH1", "N", "D"}},
      {"grow", {"G", "R", "OW1"}},
      {"open", {"OW1", "P", "AH0", "N"}},
      {"walk", {"W", "AO1", "K"}},
      {"win", {"W", "IH1", "N"}},
      {"offer", {"AO1", "F", "ER0"}},
      {"remember", {"R", "IH0", "M", "EH1", "M", "B", "ER0"}},
      {"love", {"L", "AH1", "V"}},
      {"consider", {"K", "AH0", "N", "S", "IH1", "D", "ER0"}},
      {"appear", {"AH0", "P", "IH1", "R"}},
      {"buy", {"B", "AY1"}},
      {"wait", {"W", "EY1", "T"}},
      {"serve", {"S", "ER1", "V"}},
      {"die", {"D", "AY1"}},
      {"send", {"S", "EH1", "N", "D"}},
      {"expect", {"IH0", "K", "S", "P", "EH1", "K", "T"}},
      {"build", {"B", "IH1", "L", "D"}},
      {"stay", {"S", "T", "EY1"}},
      {"fall", {"F", "AO1", "L"}},
      {"cut", {"K", "AH1", "T"}},
      {"reach", {"R", "IY1", "CH"}},
      {"kill", {"K", "IH1", "L"}},
      {"remain", {"R", "IH0", "M", "EY1", "N"}},
      {"suggest", {"S", "AH0", "G", "JH", "EH1", "S", "T"}},
      {"raise", {"R", "EY1", "Z"}},
      {"pass", {"P", "AE1", "S"}},
      {"sell", {"S", "EH1", "L"}},
      {"require", {"R", "IH0", "K", "W", "AY1", "ER0"}},
      {"report", {"R", "IH0", "P", "AO1", "R", "T"}},
      {"decide", {"D", "IH0", "S", "AY1", "D"}},
      {"pull", {"P", "UH1", "L"}},

      // Common nouns
      {"time", {"T", "AY1", "M"}},
      {"year", {"Y", "IH1", "R"}},
      {"people", {"P", "IY1", "P", "AH0", "L"}},
      {"way", {"W", "EY1"}},
      {"day", {"D", "EY1"}},
      {"man", {"M", "AE1", "N"}},
      {"thing", {"TH", "IH1", "NG"}},
      {"woman", {"W", "UH1", "M", "AH0", "N"}},
      {"life", {"L", "AY1", "F"}},
      {"child", {"CH", "AY1", "L", "D"}},
      {"world", {"W", "ER1", "L", "D"}},
      {"school", {"S", "K", "UW1", "L"}},
      {"state", {"S", "T", "EY1", "T"}},
      {"family", {"F", "AE1", "M", "AH0", "L", "IY0"}},
      {"student", {"S", "T", "UW1", "D", "AH0", "N", "T"}},
      {"group", {"G", "R", "UW1", "P"}},
      {"country", {"K", "AH1", "N", "T", "R", "IY0"}},
      {"problem", {"P", "R", "AA1", "B", "L", "AH0", "M"}},
      {"hand", {"HH", "AE1", "N", "D"}},
      {"part", {"P", "AA1", "R", "T"}},
      {"place", {"P", "L", "EY1", "S"}},
      {"case", {"K", "EY1", "S"}},
      {"week", {"W", "IY1", "K"}},
      {"company", {"K", "AH1", "M", "P", "AH0", "N", "IY0"}},
      {"system", {"S", "IH1", "S", "T", "AH0", "M"}},
      {"program", {"P", "R", "OW1", "G", "R", "AE2", "M"}},
      {"question", {"K", "W", "EH1", "S", "CH", "AH0", "N"}},
      {"work", {"W", "ER1", "K"}},
      {"government", {"G", "AH1", "V", "ER0", "N", "M", "AH0", "N", "T"}},
      {"number", {"N", "AH1", "M", "B", "ER0"}},
      {"night", {"N", "AY1", "T"}},
      {"point", {"P", "OY1", "N", "T"}},
      {"home", {"HH", "OW1", "M"}},
      {"water", {"W", "AO1", "T", "ER0"}},
      {"room", {"R", "UW1", "M"}},
      {"mother", {"M", "AH1", "DH", "ER0"}},
      {"area", {"EH1", "R", "IY0", "AH0"}},
      {"money", {"M", "AH1", "N", "IY0"}},
      {"story", {"S", "T", "AO1", "R", "IY0"}},
      {"fact", {"F", "AE1", "K", "T"}},
      {"month", {"M", "AH1", "N", "TH"}},
      {"lot", {"L", "AA1", "T"}},
      {"right", {"R", "AY1", "T"}},
      {"study", {"S", "T", "AH1", "D", "IY0"}},
      {"book", {"B", "UH1", "K"}},
      {"eye", {"AY1"}},
      {"job", {"JH", "AA1", "B"}},
      {"word", {"W", "ER1", "D"}},
      {"business", {"B", "IH1", "Z", "N", "AH0", "S"}},
      {"issue", {"IH1", "SH", "UW0"}},
      {"side", {"S", "AY1", "D"}},
      {"kind", {"K", "AY1", "N", "D"}},
      {"head", {"HH", "EH1", "D"}},
      {"house", {"HH", "AW1", "S"}},
      {"service", {"S", "ER1", "V", "AH0", "S"}},
      {"friend", {"F", "R", "EH1", "N", "D"}},
      {"father", {"F", "AA1", "DH", "ER0"}},
      {"power", {"P", "AW1", "ER0"}},
      {"hour", {"AW1", "ER0"}},
      {"game", {"G", "EY1", "M"}},
      {"line", {"L", "AY1", "N"}},
      {"end", {"EH1", "N", "D"}},
      {"member", {"M", "EH1", "M", "B", "ER0"}},
      {"law", {"L", "AO1"}},
      {"car", {"K", "AA1", "R"}},
      {"city", {"S", "IH1", "T", "IY0"}},
      {"community", {"K", "AH0", "M", "Y", "UW1", "N", "AH0", "T", "IY0"}},
      {"name", {"N", "EY1", "M"}},
      {"president", {"P", "R", "EH1", "Z", "IH0", "D", "AH0", "N", "T"}},
      {"team", {"T", "IY1", "M"}},
      {"minute", {"M", "IH1", "N", "AH0", "T"}},
      {"idea", {"AY0", "D", "IY1", "AH0"}},
      {"kid", {"K", "IH1", "D"}},
      {"body", {"B", "AA1", "D", "IY0"}},
      {"information", {"IH2", "N", "F", "ER0", "M", "EY1", "SH", "AH0", "N"}},
      {"back", {"B", "AE1", "K"}},
      {"parent", {"P", "EH1", "R", "AH0", "N", "T"}},
      {"face", {"F", "EY1", "S"}},
      {"others", {"AH1", "DH", "ER0", "Z"}},
      {"level", {"L", "EH1", "V", "AH0", "L"}},
      {"office", {"AO1", "F", "AH0", "S"}},
      {"door", {"D", "AO1", "R"}},
      {"health", {"HH", "EH1", "L", "TH"}},
      {"person", {"P", "ER1", "S", "AH0", "N"}},
      {"art", {"AA1", "R", "T"}},
      {"war", {"W", "AO1", "R"}},
      {"history", {"HH", "IH1", "S", "T", "ER0", "IY0"}},
      {"party", {"P", "AA1", "R", "T", "IY0"}},
      {"result", {"R", "IH0", "Z", "AH1", "L", "T"}},
      {"change", {"CH", "EY1", "N", "JH"}},
      {"morning", {"M", "AO1", "R", "N", "IH0", "NG"}},
      {"reason", {"R", "IY1", "Z", "AH0", "N"}},
      {"research", {"R", "IY0", "S", "ER1", "CH"}},
      {"girl", {"G", "ER1", "L"}},
      {"guy", {"G", "AY1"}},
      {"moment", {"M", "OW1", "M", "AH0", "N", "T"}},
      {"air", {"EH1", "R"}},
      {"teacher", {"T", "IY1", "CH", "ER0"}},
      {"force", {"F", "AO1", "R", "S"}},
      {"education", {"EH2", "JH", "AH0", "K", "EY1", "SH", "AH0", "N"}},

      // Common adjectives
      {"good", {"G", "UH1", "D"}},
      {"new", {"N", "UW1"}},
      {"first", {"F", "ER1", "S", "T"}},
      {"last", {"L", "AE1", "S", "T"}},
      {"long", {"L", "AO1", "NG"}},
      {"great", {"G", "R", "EY1", "T"}},
      {"little", {"L", "IH1", "T", "AH0", "L"}},
      {"own", {"OW1", "N"}},
      {"other", {"AH1", "DH", "ER0"}},
      {"old", {"OW1", "L", "D"}},
      {"right", {"R", "AY1", "T"}},
      {"big", {"B", "IH1", "G"}},
      {"high", {"HH", "AY1"}},
      {"different", {"D", "IH1", "F", "ER0", "AH0", "N", "T"}},
      {"small", {"S", "M", "AO1", "L"}},
      {"large", {"L", "AA1", "R", "JH"}},
      {"next", {"N", "EH1", "K", "S", "T"}},
      {"early", {"ER1", "L", "IY0"}},
      {"young", {"Y", "AH1", "NG"}},
      {"important", {"IH0", "M", "P", "AO1", "R", "T", "AH0", "N", "T"}},
      {"few", {"F", "Y", "UW1"}},
      {"public", {"P", "AH1", "B", "L", "IH0", "K"}},
      {"bad", {"B", "AE1", "D"}},
      {"same", {"S", "EY1", "M"}},
      {"able", {"EY1", "B", "AH0", "L"}},

      // Common adverbs
      {"not", {"N", "AA1", "T"}},
      {"so", {"S", "OW1"}},
      {"then", {"DH", "EH1", "N"}},
      {"now", {"N", "AW1"}},
      {"only", {"OW1", "N", "L", "IY0"}},
      {"very", {"V", "EH1", "R", "IY0"}},
      {"also", {"AO1", "L", "S", "OW0"}},
      {"just", {"JH", "AH1", "S", "T"}},
      {"well", {"W", "EH1", "L"}},
      {"even", {"IY1", "V", "AH0", "N"}},
      {"back", {"B", "AE1", "K"}},
      {"there", {"DH", "EH1", "R"}},
      {"how", {"HH", "AW1"}},
      {"too", {"T", "UW1"}},
      {"here", {"HH", "IH1", "R"}},
      {"where", {"W", "EH1", "R"}},
      {"why", {"W", "AY1"}},
      {"when", {"W", "EH1", "N"}},
      {"still", {"S", "T", "IH1", "L"}},
      {"again", {"AH0", "G", "EH1", "N"}},
      {"always", {"AO1", "L", "W", "EY2", "Z"}},
      {"never", {"N", "EH1", "V", "ER0"}},
      {"today", {"T", "AH0", "D", "EY1"}},
      {"together", {"T", "AH0", "G", "EH1", "DH", "ER0"}},
      {"already", {"AO0", "L", "R", "EH1", "D", "IY0"}},
      {"however", {"HH", "AW0", "EH1", "V", "ER0"}},
      {"often", {"AO1", "F", "AH0", "N"}},
      {"quite", {"K", "W", "AY1", "T"}},
      {"almost", {"AO1", "L", "M", "OW2", "S", "T"}},
      {"once", {"W", "AH1", "N", "S"}},
      {"really", {"R", "IY1", "L", "IY0"}},
      {"perhaps", {"P", "ER0", "HH", "AE1", "P", "S"}},
      {"probably", {"P", "R", "AA1", "B", "AH0", "B", "L", "IY0"}},
      {"certainly", {"S", "ER1", "T", "AH0", "N", "L", "IY0"}},

      // Pronouns
      {"i", {"AY1"}},
      {"you", {"Y", "UW1"}},
      {"he", {"HH", "IY1"}},
      {"she", {"SH", "IY1"}},
      {"it", {"IH1", "T"}},
      {"we", {"W", "IY1"}},
      {"they", {"DH", "EY1"}},
      {"me", {"M", "IY1"}},
      {"him", {"HH", "IH1", "M"}},
      {"her", {"HH", "ER1"}},
      {"us", {"AH1", "S"}},
      {"them", {"DH", "EH1", "M"}},
      {"my", {"M", "AY1"}},
      {"your", {"Y", "AO1", "R"}},
      {"his", {"HH", "IH1", "Z"}},
      {"its", {"IH1", "T", "S"}},
      {"our", {"AW1", "ER0"}},
      {"their", {"DH", "EH1", "R"}},
      {"this", {"DH", "IH1", "S"}},
      {"that", {"DH", "AE1", "T"}},
      {"these", {"DH", "IY1", "Z"}},
      {"those", {"DH", "OW1", "Z"}},
      {"what", {"W", "AA1", "T"}},
      {"which", {"W", "IH1", "CH"}},
      {"who", {"HH", "UW1"}},
      {"whom", {"HH", "UW1", "M"}},
      {"whose", {"HH", "UW1", "Z"}},

      // Prepositions
      {"of", {"AH1", "V"}},
      {"in", {"IH1", "N"}},
      {"to", {"T", "UW1"}},
      {"for", {"F", "AO1", "R"}},
      {"with", {"W", "IH1", "DH"}},
      {"on", {"AA1", "N"}},
      {"at", {"AE1", "T"}},
      {"from", {"F", "R", "AA1", "M"}},
      {"by", {"B", "AY1"}},
      {"about", {"AH0", "B", "AW1", "T"}},
      {"as", {"AE1", "Z"}},
      {"into", {"IH1", "N", "T", "UW0"}},
      {"through", {"TH", "R", "UW1"}},
      {"during", {"D", "UH1", "R", "IH0", "NG"}},
      {"before", {"B", "IH0", "F", "AO1", "R"}},
      {"after", {"AE1", "F", "T", "ER0"}},
      {"above", {"AH0", "B", "AH1", "V"}},
      {"below", {"B", "IH0", "L", "OW1"}},
      {"between", {"B", "IH0", "T", "W", "IY1", "N"}},
      {"under", {"AH1", "N", "D", "ER0"}},
      {"since", {"S", "IH1", "N", "S"}},
      {"without", {"W", "IH0", "DH", "AW1", "T"}},
      {"within", {"W", "IH0", "DH", "IH1", "N"}},
      {"along", {"AH0", "L", "AO1", "NG"}},
      {"toward", {"T", "AO1", "R", "D"}},
      {"against", {"AH0", "G", "EH1", "N", "S", "T"}},
      {"among", {"AH0", "M", "AH1", "NG"}},
      {"throughout", {"TH", "R", "UW0", "AW1", "T"}},
      {"despite", {"D", "IH0", "S", "P", "AY1", "T"}},
      {"upon", {"AH0", "P", "AA1", "N"}},
      {"across", {"AH0", "K", "R", "AO1", "S"}},
      {"behind", {"B", "IH0", "HH", "AY1", "N", "D"}},
      {"beyond", {"B", "IH0", "AA1", "N", "D"}},

      // Conjunctions
      {"and", {"AH0", "N", "D"}},
      {"or", {"AO1", "R"}},
      {"but", {"B", "AH1", "T"}},
      {"if", {"IH1", "F"}},
      {"because", {"B", "IH0", "K", "AO1", "Z"}},
      {"while", {"W", "AY1", "L"}},
      {"though", {"DH", "OW1"}},
      {"although", {"AO0", "L", "DH", "OW1"}},
      {"unless", {"AH0", "N", "L", "EH1", "S"}},
      {"until", {"AH0", "N", "T", "IH1", "L"}},
      {"whether", {"W", "EH1", "DH", "ER0"}},
      {"nor", {"N", "AO1", "R"}},

      // Greetings and common phrases
      {"hello", {"HH", "AH0", "L", "OW1"}},
      {"hi", {"HH", "AY1"}},
      {"hey", {"HH", "EY1"}},
      {"goodbye", {"G", "UH0", "D", "B", "AY1"}},
      {"bye", {"B", "AY1"}},
      {"thanks", {"TH", "AE1", "NG", "K", "S"}},
      {"thank", {"TH", "AE1", "NG", "K"}},
      {"please", {"P", "L", "IY1", "Z"}},
      {"sorry", {"S", "AA1", "R", "IY0"}},
      {"yes", {"Y", "EH1", "S"}},
      {"no", {"N", "OW1"}},
      {"ok", {"OW1", "K", "EY1"}},
      {"okay", {"OW1", "K", "EY1"}},
      {"sure", {"SH", "UH1", "R"}},
      {"welcome", {"W", "EH1", "L", "K", "AH0", "M"}},

      // Numbers
      {"one", {"W", "AH1", "N"}},
      {"two", {"T", "UW1"}},
      {"three", {"TH", "R", "IY1"}},
      {"four", {"F", "AO1", "R"}},
      {"five", {"F", "AY1", "V"}},
      {"six", {"S", "IH1", "K", "S"}},
      {"seven", {"S", "EH1", "V", "AH0", "N"}},
      {"eight", {"EY1", "T"}},
      {"nine", {"N", "AY1", "N"}},
      {"ten", {"T", "EH1", "N"}},
      {"eleven", {"IH0", "L", "EH1", "V", "AH0", "N"}},
      {"twelve", {"T", "W", "EH1", "L", "V"}},
      {"thirteen", {"TH", "ER1", "T", "IY2", "N"}},
      {"fourteen", {"F", "AO1", "R", "T", "IY2", "N"}},
      {"fifteen", {"F", "IH1", "F", "T", "IY2", "N"}},
      {"sixteen", {"S", "IH1", "K", "S", "T", "IY2", "N"}},
      {"seventeen", {"S", "EH1", "V", "AH0", "N", "T", "IY2", "N"}},
      {"eighteen", {"EY1", "T", "IY2", "N"}},
      {"nineteen", {"N", "AY1", "N", "T", "IY2", "N"}},
      {"twenty", {"T", "W", "EH1", "N", "T", "IY0"}},
      {"thirty", {"TH", "ER1", "T", "IY0"}},
      {"forty", {"F", "AO1", "R", "T", "IY0"}},
      {"fifty", {"F", "IH1", "F", "T", "IY0"}},
      {"sixty", {"S", "IH1", "K", "S", "T", "IY0"}},
      {"seventy", {"S", "EH1", "V", "AH0", "N", "T", "IY0"}},
      {"eighty", {"EY1", "T", "IY0"}},
      {"ninety", {"N", "AY1", "N", "T", "IY0"}},
      {"hundred", {"HH", "AH1", "N", "D", "R", "AH0", "D"}},
      {"thousand", {"TH", "AW1", "Z", "AH0", "N", "D"}},
      {"million", {"M", "IH1", "L", "Y", "AH0", "N"}},
      {"billion", {"B", "IH1", "L", "Y", "AH0", "N"}},
      {"first", {"F", "ER1", "S", "T"}},
      {"second", {"S", "EH1", "K", "AH0", "N", "D"}},
      {"third", {"TH", "ER1", "D"}},

      // Chinese names (common romanizations)
      {"yang", {"Y", "AE1", "NG"}},
      {"ming", {"M", "IH1", "NG"}},
      {"wang", {"W", "AA1", "NG"}},
      {"li", {"L", "IY1"}},
      {"zhang", {"JH", "AA1", "NG"}},
      {"liu", {"L", "Y", "UW1"}},
      {"chen", {"CH", "EH1", "N"}},
      {"huang", {"HH", "W", "AA1", "NG"}},
      {"zhao", {"JH", "AW1"}},
      {"wu", {"W", "UW1"}},
      {"zhou", {"JH", "OW1"}},
      {"xu", {"SH", "UW1"}},
      {"sun", {"S", "AH1", "N"}},
      {"ma", {"M", "AA1"}},
      {"zhu", {"JH", "UW1"}},
      {"hu", {"HH", "UW1"}},
      {"guo", {"G", "W", "OW1"}},
      {"he", {"HH", "AH0"}},
      {"gao", {"G", "AW1"}},
      {"lin", {"L", "IH1", "N"}},
      {"wei", {"W", "EY1"}},
      {"lei", {"L", "EY1"}},
      {"fang", {"F", "AE1", "NG"}},
      {"xiao", {"SH", "AW1"}},
      {"ying", {"IH1", "NG"}},
  };

  std::cout << "Loaded built-in English dictionary with " << en_dict_.size()
            << " entries" << std::endl;
}

bool TextPreprocessor::load_cmu_dict(const std::string &path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    std::cerr << "Failed to open CMU dictionary file: " << path << std::endl;
    return false;
  }

  std::string line;
  int count = 0;

  while (std::getline(file, line)) {
    // Skip comments
    if (line.empty() || line[0] == '#' || line.substr(0, 3) == ";;;") {
      continue;
    }

    // Parse line: WORD  P H O N E M E S
    std::istringstream iss(line);
    std::string word;
    iss >> word;

    // Skip variant pronunciations (e.g., WORD(2), WORD(3))
    // Only keep the primary pronunciation (no parentheses)
    // This matches Python g2p_en behavior which uses primary pronunciation
    if (word.find('(') != std::string::npos) {
      continue;
    }

    // Convert word to lowercase
    std::transform(word.begin(), word.end(), word.begin(), ::tolower);

    // Read phonemes
    std::vector<std::string> phonemes;
    std::string phoneme;
    while (iss >> phoneme) {
      phonemes.push_back(phoneme);
    }

    if (!phonemes.empty()) {
      en_dict_[word] = phonemes;
      count++;
    }
  }

  file.close();
  return count > 0;
}

std::string TextPreprocessor::ipa_to_arpa(const std::string &ipa) {
  // Simple IPA to ARPA phoneme conversion
  // This is a basic mapping - full implementation would require comprehensive IPA parsing
  static const std::map<std::string, std::string> ipa_arpa_map = {
      {"ə", "AH0"}, {"ʌ", "AH1"}, {"æ", "AE1"}, {"ɑ", "AA1"},
      {"ɔ", "AO1"}, {"aʊ", "AW1"}, {"aɪ", "AY1"}, {"ɛ", "EH1"},
      {"eɪ", "EY1"}, {"ɝ", "ER1"}, {"ɪ", "IH1"}, {"i", "IY1"},
      {"oʊ", "OW1"}, {"ɔɪ", "OY1"}, {"ʊ", "UH1"}, {"u", "UW1"},
      {"b", "B"}, {"tʃ", "CH"}, {"d", "D"}, {"ð", "DH"},
      {"f", "F"}, {"ɡ", "G"}, {"h", "HH"}, {"dʒ", "JH"},
      {"k", "K"}, {"l", "L"}, {"m", "M"}, {"n", "N"},
      {"ŋ", "NG"}, {"p", "P"}, {"ɹ", "R"}, {"r", "R"},
      {"s", "S"}, {"ʃ", "SH"}, {"t", "T"}, {"θ", "TH"},
      {"v", "V"}, {"w", "W"}, {"j", "Y"}, {"z", "Z"},
      {"ʒ", "ZH"}
  };

  auto it = ipa_arpa_map.find(ipa);
  if (it != ipa_arpa_map.end()) {
    return it->second;
  }
  return ""; // Unknown IPA symbol
}

llama_token
TextPreprocessor::get_symbol_id(const std::string &symbol) {
  auto it = symbol_to_id_.find(symbol);
  if (it != symbol_to_id_.end()) {
    return it->second;
  }
  return symbol_to_id_["UNK"];
}

std::vector<llama_token> TextPreprocessor::cleaned_text_to_sequence(
    const std::vector<std::string> &cleaned_text, const std::string &version) {
  (void)version;
  std::vector<llama_token> phones;
  for (const auto &symbol : cleaned_text) {
    phones.push_back(get_symbol_id(symbol));
  }
  return phones;
}

bool TextPreprocessor::is_chinese(const std::string &text) {
  for (size_t i = 0; i < text.length();) {
    unsigned char c = static_cast<unsigned char>(text[i]);
    if (c >= 0xE4 && c <= 0xE9) {
      return true;
    }
    if ((c & 0x80) == 0)
      i += 1;
    else if ((c & 0xE0) == 0xC0)
      i += 2;
    else if ((c & 0xF0) == 0xE0)
      i += 3;
    else if ((c & 0xF8) == 0xF0)
      i += 4;
    else
      i += 1;
  }
  return false;
}

bool TextPreprocessor::is_english(const std::string &text) {
  std::regex eng_regex("[A-Za-z]");
  return std::regex_search(text, eng_regex);
}

std::string
TextPreprocessor::replace_punctuation(const std::string &text) {
  std::string result = text;
  static const std::map<std::string, std::string> rep_map = {
      {"：", ","}, {"；", ","}, {"，", ","}, {"。", "."},
      {"！", "!"}, {"？", "?"}, {"·", ","}, {"、", ","},
      {"...", "…"}, {"—", "-"}, {"~", "…"}, {"～", "…"}};

  for (const auto &pair : rep_map) {
    size_t pos = 0;
    while ((pos = result.find(pair.first, pos)) != std::string::npos) {
      result.replace(pos, pair.first.length(), pair.second);
      pos += pair.second.length();
    }
  }
  return result;
}

std::string
TextPreprocessor::chinese_num_to_text(const std::string &text) {
  // Simple number to Chinese text conversion
  // In production, use cn2an library or similar
  std::string result = text;

  static const std::map<std::string, std::string> num_map = {
      {"0", "零"}, {"1", "一"}, {"2", "二"}, {"3", "三"}, {"4", "四"},
      {"5", "五"}, {"6", "六"}, {"7", "七"}, {"8", "八"}, {"9", "九"}};

  for (const auto &pair : num_map) {
    size_t pos = 0;
    while ((pos = result.find(pair.first, pos)) != std::string::npos) {
      result.replace(pos, pair.first.length(), pair.second);
      pos += pair.second.length();
    }
  }

  return result;
}

std::string
TextPreprocessor::normalize_chinese_text(const std::string &text) {
  std::string result = text;

  // Replace punctuation
  result = replace_punctuation(result);

  // Convert numbers to text if enabled
  if (config_.normalize_numbers) {
    result = chinese_num_to_text(result);
  }

  // Remove consecutive punctuation
  std::regex punct_regex("([!?…,.\\-])\\1+");
  result = std::regex_replace(result, punct_regex, "$1");

  return result;
}

std::string
TextPreprocessor::normalize_text(const std::string &text,
                                  const std::string &lang) {
  if (lang == "zh") {
    return normalize_chinese_text(text);
  }
  // For English, minimal normalization
  return text;
}

bool TextPreprocessor::g2p_chinese(
    const std::string &text, std::vector<std::string> &out_phones,
    std::vector<int> &out_word2ph) {

#ifdef USE_CPP_PINYIN
  if (!pinyin_engine_) {
    std::cerr << "Pinyin engine not initialized" << std::endl;
    return false;
  }

  try {
    // Use cpp-pinyin to convert Chinese text to pinyin
    auto pinyin_res = pinyin_engine_->hanziToPinyin(
        text, Pinyin::ManTone::Style::TONE3, // TONE3 = digit after vowel (hao3)
        Pinyin::Error::Default,              // Keep original on error
        false,                               // Don't need candidates
        config_.v_to_u,                      // Convert v to ü
        false                                // Don't use 5 for neutral tone
    );

    for (const auto &res : pinyin_res) {
      if (res.error) {
        // Keep original character (probably punctuation)
        out_phones.push_back(res.hanzi);
        out_word2ph.push_back(1);
        continue;
      }

      std::string pinyin = res.pinyin;

      // Check if it's punctuation
      if (pinyin.length() == 1 && std::ispunct(pinyin[0])) {
        out_phones.push_back(pinyin);
        out_word2ph.push_back(1);
        continue;
      }

      // Extract tone (last character)
      char tone = pinyin.back();
      std::string pinyin_without_tone = pinyin.substr(0, pinyin.length() - 1);

      // Convert pinyin to consonant + vowel format
      // Try to find in mapping table
      if (pinyin_to_phones_.find(pinyin_without_tone) !=
          pinyin_to_phones_.end()) {
        auto [consonant, vowel_base] = pinyin_to_phones_[pinyin_without_tone];

        if (consonant != "_") {
          out_phones.push_back(consonant);
        }

        // Add tone to vowel
        std::string vowel_with_tone = vowel_base + std::string(1, tone);
        out_phones.push_back(vowel_with_tone);

        // word2ph: number of phones for this character
        int phone_count = (consonant == "_") ? 1 : 2;
        out_word2ph.push_back(phone_count);
      } else {
        // Fallback: use heuristic to split consonant and vowel
        // This is a simplified approach
        std::string consonant, vowel;

        // Common consonant patterns
        if (pinyin_without_tone.substr(0, 2) == "zh" ||
            pinyin_without_tone.substr(0, 2) == "ch" ||
            pinyin_without_tone.substr(0, 2) == "sh") {
          consonant = pinyin_without_tone.substr(0, 2);
          vowel = pinyin_without_tone.substr(2);
        } else if (!pinyin_without_tone.empty() &&
                   std::string("bpmfdtnlgkhjqxzcsryw").find(
                       pinyin_without_tone[0]) != std::string::npos) {
          consonant = pinyin_without_tone.substr(0, 1);
          vowel = pinyin_without_tone.substr(1);
        } else {
          // No consonant (e.g., "a", "e", "ai")
          consonant = "";
          vowel = pinyin_without_tone;
        }

        if (!consonant.empty()) {
          out_phones.push_back(consonant);
        }

        if (!vowel.empty()) {
          out_phones.push_back(vowel + std::string(1, tone));
        }

        int phone_count = consonant.empty() ? 1 : 2;
        out_word2ph.push_back(phone_count);
      }
    }

    return true;
  } catch (const std::exception &e) {
    std::cerr << "G2P Chinese error: " << e.what() << std::endl;
    return false;
  }
#else
  std::cerr << "cpp-pinyin not available, cannot process Chinese text" << std::endl;
  return false;
#endif
}

bool TextPreprocessor::g2p_english(
    const std::string &text, std::vector<std::string> &out_phones) {

#ifdef USE_ESPEAK
  if (config_.use_espeak && espeak_handle_ && espeak_handle_->initialized) {
    // Use espeak-ng to get phonemes
    const char *phonemes =
        espeak_TextToPhonemes((const void **)&text[0], espeakCHARS_UTF8,
                              espeakPHONEMES_IPA);

    if (phonemes) {
      // Parse IPA phonemes and convert to ARPA format
      std::string phone_str(phonemes);

      // Split by word boundaries and convert IPA to ARPA
      std::istringstream iss(text);
      std::string word;
      while (iss >> word) {
        // Try dictionary first
        std::string clean_word;
        for (char c : word) {
          if (std::isalpha(c)) {
            clean_word += std::tolower(c);
          } else if (c == '\'') {
            // Keep apostrophes for contractions
            clean_word += c;
          }
        }

        if (!clean_word.empty() && en_dict_.count(clean_word)) {
          const auto &phs = en_dict_.at(clean_word);
          out_phones.insert(out_phones.end(), phs.begin(), phs.end());
        } else {
          // Use espeak phonemes (simplified conversion)
          // In production, implement proper IPA to ARPA conversion
          out_phones.push_back("UNK");
        }
      }

      return true;
    }
  }
#endif

  // Dictionary-based approach using en_dict_
  std::istringstream iss(text);
  std::string word;
  while (iss >> word) {
    std::string clean_word;
    for (char c : word) {
      if (std::isalpha(c)) {
        clean_word += std::tolower(c);
      } else if (c == '\'') {
        // Keep apostrophes for contractions (you're, it's, etc.)
        clean_word += c;
      } else if (std::ispunct(c)) {
        if (!clean_word.empty()) {
          if (en_dict_.count(clean_word)) {
            const auto &phs = en_dict_.at(clean_word);
            out_phones.insert(out_phones.end(), phs.begin(), phs.end());
          } else {
            // Unknown word - add placeholder
            out_phones.push_back("UNK");
          }
          clean_word.clear();
        }
        // Add punctuation as-is
        out_phones.push_back(std::string(1, c));
      }
    }
    if (!clean_word.empty() && en_dict_.count(clean_word)) {
      const auto &phs = en_dict_.at(clean_word);
      out_phones.insert(out_phones.end(), phs.begin(), phs.end());
    } else if (!clean_word.empty()) {
      // Unknown word - add placeholder
      out_phones.push_back("UNK");
    }
  }

  return true;
}

void TextPreprocessor::g2p(const std::string &text,
                            const std::string &lang,
                            std::vector<std::string> &out_phones,
                            std::vector<int> &out_word2ph) {
  out_phones.clear();
  out_word2ph.clear();

  if (lang == "zh") {
    if (!g2p_chinese(text, out_phones, out_word2ph)) {
      // Fallback
      out_phones.push_back("UNK");
      out_word2ph.push_back(1);
    }
  } else if (lang == "en") {
    if (!g2p_english(text, out_phones)) {
      out_phones.push_back("UNK");
    }
    // English doesn't use word2ph in Python implementation
    out_word2ph.clear();
  }
}

std::tuple<std::vector<llama_token>, std::vector<int>, std::string>
TextPreprocessor::clean_text_inf(const std::string &text,
                                  const std::string &language,
                                  const std::string &version) {
  std::string norm_text = normalize_text(text, language);
  std::vector<std::string> phone_strs;
  std::vector<int> word2ph;

  g2p(norm_text, language, phone_strs, word2ph);

  // Filter phones
  std::vector<std::string> filtered_phones;
  for (const auto &ph : phone_strs) {
    if (symbol_to_id_.find(ph) != symbol_to_id_.end()) {
      filtered_phones.push_back(ph);
    } else {
      filtered_phones.push_back("UNK");
    }
  }

  std::vector<llama_token> phones =
      cleaned_text_to_sequence(filtered_phones, version);

  return std::make_tuple(phones, word2ph, norm_text);
}

bool TextPreprocessor::get_bert_feature(
    const std::string &norm_text, const std::vector<int> &word2ph,
    std::vector<float> &out_features) {

  if (!bert_model_) {
    return false;
  }

  // Same implementation as before...
  std::vector<llama_token> tokens;
  if (!bert_model_->encode_text(norm_text, tokens, true)) {
    return false;
  }

  if (!bert_model_->eval_chunk(tokens.data(), nullptr, tokens.size(), true, 0,
                               false)) {
    return false;
  }

  std::vector<float> hidden_states;
  if (!bert_model_->get_last_hidden_state(hidden_states)) {
    return false;
  }

  int n_embd = 1024;
  int seq_len = tokens.size();

  if (hidden_states.size() != static_cast<size_t>(n_embd * seq_len)) {
    return false;
  }

  // Remove CLS and SEP
  std::vector<float> trimmed_states;
  for (int i = 1; i < seq_len - 1; ++i) {
    for (int j = 0; j < n_embd; ++j) {
      trimmed_states.push_back(hidden_states[i * n_embd + j]);
    }
  }

  // Align with word2ph
  if (word2ph.size() != norm_text.size()) {
    return false;
  }

  out_features.clear();
  for (size_t i = 0; i < word2ph.size(); ++i) {
    int repeat_count = word2ph[i];
    for (int r = 0; r < repeat_count; ++r) {
      for (int j = 0; j < n_embd; ++j) {
        out_features.push_back(trimmed_states[i * n_embd + j]);
      }
    }
  }

  return true;
}

std::vector<float> TextPreprocessor::get_bert_inf(
    const std::vector<llama_token> &phones, const std::vector<int> &word2ph,
    const std::string &norm_text, const std::string &language) {

  std::vector<float> feature;

  if (language == "zh") {
    if (!get_bert_feature(norm_text, word2ph, feature)) {
      feature.resize(1024 * phones.size(), 0.0f);
    }
  } else {
    feature.resize(1024 * phones.size(), 0.0f);
  }

  return feature;
}

std::tuple<std::vector<llama_token>, std::vector<float>, std::string>
TextPreprocessor::get_phones_and_bert(const std::string &text,
                                       const std::string &language,
                                       const std::string &version,
                                       bool final) {
  std::vector<llama_token> phones;
  std::vector<float> bert_features;
  std::string norm_text;

  // Same logic as before but using real G2P
  if (language == "en" || language == "all_zh" || language == "all_ja" ||
      language == "all_ko" || language == "all_yue") {
    std::string lang = language;
    if (lang.find("all_") == 0) {
      lang = lang.substr(4);
    }

    std::string formattext = text;

    if (lang == "zh") {
      auto [ph, w2ph, nt] = clean_text_inf(formattext, lang, version);
      phones = ph;
      norm_text = nt;
      bert_features = get_bert_inf(phones, w2ph, norm_text, lang);
    } else if (lang == "en") {
      auto [ph, w2ph, nt] = clean_text_inf(formattext, lang, version);
      phones = ph;
      norm_text = nt;
      bert_features.resize(1024 * phones.size(), 0.0f);
      // print_vector(phones, phones.size());
    }
  } else {
    auto [ph, w2ph, nt] = clean_text_inf(text, "zh", version);
    phones = ph;
    norm_text = nt;
    bert_features = get_bert_inf(phones, w2ph, norm_text, "zh");
  }

  if (!final && phones.size() < 6) {
    std::string padded_text = "." + text;
    return get_phones_and_bert(padded_text, language, version, true);
  }

  return std::make_tuple(phones, bert_features, norm_text);
}

TextPreprocessor::Result
TextPreprocessor::preprocess(const std::string &text,
                              const std::string &lang) {
  Result res;

  auto [phones, bert_features, norm_text] =
      get_phones_and_bert(text, lang, "v2", false);

  res.phones = phones;
  res.bert_features = bert_features;
  res.norm_text = norm_text;

  return res;
}
