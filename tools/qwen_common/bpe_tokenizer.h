#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <utility>

// Qwen2-style GPT-2 byte-level BPE tokenizer
// Loads vocab.json + merges.txt and provides encode(text) -> vector<int>
class BpeTokenizer {
public:
    BpeTokenizer() = default;

    // Load vocabulary and merge rules
    bool load(const std::string &vocab_path, const std::string &merges_path);

    // Encode text to token IDs
    std::vector<int> encode(const std::string &text) const;

    // Get token ID for a known token string (e.g. special tokens)
    int token_to_id(const std::string &token) const;

    // Check if loaded
    bool is_loaded() const { return loaded_; }

    // Special token IDs
    int im_start_id() const { return token_to_id("<|im_start|>"); }
    int im_end_id() const { return token_to_id("<|im_end|>"); }
    int endoftext_id() const { return token_to_id("<|endoftext|>"); }

private:
    bool loaded_ = false;

    // Token string -> ID
    std::unordered_map<std::string, int> vocab_;

    // Merge rules: (token_a, token_b) -> priority (lower = higher priority)
    std::unordered_map<std::string, int> merge_ranks_;

    // Special tokens that should be matched before BPE
    std::vector<std::pair<std::string, int>> special_tokens_;

    // GPT-2 byte-to-unicode mapping (256 entries, each is a UTF-8 string)
    std::string byte_to_unicode_[256];

    // Initialize byte_to_unicode mapping
    void init_byte_mapping();

    // Convert raw bytes to BPE-encoded unicode string
    std::string bytes_to_bpe_str(const std::string &raw) const;

    // Pre-tokenize text into chunks (simplified GPT-2 regex)
    std::vector<std::string> pre_tokenize(const std::string &text) const;

    // Apply BPE merges to a single pre-tokenized chunk
    std::vector<std::string> bpe(const std::string &token) const;
};
