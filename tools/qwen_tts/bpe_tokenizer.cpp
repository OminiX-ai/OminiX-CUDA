#include "bpe_tokenizer.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>

// ============================================================================
// UTF-8 helpers
// ============================================================================

// Encode a unicode codepoint to UTF-8 string
static std::string codepoint_to_utf8(uint32_t cp) {
    std::string s;
    if (cp < 0x80) {
        s += (char)cp;
    } else if (cp < 0x800) {
        s += (char)(0xC0 | (cp >> 6));
        s += (char)(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18));
        s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F));
        s += (char)(0x80 | (cp & 0x3F));
    }
    return s;
}

// Get length of UTF-8 character at position
static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if (c < 0xC0) return 1;  // continuation byte
    if (c < 0xE0) return 2;
    if (c < 0xF0) return 3;
    return 4;
}

// Decode one UTF-8 character, return codepoint and advance pos
static uint32_t utf8_decode(const std::string &s, size_t &pos) {
    unsigned char c = (unsigned char)s[pos];
    uint32_t cp;
    int len;
    if (c < 0x80) { cp = c; len = 1; }
    else if (c < 0xE0) { cp = c & 0x1F; len = 2; }
    else if (c < 0xF0) { cp = c & 0x0F; len = 3; }
    else { cp = c & 0x07; len = 4; }
    for (int i = 1; i < len && pos + i < s.size(); i++) {
        cp = (cp << 6) | ((unsigned char)s[pos + i] & 0x3F);
    }
    pos += len;
    return cp;
}

// ============================================================================
// Unicode character classification (simplified)
// ============================================================================

static bool is_letter(uint32_t cp) {
    // ASCII letters
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z')) return true;
    // Latin Extended
    if (cp >= 0xC0 && cp <= 0x024F) return true;
    // CJK Unified Ideographs
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    // CJK Extension A
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    // CJK Compatibility Ideographs
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    // Hiragana
    if (cp >= 0x3040 && cp <= 0x309F) return true;
    // Katakana
    if (cp >= 0x30A0 && cp <= 0x30FF) return true;
    // Hangul
    if (cp >= 0xAC00 && cp <= 0xD7AF) return true;
    // Arabic
    if (cp >= 0x0600 && cp <= 0x06FF) return true;
    // Cyrillic
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    // Greek
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    // Thai
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;
    // Devanagari
    if (cp >= 0x0900 && cp <= 0x097F) return true;
    // General: Unicode letters in various blocks
    if (cp >= 0x00C0 && cp <= 0x00FF && cp != 0x00D7 && cp != 0x00F7) return true;
    return false;
}

static bool is_digit(uint32_t cp) {
    return (cp >= '0' && cp <= '9');
}

static bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' ||
           cp == '\f' || cp == '\v' || cp == 0x00A0 || cp == 0x2000 ||
           cp == 0x2001 || cp == 0x200B || cp == 0x3000;
}

static bool is_newline(uint32_t cp) {
    return cp == '\n' || cp == '\r';
}

// ============================================================================
// GPT-2 byte-to-unicode mapping
// ============================================================================

void BpeTokenizer::init_byte_mapping() {
    // Build the standard GPT-2 bytes_to_unicode mapping
    // Printable bytes map to themselves, others map to U+0100+
    std::vector<int> bs;
    // ! to ~ (33-126)
    for (int i = 33; i <= 126; i++) bs.push_back(i);
    // Latin-1 Supplement printable (161-172, 174-255)
    for (int i = 161; i <= 172; i++) bs.push_back(i);
    for (int i = 174; i <= 255; i++) bs.push_back(i);

    // Mark which bytes are in the "direct" set
    bool direct[256] = {};
    for (int b : bs) direct[b] = true;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        if (direct[b]) {
            byte_to_unicode_[b] = codepoint_to_utf8((uint32_t)b);
        } else {
            byte_to_unicode_[b] = codepoint_to_utf8(256 + n);
            n++;
        }
    }
}

// ============================================================================
// Load vocabulary and merges
// ============================================================================

bool BpeTokenizer::load(const std::string &vocab_path,
                         const std::string &merges_path) {
    init_byte_mapping();

    // Load vocab.json
    {
        std::ifstream f(vocab_path);
        if (!f.is_open()) {
            printf("[tokenizer] cannot open vocab: %s\n", vocab_path.c_str());
            return false;
        }
        nlohmann::json j;
        f >> j;
        for (auto &[key, val] : j.items()) {
            vocab_[key] = val.get<int>();
        }
        printf("[tokenizer] loaded %zu vocab entries\n", vocab_.size());
    }

    // Load added/special tokens from tokenizer_config.json (same directory as vocab)
    {
        // Derive config path from vocab path
        std::string config_path = vocab_path;
        size_t slash = config_path.rfind('/');
        if (slash != std::string::npos) {
            config_path = config_path.substr(0, slash + 1) + "tokenizer_config.json";
        } else {
            config_path = "tokenizer_config.json";
        }
        std::ifstream cf(config_path);
        if (cf.is_open()) {
            nlohmann::json cfg;
            cf >> cfg;
            if (cfg.contains("added_tokens_decoder")) {
                for (auto &[tid_str, info] : cfg["added_tokens_decoder"].items()) {
                    int tid = std::atoi(tid_str.c_str());
                    std::string content = info["content"].get<std::string>();
                    vocab_[content] = tid;
                    special_tokens_.push_back({content, tid});
                }
            }
            printf("[tokenizer] loaded %zu special tokens\n", special_tokens_.size());
        }
    }

    // Sort special tokens by length (longest first) for greedy matching
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const auto &a, const auto &b) {
                  return a.first.size() > b.first.size();
              });

    // Load merges.txt
    {
        std::ifstream f(merges_path);
        if (!f.is_open()) {
            printf("[tokenizer] cannot open merges: %s\n", merges_path.c_str());
            return false;
        }
        std::string line;
        int rank = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            // Each line is "token_a token_b"
            size_t space = line.find(' ');
            if (space == std::string::npos) continue;
            std::string key = line;  // Use the full "a b" string as key
            merge_ranks_[key] = rank++;
        }
        printf("[tokenizer] loaded %zu merges\n", merge_ranks_.size());
    }

    loaded_ = true;
    return true;
}

// ============================================================================
// Convert raw bytes to BPE-encoded unicode string
// ============================================================================

std::string BpeTokenizer::bytes_to_bpe_str(const std::string &raw) const {
    std::string result;
    for (unsigned char c : raw) {
        result += byte_to_unicode_[c];
    }
    return result;
}

// ============================================================================
// Pre-tokenize: split text into chunks
// Simplified version of GPT-2 regex pattern
// ============================================================================

std::vector<std::string> BpeTokenizer::pre_tokenize(const std::string &text) const {
    std::vector<std::string> chunks;
    size_t pos = 0;
    size_t len = text.size();

    while (pos < len) {
        uint32_t cp = 0;
        size_t start = pos;

        // Peek at current codepoint
        size_t peek_pos = pos;
        cp = utf8_decode(text, peek_pos);

        // Pattern 1: Contractions 's 't 're 've 'm 'll 'd (case-insensitive)
        if (cp == '\'' && pos + 1 < len) {
            char next = text[pos + 1];
            char lower_next = (next >= 'A' && next <= 'Z') ? next + 32 : next;
            int contraction_len = 0;
            if (lower_next == 's' || lower_next == 't' || lower_next == 'm' || lower_next == 'd') {
                contraction_len = 2;
            } else if (pos + 2 < len) {
                char next2 = text[pos + 2];
                char lower_next2 = (next2 >= 'A' && next2 <= 'Z') ? next2 + 32 : next2;
                if ((lower_next == 'r' && lower_next2 == 'e') ||
                    (lower_next == 'v' && lower_next2 == 'e') ||
                    (lower_next == 'l' && lower_next2 == 'l')) {
                    contraction_len = 3;
                }
            }
            if (contraction_len > 0) {
                chunks.push_back(text.substr(pos, contraction_len));
                pos += contraction_len;
                continue;
            }
        }

        // Pattern 2: Optional non-letter-non-digit followed by letters
        // [^\r\n\p{L}\p{N}]?\p{L}+
        // The prefix can be space, punctuation, etc. — anything except \r\n, letters, digits
        if (is_letter(cp) || (!is_letter(cp) && !is_digit(cp) && !is_newline(cp))) {
            size_t chunk_start = pos;
            // Check if first char is non-letter-non-digit (optional prefix)
            if (!is_letter(cp) && !is_digit(cp) && !is_newline(cp)) {
                pos = peek_pos;  // consume the prefix char
                if (pos < len) {
                    peek_pos = pos;
                    cp = utf8_decode(text, peek_pos);
                }
            }
            // Now consume letters
            if (is_letter(cp)) {
                pos = peek_pos;
                while (pos < len) {
                    peek_pos = pos;
                    cp = utf8_decode(text, peek_pos);
                    if (!is_letter(cp)) break;
                    pos = peek_pos;
                }
                chunks.push_back(text.substr(chunk_start, pos - chunk_start));
                continue;
            }
            // If no letters followed the prefix, reset and try other patterns
            pos = chunk_start;
            peek_pos = pos;
            cp = utf8_decode(text, peek_pos);
        }

        // Pattern 3: Single digit \p{N}
        if (is_digit(cp)) {
            pos = peek_pos;
            chunks.push_back(text.substr(start, pos - start));
            continue;
        }

        // Pattern 4: Optional space + punctuation + optional newlines
        // ?[^\s\p{L}\p{N}]+[\r\n]*
        if (cp == ' ' || (!is_whitespace(cp) && !is_letter(cp) && !is_digit(cp))) {
            size_t chunk_start = pos;
            if (cp == ' ') {
                pos = peek_pos;
                if (pos >= len) {
                    chunks.push_back(text.substr(chunk_start, pos - chunk_start));
                    continue;
                }
                peek_pos = pos;
                cp = utf8_decode(text, peek_pos);
            }
            // Consume non-whitespace, non-letter, non-digit
            bool has_punct = false;
            while (pos < len) {
                peek_pos = pos;
                cp = utf8_decode(text, peek_pos);
                if (is_whitespace(cp) || is_letter(cp) || is_digit(cp)) break;
                pos = peek_pos;
                has_punct = true;
            }
            if (has_punct) {
                // Consume trailing newlines
                while (pos < len && (text[pos] == '\n' || text[pos] == '\r')) {
                    pos++;
                }
                chunks.push_back(text.substr(chunk_start, pos - chunk_start));
                continue;
            }
            // Reset if no punctuation found
            pos = chunk_start;
            peek_pos = pos;
            cp = utf8_decode(text, peek_pos);
        }

        // Pattern 5: Whitespace with newlines \s*[\r\n]+
        if (is_newline(cp)) {
            size_t chunk_start = pos;
            while (pos < len && (is_whitespace(text[pos]) || is_newline(text[pos]))) {
                pos++;
            }
            chunks.push_back(text.substr(chunk_start, pos - chunk_start));
            continue;
        }

        // Pattern 6: Other whitespace \s+
        if (is_whitespace(cp)) {
            size_t chunk_start = pos;
            pos = peek_pos;
            while (pos < len) {
                peek_pos = pos;
                cp = utf8_decode(text, peek_pos);
                if (!is_whitespace(cp) || is_newline(cp)) break;
                pos = peek_pos;
            }
            chunks.push_back(text.substr(chunk_start, pos - chunk_start));
            continue;
        }

        // Fallback: single character
        pos = peek_pos;
        chunks.push_back(text.substr(start, pos - start));
    }

    return chunks;
}

// ============================================================================
// BPE: apply merge operations to a single token
// ============================================================================

std::vector<std::string> BpeTokenizer::bpe(const std::string &token) const {
    // Split token into individual unicode characters
    std::vector<std::string> word;
    size_t pos = 0;
    while (pos < token.size()) {
        int clen = utf8_char_len((unsigned char)token[pos]);
        word.push_back(token.substr(pos, clen));
        pos += clen;
    }

    if (word.size() <= 1) return word;

    while (true) {
        // Find the bigram with the lowest merge rank
        int best_rank = INT32_MAX;
        int best_idx = -1;
        for (int i = 0; i < (int)word.size() - 1; i++) {
            std::string bigram = word[i] + " " + word[i + 1];
            auto it = merge_ranks_.find(bigram);
            if (it != merge_ranks_.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;  // No more merges

        // Merge all occurrences of the best bigram
        std::string merged = word[best_idx] + word[best_idx + 1];
        std::vector<std::string> new_word;
        int i = 0;
        while (i < (int)word.size()) {
            if (i < (int)word.size() - 1 &&
                word[i] + " " + word[i + 1] == word[best_idx] + " " + word[best_idx + 1]) {
                new_word.push_back(merged);
                i += 2;
            } else {
                new_word.push_back(word[i]);
                i++;
            }
        }
        word = std::move(new_word);

        if (word.size() == 1) break;
    }

    return word;
}

// ============================================================================
// Encode: full text -> token IDs
// ============================================================================

int BpeTokenizer::token_to_id(const std::string &token) const {
    auto it = vocab_.find(token);
    if (it != vocab_.end()) return it->second;
    return -1;
}

std::vector<int> BpeTokenizer::encode(const std::string &text) const {
    std::vector<int> ids;
    if (!loaded_ || text.empty()) return ids;

    // First, split on special tokens
    std::vector<std::pair<std::string, int>> segments;  // (text, special_token_id or -1)
    std::string remaining = text;

    while (!remaining.empty()) {
        // Find earliest special token
        size_t earliest_pos = std::string::npos;
        int earliest_id = -1;
        size_t earliest_len = 0;

        for (auto &[tok, id] : special_tokens_) {
            size_t pos = remaining.find(tok);
            if (pos != std::string::npos &&
                (pos < earliest_pos || (pos == earliest_pos && tok.size() > earliest_len))) {
                earliest_pos = pos;
                earliest_id = id;
                earliest_len = tok.size();
            }
        }

        if (earliest_pos == std::string::npos) {
            // No more special tokens
            segments.push_back({remaining, -1});
            remaining.clear();
        } else {
            if (earliest_pos > 0) {
                segments.push_back({remaining.substr(0, earliest_pos), -1});
            }
            segments.push_back({"", earliest_id});
            remaining = remaining.substr(earliest_pos + earliest_len);
        }
    }

    // Process each segment
    for (auto &[seg_text, special_id] : segments) {
        if (special_id >= 0) {
            ids.push_back(special_id);
            continue;
        }

        // Pre-tokenize
        auto chunks = pre_tokenize(seg_text);

        // For each chunk: convert to byte-level BPE string, apply BPE, look up vocab
        for (auto &chunk : chunks) {
            std::string bpe_str = bytes_to_bpe_str(chunk);
            auto bpe_tokens = bpe(bpe_str);
            for (auto &tok : bpe_tokens) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end()) {
                    ids.push_back(it->second);
                } else {
                    // Unknown token - encode each byte separately
                    for (unsigned char c : chunk) {
                        auto byte_it = vocab_.find(byte_to_unicode_[c]);
                        if (byte_it != vocab_.end()) {
                            ids.push_back(byte_it->second);
                        }
                    }
                }
            }
        }
    }

    return ids;
}
