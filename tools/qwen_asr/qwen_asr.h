#pragma once

#include "audio_encoder.h"
#include "mel_spectrogram.h"
#include "llama.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

#include "bpe_tokenizer.h"

struct QwenASRParams {
    std::string model_dir;           // Directory containing model files
    std::string audio_encoder_path;  // Audio encoder GGUF
    std::string decoder_path;        // Text decoder GGUF (llama.cpp format)
    std::string vocab_path;          // vocab.json
    std::string merges_path;         // merges.txt
    std::string mel_filters_path;    // mel_filters_whisper.npy (optional)
    std::string device = "CPU";      // Backend device for audio encoder
    int n_threads = 4;
    int n_gpu_layers = 0;            // Decoder layers on NPU (0=CPU, 28=full NPU)
    int max_new_tokens = 256;
};

class QwenASR {
public:
    QwenASR() = default;
    ~QwenASR();

    bool load(const QwenASRParams &params);

    // Transcribe audio file → text
    bool transcribe(const std::string &audio_path, std::string &output_text);

    // Transcribe audio samples → text
    bool transcribe(const std::vector<float> &audio_16k, std::string &output_text);

    // Transcribe from pre-computed audio features
    bool transcribe_from_features(const std::vector<float> &audio_features,
                                   int num_audio_frames, std::string &output_text);

    // Access audio encoder
    AudioEncoder &get_audio_encoder() { return audio_encoder_; }

private:
    // Components
    MelSpectrogram mel_spec_;
    AudioEncoder audio_encoder_;
    std::unique_ptr<BpeTokenizer> tokenizer_;

    // llama.cpp decoder
    llama_model *llama_model_ = nullptr;
    llama_context *llama_ctx_ = nullptr;
    int n_embd_ = 0;

    // Token IDs
    int audio_start_id_ = 151669;
    int audio_end_id_ = 151670;
    int audio_pad_id_ = 151676;
    int im_start_id_ = 151644;
    int im_end_id_ = 151645;
    int endoftext_id_ = 151643;

    // Reverse vocab for decoding (id → token bytes)
    std::unordered_map<int, std::string> id_to_bytes_;
    uint8_t unicode_to_byte_[512];
    std::string vocab_path_;

    int max_new_tokens_ = 256;

    // eval_chunk: feed tokens OR embeddings to llama.cpp (following gptsovits/llm.h pattern)
    bool eval_chunk(llama_token *tokens, float *embd, int n_tokens,
                    bool is_last, llama_pos pos_offset = 0);

    // Build prompt token segments (pre-audio, post-audio)
    void build_prompt_segments(int num_audio_frames,
                               std::vector<int> &pre_audio_tokens,
                               std::vector<int> &post_audio_tokens);

    // Decode token IDs to text string
    std::string decode_tokens(const std::vector<int> &token_ids);

    // Initialize reverse vocabulary from vocab.json
    void init_reverse_vocab(const std::string &vocab_path);

    // Convert GPT-2 BPE unicode string to raw bytes
    std::string bpe_unicode_to_bytes(const std::string &bpe_str);
};
