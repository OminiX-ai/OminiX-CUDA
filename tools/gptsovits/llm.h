#pragma once

#include "hftokenizer.hpp"
#include "llama.h"
#include "utils.h"
#include <string>

class Llm {
public:
    Llm();

    bool load_model(const std::string &model_path, LlmParam params);

    virtual std::string format_prompt(const std::string &prompt,
                                      const std::string &system_prompt = "");

    bool encode_text(const std::string &prompt,
                     std::vector<llama_token> &prompt_tokens, bool add_special);

    bool eval_chunk(llama_token *tokens, float *embd, int n_tokens,
                    bool is_last = true, llama_pos pos_offset = 0, bool is_causal = false);
    // bool eval_chunk(llama_token *tokens, float *embd, int n_tokens,
    //                 bool is_last = true, bool is_causal = false,
    //                 llama_pos pos_offset = 0);
    virtual std::string generate(const std::string &prompt,
                                 const std::string &system_prompt,
                                 std::vector<llama_token> &generated_tokens,
                                 std::vector<float> &embd,
                                 bool use_history = true);
    virtual std::string generate(float *embedding, int n_tokens,
                                 std::vector<llama_token> &generated_tokens,
                                 std::vector<float> &embd,
                                 bool use_history = true);

    std::string generate(const std::vector<llama_token> &inp_tokens,
                         std::vector<llama_token> &generated_tokens,
                         std::vector<float> &embeddings, bool use_history = true);

    // const float *get_last_hidden_state(int &n_embd) const;
    bool get_last_hidden_state(std::vector<float> &output) const;
    bool get_last_logit(std::vector<float> &output) const;
    virtual ~Llm();

    bool init_tokenizer(const std::string &tokenizer_path);
    bool encode_text_by_tokenizer_cpp(const std::string &prompt,
                                      std::vector<llama_token> &prompt_tokens,
                                      bool add_special);

protected:
    llama_model *model_ = nullptr;
    const llama_vocab *vocab_ = nullptr;
    llama_context *ctx_ = nullptr;
    llama_sampler *smpl_ = nullptr;
    int n_ctx_ = 2048;
    std::unique_ptr<HfTokenizer> tokenizer_;
    bool require_embeddings_ = false;

    Llm(const Llm &) = delete;
    Llm &operator=(const Llm &) = delete;
};

class OpenvlaLlm : public Llm {
public:
    std::string format_prompt(const std::string &prompt,
                              const std::string &system_prompt = "") override;
    bool generate(const std::string &prompt, float *img_emb, int32_t n_img_tokens,
                  std::vector<llama_token> &generated_tokens,
                  std::vector<float> &output, bool use_history = false);

    inline void set_empty_token(llama_token empty_token) {
        empty_token_ = empty_token;
    }

    llama_token empty_token_ =
        220; // 29871 for llama2 and 220 for llama3.2; // default
    int pad_to_multiple_of_ = 64;
};

class T2STrasformer : public Llm {
public:
    bool generate(float *x, int n_x, const std::vector<llama_token> &inp_tokens,
                  std::vector<llama_token> &generated_tokens,
                  bool use_history = false);
    inline bool is_eos(llama_token token) {
        return token == 1024;
    }
};

// class VoteLlm : public Llm {
// public:
//   VoteLlm() = default;
//   std::string format_prompt(const std::string &prompt,
//                             const std::string &system_prompt = "");
//   bool generate(const std::string &prompt, float *embeddings,
//                 int32_t n_img_tokens, std::vector<float> &output,
//                 bool use_history = false);
//   bool init_tokenizer(const std::string &tokenizer_path);
//   bool encode_text_by_tokenizer_cpp(const std::string &prompt,
//                                     std::vector<llama_token> &prompt_tokens,
//                                     bool add_special);

// private:
//   std::unique_ptr<HfTokenizer> tokenizer_;
//   int pad_to_multiple_of_ = 64;
//   std::vector<float> action_high_ = {0.9375,
//                                      0.8758928775787354,
//                                      0.9321428537368774,
//                                      0.1039285734295845,
//                                      0.17678570747375488,
//                                      0.14571428298950195,
//                                      1.0};
//   std::vector<float> action_low_ = {
//       -0.7454732114076613,  -0.6616071462631226, -0.9375,
//       -0.1071428582072258, -0.20678570866584778, -0.1842857152223587, 0.0};
// };