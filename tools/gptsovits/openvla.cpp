#include "openvla.h"
#include "timer.hpp"
#include "utils.h"
#include <algorithm>
#include <filesystem>
#include <functional>
#include <future>

namespace fs = std::filesystem;

bool OpenvlaProjector::run(const std::string &img_path,
                           std::vector<float> &out) {
  std::vector<float> dinov2_out, siglip_out;

  Timer t;
  t.start();
  auto fut_dinov2 = std::async(
      std::launch::async, [&]() { return dinov2_.run(img_path, dinov2_out); });
  auto fut_siglip = std::async(
      std::launch::async, [&] { return siglip_.run(img_path, siglip_out); });
  if (!fut_dinov2.get() || !fut_siglip.get()) {
    return false;
  }
  printf("vision time: %.2f ms\n", t.stop<Timer::ms>());
  if (!proj_.run(dinov2_out, siglip_out, out)) {
    return false;
  }
  printf("Proj time: %.2f ms\n", t.stop<Timer::ms>());
  // print_vector(out, 20);
  return true;
}

Openvla::Openvla(const std::string &dinov2_path, const std::string &siglip_path,
                 const std::string &proj_path, const std::string &llm_path,
                 ContextParams &ctx_params, LlmParam &llm_params)
    : proj_(dinov2_path, siglip_path, proj_path, ctx_params) {
  llm_.load_model(llm_path, llm_params);
  if (!llm_params.tokenizer_path.empty() &&
      fs::exists(llm_params.tokenizer_path)) {
    llm_.init_tokenizer(llm_params.tokenizer_path);
  }
  llm_.set_empty_token(29871);
}

bool Openvla::run(const std::string &img_path, const std::string &prompt,
                  std::vector<float> &out) {
  std::vector<float> vision_embedding;
  if (!proj_.run(img_path, vision_embedding)) {
    return false;
  }
  Timer timer;
  timer.start();
  std::vector<llama_token> generated_tokens;
  // if (!llm_.generate(prompt, vision_embedding.data(), 256, out, false)) {
  if (!llm_.generate(prompt, vision_embedding.data(), 256, generated_tokens,
                     out, false)) {
    return false;
  }
  printf("llm time: %.2f ms\n", timer.stop<Timer::ms>());
  processor_.process(generated_tokens, out);
  return true;
}

bool OpenvlaActionProcessor::init_bin_centers() {
  float start = -1.f, end = 1.f;
  int n = 256;
  float step = (end - start) / (n - 1);
  bin_centers_.resize(n - 1);
  std::vector<float> bins(n, 0.f);
  for (int i = 0; i < n; ++i) {
    bins[i] = start + i * step;
  }
  for (int i = 0; i < n - 1; ++i) {
    bin_centers_[i] = (bins[i] + bins[i + 1]) / 2.f;
  }
  return true;
}

bool OpenvlaActionProcessor::process(
    std::vector<llama_token> &predicted_action_token_ids,
    std::vector<float> &output) {
  std::transform(
      predicted_action_token_ids.begin(), predicted_action_token_ids.end(),
      predicted_action_token_ids.begin(), [&](llama_token token) {
        return std::min(std::max(token, 0), int(bin_centers_.size() - 1));
      });
  std::vector<float> normalized_actions;
  normalized_actions.reserve(predicted_action_token_ids.size());
  for (size_t i = 0; i < predicted_action_token_ids.size(); i++) {
    normalized_actions.push_back(bin_centers_[predicted_action_token_ids[i]]);
  }

  std::vector<bool> mask = {true, true, true, true, true, true, false};
  output.resize(7);
  for (int i = 0; i < 7; ++i) {
    float tmp_value = 0.5 * (normalized_actions[i] + 1) *
                          (action_high_[i] - action_low_[i] + 1e-8) +
                      action_low_[i];
    if (mask[i]) {
      output[i] = tmp_value;
    } else {
      output[i] = normalized_actions[i];
    }
  }
  return true;
}

bool VoteActionProcessor::process(const std::vector<float> &hidden_states,
                                  std::vector<float> &out) {
  std::vector<float> normalized_actions;
  reg_.run(hidden_states, normalized_actions);
  if (normalized_actions.size() != 56) {
    throw std::runtime_error("Invalid output size");
  }
  // for (int i = 0; i < 8; ++i) {
  //   print_vector(std::vector<float>(normalized_actions.begin() + i * 7,
  //                                   normalized_actions.begin() + (i + 1) * 7),
  //                7);
  // }
  out.resize(normalized_actions.size());
  // std::vector<float> actions(normalized_actions.size(), 0.f);
  std::vector<bool> mask = {true, true, true, true, true, true, false};
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 7; ++j) {
      int idx = i * 7 + j;
      float tmp_value = 0.5 * (normalized_actions[idx] + 1) *
                            (action_high_[j] - action_low_[j] + 1e-8) +
                        action_low_[j];
      if (mask[j]) {
        out[idx] = tmp_value;
      } else {
        out[idx] = normalized_actions[idx];
      }
      if (j == 6) {
        out[idx] = 2 * out[idx] - 1; // scale back to [-1, 1]
        out[idx] = out[idx] >= 0 ? -1.0f : 1.0f;
      }
    }
  }
  printf("Final actions:\n");
  for (int i = 0; i < 8; ++i) {
    print_vector(
        std::vector<float>(out.begin() + i * 7, out.begin() + (i + 1) * 7), 7);
  }
  return true;
}

OpenvlaWithRegression::OpenvlaWithRegression(const std::string &dinov2_path,
                                             const std::string &siglip_path,
                                             const std::string &proj_path,
                                             const std::string &llm_path,
                                             const std::string &reg_path,
                                             ContextParams &ctx_params,
                                             LlmParam &llm_params)
    : proj_(dinov2_path, siglip_path, proj_path, ctx_params),
      processor_(reg_path, ctx_params) {
  llm_.load_model(llm_path, llm_params);
  if (!llm_params.tokenizer_path.empty() &&
      fs::exists(llm_params.tokenizer_path)) {
    llm_.init_tokenizer(llm_params.tokenizer_path);
  }
  llm_.set_empty_token(220);
}

bool OpenvlaWithRegression::run(const std::string &img_path,
                                const std::string &prompt,
                                std::vector<float> &out) {
  // const int num_actions_chunk = 8;
  // const int num_actions_per_token = 8;
  std::vector<float> vision_embedding;
  if (!proj_.run(img_path, vision_embedding)) {
    return false;
  }
  Timer timer;
  timer.start();
  std::vector<llama_token> generated_tokens;
  std::vector<float> hidden_states;
  // load_file_to_vector(vision_embedding,
  // "/Users/wjr/Downloads/vote_model/v_e.raw"); if (!llm_.generate(prompt,
  // vision_embedding.data(), 256, out, false)) {
  if (!llm_.generate(prompt, vision_embedding.data(), 256, generated_tokens,
                     hidden_states, false)) {
    return false;
  }
  printf("llm time: %.2f ms\n", timer.stop<Timer::ms>());
  // load_file_to_vector(hidden_states,
  // "/Users/wjr/Downloads/vote_model/hid.raw"); print_vector(hidden_states,
  // 20);
  processor_.process(hidden_states, out);
  return true;
}