#pragma once

#include "llm.h"
#include "proj.hpp"
#include "reg.hpp"
#include "utils.h"
#include "vit.hpp"

class OpenvlaProjector {
public:
  OpenvlaProjector(const std::string &dinov2_path,
                   const std::string &siglip_path, const std::string &proj_path,
                   ContextParams &ctx_params)
      : proj_(proj_path, ctx_params), dinov2_(dinov2_path, ctx_params),
        siglip_(siglip_path, ctx_params) {}

  bool run(const std::string &img_path, std::vector<float> &out);

private:
  Projector proj_;
  Vit dinov2_;
  Vit siglip_;
};

class OpenvlaActionProcessor {
public:
  OpenvlaActionProcessor() { init_bin_centers(); }

  bool process(std::vector<llama_token> &predicted_action_token_ids,
               std::vector<float> &output);

private:
  bool init_bin_centers();

  std::vector<float> bin_centers_;
  std::vector<float> action_high_ = {0.028309678435325586,
                                     0.040855254605412394,
                                     0.040161586627364146,
                                     0.08192047759890528,
                                     0.07792850524187081,
                                     0.20382574498653397,
                                     1.0};
  std::vector<float> action_low_ = {-0.02872725307941437,
                                    -0.04170349963009357,
                                    -0.026093858778476715,
                                    -0.08092105075716972,
                                    -0.09288699507713317,
                                    -0.20718276381492615,
                                    0.0};
};

class VoteActionProcessor {
public:
  VoteActionProcessor(const std::string &model_path,
                      const ContextParams &params)
      : reg_(model_path, params) {}
  bool process(const std::vector<float> &hidden_states,
               std::vector<float> &out);

private:
  L1RegressionHead reg_;
  std::vector<float> action_high_ = {0.9375,
                                     0.8758928775787354,
                                     0.9321428537368774,
                                     0.1039285734295845,
                                     0.17678570747375488,
                                     0.14571428298950195,
                                     1.0};
  std::vector<float> action_low_ = {
      -0.7454732114076613,  -0.6616071462631226, -0.9375, -0.1071428582072258,
      -0.20678570866584778, -0.1842857152223587, 0.0};
};

class Openvla {
public:
  Openvla(const std::string &dinov2_path, const std::string &siglip_path,
          const std::string &proj_path, const std::string &llm_path,
          ContextParams &ctx_params, LlmParam &llm_params);
  bool run(const std::string &img_path, const std::string &prompt,
           std::vector<float> &out);

private:
  OpenvlaProjector proj_;
  OpenvlaLlm llm_;
  OpenvlaActionProcessor processor_;
};

class OpenvlaWithRegression {
public:
  OpenvlaWithRegression(const std::string &dinov2_path,
                        const std::string &siglip_path,
                        const std::string &proj_path,
                        const std::string &llm_path,
                        const std::string &reg_path, ContextParams &ctx_params,
                        LlmParam &llm_params);
  bool run(const std::string &img_path, const std::string &prompt,
           std::vector<float> &out);

private:
  OpenvlaProjector proj_;
  OpenvlaLlm llm_;
  VoteActionProcessor processor_;
};
