#pragma once

#include "infer_session.hpp"
#include "llama.h"
#include "model_defs.h"
#include <assert.h>
#include <limits>
#include "ggml.h"
#include "utils.h"

class VitsArInfer {
public:
  VitsArInfer() = default;
  VitsArInfer(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<int> &token, const std::vector<int> &pos,
           std::vector<float> &out) {
    model_.set_input("inp_token", token);
    model_.set_input("inp_pos", pos);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

private:
  InferenceSession<VitsArModel> model_;
};

class ArTextInfer {
public:
  ArTextInfer() = default;
  ArTextInfer(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<int> &token, const std::vector<int> &pos,
           const std::vector<float> &bert_feats, std::vector<float> &out) {
    assert(token.size() == pos.size());
    std::map<std::string, std::vector<int>> input_shapes;
    input_shapes["inp_token"] = {static_cast<int>(token.size())};
    input_shapes["inp_pos"] = {static_cast<int>(pos.size())};
    input_shapes["bert_feature"] = {1024, static_cast<int>(token.size())};
    set_input_shape(input_shapes);

    model_.set_input("inp_token", token);
    model_.set_input("inp_pos", pos);
    model_.set_input("bert_feature", bert_feats);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<ArTextModel> model_;
};

class MelStyleEncoder {
public:
  MelStyleEncoder() = default;
  MelStyleEncoder(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &x, std::vector<float> &out) {
    std::map<std::string, std::vector<int>> input_shapes;
    int n_tokens = x.size() / 704;
    input_shapes["x"] = {n_tokens, 704, 1};
    set_input_shape(input_shapes);

    model_.set_input("x", x);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<MelStyleEncoderModel> model_;
};

class EuclideanCodebook {
public:
  EuclideanCodebook() = default;
  EuclideanCodebook(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<llama_token> &codes, std::vector<float> &out) {
    std::map<std::string, std::vector<int>> input_shapes;
    input_shapes["codes"] = {static_cast<int>(codes.size())};
    set_input_shape(input_shapes);

    model_.set_input("codes", codes);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<EuclideanCodebookModel> model_;
};

class TextEncoder {
public:
  TextEncoder() = default;
  TextEncoder(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &quantized,
           const std::vector<llama_token> &phones, const std::vector<float> &ge,
           std::vector<float> &means, std::vector<float> &logs) {
    std::map<std::string, std::vector<int>> input_shapes;
    int n_quantized = quantized.size() / 768;
    input_shapes["quantized"] = {n_quantized, 768, 1};
    input_shapes["phones"] = {static_cast<int>(phones.size()), 1};
    input_shapes["ge"] = {512, 1};
    set_input_shape(input_shapes);

    model_.set_input("quantized", quantized);
    model_.set_input("phones", phones);
    model_.set_input("ge", ge);

    // std::vector<std::vector<float>> out;
    // if (!model_.run(means)) {
    if (!model_.run(means, logs)) {
      // if(!model_.run(out)) {
      return false;
    }
    // means = out[0];
    // logs = out[1];
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<TextEncoderModel> model_;
};

class FlowBlock {
public:
  FlowBlock() = default;
  FlowBlock(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &x, const std::vector<float> &ge,
           std::vector<float> &out) {
    std::map<std::string, std::vector<int>> input_shapes;
    int n_x = x.size() / 192;
    input_shapes["x"] = {n_x, 192, 1};
    input_shapes["ge"] = {1, 512, 1};
    set_input_shape(input_shapes);
    model_.set_input("x", x);
    model_.set_input("ge", ge);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<ResidualCouplingBlock> model_;
};

class Generator {
public:
  Generator() = default;
  Generator(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &x, const std::vector<float> &ge,
           std::vector<float> &out) {
    std::map<std::string, std::vector<int>> input_shapes;
    int n_x = x.size() / 192;
    input_shapes["x"] = {n_x, 192, 1};
    input_shapes["ge"] = {1, 512, 1};
    set_input_shape(input_shapes);

    model_.set_input("x", x);
    model_.set_input("ge", ge);

    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<GeneratorModel> model_;
};

class FakeInfer {
public:
  FakeInfer() = default;
  FakeInfer(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &x, std::vector<float> &out) {
    model_.set_input("x", x);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }
  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

private:
  InferenceSession<FakeModel> model_;
};

// CNHubert inference class
// Extracts semantic features from 16kHz audio
class CNHubertInfer {
public:
  CNHubertInfer() = default;
  CNHubertInfer(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  // Input: wav_16k (16kHz audio waveform, normalized to [-1, 1])
  // Output: features (T x 768, where T ≈ audio_len / 320)
  bool run(const std::vector<float> &wav_16k, std::vector<float> &features) {
    std::map<std::string, std::vector<int>> input_shapes;
    input_shapes["wav_16k"] = {static_cast<int>(wav_16k.size()), 1};
    set_input_shape(input_shapes);

    model_.set_input("wav_16k", wav_16k);
    if (!model_.run(features)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<CNHubertModel> model_;
};

// Residual Vector Quantizer inference class
// Projects CNHubert features and quantizes to semantic tokens using GGML graph
class ResidualVectorQuantizerInfer {
public:
  ResidualVectorQuantizerInfer() = default;
  ResidualVectorQuantizerInfer(const std::string &model_path,
                        const ContextParams &params)
      : model_(model_path, params) {}

  // Project SSL features and quantize to semantic codes
  // Input: ssl_features (768 x T)
  // Output: codes (T',) where T' = (T-1)/2 + 1 due to stride=2 in ssl_proj
  bool run(const std::vector<float> &ssl_features,
           std::vector<llama_token> &codes) {
    int T = ssl_features.size() / 768;
    std::map<std::string, std::vector<int>> input_shapes;
    input_shapes["ssl_features"] = {768, T};
    set_input_shape(input_shapes);

    model_.set_input("ssl_features", ssl_features);

    // Run the model - output is int32 indices
    std::vector<int32_t> indices;
    if (!model_.run(indices)) {
      return false;
    }

    // Convert int32 indices to llama_token
    codes.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
      codes[i] = static_cast<llama_token>(indices[i]);
    }
    return true;
  }

  // Legacy: Load codebook data from model (no longer needed for quantization)
  // Kept for backward compatibility
  bool load_codebook() {
    // Quantization is now done in GGML graph, no need to load codebook to CPU
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes) {
    model_.set_input_shape(shapes);
    return true;
  }

  void synchronize() { model_.synchronize(); }

private:
  InferenceSession<ResidualVectorQuantizer> model_;
};