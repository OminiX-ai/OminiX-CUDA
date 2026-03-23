#pragma once

#include "infer_session.hpp"
#include "model_defs.h"

class AddOneInfer {
public:
  AddOneInfer(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &x, std::vector<float> &out) {
    model_.set_input("x", x);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

  bool set_input_shape(const std::map<std::string, std::vector<int>> &shapes){
    model_.set_input_shape(shapes);
    return true;
  }

private:
  InferenceSession<AddOneModel> model_;
};