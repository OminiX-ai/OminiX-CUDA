#include "infer_session.hpp"
#include "model_defs.h"

class L1RegressionHead {
public:
  L1RegressionHead() = default;
  L1RegressionHead(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &hidden_states, std::vector<float> &out) {
    model_.set_input("inp_raw", hidden_states);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

private:
  InferenceSession<L1RegressionActionHeadFunnelModel> model_;
};