#pragma once

/*
Model inference implementation based on instantiated graph class objects.
Typically includes:
1. Model object containing required weight tensor pointers and some hparams (optional)
2. ContextManager object for context management
3. load_hparams function (optional) to load model configuration parameters like img_size, n_layers, etc.
4. load_tensors function to manually load weights into Model object based on specific Model type (may use some parameters from step 3)
5. alloc_graph function (core) to allocate memory space for graph
6. alloc_compute_meta function to resize ContextManager's buf_compute_meta for storing graph info, and call alloc_graph to allocate memory for graph
7. run function for model inference

General workflow:
1. Define a ModelLoader object to load gguf file
2. Call load_hparams function to load hyperparameters (optional)
3. Call load_tensors function to read weight parameters into model object, while transferring ctx_meta ownership to ContextManager object
4. Call alloc_compute_meta function to build graph and allocate memory
5. Execute run function for inference
*/

#include "infer_session.hpp"
#include "model_defs.h"

class Projector {
public:
  Projector() = default;
  Projector(const std::string &model_path, const ContextParams &params)
      : model_(model_path, params) {}

  bool run(const std::vector<float> &inp_dinov2,
           const std::vector<float> &inp_siglip, std::vector<float> &out) {
    model_.set_input("dinov2_feat", inp_dinov2);
    model_.set_input("siglip_feat", inp_siglip);
    if (!model_.run(out)) {
      return false;
    }
    return true;
  }

private:
  InferenceSession<ProjectorModel> model_;
};