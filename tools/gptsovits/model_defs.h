#pragma once
/*
model_defs.h defines model classes, including weight tensor pointers and model hyperparameters.
Main functions supported: load_hparams, load_tensors, get_tensors_to_load, build_graph.
- get_tensors_to_load: Gets tensor pointers to load and assigns them to the corresponding tensor* objects in the model.
- load_tensors (common): Takes a model_loader object, transfers ctx_meta ownership from model_loader to ctx_manager_, and loads tensor values into memory.
- load_hparams (optional): Reads model hyperparameters and saves them to the hparams object.
- build_graph (required): Builds the model computation graph for tensor data flow control. Takes ctx and creates a graph based on it, returns output tensor*. Graph memory allocation is controlled by InferenceSession class.

**********************************************************************************************************
To add a new model, define a Model class that inherits from BaseModel.
Implement get_tensors_to_load (required) and build_graph (required). If hyperparameters need to be loaded, implement load_hparams.
The new Model class should also contain the tensor* information needed for computation.
For example, ProjectorModel defines the tensor data needed for computation and implements get_tensors_to_load and build_graph.
VisionTransformerModel additionally implements load_hparams and build_vit functions.
*/

#include "build_graph.h"
#include "ctx_manager.h"
#include "ggml.h"
#include "model_loader.h"
#include <functional>
#include <map>
#include <unordered_set>
#include <vector>

class BaseModel {
public:
  // Transfer ctx_meta ownership from model_loader to ctx_manager_.ctx_data_, and load weights to corresponding tensors in model_
  bool load_tensors(ModelLoader &model_loader, ContextManager &ctx_manager);

  virtual bool load_hparams(const ModelLoader &model_loader) { return true; }
  // Get tensor addresses that this model needs to load
  virtual std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) {
    return {};
  }

  virtual std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) {
    return {};
  }

  virtual void reset_input_shape() {}
  void set_input_shape(const std::string &inp_name,
                       const std::vector<int> &shape) {
    if (input_shapes_.find(inp_name) == input_shapes_.end()) {
      throw std::invalid_argument("input name is not found");
    }
    // std::vector<int> old_shape = input_shapes_[inp_name];
    // int input_dim = old_shape.size();
    input_shapes_[inp_name] = shape;
  }

  virtual ~BaseModel() = default;

protected:
  std::map<std::string, std::vector<int>> input_shapes_;
};

class FakeModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;

  void reset_input_shape() override {
    input_shapes_ = {
        {"x", {6, 1}},
    };
  }

  ggml_tensor *embed_tokens = nullptr;
  ggml_tensor *llm_head = nullptr;
};

class AddOneModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    one = get_tensor(ctx, "one", tensors_to_load, true);
    return tensors_to_load;
  }
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override {
    std::vector<int> x_shape = input_shapes_["x"];
    ggml_tensor *x =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, x_shape[0], x_shape[1]);
    ggml_set_name(x, "x");
    ggml_set_input(x);
    // ggml_tensor *cur = ggml_exp(ctx0, x);
    // cur = ggml_add(ctx0, cur, one);
    // cur = ggml_log(ctx0, cur);
    // cur = ggml_tanh(ctx0, cur);
    // cur = ggml_mul(ctx0, cur, x);
    // ggml_tensor *cur = build_mish(ctx0, x, one);
    // ggml_tensor* cur = ggml_abs(ctx0, x);
    ggml_tensor *cur = ggml_flip(ctx0, x, 0);
    return {cur};
  }

  void reset_input_shape() override {
    input_shapes_ = {
        {"x", {4, 1}},
    };
  }

  ggml_tensor *one = nullptr;
};

struct VisionParams {
  int32_t image_size = 224;
  int32_t patch_size;
  int32_t n_embd;
  int32_t n_ff;
  int32_t n_head;
  int32_t n_layer;
  std::vector<float> image_mean;
  std::vector<float> image_std;
  ffn_op_type ffn_op = FFN_GELU;
  float eps = 1e-6;
  int32_t projection_dim;
  std::unordered_set<int32_t> vision_feature_layer;
};

struct ClipLayer {
  // attention
  ggml_tensor *k_w = nullptr;
  ggml_tensor *k_b = nullptr;
  ggml_tensor *q_w = nullptr;
  ggml_tensor *q_b = nullptr;
  ggml_tensor *v_w = nullptr;
  ggml_tensor *v_b = nullptr;

  ggml_tensor *o_w = nullptr;
  ggml_tensor *o_b = nullptr;

  ggml_tensor *k_norm = nullptr;
  ggml_tensor *q_norm = nullptr;

  // layernorm 1
  ggml_tensor *ln_1_w = nullptr;
  ggml_tensor *ln_1_b = nullptr;

  ggml_tensor *ff_up_w = nullptr;
  ggml_tensor *ff_up_b = nullptr;
  ggml_tensor *ff_gate_w = nullptr;
  ggml_tensor *ff_gate_b = nullptr;
  ggml_tensor *ff_down_w = nullptr;
  ggml_tensor *ff_down_b = nullptr;

  // layernorm 2
  ggml_tensor *ln_2_w = nullptr;
  ggml_tensor *ln_2_b = nullptr;

  // layer scale (no bias)
  ggml_tensor *ls_1_w = nullptr;
  ggml_tensor *ls_2_w = nullptr;
};

class ProjectorModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;

  void reset_input_shape() override {
    input_shapes_ = {{"dinov2_feat", {1024, 256}},
                     {"siglip_feat", {1152, 256}}};
  }

  ggml_tensor *fc1_weight = nullptr;
  ggml_tensor *fc1_bias = nullptr;
  ggml_tensor *fc2_weight = nullptr;
  ggml_tensor *fc2_bias = nullptr;
  ggml_tensor *fc3_weight = nullptr;
  ggml_tensor *fc3_bias = nullptr;
};

class VisionTransformerModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  void reset_input_shape() override {
    input_shapes_ = {
        {"inp_raw", {hparams.image_size, hparams.image_size, 3}},
    };
  }

  ggml_tensor *build_vit(
      ggml_context *ctx0, ggml_tensor *inp, int n_pos, norm_type norm_t,
      std::function<ggml_tensor *(ggml_tensor *, const ClipLayer &)> add_pos);

  VisionParams hparams;
  // embeddings
  ggml_tensor *class_embedding = nullptr;
  ggml_tensor *reg_embedding = nullptr;
  ggml_tensor *patch_embeddings_0 = nullptr;
  ggml_tensor *patch_embeddings_1 =
      nullptr; // second Conv2D kernel when we decouple Conv3D along temproal
               // dimension (Qwen2VL)
  ggml_tensor *patch_bias = nullptr;
  ggml_tensor *position_embeddings = nullptr;

  ggml_tensor *pre_ln_w = nullptr;
  ggml_tensor *pre_ln_b = nullptr;

  std::vector<ClipLayer> layers;

  ggml_tensor *post_ln_w;
  ggml_tensor *post_ln_b;
};

struct RegressionParams {
  int action_dim = 7;
  int num_actions_chunk = 8;
  int num_actions_per_token = 8;
  int num_blocks = 4;
  int input_dim = 2048;
  int hidden_dim = 512;
  int expansion = 4;
};

class MLPResNetBlockV2 {
public:
  /*
  class MLPResNetBlockV2(nn.Module):
      def __init__(self, dim, expansion=4, dropout=0.1):
          super().__init__()
          self.ffn = nn.Sequential(
              nn.LayerNorm(dim),
              nn.Linear(dim, dim * expansion),
              nn.SiLU(),
              nn.Linear(dim * expansion, dim)
          )
          self.dropout = nn.Dropout(dropout)

      def forward(self, x):
          identity = x
          x_ffn = self.ffn(x)
          x_dropped = self.dropout(x_ffn)
          x = x_dropped + identity
          return x
  */
  ggml_tensor *ffn_ln_w = nullptr;
  ggml_tensor *ffn_ln_b = nullptr;
  ggml_tensor *ffn_fc_w = nullptr;
  ggml_tensor *ffn_fc_b = nullptr;
  ggml_tensor *ffn_fc2_w = nullptr;
  ggml_tensor *ffn_fc2_b = nullptr;

  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp);
};

class MultiHeadAttention {
public:
  ggml_tensor *q_w = nullptr;
  ggml_tensor *q_b = nullptr;
  ggml_tensor *k_w = nullptr;
  ggml_tensor *k_b = nullptr;
  ggml_tensor *v_w = nullptr;
  ggml_tensor *v_b = nullptr;
  ggml_tensor *o_w = nullptr;
  ggml_tensor *o_b = nullptr;

  // layernorm 1
  ggml_tensor *ln_1_w = nullptr;
  ggml_tensor *ln_1_b = nullptr;
  // layernorm 2
  ggml_tensor *ln_2_w = nullptr;
  ggml_tensor *ln_2_b = nullptr;

  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp,
                           ggml_tensor *inp2, int n_q_head, int n_kv_head,
                           float kq_scale,
                           enum norm_type norm_type = NORM_TYPE_NORMAL,
                           float norm_eps = 1e-5, bool use_conv = false);
};

class MultiHeadAttention2 : public MultiHeadAttention {
public:
  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp,
                           ggml_tensor *inp2, int n_q_head, int n_kv_head,
                           float kq_scale,
                           enum norm_type norm_type = NORM_TYPE_NORMAL,
                           float norm_eps = 1e-5, bool use_conv = false);
};

class TextEncoderFFN {
public:
  ggml_tensor *conv1_weight = nullptr;
  ggml_tensor *conv1_bias = nullptr;
  ggml_tensor *conv2_weight = nullptr;
  ggml_tensor *conv2_bias = nullptr;
  ggml_tensor *ln_w = nullptr;
  ggml_tensor *ln_b = nullptr;

  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp,
                           norm_type norm_type, float norm_eps = 1e-5);
};

class L1RegressionActionHeadFunnelModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  bool load_hparams(const ModelLoader &model_loader) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;

  void reset_input_shape() override {
    input_shapes_ = {
        {"inp_raw", {hparams.input_dim, 1}},
    };
  }

  std::vector<int> default_shape() { return {hparams.input_dim, 1}; }

  RegressionParams hparams;

  /*
    self.input_proj = nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.SiLU(),
    )
  */
  ggml_tensor *input_proj_ln_w = nullptr;
  ggml_tensor *input_proj_ln_b = nullptr;
  ggml_tensor *input_proj_fc_w = nullptr;
  ggml_tensor *input_proj_fc_b = nullptr;

  std::vector<MLPResNetBlockV2> resnet_body;

  ggml_tensor *output_head_ln_w = nullptr;
  ggml_tensor *output_head_ln_b = nullptr;
  ggml_tensor *output_head_fc_w = nullptr;
  ggml_tensor *output_head_fc_b = nullptr;
};

struct VitsParams {
  int embd_dim = 512;
  float alpha = 3.478515625f;
  float x_scale = 1.f;
  int num_inputs = 1;
};

class VitsArModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  bool load_hparams(const ModelLoader &model_loader) override;

  void reset_input_shape() override {
    input_shapes_ = {
        {"inp_token", {1}},
        {"inp_pos", {1}},
    };
  }

  ggml_tensor *ar_audio_position_pe = nullptr;
  ggml_tensor *word_embeddings = nullptr;
  VitsParams hparams;
};

class ArTextModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  bool load_hparams(const ModelLoader &model_loader) override;

  void reset_input_shape() override {
    input_shapes_ = {
        {"inp_token", {1}}, {"inp_pos", {1}}, {"bert_feature", {1024, 1}}};
  }

  ggml_tensor *ar_text_position_pe = nullptr;
  ggml_tensor *word_embeddings = nullptr;
  ggml_tensor *bert_proj_weight = nullptr;
  ggml_tensor *bert_proj_bias = nullptr;
  VitsParams hparams;
};

class MelStyleEncoderModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  // bool load_hparams(const ModelLoader &model_loader) override;

  void reset_input_shape() override {
    input_shapes_ = {{"x", {468, 704, 1}}, {"codes", {50}}};
  }

  ggml_tensor *one = nullptr;

  ggml_tensor *spectral_fc1_weight = nullptr;
  ggml_tensor *spectral_fc1_bias = nullptr;
  ggml_tensor *spectral_fc2_weight = nullptr;
  ggml_tensor *spectral_fc2_bias = nullptr;

  ggml_tensor *temporal_conv1_weight = nullptr;
  ggml_tensor *temporal_conv1_bias = nullptr;
  ggml_tensor *temporal_conv2_weight = nullptr;
  ggml_tensor *temporal_conv2_bias = nullptr;

  ggml_tensor *slf_attn_w_qs_weight = nullptr;
  ggml_tensor *slf_attn_w_qs_bias = nullptr;
  ggml_tensor *slf_attn_w_ks_weight = nullptr;
  ggml_tensor *slf_attn_w_ks_bias = nullptr;
  ggml_tensor *slf_attn_w_vs_weight = nullptr;
  ggml_tensor *slf_attn_w_vs_bias = nullptr;
  ggml_tensor *slf_attn_fc_weight = nullptr;
  ggml_tensor *slf_attn_fc_bias = nullptr;

  ggml_tensor *fc_weight = nullptr;
  ggml_tensor *fc_bias = nullptr;

  ggml_tensor *codebook_embed = nullptr;
};

class EuclideanCodebookModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;

  void reset_input_shape() override { input_shapes_ = {{"codes", {50}}}; }

  // 1024x768
  ggml_tensor *embed = nullptr;
};

struct TextEncoderParams {
  // int hidden_dim = 768;
  int n_layer = 6;
  int n_q_heads = 2;
  int n_kv_heads = 2;
  float norm_eps = 1e-5;
  int n_heads_mrte = 4;
};
class TextEncoderModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  bool load_hparams(const ModelLoader &model_loader) override;

  TextEncoderParams hparams_;

  void reset_input_shape() override {
    input_shapes_ = {
        {"quantized", {100, 768, 1}}, {"phones", {50, 1}}, {"ge", {512, 1}}};
  }

  // ssl_proj
  ggml_tensor *ssl_proj_weight = nullptr;
  ggml_tensor *ssl_proj_bias = nullptr;

  // encoder_ssl
  std::vector<MultiHeadAttention> attn_layers_;
  std::vector<TextEncoderFFN> ffn_layers_;

  // encoder_text
  std::vector<MultiHeadAttention> attn_layers_text_;
  std::vector<TextEncoderFFN> ffn_layers_text_;

  // text_embedding
  ggml_tensor *text_embedding_weight = nullptr;

  // mrte
  ggml_tensor *mrte_c_pre_weight = nullptr;
  ggml_tensor *mrte_c_pre_bias = nullptr;
  ggml_tensor *mrte_text_pre_weight = nullptr;
  ggml_tensor *mrte_text_pre_bias = nullptr;
  ggml_tensor *mrte_c_post_weight = nullptr;
  ggml_tensor *mrte_c_post_bias = nullptr;

  MultiHeadAttention mrte_attn_layer_;

  // encoder2
  std::vector<MultiHeadAttention> attn_layers_encoder2_;
  std::vector<TextEncoderFFN> ffn_layers_encoder2_;

  // proj
  ggml_tensor *proj_weight = nullptr;
  ggml_tensor *proj_bias = nullptr;
};

class WN{
public:
  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *h, ggml_tensor* ge);
  
  ggml_tensor* cond_layer_weight = nullptr;
  ggml_tensor* cond_layer_bias = nullptr;

  ggml_tensor* in_layer_0_weight = nullptr;
  ggml_tensor* in_layer_0_bias = nullptr;

  ggml_tensor* in_layer_1_weight = nullptr;
  ggml_tensor* in_layer_1_bias = nullptr;

  ggml_tensor* in_layer_2_weight = nullptr;
  ggml_tensor* in_layer_2_bias = nullptr;

  ggml_tensor* in_layer_3_weight = nullptr;
  ggml_tensor* in_layer_3_bias = nullptr;

  ggml_tensor* res_skip_layers_0_weight = nullptr;
  ggml_tensor* res_skip_layers_0_bias = nullptr;
  ggml_tensor* res_skip_layers_1_weight = nullptr;
  ggml_tensor* res_skip_layers_1_bias = nullptr;
  ggml_tensor* res_skip_layers_2_weight = nullptr;
  ggml_tensor* res_skip_layers_2_bias = nullptr;
  ggml_tensor* res_skip_layers_3_weight = nullptr;
  ggml_tensor* res_skip_layers_3_bias = nullptr;
};

class ResidualCouplingLayer {
public:
  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *x, ggml_tensor* ge);

  ggml_tensor* pre_weight = nullptr;
  ggml_tensor* pre_bias = nullptr;
  ggml_tensor* post_weight = nullptr;
  ggml_tensor* post_bias = nullptr;
  WN enc;
};

class ResidualCouplingBlock : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;

  void reset_input_shape() override {
    input_shapes_ = {{"x", {100, 192, 1}}, {"ge", {1, 512, 1}}};
  }

  std::vector<ResidualCouplingLayer> layers;
};

class ResBlock1 {
public:
  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *cur);

  ggml_tensor *conv1_0_weight = nullptr;
  ggml_tensor *conv1_0_bias = nullptr;
  ggml_tensor *conv1_1_weight = nullptr;
  ggml_tensor *conv1_1_bias = nullptr;
  ggml_tensor *conv1_2_weight = nullptr;
  ggml_tensor *conv1_2_bias = nullptr;

  ggml_tensor *conv2_0_weight = nullptr;
  ggml_tensor *conv2_0_bias = nullptr;
  ggml_tensor *conv2_1_weight = nullptr;
  ggml_tensor *conv2_1_bias = nullptr;
  ggml_tensor *conv2_2_weight = nullptr;
  ggml_tensor *conv2_2_bias = nullptr;
};

class GeneratorBlock {
public:
  struct UpSampleParams {
    int stride = 1;
    int padding = 0;
  } params;
  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *inp, float slope);

  std::vector<ResBlock1> resblocks;
  ggml_tensor *up_weight = nullptr;
  ggml_tensor *up_bias = nullptr;
};

class GeneratorModel : public BaseModel {
public:
  struct GeneratorParams {
    int num_upsamples = 5;
    int num_kernels = 3;
    float lrelu_slope = 0.1f;
  } hparams_;

  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  bool load_hparams(const ModelLoader &model_loader) override;

  void reset_input_shape() override {
    input_shapes_ = {{"x", {100, 192, 1}}, {"ge", {1, 512, 1}}};
  }

  std::vector<GeneratorBlock> generate_blocks_;

  ggml_tensor *conv_pre_weight = nullptr;
  ggml_tensor *conv_pre_bias = nullptr;

  ggml_tensor *cond_weight = nullptr;
  ggml_tensor *cond_bias = nullptr;

  ggml_tensor *conv_post_weight = nullptr;
  ggml_tensor *conv_post_bias = nullptr;
};

// ==================== CNHubert Model ====================
// CNHubert (Chinese HuBERT) for semantic feature extraction from audio
// Input: 16kHz audio waveform
// Output: (batch, seq_len, 768) features at ~50Hz

struct CNHubertParams {
  int n_layer = 12;
  int hidden_size = 768;
  int n_heads = 12;
  int intermediate_size = 3072;
  float layer_norm_eps = 1e-5f;
  int num_feat_extract_layers = 7;
  std::vector<int> conv_dim = {512, 512, 512, 512, 512, 512, 512};
  std::vector<int> conv_kernel = {10, 3, 3, 3, 3, 2, 2};
  std::vector<int> conv_stride = {5, 2, 2, 2, 2, 2, 2};
};

struct CNHubertConvLayer {
  ggml_tensor *conv_weight = nullptr;
  ggml_tensor *conv_bias = nullptr;
  ggml_tensor *ln_weight = nullptr; // Group/Layer norm
  ggml_tensor *ln_bias = nullptr;
};

// Positional convolutional embedding layer for HuBERT
class HubertPositionalConvEmbedding {
public:
  ggml_tensor *build_graph(ggml_context *ctx0, ggml_tensor *hidden_states);
  
  ggml_tensor* conv_weight = nullptr;
  ggml_tensor* conv_bias = nullptr;
};

struct CNHubertEncoderLayer {
  // Self-attention
  ggml_tensor *q_w = nullptr;
  ggml_tensor *q_b = nullptr;
  ggml_tensor *k_w = nullptr;
  ggml_tensor *k_b = nullptr;
  ggml_tensor *v_w = nullptr;
  ggml_tensor *v_b = nullptr;
  ggml_tensor *o_w = nullptr;
  ggml_tensor *o_b = nullptr;

  // LayerNorm after attention
  ggml_tensor *ln_1_w = nullptr;
  ggml_tensor *ln_1_b = nullptr;

  // Feed-forward network
  ggml_tensor *ff_up_w = nullptr;   // intermediate_dense
  ggml_tensor *ff_up_b = nullptr;
  ggml_tensor *ff_down_w = nullptr; // output_dense
  ggml_tensor *ff_down_b = nullptr;

  // LayerNorm after FFN
  ggml_tensor *ln_2_w = nullptr;
  ggml_tensor *ln_2_b = nullptr;
};

class CNHubertModel : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  bool load_hparams(const ModelLoader &model_loader) override;

  void reset_input_shape() override {
    // Input: 16kHz audio, max 10 seconds
    input_shapes_ = {
        {"wav_16k", {16000 * 10, 1}},
    };
  }

  CNHubertParams hparams;

  // Feature extractor (7 conv layers)
  std::vector<CNHubertConvLayer> conv_layers;

  // Feature projection
  ggml_tensor *proj_weight = nullptr;
  ggml_tensor *proj_bias = nullptr;
  ggml_tensor *proj_ln_w = nullptr;
  ggml_tensor *proj_ln_b = nullptr;

  // Positional convolutional embedding
  HubertPositionalConvEmbedding pos_conv_embed;

  ggml_tensor *encoder_ln_weight = nullptr;
  ggml_tensor *encoder_ln_bias = nullptr;

  // Encoder layers
  std::vector<CNHubertEncoderLayer> encoder_layers;

  
};

// ==================== Residual Vector Quantizer ====================
// Projects CNHubert features and quantizes to semantic tokens

struct ResidualVectorQuantizerParams {
  int ssl_dim = 768;
  int codebook_size = 1024;
  int n_q = 1; // number of quantizer layers
};

class ResidualVectorQuantizer : public BaseModel {
public:
  std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
  std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
  bool load_hparams(const ModelLoader &model_loader) override;

  void reset_input_shape() override {
    input_shapes_ = {
        {"ssl_features", {768, 100}}, // (768, T)
    };
  }

  ResidualVectorQuantizerParams hparams;

  // ssl_proj: Conv1D(768, 768, kernel=1) equivalent to Linear
  ggml_tensor *ssl_proj_weight = nullptr; // (768, 768)
  ggml_tensor *ssl_proj_bias = nullptr;   // (768,)

  // Quantizer codebook
  ggml_tensor *codebook = nullptr; // (1024, 768)
};