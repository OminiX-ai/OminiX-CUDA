#include "model_defs.h"
#include "build_graph.h"
#include "ggml.h"
#include "utils.h"
#include <cmath>
#include <cstddef>
#include <map>
// #include <string.h>
#include <string>

bool BaseModel::load_tensors(ModelLoader &model_loader,
                             ContextManager &ctx_manager) {
    std::map<std::string, size_t> tensor_offset;
    gguf_context_ptr &ctx_gguf = model_loader.ctx_gguf_;

    // Transfer ownership of ctx_meta from model_loader
    ctx_manager.ctx_data_ = std::move(model_loader.ctx_meta_);
    ggml_context *ctx = ctx_manager.ctx_data_.get();

    // Get offsets of all weights in the file
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
        const char *name = gguf_get_tensor_name(ctx_gguf.get(), i);
        tensor_offset[name] = gguf_get_data_offset(ctx_gguf.get()) + gguf_get_tensor_offset(ctx_gguf.get(), i);
    }

    std::vector<ggml_tensor *> tensors_to_load = get_tensors_to_load(ctx);

    // Load data
    {
        std::vector<uint8_t> read_buf;

        auto fin = std::ifstream(model_loader.fname_, std::ios::binary);
        if (!fin) {
            throw std::runtime_error(string_format(
                "%s: failed to open %s\n", __func__, model_loader.fname_.c_str()));
        }

        // Allocate memory for tensors in ctx_meta on the specified backend
        ggml_backend_buffer_type_t buft =
            ggml_backend_get_default_buffer_type(ctx_manager.backend_.get());
        ctx_manager.buffer_.reset(ggml_backend_alloc_ctx_tensors_from_buft(
            ctx_manager.ctx_data_.get(), buft));
        ggml_backend_buffer_set_usage(ctx_manager.buffer_.get(),
                                      GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        // Assign values to tensors
        for (auto &cur : tensors_to_load) {
            const size_t offset = tensor_offset[cur->name];
            fin.seekg(offset, std::ios::beg);
            if (!fin) {
                throw std::runtime_error(string_format(
                    "%s: failed to seek for tensor %s\n", __func__, cur->name));
            }
            size_t num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buft_is_host(buft)) {
                // for the CPU and Metal backend, we can read directly into the tensor
                fin.read(reinterpret_cast<char *>(cur->data), num_bytes);
            } else {
                // read into a temporary buffer first, then copy to device memory
                read_buf.resize(num_bytes);
                fin.read(reinterpret_cast<char *>(read_buf.data()), num_bytes);
                // if (cur->name != NULL &&
                //     strcmp(cur->name, "flows.6.enc.cond_layer.weight") == 0) {
                //   printf("save %s\n", cur->name);
                //   save_vector_to_file(read_buf, std::string(cur->name) + ".raw");
                // }
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
        }
        fin.close();

        printf("%s: loaded %zu tensors from %s\n", __func__, tensors_to_load.size(),
               model_loader.fname_.c_str());
    }
    return true;
}
std::vector<ggml_tensor *> FakeModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    {
        // embed_tokens = get_tensor(ctx, "token_embd.weight", tensors_to_load,
        // false); llm_head = get_tensor(ctx, "output.weight", tensors_to_load,
        // false);
    }
    return tensors_to_load;
}

std::vector<ggml_tensor *> FakeModel::build_graph(ggml_context *ctx0) {
    std::vector<int> x_shape = input_shapes_["x"];
    // {
    //   ggml_tensor *x =
    //       ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, x_shape[0], x_shape[1]);
    //   ggml_set_name(x, " x");
    //   ggml_set_input(x);
    //   ggml_tensor *weight = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 256, 1024);
    //   ggml_tensor *cur = build_linear(ctx0, x, weight, nullptr);
    //   return {cur};
    // }
    {
        // test flip kernel
        ggml_tensor *x = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, x_shape[0],
                                            x_shape[1], x_shape[2], x_shape[3]);
        ggml_set_name(x, "x");
        ggml_set_input(x);
        // ggml_tensor* cur = ggml_flip(ctx0, x, 0);
        ggml_tensor *cur = ggml_abs(ctx0, x);
        return {cur};
    }
}

// ==========================================ProjectorModel===========================================
std::vector<ggml_tensor *>
ProjectorModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    fc1_weight = get_tensor(ctx, "fc1.weight", tensors_to_load);
    fc1_bias = get_tensor(ctx, "fc1.bias", tensors_to_load);
    fc2_weight = get_tensor(ctx, "fc2.weight", tensors_to_load);
    fc2_bias = get_tensor(ctx, "fc2.bias", tensors_to_load);
    fc3_weight = get_tensor(ctx, "fc3.weight", tensors_to_load);
    fc3_bias = get_tensor(ctx, "fc3.bias", tensors_to_load);
    return tensors_to_load;
}

std::vector<ggml_tensor *> ProjectorModel::build_graph(ggml_context *ctx0) {
    std::vector<int> dinov2_shape = input_shapes_["dinov2_feat"];
    std::vector<int> siglip_shape = input_shapes_["siglip_feat"];
    ggml_tensor *inp_dinov2 =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dinov2_shape[0], dinov2_shape[1]);
    ggml_set_name(inp_dinov2, "dinov2_feat");
    ggml_set_input(inp_dinov2);

    ggml_tensor *inp_siglip =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, siglip_shape[0], siglip_shape[1]);
    ggml_set_name(inp_siglip, "siglip_feat");
    ggml_set_input(inp_siglip);

    ggml_tensor *inp = ggml_concat(ctx0, inp_dinov2, inp_siglip, 0);
    inp = build_linear(ctx0, inp, fc1_weight, fc1_bias);
    inp = ggml_gelu(ctx0, inp);

    inp = build_linear(ctx0, inp, fc2_weight, fc2_bias);
    inp = ggml_gelu(ctx0, inp);

    inp = build_linear(ctx0, inp, fc3_weight, fc3_bias);
    return {inp};
}

// ==========================================VisionTransformerModel===========================================
std::vector<ggml_tensor *>
VisionTransformerModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    {
        class_embedding = get_tensor(ctx, "v.class_embd", tensors_to_load, false);
        reg_embedding = get_tensor(ctx, "v.reg_embd", tensors_to_load, false);

        pre_ln_w = get_tensor(ctx, "v.pre_ln.weight", tensors_to_load, false);
        pre_ln_b = get_tensor(ctx, "v.pre_ln.bias", tensors_to_load, false);

        post_ln_w = get_tensor(ctx, "v.post_ln.weight", tensors_to_load, false);
        post_ln_b = get_tensor(ctx, "v.post_ln.bias", tensors_to_load, false);

        patch_bias = get_tensor(ctx, "v.patch_embd.bias", tensors_to_load, false);
        patch_embeddings_0 =
            get_tensor(ctx, "v.patch_embd.weight", tensors_to_load, false);
        patch_embeddings_1 =
            get_tensor(ctx, "v.patch_embd.weight.1", tensors_to_load, false);

        position_embeddings =
            get_tensor(ctx, "v.position_embd.weight", tensors_to_load, false);

        // layers
        int n_layer = hparams.n_layer;
        layers.resize(n_layer);
        const char *prefix = "v";
        for (int il = 0; il < n_layer; ++il) {
            auto &layer = layers[il];
            layer.k_w = get_tensor(
                ctx, string_format("%s.blk.%d.attn_k.%s", prefix, il, "weight"),
                tensors_to_load);
            layer.q_w = get_tensor(
                ctx, string_format("%s.blk.%d.attn_q.%s", prefix, il, "weight"),
                tensors_to_load);
            layer.v_w = get_tensor(
                ctx, string_format("%s.blk.%d.attn_v.%s", prefix, il, "weight"),
                tensors_to_load);
            layer.o_w = get_tensor(
                ctx, string_format("%s.blk.%d.attn_out.%s", prefix, il, "weight"),
                tensors_to_load);
            layer.k_norm = get_tensor(
                ctx, string_format("%s.blk.%d.attn_k_norm.%s", prefix, il, "weight"),
                tensors_to_load, false);
            layer.q_norm = get_tensor(
                ctx, string_format("%s.blk.%d.attn_q_norm.%s", prefix, il, "weight"),
                tensors_to_load, false);
            layer.ln_1_w = get_tensor(
                ctx, string_format("%s.blk.%d.ln1.%s", prefix, il, "weight"),
                tensors_to_load, false);
            layer.ln_2_w = get_tensor(
                ctx, string_format("%s.blk.%d.ln2.%s", prefix, il, "weight"),
                tensors_to_load, false);
            layer.ls_1_w = get_tensor(
                ctx, string_format("%s.blk.%d.ls1.%s", prefix, il, "weight"),
                tensors_to_load,
                false); // no bias
            layer.ls_2_w = get_tensor(
                ctx, string_format("%s.blk.%d.ls2.%s", prefix, il, "weight"),
                tensors_to_load,
                false); // no bias

            layer.k_b = get_tensor(
                ctx, string_format("%s.blk.%d.attn_k.%s", prefix, il, "bias"),
                tensors_to_load, false);
            layer.q_b = get_tensor(
                ctx, string_format("%s.blk.%d.attn_q.%s", prefix, il, "bias"),
                tensors_to_load, false);
            layer.v_b = get_tensor(
                ctx, string_format("%s.blk.%d.attn_v.%s", prefix, il, "bias"),
                tensors_to_load, false);
            layer.o_b = get_tensor(
                ctx, string_format("%s.blk.%d.attn_out.%s", prefix, il, "bias"),
                tensors_to_load, false);
            layer.ln_1_b =
                get_tensor(ctx, string_format("%s.blk.%d.ln1.%s", prefix, il, "bias"),
                           tensors_to_load, false);
            layer.ln_2_b =
                get_tensor(ctx, string_format("%s.blk.%d.ln2.%s", prefix, il, "bias"),
                           tensors_to_load, false);

            // ffn
            layer.ff_up_w = get_tensor(
                ctx, string_format("%s.blk.%d.ffn_up.%s", prefix, il, "weight"),
                tensors_to_load);
            layer.ff_up_b = get_tensor(
                ctx, string_format("%s.blk.%d.ffn_up.%s", prefix, il, "bias"),
                tensors_to_load, false);
            layer.ff_gate_w = get_tensor(
                ctx, string_format("%s.blk.%d.ffn_gate.%s", prefix, il, "weight"),
                tensors_to_load, false);
            layer.ff_gate_b = get_tensor(
                ctx, string_format("%s.blk.%d.ffn_gate.%s", prefix, il, "bias"),
                tensors_to_load, false);
            layer.ff_down_w = get_tensor(
                ctx, string_format("%s.blk.%d.ffn_down.%s", prefix, il, "weight"),
                tensors_to_load);
            layer.ff_down_b = get_tensor(
                ctx, string_format("%s.blk.%d.ffn_down.%s", prefix, il, "bias"),
                tensors_to_load, false);
        }
    }
    return tensors_to_load;
}

bool VisionTransformerModel::load_hparams(const ModelLoader &model_loader) {
    {
        const char *prefix = "vision";
        model_loader.get_u32(string_format("clip.%s.embedding_length", prefix),
                             hparams.n_embd);
        model_loader.get_u32(string_format("clip.%s.attention.head_count", prefix),
                             hparams.n_head);
        model_loader.get_u32(string_format("clip.%s.feed_forward_length", prefix),
                             hparams.n_ff);
        model_loader.get_u32(string_format("clip.%s.block_count", prefix),
                             hparams.n_layer);
        model_loader.get_u32(string_format("clip.%s.projection_dim", prefix),
                             hparams.projection_dim);
        model_loader.get_f32(
            string_format("clip.%s.attention.layer_norm_epsilon", prefix),
            hparams.eps);
        model_loader.get_u32("clip.vision.image_size", hparams.image_size);
        model_loader.get_u32("clip.vision.patch_size", hparams.patch_size);

        // for pinpoints, we need to convert it into a list of resolution
        // candidates

        // default warmup value

        {
            bool use_gelu = false;
            bool use_silu = false;
            model_loader.get_bool("clip.use_gelu", use_gelu, false);
            model_loader.get_bool("clip.use_silu", use_silu, false);
            if (use_gelu && use_silu) {
                throw std::runtime_error(string_format(
                    "%s: both use_gelu and use_silu are set to true\n", __func__));
            }
            if (use_gelu) {
                hparams.ffn_op = FFN_GELU;
            } else if (use_silu) {
                hparams.ffn_op = FFN_SILU;
            } else {
                hparams.ffn_op = FFN_GELU_QUICK;
            }
        }

        int idx_mean =
            gguf_find_key(model_loader.ctx_gguf_.get(), "clip.vision.image_mean");
        int idx_std =
            gguf_find_key(model_loader.ctx_gguf_.get(), "clip.vision.image_std");
        GGML_ASSERT(idx_mean >= 0 && "image_mean not found");
        GGML_ASSERT(idx_std >= 0 && "image_std not found");
        const float *mean_data = (const float *)gguf_get_arr_data(
            model_loader.ctx_gguf_.get(), idx_mean);
        const float *std_data =
            (const float *)gguf_get_arr_data(model_loader.ctx_gguf_.get(), idx_std);
        hparams.image_mean.resize(3);
        hparams.image_std.resize(3);
        for (int i = 0; i < 3; ++i) {
            hparams.image_mean[i] = mean_data[i];
            hparams.image_std[i] = std_data[i];
        }

        // Load the vision feature layer indices if they are explicitly provided;
        // if multiple vision feature layers are present, the values will be
        // concatenated to form the final visual features. NOTE: gguf conversions
        // should standardize the values of the vision feature layer to be
        // non-negative, since we use -1 to mark values as unset here.
        std::vector<int> vision_feature_layer;
        model_loader.get_arr_int("clip.vision.feature_layer", vision_feature_layer,
                                 false);
        // convert std::vector to std::unordered_set
        for (auto &layer : vision_feature_layer) {
            hparams.vision_feature_layer.insert(layer);
        }
    }
    return true;
}

ggml_tensor *VisionTransformerModel::build_vit(
    ggml_context *ctx0, ggml_tensor *inp, int n_pos, norm_type norm_t,
    std::function<ggml_tensor *(ggml_tensor *, const ClipLayer &)> add_pos) {
    int n_layer = hparams.n_layer;
    float eps = hparams.eps;
    int d_head = hparams.n_embd / hparams.n_head;
    int n_head = hparams.n_head;
    float kq_scale = 1.0f / sqrtf((float)d_head);
    int n_patches = hparams.image_size * hparams.image_size / hparams.patch_size / hparams.patch_size;
    ggml_tensor *learned_pos_embd = position_embeddings;
    ffn_op_type ffn_t = hparams.ffn_op;
    if (learned_pos_embd) {
        inp = ggml_add(ctx0, inp, learned_pos_embd);
        cb(ctx0, inp, "pos_embed", -1);
    }
    {
        // concat class_embedding、reg_embedding and inp
        ggml_tensor *to_cat = nullptr;
        if (class_embedding) {
            to_cat = class_embedding;
        }
        if (reg_embedding) {
            to_cat = ggml_concat(ctx0, to_cat, reg_embedding, 1);
        }
        if (to_cat) {
            inp = ggml_concat(ctx0, to_cat, inp, 1);
        }
    }
    // pre-layernorm
    if (pre_ln_w) {
        inp = build_norm(ctx0, inp, pre_ln_w, pre_ln_b, norm_t, eps, -1);
        cb(ctx0, inp, "pre_ln", -1);
    }
    ggml_tensor *inpL = inp;
    // loop over layers, return the second-to-last layer output
    for (int il = 0; il < n_layer - 1; il++) {
        const auto &layer = layers[il];
        ggml_tensor *cur = inpL; // inpL = residual, cur = hidden_states

        // layernorm1
        cur = build_norm(ctx0, cur, layer.ln_1_w, layer.ln_1_b, norm_t, eps, il);
        cb(ctx0, cur, "layer_inp_normed", il);

        // self-attention
        {
            ggml_tensor *Qcur = ggml_mul_mat(ctx0, layer.q_w, cur);
            if (layer.q_b) {
                Qcur = ggml_add(ctx0, Qcur, layer.q_b);
            }

            ggml_tensor *Kcur = ggml_mul_mat(ctx0, layer.k_w, cur);
            if (layer.k_b) {
                Kcur = ggml_add(ctx0, Kcur, layer.k_b);
            }

            ggml_tensor *Vcur = ggml_mul_mat(ctx0, layer.v_w, cur);
            if (layer.v_b) {
                Vcur = ggml_add(ctx0, Vcur, layer.v_b);
            }

            if (layer.q_norm) {
                Qcur = build_norm(ctx0, Qcur, layer.q_norm, NULL, norm_t, eps, il);
                cb(ctx0, Qcur, "Qcur_norm", il);
            }

            if (layer.k_norm) {
                Kcur = build_norm(ctx0, Kcur, layer.k_norm, NULL, norm_t, eps, il);
                cb(ctx0, Kcur, "Kcur_norm", il);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, d_head, n_head, n_pos);
            Kcur = ggml_reshape_3d(ctx0, Kcur, d_head, n_head, n_pos);
            Vcur = ggml_reshape_3d(ctx0, Vcur, d_head, n_head, n_pos);

            cb(ctx0, Qcur, "Qcur", il);
            cb(ctx0, Kcur, "Kcur", il);
            cb(ctx0, Vcur, "Vcur", il);

            if (add_pos) {
                Qcur = add_pos(Qcur, layer);
                Kcur = add_pos(Kcur, layer);
                cb(ctx0, Qcur, "Qcur_pos", il);
                cb(ctx0, Kcur, "Kcur_pos", il);
            }

            cur = build_attn(ctx0, layer.o_w, layer.o_b, Qcur, Kcur, Vcur, nullptr,
                             kq_scale, il);
            cb(ctx0, cur, "attn_out", il);
        }

        if (layer.ls_1_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_1_w);
            cb(ctx0, cur, "attn_out_scaled", il);
        }

        // re-add the layer input, e.g., residual
        cur = ggml_add(ctx0, cur, inpL);

        inpL = cur; // inpL = residual, cur = hidden_states

        cb(ctx0, cur, "ffn_inp", il);

        // layernorm2
        cur = build_norm(ctx0, cur, layer.ln_2_w, layer.ln_2_b, norm_t, eps, il);
        cb(ctx0, cur, "ffn_inp_normed", il);

        // ffn
        cur =
            build_ffn(ctx0, cur, layer.ff_up_w, layer.ff_up_b, layer.ff_gate_w,
                      layer.ff_gate_b, layer.ff_down_w, layer.ff_down_b, ffn_t, il);

        cb(ctx0, cur, "ffn_out", il);

        if (layer.ls_2_w) {
            cur = ggml_mul(ctx0, cur, layer.ls_2_w);
            cb(ctx0, cur, "ffn_out_scaled", il);
        }

        // residual 2
        cur = ggml_add(ctx0, inpL, cur);
        cb(ctx0, cur, "layer_out", il);

        inpL = cur;
    }
    // post-layernorm
    if (post_ln_w) {
        inpL = build_norm(ctx0, inpL, post_ln_w, post_ln_b, norm_t, eps, -1);
    }
    // For dinov2, output length is 5+256, need to extract the last 256
    int offset = n_pos - n_patches;
    if (offset > 0) {
        // remove class/reg tokens
        offset = offset * inpL->nb[1];
        inpL =
            ggml_view_2d(ctx0, inpL, inpL->ne[0], n_patches, inpL->nb[1], offset);
    }
    return inpL;
}
std::vector<ggml_tensor *>
VisionTransformerModel::build_graph(ggml_context *ctx0) {
    std::vector<ggml_tensor *> outputs;
    std::vector<int> input_shape = input_shapes_["inp_raw"];
    {
        int n_patches = hparams.image_size * hparams.image_size / hparams.patch_size / hparams.patch_size;
        // build_inp
        ggml_tensor *inp_raw = nullptr;
        if (input_shape.size() == 3) {
            inp_raw = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, input_shape[0],
                                         input_shape[1], input_shape[2]);
        } else {
            inp_raw =
                ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, input_shape[0],
                                   input_shape[1], input_shape[2], input_shape[3]);
        }
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);

        // conv2d
        ggml_tensor *inp =
            ggml_conv_2d(ctx0, patch_embeddings_0, inp_raw, hparams.patch_size,
                         hparams.patch_size, 0, 0, 1, 1);
        inp = ggml_reshape_2d(ctx0, inp, n_patches, hparams.n_embd);
        inp = ggml_cont(ctx0, ggml_transpose(ctx0, inp));
        if (patch_bias) {
            inp = ggml_add(ctx0, inp, patch_bias);
            cb(ctx0, inp, "patch_bias", -1);
        }

        int n_pos = n_patches;
        if (class_embedding) {
            n_pos += 1;
        }
        if (reg_embedding) {
            n_pos += 4;
        }
        ggml_tensor *cur = build_vit(ctx0, inp, n_pos, NORM_TYPE_NORMAL, nullptr);
        outputs.push_back(cur);
    }
    return outputs;
}

// ==========================================L1RegressionActionHeadFunnelModel===========================================
std::vector<ggml_tensor *>
L1RegressionActionHeadFunnelModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    {
        input_proj_ln_w =
            get_tensor(ctx, "input_proj.0.weight", tensors_to_load, true);
        input_proj_ln_b =
            get_tensor(ctx, "input_proj.0.bias", tensors_to_load, false);

        input_proj_fc_w =
            get_tensor(ctx, "input_proj.1.weight", tensors_to_load, true);
        input_proj_fc_b =
            get_tensor(ctx, "input_proj.1.bias", tensors_to_load, false);

        // blocks
        int num_blocks = hparams.num_blocks;
        resnet_body.resize(num_blocks);
        for (int il = 0; il < num_blocks; ++il) {
            auto &blk = resnet_body[il];
            blk.ffn_ln_w =
                get_tensor(ctx, string_format("resnet_body.%d.ffn.0.weight", il),
                           tensors_to_load, true);
            blk.ffn_ln_b =
                get_tensor(ctx, string_format("resnet_body.%d.ffn.0.bias", il),
                           tensors_to_load, false);
            blk.ffn_fc_w =
                get_tensor(ctx, string_format("resnet_body.%d.ffn.1.weight", il),
                           tensors_to_load, true);
            blk.ffn_fc_b =
                get_tensor(ctx, string_format("resnet_body.%d.ffn.1.bias", il),
                           tensors_to_load, false);
            blk.ffn_fc2_w =
                get_tensor(ctx, string_format("resnet_body.%d.ffn.3.weight", il),
                           tensors_to_load, true);
            blk.ffn_fc2_b =
                get_tensor(ctx, string_format("resnet_body.%d.ffn.3.bias", il),
                           tensors_to_load, false);
        }

        output_head_ln_w =
            get_tensor(ctx, "output_head.0.weight", tensors_to_load, true);
        output_head_ln_b =
            get_tensor(ctx, "output_head.0.bias", tensors_to_load, false);
        output_head_fc_w =
            get_tensor(ctx, "output_head.1.weight", tensors_to_load, true);
        output_head_fc_b =
            get_tensor(ctx, "output_head.1.bias", tensors_to_load, false);
    }
    return tensors_to_load;
}

bool L1RegressionActionHeadFunnelModel::load_hparams(
    const ModelLoader &model_loader) {
    model_loader.get_u32("action_dim", hparams.action_dim);
    model_loader.get_u32("num_actions_chunk", hparams.num_actions_chunk);
    model_loader.get_u32("num_actions_per_token", hparams.num_actions_per_token);
    model_loader.get_u32("num_blocks", hparams.num_blocks);
    model_loader.get_u32("input_dim", hparams.input_dim);
    model_loader.get_u32("hidden_dim", hparams.hidden_dim);
    model_loader.get_u32("expansion", hparams.expansion);
    return true;
}

ggml_tensor *MLPResNetBlockV2::build_graph(ggml_context *ctx0,
                                           ggml_tensor *inp) {
    ggml_tensor *cur =
        build_norm(ctx0, inp, ffn_ln_w, ffn_ln_b, NORM_TYPE_NORMAL, 1e-5, -1);
    cur = build_linear(ctx0, cur, ffn_fc_w, ffn_fc_b, -1);
    cur = ggml_silu(ctx0, cur);
    cur = build_linear(ctx0, cur, ffn_fc2_w, ffn_fc2_b, -1);
    cur = ggml_add(ctx0, inp, cur);
    return cur;
}
std::vector<ggml_tensor *>
L1RegressionActionHeadFunnelModel::build_graph(ggml_context *ctx0) {
    std::vector<ggml_tensor *> outputs;
    std::vector<int> input_shape = input_shapes_["inp_raw"];
    {
        ggml_tensor *inp_raw =
            ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.input_dim, 1);
        ggml_set_name(inp_raw, "inp_raw");
        ggml_set_input(inp_raw);

        // input_proj
        ggml_tensor *cur = build_norm(ctx0, inp_raw, input_proj_ln_w,
                                      input_proj_ln_b, NORM_TYPE_NORMAL, 1e-5, -1);
        cur = build_linear(ctx0, cur, input_proj_fc_w, input_proj_fc_b, -1);
        cur = ggml_silu(ctx0, cur);
        // blocks
        for (size_t i = 0; i < resnet_body.size(); i++) {
            cur = resnet_body[i].build_graph(ctx0, cur);
        }
        // output_head
        cur = build_norm(ctx0, cur, output_head_ln_w, output_head_ln_b,
                         NORM_TYPE_NORMAL, 1e-5, -1);
        cur = build_linear(ctx0, cur, output_head_fc_w, output_head_fc_b, -1);
        cur = ggml_reshape_2d(ctx0, cur, hparams.num_actions_chunk,
                              hparams.action_dim);
        outputs.push_back(cur);
    }
    return outputs;
}

ggml_tensor *MultiHeadAttention::build_graph(ggml_context *ctx0,
                                             ggml_tensor *inp,
                                             ggml_tensor *inp_kv, int n_q_head,
                                             int n_kv_head, float kq_scale,
                                             enum norm_type norm_type,
                                             float norm_eps, bool use_conv) {
    // calqulate QKV
    int n_embd_dim = q_w->ne[1];
    int n_embd_head = n_embd_dim / n_q_head;
    if (n_kv_head <= 0) {
        n_kv_head = n_q_head;
    }
    if (kq_scale <= 0) {
        kq_scale = 1.0f / sqrtf(n_embd_head);
    }
    if (!inp_kv) {
        inp_kv = inp;
    }
    int n_tokens = inp->ne[1];
    int n_tokens_kv = inp_kv->ne[1];
    ggml_tensor *cur = nullptr;
    if (use_conv) {
        // TODO conv attention
        // inp = ggml_permute(ctx0, inp, 1, 0, 2, 3);
        // inp = ggml_cont(ctx0, inp);
        // cur = build_norm(ctx0, inp, ln_1_w, ln_1_b, norm_type, norm_eps, -1);
        // cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
        // cur = ggml_cont(ctx0, cur);

    } else {
        // cur = build_norm(ctx0, inp, ln_1_w, ln_1_b, norm_type, norm_eps, -1);
        // return cur;
        ggml_tensor *Qcur = build_linear(ctx0, inp, q_w, q_b);
        ggml_tensor *Kcur = build_linear(ctx0, inp_kv, k_w, k_b);
        ggml_tensor *Vcur = build_linear(ctx0, inp_kv, v_w, v_b);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_q_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_kv_head, n_tokens_kv);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_kv_head, n_tokens_kv);

        cur = build_attn(ctx0, o_w, o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, -1);
        cur = ggml_add(ctx0, cur, inp);
        if (ln_1_w) {
            cur = build_norm(ctx0, cur, ln_1_w, ln_1_b, norm_type, norm_eps, -1);
        }
        return cur;
    }

    return cur;
}

ggml_tensor *MultiHeadAttention2::build_graph(ggml_context *ctx0,
                                              ggml_tensor *inp,
                                              ggml_tensor *inp_kv, int n_q_head,
                                              int n_kv_head, float kq_scale,
                                              enum norm_type norm_type,
                                              float norm_eps, bool use_conv) {
    // calqulate QKV
    int n_embd_dim = q_w->ne[1];
    int n_embd_head = n_embd_dim / n_q_head;
    if (n_kv_head <= 0) {
        n_kv_head = n_q_head;
    }
    if (kq_scale <= 0) {
        kq_scale = 1.0f / sqrtf(n_embd_head);
    }
    if (!inp_kv) {
        inp_kv = inp;
    }
    int n_tokens = inp->ne[1];
    int n_tokens_kv = inp_kv->ne[1];
    ggml_tensor *cur = nullptr;
    if (use_conv) {
        // TODO conv attention
        // inp = ggml_permute(ctx0, inp, 1, 0, 2, 3);
        // inp = ggml_cont(ctx0, inp);
        // cur = build_norm(ctx0, inp, ln_1_w, ln_1_b, norm_type, norm_eps, -1);
        // cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
        // cur = ggml_cont(ctx0, cur);

    } else {
        // cur = build_norm(ctx0, inp, ln_1_w, ln_1_b, norm_type, norm_eps, -1);
        // return cur;
        ggml_tensor *Qcur = build_linear(ctx0, inp, q_w, q_b);
        ggml_tensor *Kcur = build_linear(ctx0, inp_kv, k_w, k_b);
        ggml_tensor *Vcur = build_linear(ctx0, inp_kv, v_w, v_b);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_q_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_kv_head, n_tokens_kv);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_kv_head, n_tokens_kv);

        cur = build_attn(ctx0, o_w, o_b, Qcur, Kcur, Vcur, nullptr, kq_scale, -1);
        cur = ggml_add(ctx0, cur, inp);
        if (ln_1_w) {
            cur = build_norm(ctx0, cur, ln_1_w, ln_1_b, norm_type, norm_eps, -1);
        }
        return cur;
    }

    return cur;
}

// ==========================================VitArModel===========================================

std::vector<ggml_tensor *> VitsArModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    {
        ar_audio_position_pe =
            get_tensor(ctx, "ar_audio_position_pe", tensors_to_load, false);
        word_embeddings =
            get_tensor(ctx, "ar_audio_embedding.word_embeddings.weight",
                       tensors_to_load, false);
    }
    return tensors_to_load;
}

std::vector<ggml_tensor *> VitsArModel::build_graph(ggml_context *ctx0) {
    std::vector<int> token_shape = input_shapes_["inp_token"];
    std::vector<int> pos_shape = input_shapes_["inp_pos"];
    ggml_tensor *token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_shape[0]);
    ggml_tensor *pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, pos_shape[0]);
    ggml_set_name(token, "inp_token");
    ggml_set_input(token);
    ggml_set_name(pos, "inp_pos");
    ggml_set_input(pos);

    ggml_tensor *token_emb = ggml_get_rows(ctx0, word_embeddings, token);
    ggml_tensor *pos_emb = ggml_get_rows(ctx0, ar_audio_position_pe, pos);
    // token_emb * scale + alpha * pos_emb
    ggml_tensor *scaled_token_emb = ggml_scale(ctx0, token_emb, hparams.x_scale);
    ggml_tensor *scaled_pos_emb = ggml_scale(ctx0, pos_emb, hparams.alpha);
    ggml_tensor *out = ggml_add(ctx0, scaled_token_emb, scaled_pos_emb);
    return {out};
}

bool VitsArModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("vits_ar.embedding_length", hparams.embd_dim);
    model_loader.get_f32("alpha", hparams.alpha);
    model_loader.get_f32("x_scale", hparams.x_scale);
    return true;
}

// ==========================================VitTextModel===========================================

std::vector<ggml_tensor *> ArTextModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    {
        ar_text_position_pe =
            get_tensor(ctx, "ar_text_position_pe", tensors_to_load, false);
        word_embeddings =
            get_tensor(ctx, "ar_text_embedding.word_embeddings.weight",
                       tensors_to_load, false);
        bert_proj_weight =
            get_tensor(ctx, "bert_proj.weight", tensors_to_load, true);
        bert_proj_bias = get_tensor(ctx, "bert_proj.bias", tensors_to_load, true);
    }
    return tensors_to_load;
}

std::vector<ggml_tensor *> ArTextModel::build_graph(ggml_context *ctx0) {
    std::vector<int> token_shape = input_shapes_["inp_token"];
    std::vector<int> pos_shape = input_shapes_["inp_pos"];
    std::vector<int> bert_shape = input_shapes_["bert_feature"];

    ggml_tensor *token = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, token_shape[0]);
    ggml_tensor *pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, pos_shape[0]);
    ggml_tensor *bert_feature =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, bert_shape[0], bert_shape[1]);
    ggml_set_name(token, "inp_token");
    ggml_set_input(token);
    ggml_set_name(pos, "inp_pos");
    ggml_set_input(pos);
    ggml_set_name(bert_feature, "bert_feature");
    ggml_set_input(bert_feature);

    ggml_tensor *cur = ggml_get_rows(ctx0, word_embeddings, token);
    ggml_tensor *bert_proj =
        build_linear(ctx0, bert_feature, bert_proj_weight, bert_proj_bias);
    cur = ggml_add(ctx0, cur, bert_proj);
    ggml_tensor *pos_emb = ggml_get_rows(ctx0, ar_text_position_pe, pos);

    // token_emb * scale + alpha * pos_emb
    ggml_tensor *scaled_token_emb = ggml_scale(ctx0, cur, hparams.x_scale);
    ggml_tensor *scaled_pos_emb = ggml_scale(ctx0, pos_emb, hparams.alpha);
    ggml_tensor *out = ggml_add(ctx0, scaled_token_emb, scaled_pos_emb);
    return {out};
}
bool ArTextModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("vits_text.embedding_length", hparams.embd_dim);
    model_loader.get_f32("alpha", hparams.alpha);
    model_loader.get_f32("x_scale", hparams.x_scale);
    return true;
}

std::vector<ggml_tensor *>
MelStyleEncoderModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    {
        one = get_tensor(ctx, "one", tensors_to_load, true);
        spectral_fc1_weight =
            get_tensor(ctx, "spectral.0.fc.weight", tensors_to_load, true);
        spectral_fc1_bias =
            get_tensor(ctx, "spectral.0.fc.bias", tensors_to_load, false);
        spectral_fc2_weight =
            get_tensor(ctx, "spectral.3.fc.weight", tensors_to_load, true);
        spectral_fc2_bias =
            get_tensor(ctx, "spectral.3.fc.bias", tensors_to_load, false);

        temporal_conv1_weight =
            get_tensor(ctx, "temporal.0.conv1.conv.weight", tensors_to_load, true);
        temporal_conv1_bias =
            get_tensor(ctx, "temporal.0.conv1.conv.bias", tensors_to_load, false);
        temporal_conv2_weight =
            get_tensor(ctx, "temporal.1.conv1.conv.weight", tensors_to_load, true);
        temporal_conv2_bias =
            get_tensor(ctx, "temporal.1.conv1.conv.bias", tensors_to_load, false);

        slf_attn_w_qs_weight =
            get_tensor(ctx, "slf_attn.w_qs.weight", tensors_to_load, true);
        slf_attn_w_qs_bias =
            get_tensor(ctx, "slf_attn.w_qs.bias", tensors_to_load, false);
        slf_attn_w_ks_weight =
            get_tensor(ctx, "slf_attn.w_ks.weight", tensors_to_load, true);
        slf_attn_w_ks_bias =
            get_tensor(ctx, "slf_attn.w_ks.bias", tensors_to_load, false);
        slf_attn_w_vs_weight =
            get_tensor(ctx, "slf_attn.w_vs.weight", tensors_to_load, true);
        slf_attn_w_vs_bias =
            get_tensor(ctx, "slf_attn.w_vs.bias", tensors_to_load, false);
        slf_attn_fc_weight =
            get_tensor(ctx, "slf_attn.fc.weight", tensors_to_load, true);
        slf_attn_fc_bias =
            get_tensor(ctx, "slf_attn.fc.bias", tensors_to_load, false);

        fc_weight = get_tensor(ctx, "fc.fc.weight", tensors_to_load, true);
        fc_bias = get_tensor(ctx, "fc.fc.bias", tensors_to_load, false);
    }

    return tensors_to_load;
}

std::vector<ggml_tensor *>
MelStyleEncoderModel::build_graph(ggml_context *ctx0) {
    std::vector<int> x_shape = input_shapes_["x"];
    std::vector<int> codes_shape = input_shapes_["codes"];
    int n_embd_head = 64;
    int n_head = 2;
    int n_head_kv = 2;
    int n_tokens = x_shape[0];
    const float kq_scale = 1.0f / sqrtf(float(n_embd_head * n_head));
    ggml_tensor *ge = nullptr;

    // nx704x1
    ggml_tensor *inp_x = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, x_shape[0],
                                            x_shape[1], x_shape[2]);
    ggml_set_name(inp_x, "x");
    ggml_set_input(inp_x);

    ggml_tensor *codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, codes_shape[0]);
    ggml_set_name(codes, "codes");
    ggml_set_input(codes);

    // nx704x1 -> 704xnx1
    ggml_tensor *cur = ggml_permute(ctx0, inp_x, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    // return {cur};

    /*
    self.spectral = nn.Sequential(
          LinearNorm(self.in_dim, self.hidden_dim),
          Mish(),
          nn.Dropout(self.dropout),
          LinearNorm(self.hidden_dim, self.hidden_dim),
          Mish(),
          nn.Dropout(self.dropout),
      )
    */
    {
        // 704xnx1 -> 128xnx1
        cur = build_linear(ctx0, cur, spectral_fc1_weight, spectral_fc1_bias);
        cur = build_mish(ctx0, cur, one);
        cur = build_linear(ctx0, cur, spectral_fc2_weight, spectral_fc2_bias);
        cur = build_mish(ctx0, cur, one);
    }
    /*
    self.temporal = nn.Sequential(
          Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size,
    self.dropout), Conv1dGLU(self.hidden_dim, self.hidden_dim, self.kernel_size,
    self.dropout),
      )
    */
    // 128xnx1 -> nx128x1
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    {
        ggml_tensor *residual = cur;
        // nx256x1
        cur = build_conv1d(ctx0, cur, temporal_conv1_weight, temporal_conv1_bias, 1,
                           2, 1);
        ggml_tensor *x1 = ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2,
                                       cur->ne[2], cur->nb[1], cur->nb[2], 0);
        ggml_tensor *x2 =
            ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->ne[2],
                         cur->nb[1], cur->nb[2], cur->ne[1] / 2 * cur->nb[1]);
        cur = ggml_mul(ctx0, x1, ggml_sigmoid(ctx0, x2));
        cur = ggml_add(ctx0, cur, residual);
    }
    {
        ggml_tensor *residual = cur;
        // nx256x1
        cur = build_conv1d(ctx0, cur, temporal_conv2_weight, temporal_conv2_bias, 1,
                           2, 1);
        ggml_tensor *x1 = ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2,
                                       cur->ne[2], cur->nb[1], cur->nb[2], 0);
        ggml_tensor *x2 =
            ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->ne[2],
                         cur->nb[1], cur->nb[2], cur->ne[1] / 2 * cur->nb[1]);

        cur = ggml_mul(ctx0, x1, ggml_sigmoid(ctx0, x2));
        cur = ggml_add(ctx0, cur, residual);
    }
    // nx128x1 -> 128xnx1
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    // return {cur};
    {
        ggml_tensor *residual = cur;
        ggml_tensor *Qcur =
            build_linear(ctx0, cur, slf_attn_w_qs_weight, slf_attn_w_qs_bias);
        ggml_tensor *Kcur =
            build_linear(ctx0, cur, slf_attn_w_ks_weight, slf_attn_w_ks_bias);
        ggml_tensor *Vcur =
            build_linear(ctx0, cur, slf_attn_w_vs_weight, slf_attn_w_vs_bias);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);
        cur = build_attn(ctx0, slf_attn_fc_weight, slf_attn_fc_bias, Qcur, Kcur,
                         Vcur, nullptr, kq_scale, -1);
        cur = ggml_add(ctx0, cur, residual);

        cur = build_linear(ctx0, cur, fc_weight, fc_bias);
    }
    {
        // temoral average pooling
        // 512xnx1
        cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
        cur = ggml_cont(ctx0, cur);
        cur = ggml_sum_rows(ctx0, cur);
        cur = ggml_scale(ctx0, cur, 1.0f / n_tokens);
        ge = cur;
    }

    // {
    //   cur = ggml_get_rows(ctx0, codebook_embed, codes);
    //   cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    //   cur = ggml_cont(ctx0, cur);
    //   // F.interpolate
    //   cur = ggml_repeat(
    //       ctx0, cur,
    //       ggml_new_tensor_2d(ctx0, cur->type, cur->ne[0] * 2, cur->ne[1]));
    // }

    return {cur};
}

std::vector<ggml_tensor *>
EuclideanCodebookModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    { embed = get_tensor(ctx, "embed", tensors_to_load, true); }
    return tensors_to_load;
}

std::vector<ggml_tensor *>
EuclideanCodebookModel::build_graph(ggml_context *ctx0) {
    std::vector<int> codes_shape = input_shapes_["codes"];
    ggml_tensor *codes = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, codes_shape[0]);
    ggml_set_name(codes, "codes");
    ggml_set_input(codes);

    ggml_tensor *cur = ggml_get_rows(ctx0, embed, codes);
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_interpolate(ctx0, cur, cur->ne[0] * 2, cur->ne[1], cur->ne[2],
                           cur->ne[3], 0);
    cur = ggml_cont(ctx0, cur);
    return {cur};
}

std::vector<ggml_tensor *>
TextEncoderModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    int n_layers = hparams_.n_layer;
    // encoder_ssl
    {
        for (int i = 0; i < n_layers / 2; i++) {
            attn_layers_.push_back(MultiHeadAttention());
            ffn_layers_.push_back(TextEncoderFFN());
        }
        ssl_proj_weight = get_tensor(ctx, "ssl_proj.weight", tensors_to_load, true);
        ssl_proj_bias = get_tensor(ctx, "ssl_proj.bias", tensors_to_load, false);

        for (size_t i = 0; i < attn_layers_.size(); i++) {
            attn_layers_[i].q_w = get_tensor(ctx,
                                             "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_q.weight",
                                             tensors_to_load, true);
            attn_layers_[i].q_b = get_tensor(
                ctx, "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_q.bias",
                tensors_to_load, false);
            attn_layers_[i].k_w = get_tensor(ctx,
                                             "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_k.weight",
                                             tensors_to_load, true);
            attn_layers_[i].k_b = get_tensor(
                ctx, "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_k.bias",
                tensors_to_load, false);
            attn_layers_[i].v_w = get_tensor(ctx,
                                             "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_v.weight",
                                             tensors_to_load, true);
            attn_layers_[i].v_b = get_tensor(
                ctx, "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_v.bias",
                tensors_to_load, false);
            attn_layers_[i].o_w = get_tensor(ctx,
                                             "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_o.weight",
                                             tensors_to_load, true);
            attn_layers_[i].o_b = get_tensor(
                ctx, "encoder_ssl.attn_layers." + std::to_string(i) + ".conv_o.bias",
                tensors_to_load, false);

            // encoder_ssl.norm_layers_1.0.gamma and encoder_ssl.norm_layers_1.0.beta
            attn_layers_[i].ln_1_w = get_tensor(
                ctx, "encoder_ssl.norm_layers_1." + std::to_string(i) + ".gamma",
                tensors_to_load, false);
            attn_layers_[i].ln_1_b = get_tensor(
                ctx, "encoder_ssl.norm_layers_1." + std::to_string(i) + ".beta",
                tensors_to_load, false);

            ffn_layers_[i].conv1_weight = get_tensor(
                ctx, "encoder_ssl.ffn_layers." + std::to_string(i) + ".conv_1.weight",
                tensors_to_load, true);
            ffn_layers_[i].conv1_bias = get_tensor(
                ctx, "encoder_ssl.ffn_layers." + std::to_string(i) + ".conv_1.bias",
                tensors_to_load, false);
            ffn_layers_[i].conv2_weight = get_tensor(
                ctx, "encoder_ssl.ffn_layers." + std::to_string(i) + ".conv_2.weight",
                tensors_to_load, true);
            ffn_layers_[i].conv2_bias = get_tensor(
                ctx, "encoder_ssl.ffn_layers." + std::to_string(i) + ".conv_2.bias",
                tensors_to_load, false);
            ffn_layers_[i].ln_w = get_tensor(
                ctx, "encoder_ssl.norm_layers_2." + std::to_string(i) + ".gamma",
                tensors_to_load, false);
            ffn_layers_[i].ln_b = get_tensor(
                ctx, "encoder_ssl.norm_layers_2." + std::to_string(i) + ".beta",
                tensors_to_load, false);
        }
    }
    text_embedding_weight =
        get_tensor(ctx, "text_embedding.weight", tensors_to_load, true);
    // encoder_text
    {
        for (int i = 0; i < n_layers; i++) {
            attn_layers_text_.push_back(MultiHeadAttention());
            ffn_layers_text_.push_back(TextEncoderFFN());
        }
        for (size_t i = 0; i < attn_layers_text_.size(); i++) {
            attn_layers_text_[i].q_w = get_tensor(
                ctx,
                "encoder_text.attn_layers." + std::to_string(i) + ".conv_q.weight",
                tensors_to_load, true);
            attn_layers_text_[i].q_b = get_tensor(
                ctx, "encoder_text.attn_layers." + std::to_string(i) + ".conv_q.bias",
                tensors_to_load, false);
            attn_layers_text_[i].k_w = get_tensor(
                ctx,
                "encoder_text.attn_layers." + std::to_string(i) + ".conv_k.weight",
                tensors_to_load, true);
            attn_layers_text_[i].k_b = get_tensor(
                ctx, "encoder_text.attn_layers." + std::to_string(i) + ".conv_k.bias",
                tensors_to_load, false);
            attn_layers_text_[i].v_w = get_tensor(
                ctx,
                "encoder_text.attn_layers." + std::to_string(i) + ".conv_v.weight",
                tensors_to_load, true);
            attn_layers_text_[i].v_b = get_tensor(
                ctx, "encoder_text.attn_layers." + std::to_string(i) + ".conv_v.bias",
                tensors_to_load, false);
            attn_layers_text_[i].o_w = get_tensor(
                ctx,
                "encoder_text.attn_layers." + std::to_string(i) + ".conv_o.weight",
                tensors_to_load, true);
            attn_layers_text_[i].o_b = get_tensor(
                ctx, "encoder_text.attn_layers." + std::to_string(i) + ".conv_o.bias",
                tensors_to_load, false);

            // encoder_text.norm_layers_1.0.gamma and encoder_text.norm_layers_1.0.beta
            attn_layers_text_[i].ln_1_w = get_tensor(
                ctx, "encoder_text.norm_layers_1." + std::to_string(i) + ".gamma",
                tensors_to_load, false);
            attn_layers_text_[i].ln_1_b = get_tensor(
                ctx, "encoder_text.norm_layers_1." + std::to_string(i) + ".beta",
                tensors_to_load, false);

            ffn_layers_text_[i].conv1_weight = get_tensor(
                ctx,
                "encoder_text.ffn_layers." + std::to_string(i) + ".conv_1.weight",
                tensors_to_load, true);
            ffn_layers_text_[i].conv1_bias = get_tensor(
                ctx, "encoder_text.ffn_layers." + std::to_string(i) + ".conv_1.bias",
                tensors_to_load, false);
            ffn_layers_text_[i].conv2_weight = get_tensor(
                ctx,
                "encoder_text.ffn_layers." + std::to_string(i) + ".conv_2.weight",
                tensors_to_load, true);
            ffn_layers_text_[i].conv2_bias = get_tensor(
                ctx, "encoder_text.ffn_layers." + std::to_string(i) + ".conv_2.bias",
                tensors_to_load, false);
            ffn_layers_text_[i].ln_w = get_tensor(
                ctx, "encoder_text.norm_layers_2." + std::to_string(i) + ".gamma",
                tensors_to_load, false);
            ffn_layers_text_[i].ln_b = get_tensor(
                ctx, "encoder_text.norm_layers_2." + std::to_string(i) + ".beta",
                tensors_to_load, false);
        }
    }
    // mrte
    {
        mrte_c_pre_weight =
            get_tensor(ctx, "mrte.c_pre.weight", tensors_to_load, true);
        mrte_c_pre_bias =
            get_tensor(ctx, "mrte.c_pre.bias", tensors_to_load, false);
        mrte_text_pre_weight =
            get_tensor(ctx, "mrte.text_pre.weight", tensors_to_load, true);
        mrte_text_pre_bias =
            get_tensor(ctx, "mrte.text_pre.bias", tensors_to_load, false);

        mrte_attn_layer_.q_w = get_tensor(ctx, "mrte.cross_attention.conv_q.weight",
                                          tensors_to_load, true);
        mrte_attn_layer_.q_b = get_tensor(ctx, "mrte.cross_attention.conv_q.bias",
                                          tensors_to_load, false);
        mrte_attn_layer_.k_w = get_tensor(ctx, "mrte.cross_attention.conv_k.weight",
                                          tensors_to_load, true);
        mrte_attn_layer_.k_b = get_tensor(ctx, "mrte.cross_attention.conv_k.bias",
                                          tensors_to_load, false);
        mrte_attn_layer_.v_w = get_tensor(ctx, "mrte.cross_attention.conv_v.weight",
                                          tensors_to_load, true);
        mrte_attn_layer_.v_b = get_tensor(ctx, "mrte.cross_attention.conv_v.bias",
                                          tensors_to_load, false);
        mrte_attn_layer_.o_w = get_tensor(ctx, "mrte.cross_attention.conv_o.weight",
                                          tensors_to_load, true);
        mrte_attn_layer_.o_b = get_tensor(ctx, "mrte.cross_attention.conv_o.bias",
                                          tensors_to_load, false);
        mrte_c_post_weight =
            get_tensor(ctx, "mrte.c_post.weight", tensors_to_load, true);
        mrte_c_post_bias =
            get_tensor(ctx, "mrte.c_post.bias", tensors_to_load, false);
    }
    // encoder2
    {
        for (int i = 0; i < n_layers / 2; i++) {
            attn_layers_encoder2_.push_back(MultiHeadAttention());
            ffn_layers_encoder2_.push_back(TextEncoderFFN());
        }
        ssl_proj_weight = get_tensor(ctx, "ssl_proj.weight", tensors_to_load, true);
        ssl_proj_bias = get_tensor(ctx, "ssl_proj.bias", tensors_to_load, false);

        for (size_t i = 0; i < attn_layers_encoder2_.size(); i++) {
            attn_layers_encoder2_[i].q_w = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_q.weight",
                tensors_to_load, true);
            attn_layers_encoder2_[i].q_b = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_q.bias",
                tensors_to_load, false);
            attn_layers_encoder2_[i].k_w = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_k.weight",
                tensors_to_load, true);
            attn_layers_encoder2_[i].k_b = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_k.bias",
                tensors_to_load, false);
            attn_layers_encoder2_[i].v_w = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_v.weight",
                tensors_to_load, true);
            attn_layers_encoder2_[i].v_b = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_v.bias",
                tensors_to_load, false);
            attn_layers_encoder2_[i].o_w = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_o.weight",
                tensors_to_load, true);
            attn_layers_encoder2_[i].o_b = get_tensor(
                ctx, "encoder2.attn_layers." + std::to_string(i) + ".conv_o.bias",
                tensors_to_load, false);

            // encoder2.norm_layers_1.0.gamma and encoder2.norm_layers_1.0.beta
            attn_layers_encoder2_[i].ln_1_w = get_tensor(
                ctx, "encoder2.norm_layers_1." + std::to_string(i) + ".gamma",
                tensors_to_load, false);
            attn_layers_encoder2_[i].ln_1_b = get_tensor(
                ctx, "encoder2.norm_layers_1." + std::to_string(i) + ".beta",
                tensors_to_load, false);

            ffn_layers_encoder2_[i].conv1_weight = get_tensor(
                ctx, "encoder2.ffn_layers." + std::to_string(i) + ".conv_1.weight",
                tensors_to_load, true);
            ffn_layers_encoder2_[i].conv1_bias = get_tensor(
                ctx, "encoder2.ffn_layers." + std::to_string(i) + ".conv_1.bias",
                tensors_to_load, false);
            ffn_layers_encoder2_[i].conv2_weight = get_tensor(
                ctx, "encoder2.ffn_layers." + std::to_string(i) + ".conv_2.weight",
                tensors_to_load, true);
            ffn_layers_encoder2_[i].conv2_bias = get_tensor(
                ctx, "encoder2.ffn_layers." + std::to_string(i) + ".conv_2.bias",
                tensors_to_load, false);
            ffn_layers_encoder2_[i].ln_w = get_tensor(
                ctx, "encoder2.norm_layers_2." + std::to_string(i) + ".gamma",
                tensors_to_load, false);
            ffn_layers_encoder2_[i].ln_b = get_tensor(
                ctx, "encoder2.norm_layers_2." + std::to_string(i) + ".beta",
                tensors_to_load, false);
        }
    }
    proj_weight = get_tensor(ctx, "proj.weight", tensors_to_load, true);
    proj_bias = get_tensor(ctx, "proj.bias", tensors_to_load, false);
    return tensors_to_load;
}

ggml_tensor *TextEncoderFFN::build_graph(ggml_context *ctx0, ggml_tensor *inp,
                                         norm_type norm_type, float norm_eps) {
    inp = ggml_permute(ctx0, inp, 1, 0, 2, 3);
    inp = ggml_cont(ctx0, inp);
    int pad_size = (conv1_weight->ne[0] - 1) / 2;
    ggml_tensor *cur =
        build_conv1d(ctx0, inp, conv1_weight, conv1_bias, 1, pad_size, 1);
    cur = ggml_relu(ctx0, cur);
    pad_size = (conv2_weight->ne[0] - 1) / 2;
    cur = build_conv1d(ctx0, cur, conv2_weight, conv2_bias, 1, pad_size, 1);
    cur = ggml_add(ctx0, cur, inp);
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    cur = build_norm(ctx0, cur, ln_w, ln_b, norm_type, norm_eps);
    return cur;
}

std::vector<ggml_tensor *> TextEncoderModel::build_graph(ggml_context *ctx0) {
    std::vector<int> quantized_shape = input_shapes_["quantized"];
    std::vector<int> phones_shape = input_shapes_["phones"];
    std::vector<int> ge_shape = input_shapes_["ge"];

    ggml_tensor *quantized =
        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, quantized_shape[0],
                           quantized_shape[1], quantized_shape[2]);
    ggml_set_name(quantized, "quantized");
    ggml_set_input(quantized);
    ggml_tensor *phones =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, phones_shape[0], phones_shape[1]);
    ggml_set_name(phones, "phones");
    ggml_set_input(phones);

    ggml_tensor *ge =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, ge_shape[0], ge_shape[1]);
    ggml_set_name(ge, "ge");
    ggml_set_input(ge);

    // output
    ggml_tensor *means = nullptr;
    ggml_tensor *logs = nullptr;

    ggml_tensor *cur =
        build_conv1d(ctx0, quantized, ssl_proj_weight, ssl_proj_bias, 1, 0, 1);
    cur = ggml_permute(ctx0, cur, 1, 0, 2, 3);
    cur = ggml_cont(ctx0, cur);
    // encoder_ssl
    for (size_t i = 0; i < attn_layers_.size(); i++) {
        cur = attn_layers_[i].build_graph(ctx0, cur, nullptr, hparams_.n_q_heads,
                                          hparams_.n_kv_heads, -1, NORM_TYPE_NORMAL,
                                          hparams_.norm_eps);
        cur = ffn_layers_[i].build_graph(ctx0, cur, NORM_TYPE_NORMAL,
                                         hparams_.norm_eps);
    }
    ggml_tensor *out_ssl = cur;

    // phones
    cur = ggml_get_rows(ctx0, text_embedding_weight, phones);
    for (size_t i = 0; i < attn_layers_text_.size(); i++) {
        cur = attn_layers_text_[i].build_graph(
            ctx0, cur, nullptr, hparams_.n_q_heads, hparams_.n_kv_heads, -1,
            NORM_TYPE_NORMAL, hparams_.norm_eps);
        cur = ffn_layers_text_[i].build_graph(ctx0, cur, NORM_TYPE_NORMAL,
                                              hparams_.norm_eps);
    }
    ggml_tensor *out_text = cur;
    // return {out_ssl, out_text};

    // mrte
    {
        out_ssl = ggml_cont(ctx0, ggml_permute(ctx0, out_ssl, 1, 0, 2, 3));
        out_text = ggml_cont(ctx0, ggml_permute(ctx0, out_text, 1, 0, 2, 3));
        ggml_tensor *ssl_enc = build_conv1d(ctx0, out_ssl, mrte_c_pre_weight,
                                            mrte_c_pre_bias, 1, 0, 1);
        ggml_tensor *text_enc = build_conv1d(ctx0, out_text, mrte_text_pre_weight,
                                             mrte_text_pre_bias, 1, 0, 1);
        // return {ssl_enc, text_enc};
        // cross attention
        ssl_enc = ggml_cont(ctx0, ggml_permute(ctx0, ssl_enc, 1, 0, 2, 3));
        text_enc = ggml_cont(ctx0, ggml_permute(ctx0, text_enc, 1, 0, 2, 3));
        // return {ssl_enc, text_enc};
        cur = mrte_attn_layer_.build_graph(
            ctx0, ssl_enc, text_enc, hparams_.n_heads_mrte, hparams_.n_heads_mrte,
            -1, NORM_TYPE_NORMAL, hparams_.norm_eps);
        // return {cur, text_enc};
        cur = ggml_add(ctx0, cur, ge);
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
        cur =
            build_conv1d(ctx0, cur, mrte_c_post_weight, mrte_c_post_bias, 1, 0, 1);
    }

    // encoder2
    {
        cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
        for (size_t i = 0; i < attn_layers_encoder2_.size(); i++) {
            cur = attn_layers_encoder2_[i].build_graph(
                ctx0, cur, nullptr, hparams_.n_q_heads, hparams_.n_kv_heads, -1,
                NORM_TYPE_NORMAL, hparams_.norm_eps);
            cur = ffn_layers_encoder2_[i].build_graph(ctx0, cur, NORM_TYPE_NORMAL,
                                                      hparams_.norm_eps);
        }
    }
    // proj
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = build_conv1d(ctx0, cur, proj_weight, proj_bias, 1, 0, 1);

    // split
    means = ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->ne[2],
                         cur->nb[1], cur->nb[2], 0);
    logs = ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->ne[2],
                        cur->nb[1], cur->nb[2], cur->ne[1] / 2 * cur->nb[1]);
    logs = ggml_exp(ctx0, logs);

    return {means, logs};
}

bool TextEncoderModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("n_layer", hparams_.n_layer);
    model_loader.get_u32("n_q_heads", hparams_.n_q_heads);
    model_loader.get_u32("n_kv_heads", hparams_.n_kv_heads);
    model_loader.get_u32("n_heads_mrte", hparams_.n_heads_mrte);
    model_loader.get_f32("norm_eps", hparams_.norm_eps);

    return true;
}

std::vector<ggml_tensor *>
GeneratorModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    generate_blocks_.resize(hparams_.num_upsamples);
    generate_blocks_[0].params = {.stride = 10, .padding = 0};
    generate_blocks_[1].params = {.stride = 8, .padding = 0};
    generate_blocks_[2].params = {.stride = 2, .padding = 0};
    generate_blocks_[3].params = {.stride = 2, .padding = 0};
    generate_blocks_[4].params = {.stride = 2, .padding = 0};
    // generate_blocks_[0].params = {.stride = 10, .padding = 3};
    // generate_blocks_[1].params = {.stride = 8, .padding = 4};
    // generate_blocks_[2].params = {.stride = 2, .padding = 3};
    // generate_blocks_[3].params = {.stride = 2, .padding = 0};
    // generate_blocks_[4].params = {.stride = 2, .padding = 0};
    for (size_t i = 0; i < generate_blocks_.size(); i++) {
        generate_blocks_[i].resblocks.resize(hparams_.num_kernels);
    }

    conv_pre_weight = get_tensor(ctx, "conv_pre.weight", tensors_to_load, true);
    conv_pre_bias = get_tensor(ctx, "conv_pre.bias", tensors_to_load, false);

    cond_weight = get_tensor(ctx, "cond.weight", tensors_to_load, true);
    cond_bias = get_tensor(ctx, "cond.bias", tensors_to_load, false);

    for (int i = 0; i < generate_blocks_.size(); i++) {
        generate_blocks_[i].up_weight = get_tensor(
            ctx, "ups." + std::to_string(i) + ".weight", tensors_to_load, true);
        generate_blocks_[i].up_bias = get_tensor(
            ctx, "ups." + std::to_string(i) + ".bias", tensors_to_load, false);
        for (int j = 0; j < generate_blocks_[i].resblocks.size(); j++) {
            int blk_idx = i * hparams_.num_kernels + j;
            std::string blk_prefix = "resblocks." + std::to_string(blk_idx);
            generate_blocks_[i].resblocks[j].conv1_0_weight = get_tensor(
                ctx, blk_prefix + ".convs1.0.weight", tensors_to_load, true);
            generate_blocks_[i].resblocks[j].conv1_0_bias = get_tensor(
                ctx, blk_prefix + ".convs1.0.bias", tensors_to_load, false);
            generate_blocks_[i].resblocks[j].conv1_1_weight = get_tensor(
                ctx, blk_prefix + ".convs1.1.weight", tensors_to_load, true);
            generate_blocks_[i].resblocks[j].conv1_1_bias = get_tensor(
                ctx, blk_prefix + ".convs1.1.bias", tensors_to_load, false);
            generate_blocks_[i].resblocks[j].conv1_2_weight = get_tensor(
                ctx, blk_prefix + ".convs1.2.weight", tensors_to_load, true);
            generate_blocks_[i].resblocks[j].conv1_2_bias = get_tensor(
                ctx, blk_prefix + ".convs1.2.bias", tensors_to_load, false);
            generate_blocks_[i].resblocks[j].conv2_0_weight = get_tensor(
                ctx, blk_prefix + ".convs2.0.weight", tensors_to_load, true);
            generate_blocks_[i].resblocks[j].conv2_0_bias = get_tensor(
                ctx, blk_prefix + ".convs2.0.bias", tensors_to_load, false);
            generate_blocks_[i].resblocks[j].conv2_1_weight = get_tensor(
                ctx, blk_prefix + ".convs2.1.weight", tensors_to_load, true);
            generate_blocks_[i].resblocks[j].conv2_1_bias = get_tensor(
                ctx, blk_prefix + ".convs2.1.bias", tensors_to_load, false);
            generate_blocks_[i].resblocks[j].conv2_2_weight = get_tensor(
                ctx, blk_prefix + ".convs2.2.weight", tensors_to_load, true);
            generate_blocks_[i].resblocks[j].conv2_2_bias = get_tensor(
                ctx, blk_prefix + ".convs2.2.bias", tensors_to_load, false);
        }
    }

    conv_post_weight = get_tensor(ctx, "conv_post.weight", tensors_to_load, true);
    conv_post_bias = get_tensor(ctx, "conv_post.bias", tensors_to_load, false);

    return tensors_to_load;
}

std::vector<ggml_tensor *> GeneratorModel::build_graph(ggml_context *ctx0) {
    std::vector<int> ge_shape = input_shapes_["ge"];
    std::vector<int> x_shape = input_shapes_["x"];

    ggml_tensor *ge = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ge_shape[0],
                                         ge_shape[1], ge_shape[2]);
    ggml_set_name(ge, "ge");
    ggml_set_input(ge);

    ggml_tensor *inp_x = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, x_shape[0],
                                            x_shape[1], x_shape[2]);
    ggml_set_name(inp_x, "x");
    ggml_set_input(inp_x);

    // x = conv_pre(x)
    ggml_tensor *x =
        build_conv1d(ctx0, inp_x, conv_pre_weight, conv_pre_bias, 1, 3, 1);
    // x = x + cond(g)
    ggml_tensor *cur = build_conv1d(ctx0, ge, cond_weight, cond_bias, 1, 0, 1);
    cur = ggml_repeat(ctx0, cur, x);
    cur = ggml_add(ctx0, cur, x);

    for (size_t i = 0; i < generate_blocks_.size(); i++) {
        // x = generate_block(x)
        cur = generate_blocks_[i].build_graph(ctx0, cur, hparams_.lrelu_slope);
    }

    cur = ggml_leaky_relu(ctx0, cur, 0.01, false);
    cur = build_conv1d(ctx0, cur, conv_post_weight, conv_post_bias, 1, 3, 1);
    cur = ggml_tanh(ctx0, cur);
    return {cur};
}

bool GeneratorModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("num_upsamples", hparams_.num_upsamples, false);
    model_loader.get_u32("num_kernels", hparams_.num_kernels, false);
    model_loader.get_f32("norm_eps", hparams_.lrelu_slope, false);

    return true;
}

ggml_tensor *ResBlock1::build_graph(ggml_context *ctx0, ggml_tensor *inp) {
    ggml_tensor *x = inp;
    ggml_tensor *xt = ggml_leaky_relu(ctx0, x, 0.1, false);

    const int kernel_size = conv1_0_weight->ne[0];

    std::vector<int> dilations = {1, 3, 5};

    int idx_conv = 0;
    int dilation = dilations[idx_conv];
    int padding = (kernel_size * dilation - dilation) / 2;
    xt = build_conv1d(ctx0, xt, conv1_0_weight, conv1_0_bias, 1, padding,
                      dilation);
    xt = ggml_leaky_relu(ctx0, xt, 0.1, false);

    dilation = 1;
    padding = (kernel_size * dilation - dilation) / 2;
    xt = build_conv1d(ctx0, xt, conv2_0_weight, conv2_0_bias, 1, padding,
                      dilation);
    x = ggml_add(ctx0, x, xt);

    idx_conv = 1;
    dilation = dilations[idx_conv];
    padding = (kernel_size * dilation - dilation) / 2;
    xt = ggml_leaky_relu(ctx0, x, 0.1, false);
    xt = build_conv1d(ctx0, xt, conv1_1_weight, conv1_1_bias, 1, padding,
                      dilation);
    xt = ggml_leaky_relu(ctx0, xt, 0.1, false);
    dilation = 1;
    padding = (kernel_size * dilation - dilation) / 2;
    xt = build_conv1d(ctx0, xt, conv2_1_weight, conv2_1_bias, 1, padding,
                      dilation);
    x = ggml_add(ctx0, x, xt);

    idx_conv = 2;
    dilation = dilations[idx_conv];
    padding = (kernel_size * dilation - dilation) / 2;
    xt = ggml_leaky_relu(ctx0, x, 0.1, false);
    xt = build_conv1d(ctx0, xt, conv1_2_weight, conv1_2_bias, 1, padding,
                      dilation);
    xt = ggml_leaky_relu(ctx0, xt, 0.1, false);
    dilation = 1;
    padding = (kernel_size * dilation - dilation) / 2;
    xt = build_conv1d(ctx0, xt, conv2_2_weight, conv2_2_bias, 1, padding,
                      dilation);
    x = ggml_add(ctx0, x, xt);

    return x;
}

ggml_tensor *GeneratorBlock::build_graph(ggml_context *ctx0, ggml_tensor *inp,
                                         float slope) {
    float scale = 1.f / resblocks.size();
    ggml_tensor *x = ggml_leaky_relu(ctx0, inp, slope, false);
    x = build_conv_transpose1d(ctx0, x, up_weight, up_bias, params.stride,
                               params.padding, 1);

    ggml_tensor *xs = nullptr;
    if (resblocks.size() > 0) {
        xs = resblocks[0].build_graph(ctx0, x);
    }
    for (int i = 1; i < resblocks.size(); i++) {
        ggml_tensor *cur = resblocks[i].build_graph(ctx0, x);
        xs = ggml_add(ctx0, xs, cur);
    }
    x = ggml_scale(ctx0, xs, scale);
    return x;
}

std::vector<ggml_tensor *>
ResidualCouplingBlock::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;
    layers.resize(4);
    {
        for (int i = 0; i < layers.size(); ++i) {
            int blk_idx = i * 2;
            layers[i].pre_weight =
                get_tensor(ctx, "flows." + std::to_string(blk_idx) + ".pre.weight",
                           tensors_to_load, true);
            layers[i].pre_bias =
                get_tensor(ctx, "flows." + std::to_string(blk_idx) + ".pre.bias",
                           tensors_to_load, false);
            layers[i].post_weight =
                get_tensor(ctx, "flows." + std::to_string(blk_idx) + ".post.weight",
                           tensors_to_load, true);
            layers[i].post_bias =
                get_tensor(ctx, "flows." + std::to_string(blk_idx) + ".post.bias",
                           tensors_to_load, false);
            layers[i].enc.cond_layer_weight = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.cond_layer.weight",
                tensors_to_load, true);
            layers[i].enc.cond_layer_bias = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.cond_layer.bias",
                tensors_to_load, false);

            layers[i].enc.in_layer_0_weight = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.0.weight",
                tensors_to_load, true);
            layers[i].enc.in_layer_0_bias = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.0.bias",
                tensors_to_load, false);
            layers[i].enc.in_layer_1_weight = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.1.weight",
                tensors_to_load, true);
            layers[i].enc.in_layer_1_bias = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.1.bias",
                tensors_to_load, false);
            layers[i].enc.in_layer_2_weight = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.2.weight",
                tensors_to_load, true);
            layers[i].enc.in_layer_2_bias = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.2.bias",
                tensors_to_load, false);
            layers[i].enc.in_layer_3_weight = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.3.weight",
                tensors_to_load, true);
            layers[i].enc.in_layer_3_bias = get_tensor(
                ctx, "flows." + std::to_string(blk_idx) + ".enc.in_layers.3.bias",
                tensors_to_load, false);

            layers[i].enc.res_skip_layers_0_weight = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.0.weight",
                tensors_to_load, true);
            layers[i].enc.res_skip_layers_0_bias = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.0.bias",
                tensors_to_load, false);
            layers[i].enc.res_skip_layers_1_weight = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.1.weight",
                tensors_to_load, true);
            layers[i].enc.res_skip_layers_1_bias = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.1.bias",
                tensors_to_load, false);
            layers[i].enc.res_skip_layers_2_weight = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.2.weight",
                tensors_to_load, true);
            layers[i].enc.res_skip_layers_2_bias = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.2.bias",
                tensors_to_load, false);
            layers[i].enc.res_skip_layers_3_weight = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.3.weight",
                tensors_to_load, true);
            layers[i].enc.res_skip_layers_3_bias = get_tensor(
                ctx,
                "flows." + std::to_string(blk_idx) + ".enc.res_skip_layers.3.bias",
                tensors_to_load, false);
        }
    }
    return tensors_to_load;
}

ggml_tensor *WN::build_graph(ggml_context *ctx0, ggml_tensor *x,
                             ggml_tensor *ge) {
    ge = build_conv1d(ctx0, ge, cond_layer_weight, cond_layer_bias, 1, 0, 1);
    ggml_tensor *g0 = ggml_view_3d(ctx0, ge, ge->ne[0], ge->ne[1] / 4, ge->ne[2],
                                   ge->nb[1], ge->nb[2], 0);
    ggml_tensor *g1 =
        ggml_view_3d(ctx0, ge, ge->ne[0], ge->ne[1] / 4, ge->ne[2], ge->nb[1],
                     ge->nb[2], ge->ne[1] / 4 * ge->nb[1]);
    ggml_tensor *g2 =
        ggml_view_3d(ctx0, ge, ge->ne[0], ge->ne[1] / 4, ge->ne[2], ge->nb[1],
                     ge->nb[2], ge->ne[1] / 2 * ge->nb[1]);
    ggml_tensor *g3 =
        ggml_view_3d(ctx0, ge, ge->ne[0], ge->ne[1] / 4, ge->ne[2], ge->nb[1],
                     ge->nb[2], ge->ne[1] * 3 / 4 * ge->nb[1]);
    ggml_tensor *x_in, *in_act, *t_act, *s_act, *acts, *res_skip_acts, *res_acts,
        *res_acts2, *output, *gl;
    // layer0
    gl = g0;
    x_in = build_conv1d(ctx0, x, in_layer_0_weight, in_layer_0_bias, 1, 2, 1);
    in_act = ggml_add(ctx0, x_in, gl);
    t_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2], 0);
    s_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2],
                         in_act->ne[1] / 2 * in_act->nb[1]);
    t_act = ggml_tanh(ctx0, t_act);
    s_act = ggml_sigmoid(ctx0, s_act);
    acts = ggml_mul(ctx0, t_act, s_act);

    res_skip_acts = build_conv1d(ctx0, acts, res_skip_layers_0_weight,
                                 res_skip_layers_0_bias, 1, 0, 1);
    res_acts = ggml_view_3d(ctx0, res_skip_acts, res_skip_acts->ne[0],
                            res_skip_acts->ne[1] / 2, res_skip_acts->ne[2],
                            res_skip_acts->nb[1], res_skip_acts->nb[2], 0);
    res_acts2 = ggml_view_3d(ctx0, res_skip_acts, res_skip_acts->ne[0],
                             res_skip_acts->ne[1] / 2, res_skip_acts->ne[2],
                             res_skip_acts->nb[1], res_skip_acts->nb[2],
                             res_skip_acts->ne[1] / 2 * res_skip_acts->nb[1]);

    x = ggml_add(ctx0, x, res_acts);
    output = res_acts2;

    // layer1
    gl = g1;
    x_in = build_conv1d(ctx0, x, in_layer_1_weight, in_layer_1_bias, 1, 2, 1);
    in_act = ggml_add(ctx0, x_in, gl);
    t_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2], 0);
    s_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2],
                         in_act->ne[1] / 2 * in_act->nb[1]);
    t_act = ggml_tanh(ctx0, t_act);
    s_act = ggml_sigmoid(ctx0, s_act);
    acts = ggml_mul(ctx0, t_act, s_act);
    res_skip_acts = build_conv1d(ctx0, acts, res_skip_layers_1_weight,
                                 res_skip_layers_1_bias, 1, 0, 1);
    res_acts = ggml_view_3d(ctx0, res_skip_acts, res_skip_acts->ne[0],
                            res_skip_acts->ne[1] / 2, res_skip_acts->ne[2],
                            res_skip_acts->nb[1], res_skip_acts->nb[2], 0);
    res_acts2 = ggml_view_3d(ctx0, res_skip_acts, res_skip_acts->ne[0],
                             res_skip_acts->ne[1] / 2, res_skip_acts->ne[2],
                             res_skip_acts->nb[1], res_skip_acts->nb[2],
                             res_skip_acts->ne[1] / 2 * res_skip_acts->nb[1]);

    x = ggml_add(ctx0, x, res_acts);
    output = ggml_add(ctx0, output, res_acts2);

    // layer2
    gl = g2;
    x_in = build_conv1d(ctx0, x, in_layer_2_weight, in_layer_2_bias, 1, 2, 1);
    in_act = ggml_add(ctx0, x_in, gl);
    t_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2], 0);
    s_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2],
                         in_act->ne[1] / 2 * in_act->nb[1]);
    t_act = ggml_tanh(ctx0, t_act);
    s_act = ggml_sigmoid(ctx0, s_act);
    acts = ggml_mul(ctx0, t_act, s_act);
    res_skip_acts = build_conv1d(ctx0, acts, res_skip_layers_2_weight,
                                 res_skip_layers_2_bias, 1, 0, 1);
    res_acts = ggml_view_3d(ctx0, res_skip_acts, res_skip_acts->ne[0],
                            res_skip_acts->ne[1] / 2, res_skip_acts->ne[2],
                            res_skip_acts->nb[1], res_skip_acts->nb[2], 0);
    res_acts2 = ggml_view_3d(ctx0, res_skip_acts, res_skip_acts->ne[0],
                             res_skip_acts->ne[1] / 2, res_skip_acts->ne[2],
                             res_skip_acts->nb[1], res_skip_acts->nb[2],
                             res_skip_acts->ne[1] / 2 * res_skip_acts->nb[1]);

    x = ggml_add(ctx0, x, res_acts);
    output = ggml_add(ctx0, output, res_acts2);

    // layer3
    gl = g3;
    x_in = build_conv1d(ctx0, x, in_layer_3_weight, in_layer_3_bias, 1, 2, 1);
    in_act = ggml_add(ctx0, x_in, gl);
    t_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2], 0);
    s_act = ggml_view_3d(ctx0, in_act, in_act->ne[0], in_act->ne[1] / 2,
                         in_act->ne[2], in_act->nb[1], in_act->nb[2],
                         in_act->ne[1] / 2 * in_act->nb[1]);
    t_act = ggml_tanh(ctx0, t_act);
    s_act = ggml_sigmoid(ctx0, s_act);
    acts = ggml_mul(ctx0, t_act, s_act);

    res_skip_acts = build_conv1d(ctx0, acts, res_skip_layers_3_weight,
                                 res_skip_layers_3_bias, 1, 0, 1);

    output = ggml_add(ctx0, output, res_skip_acts);
    return output;
}

// HubertPositionalConvEmbedding implementation
ggml_tensor *
HubertPositionalConvEmbedding::build_graph(ggml_context *ctx0,
                                           ggml_tensor *hidden_states) {
    // hidden_states shape: (hidden_size, seq_len)
    // Transpose from (hidden_size, seq_len) to (seq_len, hidden_size)
    hidden_states = ggml_permute(ctx0, hidden_states, 1, 0, 2, 3);
    hidden_states = ggml_cont(ctx0, hidden_states);

    // Apply Conv1D with groups=16
    // Conv1d(768, 768, kernel_size=128, stride=1, padding=64, groups=16)
    // Note: weight_norm is applied during weight loading
    hidden_states = build_conv1d_grouped(ctx0, hidden_states, conv_weight,
                                         conv_bias, 16, 1, 64, 1);

    // Remove padding: hidden_states[:, :, :-num_pad_remove] (num_pad_remove=1)
    int64_t num_pad_remove = 1;
    hidden_states = ggml_view_2d(ctx0, hidden_states,
                                 hidden_states->ne[0] - num_pad_remove,
                                 hidden_states->ne[1],
                                 hidden_states->nb[1],
                                 0);
    hidden_states = ggml_cont(ctx0, hidden_states);

    // Apply GELU activation
    hidden_states = ggml_gelu(ctx0, hidden_states);

    // Transpose back from (seq_len, hidden_size) to (hidden_size, seq_len)
    hidden_states = ggml_permute(ctx0, hidden_states, 1, 0, 2, 3);
    hidden_states = ggml_cont(ctx0, hidden_states);

    return hidden_states;
}

ggml_tensor *ResidualCouplingLayer::build_graph(ggml_context *ctx0,
                                                ggml_tensor *cur,
                                                ggml_tensor *ge) {
    ggml_tensor *x0 = ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2,
                                   cur->ne[2], cur->nb[1], cur->nb[2], 0);
    ggml_tensor *x1 =
        ggml_view_3d(ctx0, cur, cur->ne[0], cur->ne[1] / 2, cur->ne[2],
                     cur->nb[1], cur->nb[2], cur->ne[1] / 2 * cur->nb[1]);
    ggml_tensor *h = build_conv1d(ctx0, x0, pre_weight, pre_bias, 1, 0, 1);
    h = enc.build_graph(ctx0, h, ge);
    ggml_tensor *stats = build_conv1d(ctx0, h, post_weight, post_bias, 1, 0, 1);
    ggml_tensor *m = stats;
    x1 = ggml_sub(ctx0, x1, m);
    x1 = ggml_concat(ctx0, x0, x1, 1);
    return x1;
}
std::vector<ggml_tensor *>
ResidualCouplingBlock::build_graph(ggml_context *ctx0) {
    std::vector<int> ge_shape = input_shapes_["ge"];
    std::vector<int> x_shape = input_shapes_["x"];

    ggml_tensor *ge = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, ge_shape[0],
                                         ge_shape[1], ge_shape[2]);
    ggml_set_name(ge, "ge");
    ggml_set_input(ge);

    ggml_tensor *inp_x = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, x_shape[0],
                                            x_shape[1], x_shape[2]);
    ggml_set_name(inp_x, "x");
    ggml_set_input(inp_x);

    ggml_tensor *cur = inp_x;

    // cur = ggml_flip(ctx0, cur, 1);
    // cur = layers[3].build_graph(ctx0, cur, ge);

    for (int i = layers.size() - 1; i >= 0; --i) {
        cur = ggml_flip(ctx0, cur, 1);
        cur = layers[i].build_graph(ctx0, cur, ge);
        // if(i==1)
        //   return {cur};
    }
    return {cur};
}

// ==================== CNHubertModel Implementation ====================

bool CNHubertModel::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("n_layer", hparams.n_layer, false);
    model_loader.get_u32("hidden_size", hparams.hidden_size, false);
    model_loader.get_u32("n_heads", hparams.n_heads, false);
    model_loader.get_u32("intermediate_size", hparams.intermediate_size, false);
    model_loader.get_f32("layer_norm_eps", hparams.layer_norm_eps, false);
    model_loader.get_u32("num_feat_extract_layers",
                         hparams.num_feat_extract_layers, false);
    model_loader.get_arr_int("conv_dim", hparams.conv_dim, false);
    model_loader.get_arr_int("conv_kernel", hparams.conv_kernel, false);
    model_loader.get_arr_int("conv_stride", hparams.conv_stride, false);
    return true;
}

std::vector<ggml_tensor *>
CNHubertModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;

    // Feature extractor conv layers
    int n_conv = hparams.num_feat_extract_layers;
    conv_layers.resize(n_conv);
    for (int i = 0; i < n_conv; ++i) {
        auto &layer = conv_layers[i];
        std::string prefix = string_format("feature_extractor.conv_layers.%d", i);
        layer.conv_weight = get_tensor(
            ctx, string_format("%s.conv.weight", prefix.c_str()), tensors_to_load);
        layer.conv_bias =
            get_tensor(ctx, string_format("%s.conv.bias", prefix.c_str()),
                       tensors_to_load, false);
        // Layer 0 has group norm, others have layer norm
        if (i == 0) {
            layer.ln_weight =
                get_tensor(ctx, string_format("%s.layer_norm.weight", prefix.c_str()),
                           tensors_to_load, false);
            layer.ln_bias =
                get_tensor(ctx, string_format("%s.layer_norm.bias", prefix.c_str()),
                           tensors_to_load, false);
        }
    }

    // Feature projection
    proj_weight =
        get_tensor(ctx, "feature_projection.projection.weight", tensors_to_load);
    proj_bias = get_tensor(ctx, "feature_projection.projection.bias",
                           tensors_to_load, false);
    proj_ln_w =
        get_tensor(ctx, "feature_projection.layer_norm.weight", tensors_to_load);
    proj_ln_b = get_tensor(ctx, "feature_projection.layer_norm.bias",
                           tensors_to_load, false);

    // Encoder positional convolutional embedding
    pos_conv_embed.conv_weight =
        get_tensor(ctx, "encoder.pos_conv_embed.conv.weight", tensors_to_load);
    pos_conv_embed.conv_bias = get_tensor(ctx, "encoder.pos_conv_embed.conv.bias",
                                          tensors_to_load, false);

    // Encoder layers
    int n_layer = hparams.n_layer;
    encoder_layers.resize(n_layer);
    for (int i = 0; i < n_layer; ++i) {
        auto &layer = encoder_layers[i];
        std::string prefix = string_format("encoder.layers.%d", i);

        // Self-attention
        layer.q_w = get_tensor(
            ctx, string_format("%s.attention.q_proj.weight", prefix.c_str()),
            tensors_to_load);
        layer.q_b = get_tensor(
            ctx, string_format("%s.attention.q_proj.bias", prefix.c_str()),
            tensors_to_load, false);
        layer.k_w = get_tensor(
            ctx, string_format("%s.attention.k_proj.weight", prefix.c_str()),
            tensors_to_load);
        layer.k_b = get_tensor(
            ctx, string_format("%s.attention.k_proj.bias", prefix.c_str()),
            tensors_to_load, false);
        layer.v_w = get_tensor(
            ctx, string_format("%s.attention.v_proj.weight", prefix.c_str()),
            tensors_to_load);
        layer.v_b = get_tensor(
            ctx, string_format("%s.attention.v_proj.bias", prefix.c_str()),
            tensors_to_load, false);
        layer.o_w = get_tensor(
            ctx, string_format("%s.attention.out_proj.weight", prefix.c_str()),
            tensors_to_load);
        layer.o_b = get_tensor(
            ctx, string_format("%s.attention.out_proj.bias", prefix.c_str()),
            tensors_to_load, false);

        // Layer norm after attention
        layer.ln_1_w =
            get_tensor(ctx, string_format("%s.layer_norm.weight", prefix.c_str()),
                       tensors_to_load);
        layer.ln_1_b =
            get_tensor(ctx, string_format("%s.layer_norm.bias", prefix.c_str()),
                       tensors_to_load, false);

        // Feed-forward
        layer.ff_up_w =
            get_tensor(ctx,
                       string_format("%s.feed_forward.intermediate_dense.weight",
                                     prefix.c_str()),
                       tensors_to_load);
        layer.ff_up_b =
            get_tensor(ctx,
                       string_format("%s.feed_forward.intermediate_dense.bias",
                                     prefix.c_str()),
                       tensors_to_load, false);
        layer.ff_down_w = get_tensor(
            ctx,
            string_format("%s.feed_forward.output_dense.weight", prefix.c_str()),
            tensors_to_load);
        layer.ff_down_b = get_tensor(
            ctx, string_format("%s.feed_forward.output_dense.bias", prefix.c_str()),
            tensors_to_load, false);

        // Final layer norm
        layer.ln_2_w = get_tensor(
            ctx, string_format("%s.final_layer_norm.weight", prefix.c_str()),
            tensors_to_load);
        layer.ln_2_b = get_tensor(
            ctx, string_format("%s.final_layer_norm.bias", prefix.c_str()),
            tensors_to_load, false);
    }

    // Final layer norm
    encoder_ln_weight =
        get_tensor(ctx, "encoder.layer_norm.weight", tensors_to_load, false);
    encoder_ln_bias =
        get_tensor(ctx, "encoder.layer_norm.bias", tensors_to_load, false);

    return tensors_to_load;
}

std::vector<ggml_tensor *> CNHubertModel::build_graph(ggml_context *ctx0) {
    // CNHubert forward pass:
    // 1. Feature extractor: 7 Conv1D layers with different kernel sizes/strides
    // 2. Feature projection: Linear + LayerNorm
    // 3. Encoder: 12 Transformer layers with self-attention and FFN

    std::vector<int> wav_shape = input_shapes_["wav_16k"];

    // Create input tensor: (input_len,) audio waveform
    ggml_tensor *wav_input =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, wav_shape[0], wav_shape[1]);
    ggml_set_name(wav_input, "wav_16k");
    ggml_set_input(wav_input);

    ggml_tensor *cur = wav_input;

    // ==================== Feature Extractor ====================
    // 7 conv layers with kernels [10, 3, 3, 3, 3, 2, 2] and strides [5, 2, 2, 2,
    // 2, 2, 2]
    //   int n_conv = hparams.num_feat_extract_layers;
    //   int cur_len = input_len;

    for (int i = 0; i < conv_layers.size(); ++i) {
        auto &layer = conv_layers[i];
        int stride = hparams.conv_stride[i];

        // Conv1D
        cur = build_conv1d(ctx0, cur, layer.conv_weight, layer.conv_bias, stride, 0,
                           1);
        if (0 == i) {
            cur = build_norm(ctx0, cur, layer.ln_weight, layer.ln_bias,
                             NORM_TYPE_GROUP, hparams.layer_norm_eps, cur->ne[1]);
        }
        cur = ggml_gelu(ctx0, cur);
    }

    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur)); // (512, seq_len)

    // ==================== Feature Projection ====================
    // Layer normalization
    cur = build_norm(ctx0, cur, proj_ln_w, proj_ln_b, NORM_TYPE_NORMAL,
                     hparams.layer_norm_eps);
    // Project from 512 to 768 dimensions
    cur = build_linear(ctx0, cur, proj_weight, proj_bias);

    // Save hidden_states before pos_conv_embed
    ggml_tensor *hidden_states = cur;
    ggml_tensor *position_embeddings = pos_conv_embed.build_graph(ctx0, cur);

    // hidden_states = hidden_states + position_embeddings
    cur = ggml_add(ctx0, hidden_states, position_embeddings);

    // hidden_states = self.layer_norm(hidden_states)
    cur = build_norm(ctx0, cur, encoder_ln_weight, encoder_ln_bias, NORM_TYPE_NORMAL,
                     hparams.layer_norm_eps);

    // cur is now (768, seq_len) after projection
    int hidden_size = hparams.hidden_size;
    int n_heads = hparams.n_heads;
    int head_dim = hidden_size / n_heads;
    int seq_len = cur->ne[1]; // ne[0]=768, ne[1]=seq_len

    // ==================== Transformer Encoder ====================
    for (int il = 0; il < hparams.n_layer; ++il) {
        auto &layer = encoder_layers[il];
        ggml_tensor *residual = cur;

        // Self-attention
        // Q, K, V projections
        ggml_tensor *q = build_linear(ctx0, cur, layer.q_w, layer.q_b, il);
        ggml_tensor *k = build_linear(ctx0, cur, layer.k_w, layer.k_b, il);
        ggml_tensor *v = build_linear(ctx0, cur, layer.v_w, layer.v_b, il);

        // Reshape for multi-head attention: (seq_len, n_heads, head_dim)
        q = ggml_reshape_3d(ctx0, q, head_dim, n_heads, seq_len);
        k = ggml_reshape_3d(ctx0, k, head_dim, n_heads, seq_len);
        v = ggml_reshape_3d(ctx0, v, head_dim, n_heads, seq_len);

        // Attention: no mask for encoder self-attention
        float kq_scale = 1.0f / sqrtf(static_cast<float>(head_dim));
        cur =
            build_attn(ctx0, layer.o_w, layer.o_b, q, k, v, nullptr, kq_scale, il);

        // Residual connection
        cur = ggml_add(ctx0, cur, residual);

        // Layer norm after attention
        cur = build_norm(ctx0, cur, layer.ln_1_w, layer.ln_1_b, NORM_TYPE_NORMAL,
                         hparams.layer_norm_eps, il);

        residual = cur;

        // Feed-forward network
        cur = build_ffn(ctx0, cur, layer.ff_up_w, layer.ff_up_b, nullptr, nullptr,
                        layer.ff_down_w, layer.ff_down_b, FFN_GELU, il);

        // Residual connection
        cur = ggml_add(ctx0, cur, residual);

        // Layer norm after FFN
        cur = build_norm(ctx0, cur, layer.ln_2_w, layer.ln_2_b, NORM_TYPE_NORMAL,
                         hparams.layer_norm_eps, il);
    }
    
    // Output: (seq_len, 768) -> transpose to (768, seq_len) to match expected
    // format
    cur = ggml_cont(ctx0, ggml_transpose(ctx0, cur));

    return {cur};
}

// ==================== ResidualVectorQuantizer Implementation
// ====================

bool ResidualVectorQuantizer::load_hparams(const ModelLoader &model_loader) {
    model_loader.get_u32("ssl_dim", hparams.ssl_dim, false);
    model_loader.get_u32("codebook_size", hparams.codebook_size, false);
    model_loader.get_u32("n_q", hparams.n_q, false);
    return true;
}

std::vector<ggml_tensor *>
ResidualVectorQuantizer::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors_to_load;

    ssl_proj_weight = get_tensor(ctx, "ssl_proj.weight", tensors_to_load);
    ssl_proj_bias = get_tensor(ctx, "ssl_proj.bias", tensors_to_load, false);
    codebook = get_tensor(ctx, "quantizer.codebook", tensors_to_load);

    return tensors_to_load;
}

std::vector<ggml_tensor *>
ResidualVectorQuantizer::build_graph(ggml_context *ctx0) {
    std::vector<int> ssl_shape = input_shapes_["ssl_features"];
    int ssl_dim = ssl_shape[0];
    int seq_len = ssl_shape[1];

    // Create input tensor (T, 768)
    ggml_tensor *ssl_input =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_len, ssl_dim);
    ggml_set_name(ssl_input, "ssl_features");
    ggml_set_input(ssl_input);

    // ssl_proj: Conv1D(768, 768, kernel=1, stride=2)
    // Input: (T, 768) -> Output: (T', 768) where T' = (T-1)/2 + 1
    ggml_tensor *projected =
        build_conv1d(ctx0, ssl_input, ssl_proj_weight, ssl_proj_bias, 2, 0, 1);

    // Quantize: find nearest codebook entry for each time step using
    // squared Euclidean distance: ||x - c||^2 = ||x||^2 - 2*x·c + ||c||^2
    // We compute negated distance and find argmax (= argmin of distance)

    // projected shape: (T', 768) where ne[0]=T', ne[1]=768
    // codebook shape: (768, 1024) where ne[0]=768, ne[1]=1024 (from model)

    // Transpose projected for matmul: (T', 768) -> (768, T')
    ggml_tensor *proj_t = ggml_cont(ctx0, ggml_transpose(ctx0, projected));

    // Step 1: ||x||^2 for each time step
    // proj_t: (768, T') -> square and sum over 768 -> (1, T')
    ggml_tensor *x_sq = ggml_sqr(ctx0, proj_t);
    ggml_tensor *x_sq_sum = ggml_sum_rows(ctx0, x_sq);  // (1, T')

    // Step 2: ||embed||^2 for each codebook entry
    // codebook: (768, 1024) -> square and sum over 768 -> (1, 1024)
    ggml_tensor *embed_sq = ggml_sqr(ctx0, codebook);
    ggml_tensor *embed_sq_sum = ggml_sum_rows(ctx0, embed_sq);  // (1, 1024)

    // Step 3: 2 * x @ embed cross term
    // ggml_mul_mat(a, b) computes a^T @ b
    // codebook: (768, 1024), proj_t: (768, T')
    // result = codebook^T @ proj_t = (1024, 768) @ (768, T') = (1024, T')
    ggml_tensor *cross = ggml_mul_mat(ctx0, codebook, proj_t);  // (1024, T')

    // Step 4: Compute negated squared distance
    // dist = 2*cross - ||x||^2 - ||embed||^2
    // cross: (1024, T'), x_sq_sum: (1, T'), embed_sq_sum: (1, 1024)

    // Scale cross by 2
    ggml_tensor *cross_2 = ggml_scale(ctx0, cross, 2.0f);  // (1024, T')

    // Broadcast x_sq_sum (1, T') to (1024, T') and subtract
    ggml_tensor *x_sq_rep = ggml_repeat(ctx0, x_sq_sum, cross_2);  // (1024, T')
    ggml_tensor *dist = ggml_sub(ctx0, cross_2, x_sq_rep);

    // Broadcast embed_sq_sum (1, 1024) to (1024, T')
    // Need to transpose first: (1, 1024) -> (1024, 1), then repeat
    ggml_tensor *embed_sq_t =
        ggml_cont(ctx0, ggml_transpose(ctx0, embed_sq_sum));  // (1024, 1)
    ggml_tensor *embed_sq_rep = ggml_repeat(ctx0, embed_sq_t, dist);  // (1024, T')
    dist = ggml_sub(ctx0, dist, embed_sq_rep);

    // Step 5: argmax to get indices
    // dist: (1024, T') -> argmax along ne[0]=1024 -> (T',) int32
    ggml_tensor *indices = ggml_argmax(ctx0, dist);

    return {indices};
}
