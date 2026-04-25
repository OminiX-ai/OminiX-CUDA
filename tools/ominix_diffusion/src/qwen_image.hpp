#ifndef __QWEN_IMAGE_HPP__
#define __QWEN_IMAGE_HPP__

#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "common_block.hpp"
#include "flux.hpp"

namespace Qwen {
    constexpr int QWEN_IMAGE_GRAPH_SIZE = 20480;

    struct TimestepEmbedding : public GGMLBlock {
    public:
        TimestepEmbedding(int64_t in_channels,
                          int64_t time_embed_dim,
                          int64_t out_dim       = 0,
                          int64_t cond_proj_dim = 0,
                          bool sample_proj_bias = true) {
            blocks["linear_1"] = std::shared_ptr<GGMLBlock>(new Linear(in_channels, time_embed_dim, sample_proj_bias));
            if (cond_proj_dim > 0) {
                blocks["cond_proj"] = std::shared_ptr<GGMLBlock>(new Linear(cond_proj_dim, in_channels, false));
            }
            if (out_dim <= 0) {
                out_dim = time_embed_dim;
            }
            blocks["linear_2"] = std::shared_ptr<GGMLBlock>(new Linear(time_embed_dim, out_dim, sample_proj_bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* sample,
                                    struct ggml_tensor* condition = nullptr) {
            if (condition != nullptr) {
                auto cond_proj = std::dynamic_pointer_cast<Linear>(blocks["cond_proj"]);
                sample         = ggml_add(ctx->ggml_ctx, sample, cond_proj->forward(ctx, condition));
            }
            auto linear_1 = std::dynamic_pointer_cast<Linear>(blocks["linear_1"]);
            auto linear_2 = std::dynamic_pointer_cast<Linear>(blocks["linear_2"]);

            sample = linear_1->forward(ctx, sample);
            sample = ggml_silu_inplace(ctx->ggml_ctx, sample);
            sample = linear_2->forward(ctx, sample);
            return sample;
        }
    };

    struct QwenTimestepProjEmbeddings : public GGMLBlock {
    public:
        QwenTimestepProjEmbeddings(int64_t embedding_dim) {
            blocks["timestep_embedder"] = std::shared_ptr<GGMLBlock>(new TimestepEmbedding(256, embedding_dim));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* timesteps) {
            // timesteps: [N,]
            // return: [N, embedding_dim]
            auto timestep_embedder = std::dynamic_pointer_cast<TimestepEmbedding>(blocks["timestep_embedder"]);

            auto timesteps_proj = ggml_ext_timestep_embedding(ctx->ggml_ctx, timesteps, 256, 10000, 1.f);
            auto timesteps_emb  = timestep_embedder->forward(ctx, timesteps_proj);
            return timesteps_emb;
        }
    };

    struct QwenImageAttention : public GGMLBlock {
    protected:
        int64_t dim_head;

    public:
        QwenImageAttention(int64_t query_dim,
                           int64_t dim_head,
                           int64_t num_heads,
                           int64_t out_dim         = 0,
                           int64_t out_context_dim = 0,
                           bool bias               = true,
                           bool out_bias           = true,
                           float eps               = 1e-6)
            : dim_head(dim_head) {
            int64_t inner_dim = out_dim > 0 ? out_dim : dim_head * num_heads;
            out_dim           = out_dim > 0 ? out_dim : query_dim;
            out_context_dim   = out_context_dim > 0 ? out_context_dim : query_dim;

            blocks["to_q"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["to_k"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["to_v"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));

            blocks["norm_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));
            blocks["norm_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));

            blocks["add_q_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["add_k_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));
            blocks["add_v_proj"] = std::shared_ptr<GGMLBlock>(new Linear(query_dim, inner_dim, bias));

            blocks["norm_added_q"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));
            blocks["norm_added_k"] = std::shared_ptr<GGMLBlock>(new RMSNorm(dim_head, eps));

            float scale         = 1.f / 32.f;
            bool force_prec_f32 = false;
#ifdef SD_USE_VULKAN
            force_prec_f32 = true;
#endif
            // The purpose of the scale here is to prevent NaN issues in certain situations.
            // For example when using CUDA but the weights are k-quants (not all prompts).
            blocks["to_out.0"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_dim, out_bias, false, force_prec_f32, scale));
            // to_out.1 is nn.Dropout

            blocks["to_add_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, out_context_dim, out_bias, false, false, scale));
        }

        std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                      struct ggml_tensor* img,
                                                      struct ggml_tensor* txt,
                                                      struct ggml_tensor* pe,
                                                      struct ggml_tensor* mask = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])

            auto norm_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_q"]);
            auto norm_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_k"]);

            auto to_q     = std::dynamic_pointer_cast<Linear>(blocks["to_q"]);
            auto to_k     = std::dynamic_pointer_cast<Linear>(blocks["to_k"]);
            auto to_v     = std::dynamic_pointer_cast<Linear>(blocks["to_v"]);
            auto to_out_0 = std::dynamic_pointer_cast<Linear>(blocks["to_out.0"]);

            auto norm_added_q = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_q"]);
            auto norm_added_k = std::dynamic_pointer_cast<UnaryBlock>(blocks["norm_added_k"]);

            auto add_q_proj = std::dynamic_pointer_cast<Linear>(blocks["add_q_proj"]);
            auto add_k_proj = std::dynamic_pointer_cast<Linear>(blocks["add_k_proj"]);
            auto add_v_proj = std::dynamic_pointer_cast<Linear>(blocks["add_v_proj"]);
            auto to_add_out = std::dynamic_pointer_cast<Linear>(blocks["to_add_out"]);

            int64_t N           = img->ne[2];
            int64_t n_img_token = img->ne[1];
            int64_t n_txt_token = txt->ne[1];

            auto img_q        = to_q->forward(ctx, img);
            int64_t num_heads = img_q->ne[0] / dim_head;
            img_q             = ggml_reshape_4d(ctx->ggml_ctx, img_q, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]
            auto img_k        = to_k->forward(ctx, img);
            img_k             = ggml_reshape_4d(ctx->ggml_ctx, img_k, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]
            auto img_v        = to_v->forward(ctx, img);
            img_v             = ggml_reshape_4d(ctx->ggml_ctx, img_v, dim_head, num_heads, n_img_token, N);  // [N, n_img_token, n_head, d_head]

            img_q = norm_q->forward(ctx, img_q);
            img_k = norm_k->forward(ctx, img_k);

            auto txt_q = add_q_proj->forward(ctx, txt);
            txt_q      = ggml_reshape_4d(ctx->ggml_ctx, txt_q, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]
            auto txt_k = add_k_proj->forward(ctx, txt);
            txt_k      = ggml_reshape_4d(ctx->ggml_ctx, txt_k, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]
            auto txt_v = add_v_proj->forward(ctx, txt);
            txt_v      = ggml_reshape_4d(ctx->ggml_ctx, txt_v, dim_head, num_heads, n_txt_token, N);  // [N, n_txt_token, n_head, d_head]

            txt_q = norm_added_q->forward(ctx, txt_q);
            txt_k = norm_added_k->forward(ctx, txt_k);

            auto q = ggml_concat(ctx->ggml_ctx, txt_q, img_q, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto k = ggml_concat(ctx->ggml_ctx, txt_k, img_k, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]
            auto v = ggml_concat(ctx->ggml_ctx, txt_v, img_v, 2);  // [N, n_txt_token + n_img_token, n_head, d_head]

            auto attn         = Rope::attention(ctx, q, k, v, pe, mask, (1.0f / 128.f));  // [N, n_txt_token + n_img_token, n_head*d_head]
            auto txt_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             txt->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             0);  // [N, n_txt_token, n_head*d_head]
            auto img_attn_out = ggml_view_3d(ctx->ggml_ctx,
                                             attn,
                                             attn->ne[0],
                                             img->ne[1],
                                             attn->ne[2],
                                             attn->nb[1],
                                             attn->nb[2],
                                             txt->ne[1] * attn->nb[1]);  // [N, n_img_token, n_head*d_head]
            img_attn_out      = ggml_cont(ctx->ggml_ctx, img_attn_out);
            txt_attn_out      = ggml_cont(ctx->ggml_ctx, txt_attn_out);

            img_attn_out = to_out_0->forward(ctx, img_attn_out);
            txt_attn_out = to_add_out->forward(ctx, txt_attn_out);

            return {img_attn_out, txt_attn_out};
        }
    };

    class QwenImageTransformerBlock : public GGMLBlock {
    protected:
        bool zero_cond_t;

    public:
        QwenImageTransformerBlock(int64_t dim,
                                  int64_t num_attention_heads,
                                  int64_t attention_head_dim,
                                  float eps        = 1e-6,
                                  bool zero_cond_t = false)
            : zero_cond_t(zero_cond_t) {
            // img_mod.0 is nn.SiLU()
            blocks["img_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, 6 * dim, true));

            blocks["img_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["img_mlp"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim, 4, FeedForward::Activation::GELU, true));

            // txt_mod.0 is nn.SiLU()
            blocks["txt_mod.1"] = std::shared_ptr<GGMLBlock>(new Linear(dim, 6 * dim, true));

            blocks["txt_norm1"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_norm2"] = std::shared_ptr<GGMLBlock>(new LayerNorm(dim, eps, false));
            blocks["txt_mlp"]   = std::shared_ptr<GGMLBlock>(new FeedForward(dim, dim, 4, FeedForward::Activation::GELU, true));

            blocks["attn"] = std::shared_ptr<GGMLBlock>(new QwenImageAttention(dim,
                                                                               attention_head_dim,
                                                                               num_attention_heads,
                                                                               0,     // out_dim
                                                                               0,     // out_context-dim
                                                                               true,  // bias
                                                                               true,  // out_bias
                                                                               eps));
        }

        std::vector<ggml_tensor*> get_mod_params_vec(ggml_context* ctx, ggml_tensor* mod_params, ggml_tensor* index = nullptr) {
            // index: [N, n_img_token]
            // mod_params: [N, hidden_size * 12]
            if (index == nullptr) {
                return ggml_ext_chunk(ctx, mod_params, 6, 0);
            }
            mod_params          = ggml_reshape_1d(ctx, mod_params, ggml_nelements(mod_params));
            auto mod_params_vec = ggml_ext_chunk(ctx, mod_params, 12, 0);
            index               = ggml_reshape_3d(ctx, index, 1, index->ne[0], index->ne[1]);                                      // [N, n_img_token, 1]
            index               = ggml_repeat_4d(ctx, index, mod_params_vec[0]->ne[0], index->ne[1], index->ne[2], index->ne[3]);  // [N, n_img_token, hidden_size]
            std::vector<ggml_tensor*> mod_results;
            for (int i = 0; i < 6; i++) {
                auto mod_0 = mod_params_vec[i];
                auto mod_1 = mod_params_vec[i + 6];

                // mod_result = torch.where(index == 0, mod_0, mod_1)
                // mod_result = (1 - index)*mod_0 + index*mod_1
                mod_0           = ggml_sub(ctx, ggml_repeat(ctx, mod_0, index), ggml_mul(ctx, index, mod_0));  // [N, n_img_token, hidden_size]
                mod_1           = ggml_mul(ctx, index, mod_1);                                                 // [N, n_img_token, hidden_size]
                auto mod_result = ggml_add(ctx, mod_0, mod_1);
                mod_results.push_back(mod_result);
            }
            return mod_results;
        }

        virtual std::pair<ggml_tensor*, ggml_tensor*> forward(GGMLRunnerContext* ctx,
                                                              struct ggml_tensor* img,
                                                              struct ggml_tensor* txt,
                                                              struct ggml_tensor* t_emb,
                                                              struct ggml_tensor* pe,
                                                              struct ggml_tensor* modulate_index = nullptr,
                                                              struct ggml_tensor* attention_mask = nullptr) {
            // img: [N, n_img_token, hidden_size]
            // txt: [N, n_txt_token, hidden_size]
            // pe: [n_img_token + n_txt_token, d_head/2, 2, 2]
            // attention_mask: [N, L_q, L_k] additive mask (0 keep, -inf mask); optional — only needed when N>1 with unequal per-batch valid-token counts (Q4 CFG batching).
            // return: ([N, n_img_token, hidden_size], [N, n_txt_token, hidden_size])

            auto img_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["img_mod.1"]);
            auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm1"]);
            auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["img_norm2"]);
            auto img_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["img_mlp"]);

            auto txt_mod_1 = std::dynamic_pointer_cast<Linear>(blocks["txt_mod.1"]);
            auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm1"]);
            auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(blocks["txt_norm2"]);
            auto txt_mlp   = std::dynamic_pointer_cast<FeedForward>(blocks["txt_mlp"]);

            auto attn = std::dynamic_pointer_cast<QwenImageAttention>(blocks["attn"]);

            auto img_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            img_mod_params         = img_mod_1->forward(ctx, img_mod_params);
            auto img_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, img_mod_params, modulate_index);

            if (zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            auto txt_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
            txt_mod_params         = txt_mod_1->forward(ctx, txt_mod_params);
            auto txt_mod_param_vec = get_mod_params_vec(ctx->ggml_ctx, txt_mod_params);

            auto img_normed    = img_norm1->forward(ctx, img);
            auto img_modulated = Flux::modulate(ctx->ggml_ctx, img_normed, img_mod_param_vec[0], img_mod_param_vec[1], modulate_index != nullptr);
            auto img_gate1     = img_mod_param_vec[2];

            auto txt_normed    = txt_norm1->forward(ctx, txt);
            auto txt_modulated = Flux::modulate(ctx->ggml_ctx, txt_normed, txt_mod_param_vec[0], txt_mod_param_vec[1]);
            auto txt_gate1     = txt_mod_param_vec[2];

            // Q4 CFG batching: gates arrive as [hidden, N, 1, 1]. At N==1 the subsequent
            // mul with img/txt [hidden, n_tokens, 1, 1] broadcasts correctly via the
            // ne[1]=1 slot. At N>1 we must re-home N to ne[2] so ne[1]=1 broadcasts
            // across tokens and ne[2]=N differentiates batch. Reshape is a no-op at N=1
            // (ne becomes [hidden, 1, 1, 1]) — byte-identical to pre-patch. In the
            // modulate_index != nullptr branch these gates are used differently
            // (get_mod_params_vec returns [N*hidden*n_img_token] layouts) and should
            // not be reshaped here.
            if (modulate_index == nullptr) {
                img_gate1 = ggml_reshape_4d(ctx->ggml_ctx, img_gate1,
                                            img_gate1->ne[0], 1, img_gate1->ne[1], img_gate1->ne[2]);
                txt_gate1 = ggml_reshape_4d(ctx->ggml_ctx, txt_gate1,
                                            txt_gate1->ne[0], 1, txt_gate1->ne[1], txt_gate1->ne[2]);
            }

            auto [img_attn_output, txt_attn_output] = attn->forward(ctx, img_modulated, txt_modulated, pe, attention_mask);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_attn_output, img_gate1));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_attn_output, txt_gate1));

            auto img_normed2    = img_norm2->forward(ctx, img);
            auto img_modulated2 = Flux::modulate(ctx->ggml_ctx, img_normed2, img_mod_param_vec[3], img_mod_param_vec[4], modulate_index != nullptr);
            auto img_gate2      = img_mod_param_vec[5];

            auto txt_normed2    = txt_norm2->forward(ctx, txt);
            auto txt_modulated2 = Flux::modulate(ctx->ggml_ctx, txt_normed2, txt_mod_param_vec[3], txt_mod_param_vec[4]);
            auto txt_gate2      = txt_mod_param_vec[5];

            if (modulate_index == nullptr) {
                img_gate2 = ggml_reshape_4d(ctx->ggml_ctx, img_gate2,
                                            img_gate2->ne[0], 1, img_gate2->ne[1], img_gate2->ne[2]);
                txt_gate2 = ggml_reshape_4d(ctx->ggml_ctx, txt_gate2,
                                            txt_gate2->ne[0], 1, txt_gate2->ne[1], txt_gate2->ne[2]);
            }

            auto img_mlp_out = img_mlp->forward(ctx, img_modulated2);
            auto txt_mlp_out = txt_mlp->forward(ctx, txt_modulated2);

            img = ggml_add(ctx->ggml_ctx, img, ggml_mul(ctx->ggml_ctx, img_mlp_out, img_gate2));
            txt = ggml_add(ctx->ggml_ctx, txt, ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_gate2));

            return {img, txt};
        }
    };

    struct AdaLayerNormContinuous : public GGMLBlock {
    public:
        AdaLayerNormContinuous(int64_t embedding_dim,
                               int64_t conditioning_embedding_dim,
                               bool elementwise_affine = true,
                               float eps               = 1e-5f,
                               bool bias               = true) {
            blocks["norm"]   = std::shared_ptr<GGMLBlock>(new LayerNorm(conditioning_embedding_dim, eps, elementwise_affine, bias));
            blocks["linear"] = std::shared_ptr<GGMLBlock>(new Linear(conditioning_embedding_dim, embedding_dim * 2, bias));
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* c) {
            // x: [N, n_token, hidden_size]
            // c: [N, hidden_size]
            // return: [N, n_token, patch_size * patch_size * out_channels]

            auto norm   = std::dynamic_pointer_cast<LayerNorm>(blocks["norm"]);
            auto linear = std::dynamic_pointer_cast<Linear>(blocks["linear"]);

            auto emb   = linear->forward(ctx, ggml_silu(ctx->ggml_ctx, c));
            auto mods  = ggml_ext_chunk(ctx->ggml_ctx, emb, 2, 0);
            auto scale = mods[0];
            auto shift = mods[1];

            x = norm->forward(ctx, x);
            x = Flux::modulate(ctx->ggml_ctx, x, shift, scale);

            return x;
        }
    };

    struct QwenImageParams {
        int patch_size              = 2;
        int64_t in_channels         = 64;
        int64_t out_channels        = 16;
        int num_layers              = 60;
        int64_t attention_head_dim  = 128;
        int64_t num_attention_heads = 24;
        int64_t joint_attention_dim = 3584;
        int theta                   = 10000;
        std::vector<int> axes_dim   = {16, 56, 56};
        int axes_dim_sum            = 128;
        bool zero_cond_t            = false;
    };

    class QwenImageModel : public GGMLBlock {
    protected:
        QwenImageParams params;

    public:
        QwenImageModel() {}
        QwenImageModel(QwenImageParams params)
            : params(params) {
            int64_t inner_dim         = params.num_attention_heads * params.attention_head_dim;
            blocks["time_text_embed"] = std::shared_ptr<GGMLBlock>(new QwenTimestepProjEmbeddings(inner_dim));
            blocks["txt_norm"]        = std::shared_ptr<GGMLBlock>(new RMSNorm(params.joint_attention_dim, 1e-6f));
            blocks["img_in"]          = std::shared_ptr<GGMLBlock>(new Linear(params.in_channels, inner_dim));
            blocks["txt_in"]          = std::shared_ptr<GGMLBlock>(new Linear(params.joint_attention_dim, inner_dim));

            // blocks
            for (int i = 0; i < params.num_layers; i++) {
                auto block                                        = std::shared_ptr<GGMLBlock>(new QwenImageTransformerBlock(inner_dim,
                                                                                                                             params.num_attention_heads,
                                                                                                                             params.attention_head_dim,
                                                                                                                             1e-6f,
                                                                                                                             params.zero_cond_t));
                blocks["transformer_blocks." + std::to_string(i)] = block;
            }

            blocks["norm_out"] = std::shared_ptr<GGMLBlock>(new AdaLayerNormContinuous(inner_dim, inner_dim, false, 1e-6f));
            blocks["proj_out"] = std::shared_ptr<GGMLBlock>(new Linear(inner_dim, params.patch_size * params.patch_size * params.out_channels));
        }

        struct ggml_tensor* forward_orig(GGMLRunnerContext* ctx,
                                         struct ggml_tensor* x,
                                         struct ggml_tensor* timestep,
                                         struct ggml_tensor* context,
                                         struct ggml_tensor* pe,
                                         struct ggml_tensor* modulate_index = nullptr,
                                         struct ggml_tensor* attention_mask = nullptr) {
            auto time_text_embed = std::dynamic_pointer_cast<QwenTimestepProjEmbeddings>(blocks["time_text_embed"]);
            auto txt_norm        = std::dynamic_pointer_cast<RMSNorm>(blocks["txt_norm"]);
            auto img_in          = std::dynamic_pointer_cast<Linear>(blocks["img_in"]);
            auto txt_in          = std::dynamic_pointer_cast<Linear>(blocks["txt_in"]);
            auto norm_out        = std::dynamic_pointer_cast<AdaLayerNormContinuous>(blocks["norm_out"]);
            auto proj_out        = std::dynamic_pointer_cast<Linear>(blocks["proj_out"]);

            auto t_emb = time_text_embed->forward(ctx, timestep);
            if (params.zero_cond_t) {
                auto t_emb_0 = time_text_embed->forward(ctx, ggml_ext_zeros(ctx->ggml_ctx, timestep->ne[0], timestep->ne[1], timestep->ne[2], timestep->ne[3]));
                t_emb        = ggml_concat(ctx->ggml_ctx, t_emb, t_emb_0, 1);
            }
            auto img = img_in->forward(ctx, x);
            auto txt = txt_norm->forward(ctx, context);
            txt      = txt_in->forward(ctx, txt);

            for (int i = 0; i < params.num_layers; i++) {
                auto block = std::dynamic_pointer_cast<QwenImageTransformerBlock>(blocks["transformer_blocks." + std::to_string(i)]);

                auto result = block->forward(ctx, img, txt, t_emb, pe, modulate_index, attention_mask);
                img         = result.first;
                txt         = result.second;
            }

            if (params.zero_cond_t) {
                t_emb = ggml_ext_chunk(ctx->ggml_ctx, t_emb, 2, 1)[0];
            }

            img = norm_out->forward(ctx, img, t_emb);
            img = proj_out->forward(ctx, img);

            return img;
        }

        struct ggml_tensor* forward(GGMLRunnerContext* ctx,
                                    struct ggml_tensor* x,
                                    struct ggml_tensor* timestep,
                                    struct ggml_tensor* context,
                                    struct ggml_tensor* pe,
                                    std::vector<ggml_tensor*> ref_latents = {},
                                    struct ggml_tensor* modulate_index    = nullptr,
                                    struct ggml_tensor* attention_mask    = nullptr) {
            // Forward pass of DiT.
            // x: [N, C, H, W]
            // timestep: [N,]
            // context: [N, L, D]
            // pe: [L, d_head/2, 2, 2]
            // attention_mask: [N, L_q, L_k] additive mask; optional, only required for Q4 CFG batching with unequal text lengths.
            // return: [N, C, H, W]

            int64_t W = x->ne[0];
            int64_t H = x->ne[1];
            int64_t C = x->ne[2];
            int64_t N = x->ne[3];

            auto img           = DiT::pad_and_patchify(ctx, x, params.patch_size, params.patch_size);
            int64_t img_tokens = img->ne[1];

            if (ref_latents.size() > 0) {
                for (ggml_tensor* ref : ref_latents) {
                    ref = DiT::pad_and_patchify(ctx, ref, params.patch_size, params.patch_size);
                    img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
                }
            }

            auto out = forward_orig(ctx, img, timestep, context, pe, modulate_index, attention_mask);  // [N, h_len*w_len, ph*pw*C]

            if (out->ne[1] > img_tokens) {
                // Q4 CFG batching: N may be 2. The post-block reshape slices the token dim
                // to discard ref_latents/txt trailing tokens. Use view_4d so the trailing ne[3]
                // (always 1 at this point — N is in ne[2]) survives the slice unchanged.
                // At N=1 this is shape-equivalent to the original view_3d.
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [C*ph*pw, N, total_tokens, 1]
                out = ggml_view_4d(ctx->ggml_ctx, out,
                                   out->ne[0], out->ne[1], img_tokens, out->ne[3],
                                   out->nb[1], out->nb[2], out->nb[3], 0);                    // [C*ph*pw, N, img_tokens, 1]
                out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));  // [C*ph*pw, img_tokens, N, 1]
            }

            out = DiT::unpatchify_and_crop(ctx->ggml_ctx, out, H, W, params.patch_size, params.patch_size);  // [N, C, H, W]

            return out;
        }
    };

    struct QwenImageRunner : public GGMLRunner {
    public:
        QwenImageParams qwen_image_params;
        QwenImageModel qwen_image;
        std::vector<float> pe_vec;
        std::vector<float> modulate_index_vec;
        SDVersion version;

        // ----------------------------------------------------------------
        // RoPE pe-vector cache (OMINIX_QIE_ROPE_PRECOMPUTE).
        // `gen_qwen_image_pe` depends only on positional metadata
        // (h, w, bs, context_len, ref-latent shapes, circular flags,
        // theta, axes_dim, patch_size). Most of these are stable across
        // denoise steps; CFG branches may differ by context_len, so keep
        // a small shape-keyed cache instead of a single last-value slot.
        // See docs/qie_rope_precompute_lift.md for receipts.
        // MLX-measured analogous refactor: +10-25% per step.
        // ----------------------------------------------------------------
        // Cache key excludes bs: gen_qwen_image_pe is now always called with bs=1
        // (CFG batching relies on pe broadcasting across ne[3], so pe is shape-
        // independent of N). See build_graph below.
        struct PeCacheEntry {
            int h                                         = 0;
            int w                                         = 0;
            int context_len                               = 0;
            std::vector<std::pair<int64_t, int64_t>> ref_shapes;
            bool increase_ref_index                       = false;
            bool circular_x                               = false;
            bool circular_y                               = false;
            std::vector<float> pe_vec;

            bool matches(int cur_h,
                         int cur_w,
                         int cur_context_len,
                         const std::vector<std::pair<int64_t, int64_t>>& cur_ref_shapes,
                         bool cur_increase_ref_index,
                         bool cur_circular_x,
                         bool cur_circular_y) const {
                return h == cur_h &&
                       w == cur_w &&
                       context_len == cur_context_len &&
                       ref_shapes == cur_ref_shapes &&
                       increase_ref_index == cur_increase_ref_index &&
                       circular_x == cur_circular_x &&
                       circular_y == cur_circular_y;
            }
        };
        static constexpr size_t PE_CACHE_MAX_ENTRIES = 4;
        std::vector<PeCacheEntry> pe_cache_entries_;
        uint64_t pe_cache_miss_count_                = 0;
        uint64_t pe_cache_hit_count_                 = 0;

        QwenImageRunner(ggml_backend_t backend,
                        bool offload_params_to_cpu,
                        const String2TensorStorage& tensor_storage_map = {},
                        const std::string prefix                       = "",
                        SDVersion version                              = VERSION_QWEN_IMAGE,
                        bool zero_cond_t                               = false)
            : GGMLRunner(backend, offload_params_to_cpu) {
            qwen_image_params.num_layers  = 0;
            qwen_image_params.zero_cond_t = zero_cond_t;
            for (auto pair : tensor_storage_map) {
                std::string tensor_name = pair.first;
                if (tensor_name.find(prefix) == std::string::npos)
                    continue;
                if (tensor_name.find("__index_timestep_zero__") != std::string::npos) {
                    qwen_image_params.zero_cond_t = true;
                }
                size_t pos = tensor_name.find("transformer_blocks.");
                if (pos != std::string::npos) {
                    tensor_name = tensor_name.substr(pos);  // remove prefix
                    auto items  = split_string(tensor_name, '.');
                    if (items.size() > 1) {
                        int block_index = atoi(items[1].c_str());
                        if (block_index + 1 > qwen_image_params.num_layers) {
                            qwen_image_params.num_layers = block_index + 1;
                        }
                    }
                    continue;
                }
            }
            LOG_INFO("qwen_image_params.num_layers: %ld", qwen_image_params.num_layers);
            if (qwen_image_params.zero_cond_t) {
                LOG_INFO("use zero_cond_t");
            }
            qwen_image = QwenImageModel(qwen_image_params);
            qwen_image.init(params_ctx, tensor_storage_map, prefix);
        }

        std::string get_desc() override {
            return "qwen_image";
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            qwen_image.get_param_tensors(tensors, prefix);
        }

        struct ggml_cgraph* build_graph(struct ggml_tensor* x,
                                        struct ggml_tensor* timesteps,
                                        struct ggml_tensor* context,
                                        std::vector<ggml_tensor*> ref_latents = {},
                                        bool increase_ref_index               = false,
                                        struct ggml_tensor* attention_mask    = nullptr) {
            // Q4 CFG batching: N==2 is permitted when the host has stacked cond+uncond
            // along ne[3] and supplied an attention_mask that masks padded-text positions
            // per batch. All other axes (context->ne[1], timesteps, ref_latents) must be
            // consistent across batches (host responsibility).
            GGML_ASSERT(x->ne[3] == 1 || x->ne[3] == 2);
            struct ggml_cgraph* gf = new_graph_custom(QWEN_IMAGE_GRAPH_SIZE);

            x         = to_backend(x);
            context   = to_backend(context);
            timesteps = to_backend(timesteps);
            if (attention_mask != nullptr) {
                attention_mask = to_backend(attention_mask);
            }

            for (int i = 0; i < ref_latents.size(); i++) {
                ref_latents[i] = to_backend(ref_latents[i]);
            }

            // RoPE pe-vector: either re-compute every step (legacy path,
            // default) or read from shape-keyed cache (Q2 must-have
            // structural refactor). Gated by OMINIX_QIE_ROPE_PRECOMPUTE=1
            // so flag-unset builds remain byte-identical to pre-patch.
            //
            // Q4 CFG batching note: pe is broadcast across the batch dim N
            // in apply_rope (pe is [2, L, d_head/2, 2], independent of N).
            // Passing bs=1 yields a single copy of positions that applies
            // to both batch items; passing bs=x->ne[3] would double the
            // positions for N=2 and mis-shape the pe tensor
            // (ne[3]=2*pos_len instead of pos_len). For N=1, bs=1 is byte
            // identical to the historical behaviour. Because bs is now
            // always 1, the cache key excludes it.
            const char* qie_rope_precompute_env   = std::getenv("OMINIX_QIE_ROPE_PRECOMPUTE");
            const bool qie_rope_precompute_on    = qie_rope_precompute_env != nullptr &&
                                                   std::string(qie_rope_precompute_env) == "1";

            const std::vector<float>* pe_values = &pe_vec;
            if (qie_rope_precompute_on) {
                const int cur_h           = static_cast<int>(x->ne[1]);
                const int cur_w           = static_cast<int>(x->ne[0]);
                const int cur_context_len = static_cast<int>(context->ne[1]);
                std::vector<std::pair<int64_t, int64_t>> cur_ref_shapes;
                cur_ref_shapes.reserve(ref_latents.size());
                for (ggml_tensor* ref : ref_latents) {
                    cur_ref_shapes.emplace_back(ref ? ref->ne[1] : 0,
                                                ref ? ref->ne[0] : 0);
                }

                PeCacheEntry* cache_hit = nullptr;
                for (auto& entry : pe_cache_entries_) {
                    if (entry.matches(cur_h,
                                      cur_w,
                                      cur_context_len,
                                      cur_ref_shapes,
                                      increase_ref_index,
                                      circular_x_enabled,
                                      circular_y_enabled)) {
                        cache_hit = &entry;
                        break;
                    }
                }

                if (cache_hit == nullptr) {
                    PeCacheEntry entry;
                    entry.h                  = cur_h;
                    entry.w                  = cur_w;
                    entry.context_len        = cur_context_len;
                    entry.ref_shapes         = std::move(cur_ref_shapes);
                    entry.increase_ref_index = increase_ref_index;
                    entry.circular_x         = circular_x_enabled;
                    entry.circular_y         = circular_y_enabled;
                    entry.pe_vec             = Rope::gen_qwen_image_pe(cur_h,
                                                                       cur_w,
                                                                       qwen_image_params.patch_size,
                                                                       1,
                                                                       cur_context_len,
                                                                       ref_latents,
                                                                       increase_ref_index,
                                                                       qwen_image_params.theta,
                                                                       circular_y_enabled,
                                                                       circular_x_enabled,
                                                                       qwen_image_params.axes_dim);
                    if (pe_cache_entries_.size() >= PE_CACHE_MAX_ENTRIES) {
                        pe_cache_entries_.erase(pe_cache_entries_.begin());
                    }
                    pe_cache_entries_.push_back(std::move(entry));
                    const PeCacheEntry& inserted = pe_cache_entries_.back();
                    pe_values = &inserted.pe_vec;
                    ++pe_cache_miss_count_;
                    LOG_INFO("qwen_image pe_cache MISS (refill) #%llu: h=%d w=%d ctx=%d nrefs=%zu entries=%zu -> pe_vec=%zu floats",
                             (unsigned long long)pe_cache_miss_count_,
                             cur_h, cur_w, cur_context_len,
                             inserted.ref_shapes.size(),
                             pe_cache_entries_.size(),
                             inserted.pe_vec.size());
                } else {
                    pe_values = &cache_hit->pe_vec;
                    ++pe_cache_hit_count_;
                    if (pe_cache_hit_count_ == 1 ||
                        pe_cache_hit_count_ % 10 == 0) {
                        LOG_INFO("qwen_image pe_cache HIT count=%llu (miss=%llu)",
                                 (unsigned long long)pe_cache_hit_count_,
                                 (unsigned long long)pe_cache_miss_count_);
                    }
                }
            } else {
                pe_vec = Rope::gen_qwen_image_pe(static_cast<int>(x->ne[1]),
                                                 static_cast<int>(x->ne[0]),
                                                 qwen_image_params.patch_size,
                                                 1,
                                                 static_cast<int>(context->ne[1]),
                                                 ref_latents,
                                                 increase_ref_index,
                                                 qwen_image_params.theta,
                                                 circular_y_enabled,
                                                 circular_x_enabled,
                                                 qwen_image_params.axes_dim);
                pe_values = &pe_vec;
            }
            int pos_len = static_cast<int>(pe_values->size() / qwen_image_params.axes_dim_sum / 2);
            // LOG_DEBUG("pos_len %d", pos_len);
            auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32, 2, 2, qwen_image_params.axes_dim_sum / 2, pos_len);
            // pe->data = pe_vec.data();
            // print_ggml_tensor(pe, true, "pe");
            // pe->data = nullptr;
            set_backend_tensor_data(pe, pe_values->data());

            ggml_tensor* modulate_index = nullptr;
            if (qwen_image_params.zero_cond_t) {
                modulate_index_vec.clear();

                int64_t h_len          = ((x->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t w_len          = ((x->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                int64_t num_img_tokens = h_len * w_len;

                modulate_index_vec.insert(modulate_index_vec.end(), num_img_tokens, 0.f);
                int64_t num_ref_img_tokens = 0;
                for (ggml_tensor* ref : ref_latents) {
                    int64_t h_len = ((ref->ne[1] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);
                    int64_t w_len = ((ref->ne[0] + (qwen_image_params.patch_size / 2)) / qwen_image_params.patch_size);

                    num_ref_img_tokens += h_len * w_len;
                }

                if (num_ref_img_tokens > 0) {
                    modulate_index_vec.insert(modulate_index_vec.end(), num_ref_img_tokens, 1.f);
                }

                modulate_index = ggml_new_tensor_1d(compute_ctx, GGML_TYPE_F32, modulate_index_vec.size());
                set_backend_tensor_data(modulate_index, modulate_index_vec.data());
            }

            auto runner_ctx = get_context();

            struct ggml_tensor* out = qwen_image.forward(&runner_ctx,
                                                         x,
                                                         timesteps,
                                                         context,
                                                         pe,
                                                         ref_latents,
                                                         modulate_index,
                                                         attention_mask);

            ggml_build_forward_expand(gf, out);

            return gf;
        }

        bool compute(int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     std::vector<ggml_tensor*> ref_latents = {},
                     bool increase_ref_index               = false,
                     struct ggml_tensor** output           = nullptr,
                     struct ggml_context* output_ctx       = nullptr,
                     struct ggml_tensor* attention_mask    = nullptr) {
            // x: [N, in_channels, h, w]
            // timesteps: [N, ]
            // context: [N, max_position, hidden_size]
            // attention_mask: optional [N, L_q, L_k] additive mask for Q4 CFG batching.
            auto get_graph = [&]() -> struct ggml_cgraph* {
                return build_graph(x, timesteps, context, ref_latents, increase_ref_index, attention_mask);
            };

            return GGMLRunner::compute(get_graph, n_threads, false, output, output_ctx);
        }

        void test() {
            struct ggml_init_params params;
            params.mem_size   = static_cast<size_t>(1024 * 1024) * 1024;  // 1GB
            params.mem_buffer = nullptr;
            params.no_alloc   = false;

            struct ggml_context* work_ctx = ggml_init(params);
            GGML_ASSERT(work_ctx != nullptr);

            {
                // auto x = ggml_new_tensor_4d(work_ctx, GGML_TYPE_F32, 16, 16, 16, 1);
                // ggml_set_f32(x, 0.01f);
                auto x = load_tensor_from_file(work_ctx, "./qwen_image_x.bin");
                print_ggml_tensor(x);

                std::vector<float> timesteps_vec(1, 1000.f);
                auto timesteps = vector_to_ggml_tensor(work_ctx, timesteps_vec);

                // auto context = ggml_new_tensor_3d(work_ctx, GGML_TYPE_F32, 3584, 256, 1);
                // ggml_set_f32(context, 0.01f);
                auto context = load_tensor_from_file(work_ctx, "./qwen_image_context.bin");
                print_ggml_tensor(context);

                struct ggml_tensor* out = nullptr;

                int64_t t0 = ggml_time_ms();
                compute(8, x, timesteps, context, {}, false, &out, work_ctx);
                int64_t t1 = ggml_time_ms();

                print_ggml_tensor(out);
                LOG_DEBUG("qwen_image test done in %lldms", t1 - t0);
            }
        }

        static void load_from_file_and_test(const std::string& file_path) {
            // cuda q8: pass
            // cuda q8 fa: pass
            // ggml_backend_t backend    = ggml_backend_cuda_init(0);
            ggml_backend_t backend    = ggml_backend_cpu_init();
            ggml_type model_data_type = GGML_TYPE_Q8_0;

            ModelLoader model_loader;
            if (!model_loader.init_from_file_and_convert_name(file_path, "model.diffusion_model.")) {
                LOG_ERROR("init model loader from file failed: '%s'", file_path.c_str());
                return;
            }

            auto& tensor_storage_map = model_loader.get_tensor_storage_map();
            for (auto& [name, tensor_storage] : tensor_storage_map) {
                if (ends_with(name, "weight")) {
                    tensor_storage.expected_type = model_data_type;
                }
            }

            std::shared_ptr<QwenImageRunner> qwen_image = std::make_shared<QwenImageRunner>(backend,
                                                                                            false,
                                                                                            tensor_storage_map,
                                                                                            "model.diffusion_model",
                                                                                            VERSION_QWEN_IMAGE);

            qwen_image->alloc_params_buffer();
            std::map<std::string, ggml_tensor*> tensors;
            qwen_image->get_param_tensors(tensors, "model.diffusion_model");

            bool success = model_loader.load_tensors(tensors);

            if (!success) {
                LOG_ERROR("load tensors from model loader failed");
                return;
            }

            LOG_INFO("qwen_image model loaded");
            qwen_image->test();
        }
    };

}  // namespace name

#endif  // __QWEN_IMAGE_HPP__
