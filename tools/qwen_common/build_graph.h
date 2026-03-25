#pragma once

#include "ggml.h"
#include <unordered_set>
#include <vector>

enum norm_type {
  NORM_TYPE_NORMAL,
  NORM_TYPE_RMS,
  NORM_TYPE_GROUP,
};

enum ffn_op_type {
  FFN_GELU,
  FFN_GELU_ERF,
  FFN_SILU,
  FFN_GELU_QUICK,
};

void cb(ggml_context *ctx0, ggml_tensor *cur0, const char *name, int il = -1);

ggml_tensor *build_norm(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *mw,
                        ggml_tensor *mb, norm_type type, float norm_eps,
                        int n_groups = 1, int il = -1);

ggml_tensor *build_linear(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int il = -1);

ggml_tensor *build_ffn(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *up,
                       ggml_tensor *up_b, ggml_tensor *gate,
                       ggml_tensor *gate_b, ggml_tensor *down,
                       ggml_tensor *down_b, ffn_op_type type_op, int il = -1);

ggml_tensor *build_attn(ggml_context *ctx0, ggml_tensor *wo, ggml_tensor *wo_b,
                        ggml_tensor *q_cur, ggml_tensor *k_cur,
                        ggml_tensor *v_cur, ggml_tensor *kq_mask,
                        float kq_scale, int il = -1);

ggml_tensor *build_mish(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *one);

ggml_tensor *build_conv1d(ggml_context *ctx0, ggml_tensor *cur, ggml_tensor *w,
                          ggml_tensor *b, int s = 1, int p = 0, int d = 1);

// Grouped Conv1d: splits input channels into groups and applies separate conv per group
ggml_tensor *build_conv1d_grouped(ggml_context *ctx0, ggml_tensor *cur,
                                  ggml_tensor *w, ggml_tensor *b, int groups,
                                  int s = 1, int p = 0, int d = 1);

// TODO: to implement
ggml_tensor *build_flip(ggml_context *ctx0, ggml_tensor *cur, int ic);
// TODO: to implement
ggml_tensor *build_conv_transpose1d(ggml_context *ctx0, ggml_tensor *cur,
                                    ggml_tensor *w, ggml_tensor *b, int s,
                                    int p, int d);