#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// QIE-Edit fused LayerNorm + modulate: out = (1 + scale) * LayerNorm(x) + shift.
// Matches the graph chain { GGML_OP_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD }
// produced by Flux::modulate(LayerNorm(x), shift, scale) when LayerNorm has
// elementwise_affine=false (i.e. no per-channel weight/bias on the norm itself).
//
// norm_node:    the GGML_OP_NORM node (src[0] = x)
// mul_node:     the GGML_OP_MUL node   (one src is norm_node, other is `scale`)
// add_inner:    the inner GGML_OP_ADD  (srcs are norm_node and mul_node)
// add_outer:    the outer GGML_OP_ADD  (srcs are add_inner and `shift`); writes dst.
void ggml_cuda_op_norm_modulate(ggml_backend_cuda_context & ctx,
                                ggml_tensor *               norm_node,
                                ggml_tensor *               mul_node,
                                ggml_tensor *               add_inner,
                                ggml_tensor *               add_outer);
