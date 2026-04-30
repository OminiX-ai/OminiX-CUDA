#include "common.cuh"

void ggml_cuda_op_repeat(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_sub(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_mul(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_div(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_repeat_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_fused_add(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int n_fuse);

// QIE-Edit fused gated residual: out = x + y * gate.
// Matches the graph chain { GGML_OP_MUL, GGML_OP_ADD } where:
//   mul_node->src[0] = y           (token activations)
//   mul_node->src[1] = gate        (broadcasts over tokens)
//   add_node->src[0] = x           (skip connection)
//   add_node->src[1] = mul_node    (gated activations)
// The kernel folds the MUL output away — never lands in DRAM.
// All tensors must be GGML_TYPE_F32 and contiguous.
void ggml_cuda_op_mul_add_gated(ggml_backend_cuda_context & ctx,
                                ggml_tensor *               mul_node,
                                ggml_tensor *               add_node);
