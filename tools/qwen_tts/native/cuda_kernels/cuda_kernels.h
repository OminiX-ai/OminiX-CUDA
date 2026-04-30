#pragma once
// ============================================================================
// Custom CUDA kernels for TalkerCudaEngine (Phase 2.2 forward decode body).
//
// Each launcher takes raw pointers + a stream and runs a single op. They are
// intentionally narrow-scoped (S=1 decode-path shapes) so the inner loop is
// just `cublasGemmEx` calls plus a handful of these. No ggml graph runtime
// is reachable from here.
//
// All buffers are device pointers. F16 = __half. F32 = float.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ominix_cuda {

// F32 -> F16 cast.
//   in  : F32 [n]
//   out : F16 [n]
void launch_cast_f32_to_f16(const float *in, __half *out, int n,
                            cudaStream_t stream);

// F16 -> F32 cast.
//   in  : F16 [n]
//   out : F32 [n]
void launch_cast_f16_to_f32(const __half *in, float *out, int n,
                            cudaStream_t stream);

// y = a + b (F16 elementwise add).
//   a, b, y : F16 [n]
void launch_add_f16(const __half *a, const __half *b, __half *y, int n,
                    cudaStream_t stream);

// F32 -> FP8 E4M3 cast with input pre-scaling. Output is __nv_fp8_e4m3 packed
// as 1 byte per element. The kernel multiplies each input by `inv_scale`
// before quantizing, so `out_byte = E4M3(input * inv_scale)`. Caller passes
// `inv_scale = 1.0 / activation_scale` to land FP8 inputs near unity range.
//   in     : F32 [n]
//   out    : __nv_fp8_e4m3 [n]   (typed as void* in the launcher to keep the
//                                  cuda_fp8.h dependency out of this header)
void launch_cast_f32_to_fp8_e4m3_scaled(const float *in, void *out,
                                         float inv_scale, int n,
                                         cudaStream_t stream);

// RmsNorm: y = (x / sqrt(mean(x^2) + eps)) * gamma.
//   x      : F16 [rows, cols]
//   gamma  : F32 [cols]   (Qwen3 norm gammas are F32 on disk)
//   y      : F16 [rows, cols]   (may alias x)
// Each row is normed independently. Implementation does one block per row.
void launch_rmsnorm_f16_g32(const __half *x, const float *gamma, __half *y,
                            int rows, int cols, float eps,
                            cudaStream_t stream);

// NEOX-mode RoPE on a [n_heads, head_dim] tile (single token).
//   x     : F16 [n_heads, head_dim]   (in-place permitted iff y == x)
//   cos   : F16 [head_dim]            (row of precomputed cos table at pos)
//   sin   : F16 [head_dim]            (row of precomputed sin table at pos)
//   y     : F16 [n_heads, head_dim]
//
// NEOX mode: tile head_dim into two halves [low, high], where
//   y[low ] = x[low ] * cos[low ] - x[high] * sin[low ]
//   y[high] = x[low ] * sin[high] + x[high] * cos[high]
// This matches the cos/sin table layout (halves duplicated) used in Ascend.
void launch_rope_neox_f16(const __half *x, const __half *cos,
                          const __half *sin, __half *y,
                          int n_heads, int head_dim, cudaStream_t stream);

// Fused SwiGLU: y[i] = silu(gate[i]) * up[i] = (gate[i] / (1+exp(-gate[i]))) * up[i].
//   gate, up, y : F16 [n]   (y may alias gate)
void launch_swiglu_f16(const __half *gate, const __half *up, __half *y, int n,
                       cudaStream_t stream);

// Single-token GQA attention (decode path, S=1).
//   q       : F16 [n_heads,   head_dim]
//   k_cache : F16 [seq_len,   n_kv,  head_dim]   (contiguous, MAX_SEQ stride
//                                                  in the leading dim, only
//                                                  the first seq_len rows are
//                                                  valid)
//   v_cache : F16 [seq_len,   n_kv,  head_dim]   (same layout as K)
//   y       : F16 [n_heads,   head_dim]
//
//   seq_len    : current valid KV length (== pos + 1)
//   n_heads    : Q heads (e.g. 16)
//   n_kv       : KV heads (e.g. 8); GQA group size = n_heads / n_kv.
//   head_dim   : per-head dim (e.g. 128)
//   inv_sqrt_d : 1.0f / sqrtf(head_dim)
//
// Implementation: one block per Q head. Each block:
//   1. Loads q[h] (head_dim floats) into shmem.
//   2. For each token t in [0, seq_len): compute score[t] = dot(q[h], k[t, h_kv]) * inv_sqrt_d
//      where h_kv = h / group_size.
//   3. softmax(scores) over the seq_len axis.
//   4. Reduces sum_t softmax[t] * v[t, h_kv, :] to produce y[h].
//
// Numerically stable: max-subtract before exp.
void launch_attn_decode_gqa_f16(const __half *q,
                                const __half *k_cache,
                                const __half *v_cache,
                                __half *y,
                                int seq_len, int n_heads, int n_kv,
                                int head_dim, float inv_sqrt_d,
                                cudaStream_t stream);

// ============================================================================
// P1 (April 2026) — device-resident-pos variants for capture-once decode graph.
//
// These mirror the host-pos launchers above but read `pos` from a device-side
// const int* at kernel runtime. That makes the captured cudaGraph topology
// truly static across positions: the same graph can replay every step with
// only a tiny (4-byte) H2D memcpy of the new pos before launch. Without this,
// rope_cos_row / k_slot / v_slot / seq_len_total are baked into kernel
// argument slots at capture time and the graph has to be re-captured (or
// patched via cudaGraphExecUpdate) every step, cancelling the launch-overhead
// win.
//
// Used only when OMNX_TTS_DECODE_GRAPH_ONCE=1; legacy host-pos paths above
// are unchanged.
// ============================================================================

// NEOX RoPE with device-resident pos. Same math as launch_rope_neox_f16, but
// the cos/sin row offset is computed inside the kernel from *pos_dev. The
// destination y is also offset internally by *pos_dev * y_stride iff
// y_stride > 0 (used for the K-cache slot variant); pass y_stride=0 to write
// to the unstrided q-output buffer.
//
//   x         : F16 [n_heads, head_dim]
//   cos_base  : F16 [MAX_SEQ, head_dim]   (full table)
//   sin_base  : F16 [MAX_SEQ, head_dim]   (full table)
//   y_base    : F16 [n_heads, head_dim] when y_stride==0
//             : F16 [MAX_SEQ, n_heads*head_dim] when y_stride>0
//   pos_dev   : const int *               (device, size 1)
//   y_stride  : int                       (elements per row when offsetting y;
//                                          0 = no offset, write to y_base)
void launch_rope_neox_f16_dev(const __half *x,
                              const __half *cos_base,
                              const __half *sin_base,
                              __half *y_base,
                              const int *pos_dev,
                              int n_heads, int head_dim,
                              int y_stride,
                              cudaStream_t stream);

// V-cache slot write. Replaces a host-pos cudaMemcpyAsync of v_dev_ into
// (v_cache + pos*kv_dim). Tiny copy kernel that reads pos at runtime so the
// destination offset isn't baked into a captured memcpy node.
//
//   src      : F16 [kv_dim]
//   dst_base : F16 [MAX_SEQ, kv_dim]
//   pos_dev  : const int *  (device, size 1)
void launch_v_write_kv_dev(const __half *src, __half *dst_base,
                           const int *pos_dev, int kv_dim,
                           cudaStream_t stream);

// Single-token GQA attention with device-resident pos. seq_len is read inside
// the kernel as (*pos_dev) + 1. Shared-memory scores buffer is allocated at
// MAX_SEQ * sizeof(float) at launch time (constant across positions, so the
// captured graph node is static); only the first seq_len entries are used at
// runtime.
//
//   q        : F16 [n_heads, head_dim]
//   k_cache  : F16 [MAX_SEQ, n_kv, head_dim]
//   v_cache  : F16 [MAX_SEQ, n_kv, head_dim]
//   y        : F16 [n_heads, head_dim]
//   pos_dev  : const int *  (device, size 1; seq_len = *pos_dev + 1)
//   max_seq  : int          (upper bound for shmem allocation)
void launch_attn_decode_gqa_f16_dev(const __half *q,
                                    const __half *k_cache,
                                    const __half *v_cache,
                                    __half *y,
                                    const int *pos_dev,
                                    int max_seq,
                                    int n_heads, int n_kv,
                                    int head_dim, float inv_sqrt_d,
                                    cudaStream_t stream);

// ============================================================================
// SpeechTokenizerDecoder F32 ops (Phase 2.7b).
//
// All shapes are row-major [T, C] with C fastest. See decoder_ops.cu for
// per-op semantics; the comments below mirror that file's contract.
// ============================================================================

// Causal Conv1d im2col: zero-pad-left + window of size K.
//   in  : F32 [T, C_in]
//   out : F32 [T, K * C_in]   (out[t, k*C_in + c] = in_padded[t + k - (K-1), c])
void launch_causal_conv1d_im2col_f32(const float *in, float *out,
                                      int T, int C_in, int K,
                                      cudaStream_t stream);

// Dilated causal Conv1d im2col: same as above but with dilation D.
//   out[t, k*C_in + c] = in_padded[t + (k - (K-1)) * D, c]
// Padding on the left is (K-1)*D zeros (matches Ascend build_causal_conv1d).
// Reuses the K=1 / D=1 fast path implicitly (caller can just use a GEMM).
void launch_dilated_causal_conv1d_im2col_f32(const float *in, float *out,
                                              int T, int C_in, int K, int D,
                                              cudaStream_t stream);

// Generic causal ConvTranspose1d (stride=S, kernel=K). PyTorch semantics:
// for k in [0,K): for t in [0,T): out[t*S + k, oc] += sum_ic in[t, ic] * w[k, oc, ic]
// (assuming GGUF row-major [K, C_out, C_in] with C_in fastest).
// Pre-trim length is (T-1)*S + K; we trim (K-S) samples from the RIGHT to match
// Ascend's "causal" convention. Final out length is T*S, written into `out`.
//   in   : F32 [T, C_in]
//   w    : F32 [K, C_out, C_in]
//   b    : F32 [C_out]   (or nullptr)
//   out  : F32 [T*S, C_out]
void launch_causal_conv_transpose1d_f32(const float *in, const float *w,
                                          const float *b, float *out,
                                          int T, int C_in, int C_out,
                                          int K, int S,
                                          cudaStream_t stream);

// SnakeBeta activation (Ascend reference build_snake_beta):
//   y[t, c] = x[t, c] + exp(-beta[c]) * sin(x[t, c] * exp(alpha[c]))^2
// alpha, beta are per-channel learnable params [C].
//   x, y : F32 [T, C]   (y may alias x)
void launch_snake_beta_f32(const float *x, const float *alpha,
                            const float *beta, float *y,
                            int T, int C, cudaStream_t stream);

// Tanh elementwise: y[i] = tanhf(x[i]). In-place permitted.
void launch_tanh_f32(const float *x, float *y, int n, cudaStream_t stream);

// Depthwise causal Conv1d.
//   in  : F32 [T, C]
//   w   : F32 [K, C]   (per-channel filter; GGUF [K, 1, C] with the singleton
//                       in-channels-per-group axis collapsed.)
//   b   : F32 [C]   or nullptr
//   out : F32 [T, C]
void launch_depthwise_conv1d_causal_f32(const float *in, const float *w,
                                         const float *b, float *out,
                                         int T, int C, int K,
                                         cudaStream_t stream);

// ConvTranspose1d, kernel=2, stride=2 (matches the upsample.X.0.conv shape).
//   in  : F32 [T, C_in]
//   w   : F32 [K=2, C_out, C_in]   (GGUF row-major; C_in fastest)
//   b   : F32 [C_out]   or nullptr
//   out : F32 [2*T, C_out]
void launch_conv_transpose1d_k2s2_f32(const float *in, const float *w,
                                       const float *b, float *out,
                                       int T, int C_in, int C_out,
                                       cudaStream_t stream);

// LayerNorm (affine) over the last axis. One block per row.
//   x      : F32 [T, C]
//   gamma  : F32 [C]
//   beta   : F32 [C]   or nullptr
//   y      : F32 [T, C]   (may alias x)
void launch_layernorm_f32(const float *x, const float *gamma, const float *beta,
                           float *y, int T, int C, float eps,
                           cudaStream_t stream);

// GELU (erf form, ConvNeXt default).
//   y[i] = 0.5 * x[i] * (1 + erf(x[i] / sqrt(2)))
void launch_gelu_erf_f32(const float *in, float *out, int n,
                          cudaStream_t stream);

// Bias add over [T, C]: y[t, c] = x[t, c] + bias[c]. In-place OK.
void launch_bias_add_f32(const float *x, const float *bias, float *y,
                          int T, int C, cudaStream_t stream);

// Channel scale: y[t, c] = x[t, c] * gamma[c]. ConvNeXt gamma residual scale.
void launch_channel_scale_f32(const float *x, const float *gamma, float *y,
                               int T, int C, cudaStream_t stream);

// y = a + b, flat [n] F32 (residual).
void launch_add_f32(const float *a, const float *b, float *y, int n,
                     cudaStream_t stream);

// ============================================================================
// P2 (April 2026) — on-device sampling kernels.
//
// Eliminate the per-step D2H of logits + host sampling + H2D of next embedding
// by sampling directly on device and looking up the next embedding row from a
// device-resident table. Used inside the captured decode graph.
// ============================================================================

// In-place repetition penalty: for each id in `recent_tokens_dev`, divide
// (or multiply if logit<=0) `logits[id + recent_offset]` by `penalty`,
// clamped to the [lo, hi) range. Mirrors the host apply_repetition_penalty
// in tts_server.cpp.
//   logits             : F32 [vocab]
//   recent_tokens_dev  : int [n_recent]   (device)
//   recent_offset      : int              (added to each recent_tokens_dev[i]
//                                          before clamp; e.g. lo for predictor)
void launch_apply_repetition_penalty_f32(float *logits, int lo, int hi,
                                          const int *recent_tokens_dev,
                                          int n_recent, int recent_offset,
                                          float penalty,
                                          cudaStream_t stream);

// Top-K sampler with temperature. Single-block design; reads `logits[lo..hi)`,
// finds top-K by repeated max with exclusion (K small, default 50), softmaxes
// over the top-K with the temperature divisor, then samples either greedy
// (do_sample==0 || temperature<=0) or stochastic via xorshift64* PRNG seeded
// by (seed XOR step_idx).
//   logits         : F32 [vocab]
//   lo, hi         : sub-range
//   top_k          : 1..256; <=0 -> full
//   temperature    : >0 stochastic divisor; <=0 forces greedy
//   do_sample      : 0=greedy, 1=stochastic
//   seed           : per-request RNG seed
//   step_idx       : output slot AND RNG mix
//   out_token_dev  : int [pending_slots]  (writes [step_idx])
void launch_sample_top_k_f32(const float *logits,
                              int lo, int hi,
                              int top_k,
                              float temperature,
                              int do_sample,
                              uint64_t seed,
                              int step_idx,
                              int *out_token_dev,
                              cudaStream_t stream);

// Atomic-style increment of a single int in device memory: *p += 1. Used by
// the on-device decode loop to step `pos_dev_` between captured replays
// without the captured-H2D-from-pinned-host race condition (the H2D node's
// source pointer is fixed at capture time, but on multi-step replay the
// caller would have to ensure pos_host_pin_ stays valid until the H2D
// actually executes — not safe without per-step sync).
void launch_increment_int(int *p_dev, cudaStream_t stream);

// Set a single device int to `value`. Used by the predictor decode loop to
// reset pos_dev_ to -1 at the start of each frame without an H2D-from-host
// (the chain graph's first node will then increment pos_dev_ to 0 on the
// first replay). Race-safe: the kernel runs on `stream` and reads no host
// memory.
void launch_set_int_value(int *p_dev, int value, cudaStream_t stream);

// Record one slot of the per-group rep-history buffer. Writes
// `slot_dev[0] = src_token_dev[src_index] - lo`. Tiny single-thread kernel
// designed to chain after the sampler kernel inside the predictor's
// decode loop, so the next frame's rep-penalty kernel sees up-to-date
// history without any H2D round-trip.
//   src_token_dev : int [N]   (sampler output buffer, absolute ids)
//   src_index     : int       (which sampler slot to read)
//   lo            : int       (group's low bound, subtracted to get
//                              zero-based codec id in [0, group_size))
//   slot_dev      : int *     (rep_history_dev_ + g * max_frames + frame_t)
void launch_record_rep_history(const int *src_token_dev, int src_index,
                                int lo, int *slot_dev,
                                cudaStream_t stream);

// Embedding lookup: out[0..hidden) = table[tok_dev[slot] * hidden + i].
// Used to populate the next decode step's input embedding from a sampled token.
//   tok_dev    : int [N]   (device)
//   slot       : int       (which slot to read)
//   table_dev  : F32 [vocab, hidden]   (row-major)
//   out_dev    : F32 [hidden]
void launch_embedding_lookup_f32(const int *tok_dev,
                                  int slot,
                                  const float *table_dev,
                                  int vocab,
                                  int hidden,
                                  float *out_dev,
                                  cudaStream_t stream);

}  // namespace ominix_cuda
