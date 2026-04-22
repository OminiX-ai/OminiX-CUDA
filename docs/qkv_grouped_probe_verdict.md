# QKV Grouped Matmul Probe Verdict

Agent: QKV-grouped-probe (FO-audit style). Host: ac01 (Atlas A2 / Ascend
910B). CANN 8.3.RC1. Probe at `/tmp/qkv_grouped_probe/test_qkv_grouped.cpp`.

## Grouped-matmul family on ac01

Header inventory (`/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/`)
and resolved symbols in `libopapi.so`:

| Op | Header | Symbol resolved |
|---|---|---|
| `aclnnGroupedMatmul` (v1) | `aclnn_grouped_matmul.h` | yes |
| `aclnnGroupedMatmulV2` | `aclnn_grouped_matmul_v2.h` | yes |
| `aclnnGroupedMatmulV3` | `aclnn_grouped_matmul_v3.h` | yes |
| `aclnnGroupedMatmulV4` | `aclnn_grouped_matmul_v4.h` | yes |
| `aclnnGroupedMatmulV5` | `aclnn_grouped_matmul_v5.h` | yes |
| plus 14 variants (swiglu-quant, finalize-routing, alltoall, allreduce, weight-nz, add) | — | — |

Probed: **V3** (non-MoE canonical) and **V4** (adds perTokenScale / actType).
Both accept `antiquantScale` aclTensorList of FLOAT16 — exactly the A16W8
pseudo-quant lane.

## Dtype matrix analysis

Header declares `x` FLOAT16/BFLOAT16/INT8/FLOAT32 and `weight`
FLOAT16/BFLOAT16/INT8/FLOAT32 (V4 adds INT4/FP8). `antiquantScale` FLOAT16/
BFLOAT16 — the A16W8 signal. Runtime check `CheckGroupedMatmulAntiQuant` in
`aclnn_grouped_matmul.cpp:742` **requires `antiquantOffset` non-null** when
antiquant is active, even though the header marks it Optional. Supplying a
zero-filled FLOAT16 offset tensor list (one per group) unblocks the op.

No MoE / `expertTokens` gate on the non-MoE dispatch path when `groupType=-1`
and `groupListOptional=nullptr`. V3/V4 both accept this configuration.

## Shape semantics for "3 matmuls sharing input, 3 different weights"

The grouped-matmul contract is `y[i] = x[i] @ weight[i]` — one pairing per
group. For "shared activation, different weights, different N" we pass
`x_list = {hidden, hidden, hidden}` (3 views of the same device buffer),
`weight_list = {w_q, w_k, w_v}` (individual aclTensors, independent N),
`antiquantScale_list = {s_q, s_k, s_v}`, `antiquantOffset_list` of 3 zero-
F16 tensors, `y_list = {q, k, v}`. With `groupType = -1` (no axis split) and
`splitItem = 0` (multi-tensor output), asymmetric N (Nq=2048, Nk=Nv=1024) is
accepted without re-padding.

Weight tensor layout identical to WQBMMv3: logical `[K, N]` shape with
`[1, K]` strides (column-major on top of a `[N, K]` physical blob). No
weight re-pack needed for prod wiring.

## Standalone correctness + wall (ac01, 3-run median)

| Path | Q max_abs_diff | K max_abs_diff | V max_abs_diff | Wall μs median |
|---|---|---|---|---|
| 3 × WQBMMv3 (ref) | 0 (baseline) | 0 (baseline) | 0 (baseline) | 98-102 |
| aclnnGroupedMatmulV3 | 1.53e-5 | 6.13e-5 | 3.05e-5 | 94-100 |
| aclnnGroupedMatmulV4 | 1.53e-5 | 6.13e-5 | 3.05e-5 | 94-106 |

Reference magnitude `ref_max ~ 0.036` → relative diff ~1.7e-3 on K (worst),
consistent with 1-ulp-class F16 quantization noise from a different kernel
tiling. ~36% of output bits differ but all within 1–2 ULP — acceptable for a
non-bit-exact fused-op substitution. **No NaN, no out-of-bound values.**

Grouped V3 is **3-8 μs faster** per QKV triplet than 3 × WQBMMv3 at host-side
wall time, p90 tighter (96–106 μs vs 104–130 μs).

## Weight re-pack requirement

**None.** The WQBMMv3 layout (logical `[K,N]`, strides `[1,K]`, underlying
physical `[N,K]` INT8 blob) is a valid entry in the grouped weight list as-is.
Prod wiring would reuse `layer_w_[il].q_proj_w` / `k_proj_w` / `v_proj_w`
device buffers and their per-channel F16 scale buffers without any data
transformation. Need only allocate a shared zero-F16 offset buffer of length
`max(Nq, Nk, Nv)` (trivial).

## Verdict

- [x] **GREEN**: grouped works at A16W8 with the `antiquantOffset=zeros`
      workaround; matches reference within expected ULP noise; wall is
      consistently faster than 3 × WQBMMv3 on host side. Ship candidate.
- [ ] YELLOW
- [ ] RED

## Recommendation for PM

Wire `aclnnGroupedMatmulV3` for the three Q/K/V projections in the CP
attention sublayer. Budget:

- **Projected fps gain**: +0.3 fps at 30-32 fps baseline under TQE=2
  dispatch amortization (5 layers × 15 forwards × 4 μs saved = 0.3 ms/frame,
  ~1% of a 33 ms frame). Host-side wall delta is ~6 μs per triplet;
  TQE-amortized delta is ~4 μs per triplet.
- **Implementation cost**: ~3-4 hours. No weight re-pack. Add a symbol-
  resolution stub in `cp_cann_symbols.cpp` for V3 + V3GetWorkspaceSize, a
  shared zero-F16 offset buffer (one-time alloc), and replace the 3 calls in
  `w8_matmul_(q)`, `w8_matmul_(k)`, `w8_matmul_(v)` at the attn block
  entry with one `gmm_qkv_()` helper that builds the 3-list.
- **Risk**: low. Output differs by 1-2 ULP vs reference, so byte-identical
  parity gates (G2/G3 aclgraph harness) will need a tolerance bump (1e-3
  relative). Confirm on end-to-end audio output before locking in.
- **Blocking concerns**: none. The `antiquantOffset` must-be-non-null runtime
  check contradicts the header's `Optional` annotation — worth a one-line
  upstream note, but not a ship-blocker.

**A16W8 gap narrative update**: this is the **first** grouped fused-op we've
found on ac01 that accepts A16W8 without MoE gating (FFNV3 rejected it
earlier; `aclnnGroupedMatmulV3/V4` with `groupType=-1` does not). Worth
calling out in the vendor deck as evidence that the A16W8 surface does
exist on grouped-op family — the gap was in the FFN / SwiGLU fused-op
variants, not in GroupedMatmul itself.
