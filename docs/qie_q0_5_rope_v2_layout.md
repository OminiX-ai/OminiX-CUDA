# Q0.5.3 — RoPE-V2 3D-axial layout probe

**Agent**: QIE-Q0.5
**Date**: 2026-04-22
**Mode**: source-read only. Runtime validation gated on Q1 backend unblock + Q2 native engine. Prior-art anchored on `docs/v2_rope_reopen_verdict.md` (TTS YELLOW, op-numeric-GREEN) and `docs/qwen_tts_optimization_learnings.md` §V2 RoPE (3× closed-contract history).
**Question**: Does `aclnnApplyRotaryPosEmbV2` accept QIE's non-standard 3D-axial RoPE (`axes_dim = {16, 56, 56}` per `rope.hpp:655-658` + `qwen_image.hpp:359`), or does layout conversion on the `pe` vector eat all the savings?

## TL;DR

**VERDICT: CONDITIONAL → RETIRE as a day-1 lever; RoPE pre-compute at Q2 captures most of the real win anyway.**

Two independent reasons:
1. **Layout mismatch is real.** QIE's `pe` is a 4D tensor `[2, 2, axes_dim_sum/2, pos_len] = [2, 2, 64, pos_len]` holding per-(pos, freq) quadruplets `(cos, -sin, sin, cos)` assembled from 3 separate axis RoPE chains concatenated along the freq axis. V2's expected cos/sin table layout (from TTS-side `v2_rope_reopen_verdict.md`) is 1D `[seq, head_dim]` with a half-duplicated or half-half pattern. Conversion is non-trivial, NOT a single permute.
2. **TTS history: closed 3×.** V2 RoPE parity diverged on prod wiring three times on simpler 1D GQA/MHA shapes. The op itself is numerically sound in isolation (proven 2026-04-21 on ac02), but the production-side wiring failed. Adding 3D-axial layout complexity on top of a contract that TTS already struggled with is a poor ROI bet. RoPE pre-compute at Q2 (moving `gen_qwen_image_pe` out of the step loop) delivers +10-25% per step per MLX measurement — strictly larger than V2's +0.5-1.5% per step estimate at QIE scale.

## Source evidence

### Current RoPE generation — non-standard 3D axial

`qwen_image.hpp:358-362`:

```cpp
int theta                   = 10000;
std::vector<int> axes_dim   = {16, 56, 56};   // <-- 3D decomposition
int axes_dim_sum            = 128;            // = 16 + 56 + 56
```

Each head's 128 channels are split into three axis blocks:
- Axis 0 (frequency-idx 0..7 pair): encodes a **document/index** coordinate. For QIE-Edit specifically, this is the ref-image index counter when `increase_ref_index=true` — i.e. "this token is from ref image #1 / #2 / ..." at `rope.hpp:136` `img_ids[...][0] = 1.f * index`.
- Axis 1 (frequency-idx 8..35 pair): encodes the **row coordinate** (h-axis of the patch grid).
- Axis 2 (frequency-idx 36..63 pair): encodes the **column coordinate** (w-axis of the patch grid).

Per-axis theta is shared (all at 10000). Per-axis widths are **non-uniform** (16 / 56 / 56) — so channel count per axis is unequal.

### Current `pe` tensor layout (packed quadruplet)

`qwen_image.hpp:544-561`:

```cpp
pe_vec      = Rope::gen_qwen_image_pe(...);   // flat float vector, host-side
int pos_len = static_cast<int>(pe_vec.size() / qwen_image_params.axes_dim_sum / 2);
auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                             2,                         // dim 0: quadruplet-lane (cos row / sin row)
                             2,                         // dim 1: quadruplet-lane (within-pair)
                             qwen_image_params.axes_dim_sum / 2,   // = 64 freqs
                             pos_len);                  // sequence length
```

Per-(pos, freq) the vector holds 4 floats: `(cos, -sin, sin, cos)` — see `rope.hpp:83-90`. This 4-float "rotation block" is consumed by `apply_rope` at `rope.hpp:622-640` via a `[N*n_head, L, d_head/2, 2] × [L, d_head/2, 2]` broadcast-multiply-then-sum over the last dim. It is **NOT** the canonical cos/sin table that V2 expects.

### `apply_rope` consumes the packed pe layout directly

`rope.hpp:622-640` shows pe is permuted `[3,0,1,2]` then split into `pe_0` and `pe_1` along the `pe->nb[2]` axis, each `[L, d_head/2, 2]`. The input tensor `x` is split into `x_0` / `x_1` along the `d_head` axis, each reshaped to `[N*n_head, L, d_head/2, 1]` then repeated to `[..., 2]`. The rotation is:

```
x_out = (x_0 * pe_0) + (x_1 * pe_1)   // [N*n_head, L, d_head/2, 2]
```

where `pe_0 = (cos, -sin)` and `pe_1 = (sin, cos)` per quadruplet. This is mathematically equivalent to a standard rotary embedding but the **physical memory layout** is 4-float-block packed, not the 2-separate-tables layout that V2 expects.

### V2's expected layout (from TTS-side probe)

Per `v2_rope_reopen_verdict.md` (2026-04-21, ac02, standalone harness that confirmed V2 is numerically correct on TTS's GQA shape):

```
Q shape:   [1, 1, NQ=16, DH=128] BSND, F16
K shape:   [1, 1, NK=8,  DH=128] BSND, F16
cos:       [MAX_SEQ, DH=128] F16, half-duplicated ([c(j), ..., c(half-1), c(0), ..., c(half-1)])
sin:       [MAX_SEQ, DH=128] F16, half-duplicated (same pattern)
pos:       int64 scalar (single position per call)
rotaryMode: "half"
layout:    1 (BSND)
```

V2 takes **two 1D tables** (cos, sin) indexed by position, one pair of Q/K tensors, and a scalar position. It produces in-place rotated Q/K.

### Layout-conversion cost analysis

QIE's `pe` vector encodes **per-token** positional info (pos varies across the sequence, not across invocations — unlike TTS decode where pos is a single scalar and the cos/sin table is position-indexed). To convert QIE's packed-quadruplet `pe` to V2's cos/sin pair-table format:

1. **Split the 4-float block into cos and sin halves.** Pe_vec stores `(cos, -sin, sin, cos)` per freq; cos is at offsets `4k+0` and `4k+3`, sin at `4k+1` (negated) and `4k+2`. This is an elementwise gather — doable as an aclnn custom op but not free.
2. **Reconcile the 3-axis concat.** V2 expects a single per-token cos/sin table. QIE's pe is the concat of three per-axis tables (width 16 / 56 / 56). The concat dimension matches head_dim=128, so in principle a single V2 call can consume the concatenated cos/sin as long as the freq values at each channel correspond correctly. They do — QIE's rope is just "a different schedule of frequencies per channel," and V2 doesn't care what the cos/sin values are as long as the table shape matches. So the axis concat is already absorbed.
3. **Per-token position argument.** V2 takes a scalar position; QIE has per-token positions (varying along seq). This is the killer. For QIE, each token has a different `[axis0, axis1, axis2]` coordinate triple, so the cos/sin values at head_dim position `d` depend on which axis-block `d` falls into and what that axis's position value is for this particular token. You can't call V2 once per sequence — you'd have to loop or use V2's sequence-aware variant if one exists.

**V2 is designed for scalar-pos decode** (one rotation per call, same position for every Q/K row — standard LLM decode). QIE needs **per-token rotation with different positions per token**, which is the prefill/image-attention regime. V2's `aclnnApplyRotaryPosEmbV2` does accept a `seq` dimension in Q/K descriptors, but pos is still a single int64 scalar in the probe-tested signature. Extending to per-token pos would require either:

- Loop V2 per-position (4096 calls per apply_rope, dispatch-floor explodes — fatal)
- Use a different vendor op that takes cos/sin **tables** matching Q/K seq length (which is what the **current** ggml-cann RoPE impl already does)

### Is `pe` re-computed per step?

`qwen_image.hpp:528-554` shows `build_graph` is called per-forward, per-step. `pe_vec = Rope::gen_qwen_image_pe(...)` is called inside `build_graph` — **yes, per step**. This is exactly the MLX-documented structural inefficiency (`qwen-image-mlx` docs §RoPE pre-compute). Moving this out (Q2 lever, +10-25% realistic) is a strictly larger win than V2 wiring even if V2 worked.

## Verdict

**CONDITIONAL → effectively RETIRE as a Q2/Q5 lever.** Reasoning:

- **Layout mismatch is semantic, not cosmetic.** QIE needs per-token per-axis cos/sin tables. V2 is a scalar-pos op. The layouts are not permute-compatible; they're algorithmically different use cases.
- **RoPE pre-compute at Q2 (MLX import) captures the real win.** +10-25% per step per MLX's own measurement of the same optimization. Structural refactor (2-3 days). Low risk. **This is the lever.**
- **V2 estimated upside at QIE scale is +0.5-1.5% per step** (per `qie_optimization_learnings.md` §3 lever ranking, line 10). Even if wiring cost zero, the end-to-end delta is noise-band adjacent at QIE's step count. The wiring cost is NOT zero given the TTS 3× closed-contract history.
- **TTS V2 RoPE reopen probe (2026-04-21) confirmed the op is sound** on 1D GQA decode shapes. That's a useful probe outcome for MLA/LLM work but it doesn't extend to QIE's 3D-axial per-token-pos case. No prior-art for V2 at image-DiT scale.

### Precedent ruling

The TTS playbook closed V2 RoPE three times:
1. Original A.2 contract: 457 vs 434 frame divergence, attributed to GQA packed-UB.
2. A.2 reopen (2026-04-21): op is numerically correct on TTS GQA shape in isolation; production wiring bug. Reopened as "rewiring probe" — not yet re-landed.
3. Implicit: V2 has zero wire-in-prod successes on our fork; every attempt has bounced.

QIE cannot afford to repeat that pattern on a lever estimated at sub-2% per step. The engineering bar for a +1% per step lever is "costs less than 2 engineer-days." V2 wiring on 3D-axial layout costs substantially more.

## Recommendation for PM

- **Retire RoPE-V2 wiring from Q5 scope.** Remove the bullet from the Q5 contract section.
- **Promote RoPE pre-compute to Q2 must-have** (already flagged in `qie_optimization_learnings.md` §3.8 as "must-have, low risk"). 2-3 engineer-days. +10-25% per step realistic.
- **Keep the door open for a future revisit** if CANN ships a `aclnnApplyRotaryPosEmbTabular` or equivalent that takes per-token cos/sin tables at arbitrary head-dim-axis decomposition. Flag in the vendor-ask list (`qie_optimization_learnings.md` §8.2 item 2 — `aclnnFusedAttentionWithRope2D` would subsume this).
- **Do not re-dispatch** a V2-RoPE agent on QIE without a fresh probe specifically at the 3D-axial-per-token-pos scenario. The generic V2 op's capability is not the question; the question is whether a different sibling op (future `aclnnApplyRotaryPosEmbNd` / `V3`) ships that matches the QIE use case. That belongs in vendor-channel tracking, not contract scope.

## Residual items (for the vendor-ask ledger)

1. **`aclnnApplyRotaryPosEmbNd`** — hypothetical future vendor op accepting N-dim axis decomposition with per-token coordinate tables. Would unblock V2-class wiring on QIE, Flux, SD3 (all use 2D/3D axial RoPE). Raise at CANN roadmap review.
2. **`aclnnFusedAttentionWithRope2D`** — already on the QIE-LEARN vendor-ask list (§8.2 item 2). Subsumes per-token RoPE + attention into one op for DiT image-token workloads. If that ships, both the RoPE-V2 and the FIAv2 discussions collapse into one wiring call.

Neither is a 8.3.RC1 ask; both are 8.5+ roadmap signals.
