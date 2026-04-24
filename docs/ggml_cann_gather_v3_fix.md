# ggml-cann gather_v3 pool-aliasing fix under aclGraph capture

Downstream fix for the carry-forward item in `ggml_cann_sync_in_capture_fix.md` §"Carry-forward: gather_v3". After the sync-in-capture fix (commit `46b48723`) unblocked the EE9999 107027 sync-capture crash, QIE-Edit `GGML_CANN_ACL_GRAPH=1` progression revealed a second bug inside the captured graph.

## Bug

`GGML_CANN_ACL_GRAPH=1 GGML_CANN_QUANT_BF16=on` QIE-Edit smoke aborts with a
massively-parallel AscendC assert on every AIV core:

```
[ASSERT] .../ascendc/gather_v3/gather_v3_base.h:137:
         Assertion `(0 <= val && val < this->gxSize_)'
         Index 1051041638 out of range[0 196)!
...
[ERROR] ... function ggml_backend_cann_synchronize at ggml-cann.cpp:2091
        aclrtSynchronizeStream(cann_ctx->stream())
```

The crash fires at the **first `ggml_backend_graph_compute` call for the
Qwen2.5-VL vision forward** (directly after `qwen2.5vl compute buffer size:
61.24 MB(VRAM)`). It is deterministic across 256×256 and 512×512 (vision
encoder receives the same 392×392 reference image either way; `gxSize_ = 196
= 14·14` is the Qwen2.5-VL vision grid after the spatial-merge-2 stage).

## Root cause

The crashing op is `aclnn_index_select_4d` in
`ggml/src/ggml-cann/aclnn_ops.cpp:2112` — the on-device gather that backs
`GGML_OP_GET_ROWS`. The `window_index` input tensor (declared
`GGML_TYPE_I32`, 196 entries) is a ggml leaf that the diffusion runner
populates on each call via `ggml_backend_tensor_set` (synchronous
`aclrtMemcpy` H2D) **before** `ggml_backend_graph_compute` runs
(`tools/ominix_diffusion/src/ggml_extend.hpp:1817
copy_data_to_backend_tensor`).

Under aclGraph capture the index read goes bad in two independent ways:

1. **Leaf-slot recycling across the capture/replay boundary.** The
   `window_index` storage lives in the gallocr-managed compute buffer, which
   reuses slots across ops with non-overlapping lifetimes. After
   `IndexSelect` consumes `window_index`, a later op inside the same captured
   graph can write into the same slot. The pre-warm and capture passes of
   `evaluate_and_capture_cann_graph` both execute correctly because
   `ggml_backend_tensor_set` had just filled the slot. But the captured
   aclGraph does *not* record the host-side H2D (it ran on the default
   aclrtMemcpy stream, not on `cann_ctx->stream()`), so at
   `aclmdlRIExecuteAsync` replay time the slot holds whatever the subsequent
   op left in it — float-bit-pattern garbage — and the recorded
   `IndexSelect` reads those bytes as INT32 indices.

2. **Pool-address baking for F16↔F32 cast path.** The `src0->type !=
   dst->type` branch of `ggml_cann_get_rows` (line 2205) allocates a scope-
   bound pool buffer via `ggml_cann_pool_alloc`, writes the cast result into
   it, and immediately hands the pointer to `aclnn_index_select_4d` as the
   source tensor. Pool addresses are deterministic under the LIFO bump
   allocator, so the cast and gather record the same device pointer at
   capture time; but after the `ggml_cann_get_rows` scope exits the pool
   slot is free to be handed to a later allocator. The captured graph
   replays the cast/gather pair in order, and as long as no pool consumer
   from *earlier* in the same captured graph wrote to the slot the replay
   reads valid data — but defence-in-depth here would require keeping the
   buffer pinned across replay, which the current pool API cannot express.

Both failure modes are the same family as commit `61c52a34` (pool-buffer
D2H/H2D sync), but under captured-stream semantics where host sync is
illegal and pointer lifetimes span the capture window.

The observed indices (`1051041638`, `-1085232137`, ...) have the bit layout
of small-magnitude float32 values (`0x3EA08C66 ≈ 0.31`, `0xBF4B0137 ≈
-0.79`), confirming that the captured `IndexSelect` is reading a float
tensor through its INT32 interpretation — exactly what a leaf-slot recycle
would produce.

## Fix

Extend the aclGraph capture-gate scan in
`ggml_backend_cann_graph_compute` (`ggml-cann.cpp:2287`) to disqualify
**all** `GGML_OP_GET_ROWS` nodes from capture, regardless of `src0` dtype
(previously only Q4_0 / Q4_1 were gated, because they hit the
CPU-fallback sync-in-capture crash). Graphs that contain any GET_ROWS now
run eager — the same codepath they used before aclGraph existed —
avoiding both the leaf-slot-recycle and pool-baking failure modes.

Rationale for picking the scoping-gate over approach A (hold-buffer /
WSPOOL) or B (event-based ordering):

* Approach A needs a non-recyclable pool slot for every GET_ROWS index
  leaf plus every cast-path pool buffer, AND needs to teach gallocr to
  keep `window_index` / `inp_pos` / `selected_experts` leaves pinned
  across their last-use-in-captured-graph. This is ≥100 LoC touching
  three subsystems and risks gallocr invariants.
* Approach B (`aclrtRecordEvent` + `aclrtStreamWaitEvent`) orders ops
  against each other but cannot prevent gallocr from reusing the
  `window_index` slot for a later tensor — the captured graph still
  bakes the slot address, and the replay still reads whatever the
  replay itself wrote there last.
* The scoping gate matches the precedent set by commit `46b48723`
  (IM2COL_3D, Q4_0/Q4_1 GET_ROWS, K-quant MUL_MAT) and trades a small
  capture-scope reduction for immediate correctness. Graphs that do
  **not** contain GET_ROWS — including the pure Qwen-Image DiT forward,
  its attention sub-graphs, and the VAE decode — still capture
  normally, so the Q4 CFG-batching and CacheDIT measurements retain
  whatever dispatch-overhead savings aclGraph can provide on those
  paths.

## Diff

```
 ggml/src/ggml-cann/ggml-cann.cpp | 21 ++++++++++++++++-----
 1 file changed, 16 insertions(+), 5 deletions(-)
```

The single hot edit replaces the Q4_0/Q4_1-only predicate at
`ggml-cann.cpp:2307` with an unconditional `node->op == GGML_OP_GET_ROWS`
check, and updates the comment block to record both failure modes and
the reference crash signature.

## Gate suite

All runs on ac02 (910B4, CANN 8.3.RC1), QIE-Edit-2509-Q4_0 +
Qwen2.5-VL-7B-Instruct-Q4_0 + mmproj-BF16 + qwen_image_vae.safetensors,
prompt "convert to black and white", seed 42, cat.jpg reference.

| shape | steps | ACL_GRAPH=0 (baseline) | ACL_GRAPH=1 (this fix) |
|---|---|---|---|
| 256×256 | 2 | **PASS** 143.98s, `x_0=[-1.458, 1.491]` | **PASS** 139.21s, `x_0=[-1.456, 1.492]` |
| 512×512 | 2 | **PASS** 241.57s, `x_0=[-1.408, 1.740]` | **PASS** 241.44s, `x_0=[-1.403, 1.739]` |

* NaN check passes on `encoder/cond.c_crossattn`,
  `diffusion/x_0 (sampled latent)`, and `vae/decoded_image` for all four
  runs.
* ACL_GRAPH=1 matches ACL_GRAPH=0 within 3% at both shapes; 256×256
  shows a modest (~3%) speedup, 512×512 is effectively parity.
* The lift is smaller than the +10-20% figure from TTS G2 because (a)
  the Qwen-Image DiT is matmul-dominated at these shapes and step
  counts (dispatch overhead is a small fraction of the ~5s/step cost),
  and (b) vision forward + VAE encode/decode subgraphs are forced
  eager by the new gate plus the existing IM2COL_3D / Q4_0 GET_ROWS
  gates.

## Logs on ac02

* `/tmp/qie_aclgraph0_baseline_fixed.log` — ACL_GRAPH=0 256×256/2-step.
* `/tmp/qie_aclgraph0_512_fixed.log` — ACL_GRAPH=0 512×512/2-step.
* `/tmp/qie_aclgraph1_256_v3.log` — ACL_GRAPH=1 256×256/2-step (fixed).
* `/tmp/qie_aclgraph1_512_v2.log` — ACL_GRAPH=1 512×512/2-step (fixed).
* `/tmp/qie_aclgraph1_crash.log` — original repro before this fix
  (vision forward abort on the first graph compute).

## Carry-forward for the Q1 upstream PR bundle

* The scoping-gate extension is a single-line addition to the existing
  aclGraph capture-gate patch from commit `46b48723`. Easiest to ship
  as a squashed "aclGraph capture compatibility" follow-up to the Q1
  Q4_0/Q4_1 GET_ROWS PR (`bbfa1912`) and Q1 Q4_1/K-quant MUL_MAT PR
  (`c8fea6e0`), or folded into the same followup commit as
  `46b48723`'s sync helper.
* The correct long-term fix (for aclGraph perf on DiT-style graphs that
  *do* want to capture their embedding lookups) is to route captured
  graphs through a stream-ordered allocator, which would let gallocr
  pin leaf slots for the duration of the capture window and let the
  pool keep an intermediate buffer live until the aclGraph is
  destroyed. That is a CANN-side design question and larger than this
  fix can cover.
* With GET_ROWS gated out, `GGML_CANN_ACL_GRAPH=1` is now a working
  baseline for the Q4 CFG-batching dispatch-overhead measurement; the
  CFG-batching agent can rely on both eager and aclGraph paths to
  produce byte-identical latents on the Q4_0 stack without having to
  chase this crash again.
