# Path C #4 — ADD/MUL/NORM F32 widening audit (PIVOTED)

**Status**: AUDIT DONE, MISSION SCOPE INVALIDATED BY RECEIPTS. Pivot plan
below. No dispatcher code changed; no gate run. Fresh worktree
`~/work/OminiX-Ascend-c4` on `path-c-4-probe` branch from `origin/main`
(722d1cf9).

## TL;DR

1. **`GGML_PREC_F32` is not settable on ADD/MUL/NORM in ggml core.** Only
   `GGML_OP_MUL_MAT`, `GGML_OP_MUL_MAT_ID`, and `GGML_OP_FLASH_ATTN_EXT`
   carry a precision hint through `op_params` (see `ggml/src/ggml.c`
   setters `ggml_mul_mat_set_prec` @ 3242 and `ggml_flash_attn_ext_set_prec`
   @ 5380; no equivalent for ADD/MUL/NORM). Extending the Path C #3 helper
   switch to these ops returns false regardless of op_params — there's no
   upstream hint to honor.
2. **The per-op trace receipts prove the residual stream is already in
   F32 storage.** `docs/qie_leak2_per_op_trace.md` measures img_resid2
   mean-abs = 9.61e4 at block 0 and 1.89e6 at block 59. Both values
   exceed F16 saturation (65504). If the residual were F16-stored, the
   trace's `ggml_cast(t, F32) → abs → sum` pre-cast measurement would cap
   at 65504. It does not cap. Therefore the residual is F32-typed all the
   way through the 60-block loop, and widening ADD/MUL on the residual
   path to "more F32" cannot change anything.
3. **The overflow is NOT in ADD/MUL/NORM on the residual.** It is
   downstream. The mission workplan scoped the wrong op classes.

## Evidence chain

### ggml core API reality
```
$ grep -rn "set_prec" ggml/include/ggml.h ggml/src/ggml.c
ggml/include/ggml.h:1414:    // set to GGML_PREC_F32 for higher precision
ggml/include/ggml.h:1415:    GGML_API void ggml_mul_mat_set_prec(
ggml/include/ggml.h:2353:    GGML_API void ggml_flash_attn_ext_set_prec(
ggml/src/ggml.c:3242:void ggml_mul_mat_set_prec(
ggml/src/ggml.c:5380:void ggml_flash_attn_ext_set_prec(
```
Only mul_mat / mul_mat_id / flash_attn_ext. None for ADD/MUL/NORM.

### ggml-cann ADD/MUL dispatch reality
`aclnn_ops.cpp:301-327` — `aclnn_add` and `aclnn_mul` call
`GGML_CANN_CALL_ACLNN_OP(ctx, Add, ...)` / `..., Mul, ...)` with acl
tensors created directly from the ggml tensors' data pointer + dtype via
`ggml_cann_type_mapping`. If `dst->type == GGML_TYPE_F32` (which it is,
because `ggml_add` returns src0->type and src0 in the residual path is
F32 from `ggml_mul_mat` output), the acl tensor is `ACL_FLOAT`. aclnnAdd
with F32 inputs produces F32 output. The dispatcher has no F16 demotion
knob analogous to `cubeMathType` here — Add/Mul are element-wise and
there is no accumulator.

### Trace magnitude receipts (from `docs/qie_leak2_per_op_trace.md`)
```
b00 img_resid2 = 9.61e+04  (> 65504 F16 limit)
b01 img_resid1 = 1.11e+05
b59 img_resid2 = 1.89e+06
```
All post-cast-to-F32 measurements. Readable only if the underlying
storage is F32 at the measurement site. F16 storage would cap at 65504.
`OMINIX_QIE_F32_RESIDUAL=1` vs unset produce byte-identical traces
because the storage is F32 either way — the explicit cast inserts in
leak-2-f32-residual are no-ops at the type level (but also not harmful).

### Mission workplan hypothesis verdict
Proposal 1 in `qie_leak2_per_op_trace.md` Next-step section —
"GGML_OP_ADD / GGML_OP_MUL / GGML_OP_NORM F32-storage widening in
ggml-cann" — is based on the assumption that residual ops in the
backend silently demote to F16. The trace itself (which the same doc
authored) contradicts that assumption. This is a known class of
recursive-hypothesis collapse.

## What the actual leak probably is

Trace shows residual magnitude at 1.89e6 leaving block 59. `img` then
feeds into `norm_out->forward(ctx, img, t_emb)` (adaptive LayerNorm
with a modulation vector from `t_emb`) and `proj_out->forward(ctx, img)`
(Linear). Candidate leak sites, ranked by likelihood:

1. **`norm_out` variance computation in F32 but reduce accumulator in
   F16.** aclnnLayerNorm's internal mean/variance reduction may use
   F16 accumulator even for F32 input on 910B. With input magnitude
   1.89e6, `x²` is 3.6e12 — far past F16 `sum` accumulator headroom.
   Testable: instrument `img` max-abs entering and leaving `norm_out`.
2. **`proj_out` matmul's `cubeMathType`.** Despite Path C #3's fix, the
   non-quant `ggml_cann_mat_mul_fp` fast path only bumps cubeMathType
   to 1 (F32 accumulator) when `ggml_cann_prec_is_f32(dst)` OR when an
   input is BF16. `proj_out` is a plain Linear with no
   `ggml_mul_mat_set_prec(F32)` annotation upstream, so its mul_mat
   still runs at `cubeMathType=2` (F16 accumulator). With F32 input at
   magnitude 1.89e6, the accumulator overflows.
3. **Sampler step `diffusion/x_0` uses mul_mat or softmax that demotes.**
   Less likely given the per-op trace already instruments the inside of
   the block, but not the post-block chain.

## Pivot plan

Instead of editing ADD/MUL/NORM dispatchers, the probe should:

### Probe step A — extend qie_trace to measure max-abs + NaN count
Current `qie_trace` in `qwen_image.hpp` records `sum(abs(x))`. Extend
to also record `max(abs(x))` and `count(isnan(x))`. This lets us
distinguish:
- growth-only (clean F32) → `max_abs` grows with `mean_abs`, NaN=0.
- F16 saturation → `max_abs` stuck at 65504.
- Post-overflow → NaN count > 0.

### Probe step B — add trace sites around `norm_out` and `proj_out`
In `forward_orig` at line ~683:
```cpp
(void)qie_trace(ctx->ggml_ctx, img, "tail/pre_norm_out");
img = norm_out->forward(ctx, img, t_emb);
(void)qie_trace(ctx->ggml_ctx, img, "tail/post_norm_out");
img = proj_out->forward(ctx, img);
(void)qie_trace(ctx->ggml_ctx, img, "tail/post_proj_out");
```

### Probe step C — run 20-step 256×256 and inspect tail
One run. Expected outcomes:
- NaN first appears at `tail/post_norm_out` → Proposal 1 above (fix
  LayerNorm kernel precision).
- NaN first appears at `tail/post_proj_out` → Proposal 2 above (set
  `GGML_PREC_F32` hint on `proj_out` mul_mat in the model file, or a
  backend default-F32 for mul_mat with residual-path inputs).
- NaN already present at `b59/img_resid2` in some cells (but not mean)
  → overflow inside the residual path is bursty; Proposal 1 variant on
  `img_norm1` / `img_norm2` per block.
- Clean through tail, NaN later → sampler step or VAE input — out of
  ggml-cann scope.

### Probe step D — act based on where NaN first appears
Only *after* step C do we know whether the fix is:
- (a) ggml-cann `ggml_cann_norm` / `ggml_cann_rms_norm` precision (the
      originally scoped work, but on an internal accumulator, not on
      storage), or
- (b) model-side `ggml_mul_mat_set_prec(F32)` annotations on `proj_out`
      and in-block norm inputs, or
- (c) backend-default promotion of mul_mat with large-magnitude F32
      input, or
- (d) somewhere else (sampler / VAE / post-graph).

Gate suite unchanged: 256×256/2, 256×256/20, 512×512/20 on ac02.

## Minimal code artefacts shipping in this commit

- Fresh worktree on ac02 at `~/work/OminiX-Ascend-c4` on
  `path-c-4-probe` branch from `origin/main` @ `722d1cf9` (clean, no
  trace instrumentation yet — leak2 worktree's trace code remains on its
  own branch).
- This doc, committed on `path-c-4-probe`.
- HBM lock `/tmp/ac02_hbm_lock` acquired for the probe window.

No `aclnn_ops.cpp` changes. No `common.h` helper extension. The
mission's Step 2 (wire F32 dispatch for ADD/MUL/NORM) is on hold
pending the probe's finding; if probe step C points at Proposal 1
(internal LayerNorm accumulator), Step 2 comes back with a scoped
target (`ggml_cann_norm` only, internal math) instead of the broad
three-class widening the mission originally called for.

## Budget impact

- Probe steps A-C: ~1 day (trace extension + build + one smoke run).
- Fix + gate: 1-2 days depending on which proposal lands.
- Total: inside the 5-7 day budget, with a correct-target fix rather
  than a speculative three-class refactor.

## References
- Path C #3 helper pattern: `3acc62aa` + `common.h:134-152` +
  `ggml-cann.cpp:133-173`.
- Leak #2 diagnosis: `docs/qie_edit_nan_diagnosis.md`.
- Per-op trace receipts: `docs/qie_leak2_per_op_trace.md`.
- F32 residual graph cast (leak2 worktree): `docs/qie_leak2_f32_residual_graph.md`.
- Native engine precedent: commit `f0b51dc1` ("Q2.4.4d — NaN fixed @ N=60
  via F32 residual stream"). Reads as: at the native engine's level, the
  fix was F32 residual stream + F32 in-block LayerNorm. The ggml-graph
  residual is ALREADY F32 (proved above), so the ggml-cann analog is
  only the in-block LayerNorm half — the scope actually narrows.
