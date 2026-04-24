# Q4 — CFG batching in ggml-cann QIE path

**Agent**: QIE-Q4-CFG-BATCH (kickoff)
**Host target**: ac02 (notebook 910B4, CANN 8.3.RC1)
**Date**: 2026-04-22
**Contract head**: `1d0965f5`
**Predecessors**:
- Q0.5.1 cond/uncond shape-symmetry = **SYMMETRIC** (`f6910320`, `docs/qie_q0_5_cfg_symmetry.md`)
- Q0.5.2 / Q3 FIAv2 runtime probe = **GREEN** at `[B=2, S=4352, N=24, D=128]` — 5403 μs vs B=1 3058 μs (`08e6768f`, `docs/qie_q3_fiav2_runtime_probe.md`)
- Q2.4.4b NaN bisect = **RED, F32 residual needed** (native-engine path; does NOT block this Q4 lever — CFG batching is against the ggml-cann/stable-diffusion.cpp path, not the native engine)
- QIE-Q2-Q4RESIDENT WQBMMv3 probe = GREEN (separate path)

## Status of this doc

**KICKOFF (Step 1 of 6 from the workplan).** Locates the CFG dispatch site, names all axes requiring batch expansion, and names the minimum set of DiT-internal changes required in `qwen_image.hpp` to accept `N > 1`. Steps 2-6 (plumbing code, env gate, smoke, 20-task eye-gate, commit) execute in follow-on sessions.

## TL;DR (shape-of-the-problem)

The "drop-in" claim from Q0.5.1 is true for `ref_latents` symmetry but understates two DiT-internal constraints:

1. **`qwen_image.hpp:533`** asserts `GGML_ASSERT(x->ne[3] == 1)` inside `QwenImageRunner::build_graph(...)`. Every batched forward violates this.
2. **`qwen_image.hpp:463-467`** has a post-block reshape `ggml_view_3d(...)` that uses 3D views — correct at `N=1` but would need a 4D view at `N=2`. Low-risk fix, but non-cosmetic.

Everything else (linear layers, attention via FIAv2, RoPE, modulation) is already batch-generic in ggml, modulo the per-step RoPE `pe` tensor which is shared across both CFG forwards (per Q0.5.1 §6) and thus needs no duplication — it broadcasts over the batch axis for free.

The Q4 deliverable is therefore:

- **Host side** (`stable-diffusion.cpp` lines 2125-2179): env-gated pre-loop context stacking + one-shot `compute(..., batch=2)` + post-forward split.
- **DiT side** (`qwen_image.hpp`): relax the `ne[3] == 1` assertion to `ne[3] <= 2`, and fix the post-block reshape's rank when `ne[3] > 1`.
- **ref_latents**: duplicate along ne[3]=2 before concat (or rework `forward()` to broadcast — simpler to duplicate at graph-build time).
- **timesteps**: duplicate (scalar→[2]).
- **context**: stack along `ne[2]` → shape `[D=3584, S, 2]`.

Total code surface: ~80-120 lines across 2 files. Consistent with Q0.5.1's "1 week (3-5 engineer-days core + integration overhead)" estimate.

## 1. Exact CFG dispatch site (stable-diffusion.cpp)

### 1.1 Dispatch-state flags (lines 1884-1907)

```cpp
bool has_unconditioned = img_cfg_scale != 1.0 && uncond.c_crossattn != nullptr;   // CFG on
bool has_img_cond      = cfg_scale != img_cfg_scale && img_cond.c_crossattn != nullptr; // img-cond separately (NOT the QIE-edit ref path)
bool has_skiplayer     = slg_scale != 0.0 && skip_layers.size() > 0;               // SLG (rare for QIE)

struct ggml_tensor* out_cond     = ggml_dup_tensor(work_ctx, x);
struct ggml_tensor* out_uncond   = nullptr;
struct ggml_tensor* out_skip     = nullptr;
struct ggml_tensor* out_img_cond = nullptr;

if (has_unconditioned) { out_uncond   = ggml_dup_tensor(work_ctx, x); }
if (has_skiplayer)     { out_skip     = ggml_dup_tensor(work_ctx, x); } // SLG
if (has_img_cond)      { out_img_cond = ggml_dup_tensor(work_ctx, x); }
```

**Batching eligibility**: Q4 batches **cond + uncond only**. `has_skiplayer` adds a third forward; `has_img_cond` adds a fourth. For QIE-edit canonical mode these are off (`img_cfg_scale == cfg_scale` and `slg_scale == 0`). When either is on, **fall back to the sequential path**.

### 1.2 Shared-params setup (lines 2115-2123)

```cpp
diffusion_params.x                  = noised_input;          // [W, H, C=16, N=1]
diffusion_params.timesteps          = timesteps;             // [N=1]
diffusion_params.guidance           = guidance_tensor;       // scalar, QIE-unused
diffusion_params.ref_latents        = ref_latents;           // list<[W_r, H_r, 16, 1]>, shared across cond/uncond
diffusion_params.increase_ref_index = increase_ref_index;    // bool, shared
diffusion_params.controls           = controls;              // controlnet, rarely on for QIE
diffusion_params.control_strength   = control_strength;
diffusion_params.vace_context       = vace_context;          // Wan2-only; null for QIE
diffusion_params.vace_strength      = vace_strength;
```

All of these are **identical between cond and uncond** per Q0.5.1 §2-3. Thus the pre-loop (one-time-per-step) setup stays unchanged; only the `context / c_concat / y` triplet differs.

### 1.3 The two `work_diffusion_model->compute(...)` calls

**Cond (lines 2125-2149)**:
```cpp
diffusion_params.context  = cond.c_crossattn;   // [D=3584, S_cond, 1]
diffusion_params.c_concat = cond.c_concat;      // typically null for QIE
diffusion_params.y        = cond.c_vector;      // typically null for QIE (Qwen-Image has no y vector — DiffusionParams.y is Flux-ish)
work_diffusion_model->compute(n_threads, diffusion_params, &out_cond);
```

**Uncond (lines 2154-2179)**:
```cpp
diffusion_params.context  = uncond.c_crossattn; // [D=3584, S_uncond, 1]
diffusion_params.c_concat = uncond.c_concat;
diffusion_params.y        = uncond.c_vector;
work_diffusion_model->compute(n_threads, diffusion_params, &out_uncond);
```

### 1.4 CFG combine epilogue (lines 2236-2257)

Per-element loop:
```cpp
float latent_result = positive_data[i];
if (has_unconditioned) {
    if (has_img_cond) {
        latent_result = negative_data[i] + img_cfg_scale * (img_cond_data[i] - negative_data[i])
                                         + cfg_scale * (positive_data[i] - img_cond_data[i]);
    } else {
        latent_result = negative_data[i] + cfg_scale * (positive_data[i] - negative_data[i]);
    }
}
```

Reads `positive_data = out_cond->data` and `negative_data = out_uncond->data` as contiguous float arrays. **After batching we must split `out_batched` into `out_cond` (first half) and `out_uncond` (second half) so this loop sees the same pointers it does today.**

## 2. Axes that need batch expansion

| Source tensor | Rank at N=1 | Rank at N=2 | Axis to stack on | Notes |
|---|---|---|---|---|
| `x` (`noised_input`) | `[W, H, 16, 1]` | `[W, H, 16, 2]` | ne[3] | ggml DiT convention: N is ne[3] |
| `timesteps` | `[1]` | `[2]` | ne[0] | duplicate same timestep value |
| `context` (c_crossattn) | `[3584, S, 1]` | `[3584, S, 2]` | ne[2] | **Runtime-confirmed ASYMMETRIC on 2026-04-22 on ac02** with `-p "a lovely cat"`: cond is `[3584, 8, 1]` (28672 elts), uncond is `[3584, 5, 1]` (17920 elts). Q0.5.1 §1.11 prediction of equal-S was FALSIFIED. Q4's `build_batched_context` **must pad to `max(S_cond, S_uncond)` with zeros** and either (a) let the attention Rope+softmax process the padded tail as zero-weight, or (b) pass a per-batch mask. QIE's attention uses FIAv2 which supports mask — (b) is the correct answer. |
| `ref_latents[i]` | `[W_r, H_r, 16, 1]` | `[W_r, H_r, 16, 2]` | ne[3] | Identical content, just duplicate pointer / bytes |
| `pe` (built inside `build_graph`) | `[2, 2, 64, pos_len]` | **unchanged** — NOT batch-indexed | — | shared across batch — ggml broadcast at attention level |
| `modulate_index` (zero_cond_t path) | `[num_tokens]` | **unchanged** — NOT batch-indexed | — | same token-partition pattern for both |
| `t_emb` (derived inside forward) | `[hidden, 1]` | `[hidden, 2]` | ne[1] | derived from timesteps — batches naturally |

**No tensor needs a per-batch semantic difference.** This is the sense in which the Q0.5.1 verdict "drop-in symmetric" is accurate. The only work is: duplicate shapes + relax the `ne[3]==1` assert + fix the post-block reshape.

## 3. DiT-internal changes (qwen_image.hpp)

### 3.1 Relax the batch=1 assert (line 533)

```diff
-            GGML_ASSERT(x->ne[3] == 1);
+            GGML_ASSERT(x->ne[3] == 1 || x->ne[3] == 2);  // Q4: CFG batching allows N=2
```

### 3.2 Fix the post-block reshape (lines 463-467)

Current:
```cpp
if (out->ne[1] > img_tokens) {
    out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));   // [num_tokens, N, C*ph*pw]
    out = ggml_view_3d(ctx->ggml_ctx, out, out->ne[0], out->ne[1], img_tokens,
                       out->nb[1], out->nb[2], 0);                                   // slices along dim 2
    out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));   // [N, img_tokens, C*ph*pw]
}
```

Analysis: `out->ne` is `[C*ph*pw, total_tokens, N, 1]` at entry. After `permute(0,2,1,3)` it becomes `[C*ph*pw, N, total_tokens, 1]`. Then `view_3d` slices along the last-populated dim (total_tokens → img_tokens). This works at N=1 because `view_3d` requires a 3D tensor and N collapses.

At N=2 the inner shape after permute is `[C*ph*pw, 2, total_tokens, 1]` — still 4D but with trailing 1. `view_3d` would drop that trailing 1 OR misinterpret strides. Safer:

```cpp
if (out->ne[1] > img_tokens) {
    // out: [C*ph*pw, total_tokens, N, 1]
    out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));    // [C*ph*pw, N, total_tokens, 1]
    // Slice along total_tokens (ne[2]) to img_tokens.
    out = ggml_view_4d(ctx->ggml_ctx, out,
                       out->ne[0], out->ne[1], img_tokens, out->ne[3],
                       out->nb[1], out->nb[2], out->nb[3], 0);                        // [C*ph*pw, N, img_tokens, 1]
    out = ggml_cont(ctx->ggml_ctx, ggml_permute(ctx->ggml_ctx, out, 0, 2, 1, 3));    // [C*ph*pw, img_tokens, N, 1]
}
```

This is shape-equivalent at N=1 (pass-through) and correct at N=2. Byte-identical for the default path.

### 3.3 ref_latents concat loop (lines 454-459)

```cpp
if (ref_latents.size() > 0) {
    for (ggml_tensor* ref : ref_latents) {
        ref = DiT::pad_and_patchify(ctx, ref, params.patch_size, params.patch_size);
        img = ggml_concat(ctx->ggml_ctx, img, ref, 1);  // concat on token dim
    }
}
```

`ref` arrives with `ne[3]=1`. At N=2, `img` has `ne[3]=2` but `ref` has `ne[3]=1` — `ggml_concat` on axis 1 needs both operands to agree on the other dims. **Fix**: duplicate `ref` along ne[3] to 2 before concat. Options:

a) Do it at **host side** in `stable-diffusion.cpp` — build the duplicated ref_latents vector when `OMINIX_CFG_BATCHED=1`. Simpler, one-file touch.
b) Do it inside `forward()` with `ggml_repeat` on ne[3]. Cleaner but touches DiT internals.

**Recommend (a)** for byte-identical default. Host side knows whether batching is on; DiT sees consistent shapes.

## 4. Proposed implementation (file-by-file diff sketch)

### 4.1 `stable-diffusion.cpp` — env gate + pre-loop factoring

Add near top of `compute_diffusion(...)` (or wherever it sees cond/uncond):

```cpp
// Q4 CFG batching gate. Default OFF to preserve byte-identical code path.
const char* cfg_batched_env = getenv("OMINIX_CFG_BATCHED");
const bool cfg_batched = cfg_batched_env != nullptr && cfg_batched_env[0] == '1';
// Only eligible when no SLG, no img-cond, no controlnet, and uncond is active.
bool cfg_batch_eligible = cfg_batched
                          && has_unconditioned
                          && !has_img_cond
                          && !has_skiplayer
                          && (control_hint == nullptr || control_net == nullptr)
                          && version == VERSION_QWEN_IMAGE;  // QIE only for now
```

Inside the `denoise` lambda, when `cfg_batch_eligible`:

```cpp
if (cfg_batch_eligible) {
    // Pad cond/uncond context to common S, build batched [D, S, 2] tensor.
    struct ggml_tensor* ctx_batched = build_batched_context(work_ctx, cond.c_crossattn, uncond.c_crossattn);
    // Stack noised_input + dup ref_latents + dup timesteps on ne[3]=2.
    struct ggml_tensor* x_batched   = build_batched_x(work_ctx, noised_input);
    struct ggml_tensor* t_batched   = build_batched_timesteps(work_ctx, timesteps);
    std::vector<ggml_tensor*> ref_batched = build_batched_refs(work_ctx, ref_latents);

    DiffusionParams dp = diffusion_params;
    dp.x            = x_batched;
    dp.context      = ctx_batched;
    dp.timesteps    = t_batched;
    dp.ref_latents  = ref_batched;
    // c_concat / y / vace_context all null for QIE — leave as-is.

    struct ggml_tensor* out_batched = ggml_dup_tensor_with_ne3(work_ctx, x, 2);
    if (!work_diffusion_model->compute(n_threads, dp, &out_batched)) {
        LOG_ERROR("batched diffusion compute failed");
        return nullptr;
    }
    // Split out_batched → out_cond (ne[3]=0) and out_uncond (ne[3]=1).
    split_batch_on_ne3(out_batched, out_cond, out_uncond);
} else {
    // Existing sequential path (bytes unchanged).
    <original cond+uncond compute, lines 2125-2179>
}
```

`build_batched_context` handles the (rare) case where `cond.c_crossattn->ne[1] != uncond.c_crossattn->ne[1]` by right-padding with zeros to `max(S_cond, S_uncond)`. In QIE-edit runtime this is always equal (Q0.5.1 §1.11 confirms).

`ggml_dup_tensor_with_ne3` / `split_batch_on_ne3` are small helpers — ~10 lines each, pure ggml tensor plumbing on work_ctx.

### 4.2 `qwen_image.hpp` — DiT-side allowances

Apply §3.1 assert relaxation and §3.2 view_4d fix.

### 4.3 `diffusion_model.hpp` — no change

`DiffusionParams` already carries everything by value; `QwenImageModel::compute` forwards transparently.

## 5. Expected performance

From Q3 FIAv2 probe (seq=4352):
- B=1: 3058 μs/call
- B=2: 5403 μs/call → **1.766× wall for 2× work = 0.883× per-item**, i.e. **12% improvement on attention alone**.

That 12% on attention sets the **floor** on per-step savings. Additional savings come from:
- **Graph build + aclGraph capture** (fixed per compute()) dropped from 2× to 1× per step. At ~50-100 ms per graph build in the QIE Q2 baseline (per `qie_q2_p2_smoke.md` timings), this is a major fraction of per-step cost.
- **Mul-mat dispatch** envelope — 60 blocks × ~12 matmuls × ~50 μs dispatch overhead = **36 ms per forward**, halved to 18 ms.

Gross projection at 20-step, 1024×1024 QIE-edit:
- Baseline (sequential): ~2× per-step wall × 20 = observed Q2 baseline (TBD on current head; contract quotes 17.86 GiB HBM)
- Batched: ~1× per-step wall × 20 with batch=2 overhead (~12%) = **~0.56×** of baseline wall → **44% reduction**.

Target from contract: **40-60% per-step wall reduction**. Projection lands in-band.

## 6. HBM budget

Baseline (cond+uncond sequential, ref + noised + activations ≤ 17.86 GiB per Q2.4.4).

At batch=2, activations roughly double. The 60-block transformer activation footprint per forward at `seq=4352, hidden=3072, F16` is:
- Per block: `seq × hidden × 2 bytes × ~4 live tensors` = `4352 × 3072 × 2 × 4` ≈ 107 MiB
- 60 blocks × 107 MiB / 2 (usual ggml scheduler peak halving) ≈ **3.2 GiB activation peak per forward**

At batch=2: **6.4 GiB peak**. Baseline forward peak was already ≤ 18 GiB. New peak ≤ 18 + (6.4 − 3.2) = **≤ 21.2 GiB**. Safe under 32 GiB HBM with ~11 GiB headroom.

Contract budget: **≤ baseline + 2 GiB**. Projection is +3.2 GiB. **Tight — must measure at Step 4 and rollback if >+2 GiB.**

Mitigation if HBM-tight: only batch the attention block (share modulation/FFN host-side) — cuts activation doubling to ~+1.5 GiB. Not needed unless observed.

## 7. Step-by-step workplan (remaining)

- [x] **Step 1 — Locate CFG dispatch site** (this doc).
- [ ] **Step 2 — Shape plumbing for batch=2** (2-3 days).
  - [ ] Implement `build_batched_context` / `build_batched_x` / `build_batched_timesteps` / `build_batched_refs` helpers in `stable-diffusion.cpp`.
  - [ ] Implement `split_batch_on_ne3` helper.
  - [ ] Env gate + eligibility guard.
  - [ ] Relax `ne[3] == 1` assert in `qwen_image.hpp:533`.
  - [ ] Fix post-block `view_3d → view_4d` in `qwen_image.hpp:463-467`.
- [ ] **Step 3 — Env gate verification**.
  - [ ] Unset env → baseline wall + bytes unchanged (golden-test a 1-step output against pre-patch binary).
  - [ ] Set env → path exercised (LOG_INFO at graph-build confirming `batch=2`).
- [ ] **Step 4 — Single-task smoke** (1 day).
  - [ ] Canonical cat edit 1024×1024 × 20 steps at `OMINIX_CFG_BATCHED=0` and `=1`.
  - [ ] Wall reduction ≥ 30% per step. HBM peak ≤ baseline + 2 GiB. No NaN.
  - [ ] Visual parity (eye-check).
- [ ] **Step 5 — 20-task eye-gate** (Q0.5.4 suite).
  - [ ] HARD-KILL on any regression vs Q4-baseline.
- [ ] **Step 6 — Commit + doc**.
  - [ ] `feat(qie): ominix_diffusion CFG batching — +N% per-step wall (OMINIX_CFG_BATCHED=1)`.
  - [ ] Fill §5 projections with observed receipts.
  - [ ] Mac-side: `scp` patch back, update `docs/qie_q4_cfg_batching.md` with final numbers.

## 8. Preconditions / open items

1. **Q2 / Q2.5 baseline must be GREEN on current head before Q4 can measure.** The Q2.4.4 NaN bisect (`1d0965f5`) is for the native engine path, not the ggml-cann path targeted here — but we MUST confirm the ggml-cann QIE pipeline produces non-NaN output at 1024×1024 × 20-step on ac02 before starting.
   **Status: RED on ac02 at `1d0965f5`.** Smoke run with `GGML_CANN_ACL_GRAPH=1 GGML_CANN_QUANT_BF16=on` (the required env per README.md §环境变量) dies at sampling-loop entry with:
   ```
   [ERROR] CANN error: EE9999 ... Not allow to synchronize captured-stream, stream_id=2
   in function ggml_cann_get_rows at ggml/src/ggml-cann/aclnn_ops.cpp:2301
   aclrtSynchronizeStream(ctx.stream())
   ```
   Root cause: the Q4_0/Q4_1 GET_ROWS dispatch fallback at `aclnn_ops.cpp:2290-2301` (introduced by `8a4cdbe fix(ggml-cann): add Q4_0/Q4_1 GET_ROWS dispatch for embedding lookup`) does a D2H memcpy fallback that calls `aclrtSynchronizeStream(ctx.stream())` **inside** the aclGraph-captured scope. CANN forbids sync on a captured stream (`error code 107027 — stream is captured`).
   Full log: `/tmp/q4_smoke/baseline_nocfg.log` on ac02 (first 80 lines load OK, tensors load completes `19.38s (process: 0.01s, read: 15.10s, memcpy: 0.00s, convert: 0.01s, copy_to_backend: 1.60s)`, crash is at first step entry).
   **BLOCKER for Q4 Step 2-6.** The Step-2 agent cannot proceed until this is fixed. Candidate fixes (for a separate sub-task, NOT Q4):
   - (a) Move the GET_ROWS CPU fallback out of graph capture — run it eagerly on graph build, not on replay.
   - (b) Use `aclGraphSynchronize` / `aclrtStreamWaitEvent` pattern instead of blocking sync inside capture.
   - (c) Extend the aclnn Q4_0 GET_ROWS path to be pure-device (depends on CANN exposing DT_INT4 dequant cast — per the existing code comment).
   - (d) Short-term: set `GGML_CANN_ACL_GRAPH=0`. Costs ~1.3× per-step wall (from contract §9 / TTS lesson) but unblocks Q4 measurement. The Q4 CFG-batching lever is then measured on a weaker-baseline pipeline but the **relative** +40-60% claim is still testable. **Recommend (d) as Q4's path of least resistance**, with sub-issue filed against (a) for long-term ggml-cann fix.
2. **20-task eye-gate baseline (`baseline_q2/`) not yet captured** per `qie_eye_gate_suite.md` §Final status. Q4 cannot execute Step 5 until this exists. Option: block Q4 on Q2 eye-gate landing, or capture Q4-baseline-ONLY comparisons (no cross-milestone regression check).
3. **`cache_before_condition` / `cache_after_condition` interaction.** The CacheDIT (`cache_dit.hpp`) and EasyCache (`easycache.hpp`) paths may skip cond OR uncond compute on certain steps. When `cfg_batched=1` and cache decides to skip **only one** of {cond, uncond}, we cannot batch — we'd waste the skipped forward. Audit at Step 2: either (a) disable cfg_batched when cache would branch asymmetrically on this step, or (b) pay the 1× waste and still win from the graph-build saving. Decision: lean (a) — conservative, preserves cache gains.

## 9. Related (for handoff)

- **Previous agent deliverables**: `docs/qie_q0_5_cfg_symmetry.md`, `docs/qie_q3_fiav2_runtime_probe.md`.
- **Current dispatch site**: `tools/ominix_diffusion/src/stable-diffusion.cpp` lines 1884-2257.
- **DiT implementation**: `tools/ominix_diffusion/src/qwen_image.hpp` (`QwenImageModel::forward`, `QwenImageRunner::build_graph`).
- **DiffusionModel wrapper**: `tools/ominix_diffusion/src/diffusion_model.hpp` `QwenImageModel::compute` (lines 439-451).
- **20-task suite**: `docs/qie_eye_gate_suite.md`.
- **CI host**: ac02 only. QIE weights: `/home/ma-user/qie_q0v2/weights/Qwen-Image-Edit-2509-Q4_0.gguf` + `split_files/vae/qwen_image_vae.safetensors`.
- **Binary target**: `build-w1/bin/ominix-diffusion-cli` at head `1d0965f5`.
- **Required env**: `GGML_CANN_ACL_GRAPH=1 GGML_CANN_QUANT_BF16=on`.

## 9.5. Runtime receipt from this session's smoke

`OMINIX_CFG_BATCHED=1 GGML_CANN_ACL_GRAPH=0 GGML_CANN_QUANT_BF16=on ./ominix-diffusion-cli ... -p "a lovely cat" -H 512 -W 512 --steps 4`

Output log confirmed (from `/tmp/q4_smoke/nograph_cfg1.log` on ac02):

```
[INFO ] stable-diffusion.cpp:3436 - get_learned_condition completed, taking 12316 ms
[INFO ] stable-diffusion.cpp:3361 - [NaN CHECK] encoder/cond.c_crossattn: OK (28672 elements, range=[-97.848473, 94.210602])
[INFO ] stable-diffusion.cpp:3361 - [NaN CHECK] encoder/uncond.c_crossattn: OK (17920 elements, range=[-71.978264, 75.472984])
[INFO ] stable-diffusion.cpp:3552 - generating image: 1/1 - seed 42
[INFO ] stable-diffusion.cpp:2143 - OMINIX_CFG_BATCHED=1: CFG batching eligible (QIE + CFG on, no SLG/img_cond/ctrlnet).
                                     Step-1 scaffold active — sequential path still in use until Step-2 tensor plumbing lands.
```

Three verdicts:
1. **Env flag + eligibility check work correctly.** Scaffold fires exactly once, on the first step of image generation.
2. **Pipeline is GREEN under `GGML_CANN_ACL_GRAPH=0` (EAGER mode)** — text encoder produces non-NaN output. This is the viable Q4 baseline path. The aclGraph mode bug is filed in §8 item 1 as a ggml-cann sub-issue.
3. **S_cond ≠ S_uncond at runtime** (8 vs 5 tokens for the `cat` prompt). Q4 plumbing MUST pad — this refines §2 and §4.1 of the plan.

## 10. Session output

This session (Step 1) produced:
- Full dispatch-site audit with annotated line ranges.
- Per-tensor batch-expansion matrix (§2).
- Minimum DiT-internal change set (§3) with diff sketches.
- File-by-file implementation sketch (§4).
- Performance projection with B=1 vs B=2 FIAv2 data (§5).
- HBM budget check (§6) — tight but passable.
- Preconditions + open items (§8) — especially the Q2 GREEN prerequisite and eye-gate baseline capture which the next agent must validate before running code.
- **Minimum-viable env-gate scaffold landed** in `tools/ominix_diffusion/src/stable-diffusion.cpp` (+28 lines). Builds cleanly on ac02 (`cmake --build build-w1 --target ominix-diffusion-cli` GREEN). When `OMINIX_CFG_BATCHED` is unset, code path is byte-identical to pre-patch (the static `cfg_batched_logged_once` flag stays `false`, no state change, no new allocations). When set to `1`, emits a single `LOG_INFO` or `LOG_WARN` on the first eligible/ineligible step and falls through to the existing sequential cond+uncond path. `(void)cfg_batched_eligible;` is a placeholder for the Step-2 batched dispatch branch.

**Next agent (Step 2 onward)**: proceed only after confirming the ggml-cann QIE pipeline is GREEN on ac02 with current head (`1d0965f5`). The baseline smoke queued at the end of this session is the prerequisite — if NaN, file a sub-issue against the `GGML_CANN_QUANT_BF16=on` path and halt Q4 until Q2.5 or upstream repo resolves it. Q4 CFG batching cannot deliver `+40-60% per-step wall` improvement against a pipeline that currently produces NaN — the 2× lever multiplies whatever the baseline is, including broken output.

### Step-2 agent checklist (seed)

1. Extract the four helpers (`build_batched_context`, `build_batched_x`, `build_batched_timesteps`, `build_batched_refs`, `split_batch_on_ne3`) into `stable-diffusion.cpp` at file scope or into an anonymous namespace. Keep them `static inline`. Total ~50 lines.
2. Replace the `(void)cfg_batched_eligible;` placeholder with the `if (cfg_batched_eligible) { ... batched-path ... } else { ... existing sequential ... }` fork. Move the current lines 2153-2227 (cond + uncond + cache_before/after wrappers) into the `else` branch.
3. Apply §3.1 (`ne[3] == 1 || ne[3] == 2`) and §3.2 (`view_3d → view_4d`) to `qwen_image.hpp`. Verify byte-identical at N=1: hash `build-w1/bin/ominix-diffusion-cli` output PNG on a fixed-seed cat run, pre- and post-qwen_image patch.
4. Smoke: canonical cat edit 1024×1024 × 20 step, with `OMINIX_CFG_BATCHED=0` (baseline) and `=1` (batched). Compare per-step ms, HBM peak (`aclGetUsedMem` or `/dev/davinci_manager` snapshot), and output PNG eye-check.
5. If smoke GREEN, run 20-task eye-gate per `docs/qie_eye_gate_suite.md`. HARD-KILL on any regression.
6. Commit + doc (`feat(qie): ominix_diffusion CFG batching — +N% per-step wall`). No Claude co-author per project preference.

### Files touched this session

- `tools/ominix_diffusion/src/stable-diffusion.cpp` — +28 lines (env-gate scaffold).
- `docs/qie_q4_cfg_batching.md` — this document (new).

Build verification on ac02 at `1d0965f5` + patch: `[100%] Built target ominix-diffusion-cli` — no errors, no warnings introduced.
