# Qwen-TTS on Ascend: Learnings + Further Optimization Opportunities

**Date**: 2026-04-21. **Source**: retrospective over contracts
`CP_FPS_OPTIMIZATION`, `ACLGRAPH`, `PATH_C_ASCENDC`, `CANNFUSION`,
`FUSED_OP_LANDING`, `40FPS`, `ASCEND_API_BRIDGE`, and probe reports
(`aclgraph_feasibility`, `cp_group_dependency_audit`, `fused_op_audit`,
`a16w8_specific_op_audit`, `v2_rope_numerics_debug`,
`talker_aclgraph_feasibility`, `ascend_910b4_datasheet`).

## Executive summary

- **Where we are**: clean-quality canonical xvec mayun zh LONG delivers
  **32.2 fps** on Ascend 910B4 (tag `32fps-landed`, commit `a3c9ebf1`).
  Baseline was ~1 fps on llama.cpp. The 32x lift was delivered by a
  small number of large wins (W1 +8.3 fps, aclGraph G2 +1.15 fps) and a
  lot of sub-fps increments. Five parallel tracks were funded; two landed,
  three (Path C AscendC, CannFusion, V2 RoPE / FFNV3) RED-ed and closed.
- **What's next**: realistic single-card ceiling without W4 quant is
  ~33-34 fps. W4 quant is the only lever that plausibly pushes ≥ 36 fps
  at single-stream scale. **40 fps + is a cluster-TP problem**, not a
  single-card problem.
- **Meta-pattern**: the agent swarm delivered real wins when gates were
  numeric and the critical path was one-agent-per-milestone. It over-spent
  on optimism when projections were taken at face value instead of
  discounted. Every track that RED-ed had a 30-minute probe that would
  have caught it, had the probe been mandated as Gate 0.

## Part 1 — What worked (quantified, with receipts)

### W1 — NPU lm_head port: +8.3 fps

Commits `0c47e957` + `038ec1dd`. Moved 15 per-frame lm_head GEMV
dispatches from CPU NEON+OMP (14–17 ms/frame) to NPU via `aclnnMm` on
F16 weights uploaded at init. Correctness 0/16 token drift on three
canonical utterances. W1 OFF 21.9 fps → W1 ON 30.2 fps.

**Why it worked**: cross-layer port — obvious CPU bottleneck, but the
refactor was in the engine's hidden-state hand-off, not a kernel tune.
PM-gate critical catch: keeping hidden-state on-device (Q2 Option b) —
without it, D2H-per-group would have nuked most of the win.

### W3b — AddRmsNorm fusion: +0.66 fps

Commit `36d0fb7c`. `aclnnAddRmsNorm` collapsed 10 per-sublayer `Add +
RmsNorm + residual memcpy` sites × 15 forwards = 255 dispatches/frame
saved. Delta +0.66 fps (29.9 → 30.5), vs contract target +3.5.

**Why small but correct**: `TASK_QUEUE_ENABLE=2`'s two-phase submit
already amortised host-launch overhead from ~40 μs eager → ~2–3 μs.
**The dispatch-floor thesis was already 10× paid for by prior work.**
PM-gate critical catch: honest-framed below-target and moved on — no
chasing.

### G2 — aclGraph pos-keyed cache: +1.15 fps SHORT, +1.6 fps LONG

Commit `63a3d90e`. 17 pos-keyed `aclmdlRI*` graphs captured at engine
init, replayed by `aclmdlRIExecuteAsync`. **Byte-identical** CP tokens
and WAV md5 across xvec / ICL / CV. G0 projected +6–10 fps; delivered
+1.15 SHORT / +1.6 LONG — same TQE=2 amortisation root-cause as W3b.

The fallback-to-pos-keyed-cache PM move (after G1 YELLOW on multi-op
TaskUpdate 3e-3 drift) was right — Option 2 single-graph + TaskUpdate
would have needed a stream-barrier debug spike we avoided entirely.

### A.1 InplaceAddRmsNorm: +0.3 fps (within noise)

Commit `d927758f`. Drop-in replacement for W3b's `AddRmsNorm → residual-
copy`, saved 75 `aclnnAdd` zero-copies/frame. Same pattern as W3b —
residual-copy was D2D bandwidth-saturated, not dispatch-bound.

### M3'new' pos 0+1 batch: +0.3 fps (within noise)

Commit `a3c9ebf1`. Host-side chaining of two pos-keyed forwards with
async H2D + aggregated event record. Projected +1 fps, delivered +0.3
— most of the estimate was host-side only; device-side serialisation
on two launches was smaller than GD-audit assumed. True device-side
S=2 batched prefill would recover the rest but needs capture
re-inventory.

### Cumulative: ~1 → 32.2 fps ≈ 32×

| Lever | Commit | fps delta |
|---|---|---|
| llama.cpp baseline | — | ~1 |
| Native CANN port (pre-contracts) | — | ~15 |
| TQE=2 + NZ + W8 stack | — | ~22 |
| W1 NPU lm_head | `0c47e957` | +8.3 → 30.2 |
| W3b AddRmsNorm fusion | `36d0fb7c` | +0.66 → 30.5 |
| G2 aclGraph pos-keyed | `63a3d90e` | +1.15 (short) / +1.6 (long) → 31.6 |
| A.1 InplaceAddRmsNorm | `d927758f` | +0.3 → ~31.9 |
| M3'new' pos 0+1 batch | `a3c9ebf1` | +0.3 → 32.2 |

## Part 2 — What didn't work (and the lessons)

### Path C W4.1 — catastrophic drift

Hand-written AscendC fused attn kernel. Wired (`be082ed`), passed
offline numerical validation (max_abs_diff ≤ 1e-3 vs host-F32 gold on
8-token synthetic fixture), **FAILED live drift catastrophically**:
max_drift=1949, 1/256 positions matched, 2.1× slower than stock.

**Lesson — offline diff ≠ live gate**. A kernel coherent on 8 synthetic
tokens can diverge from frame 0 under full autoregressive cascade.
**Live token-drift over ≥ 10 frames is mandatory**; promote to
universal pattern.

### Path C PC-tile re-spike — gemv beats aclnn, reduce kills it

40-AIV-core tiled F16 matmul at M=1 decode shapes beat `aclnnMm` by
26% wall (324 vs 222 GB/s HBM). But cross-core F32→F16 reduce costs
18–22 μs, wiping the gain. Atomic-add race-prone at blockDim ≥ 20,
NaN-prone at blockDim=40 (hardware limit).

**Lesson**: multi-core tiling alone isn't enough; reduce must fuse
into a downstream op — which **is** the full attn-sublayer fusion
W4.1 failed numerically.

### CannFusion F1 — dtype RED

F16×INT8→F16 with per-channel F16 scale is **not in CannFusion's
validator whitelist** (`src/validate.rs:141-163`). Explicit negative
unit test `dtype_rejects_f16_int8_f32` at `:367-372` confirms the
rejection is **intentional design**.

**Lesson — validator whitelists are intentional design, not
omissions**. The 25-minute probe would have saved two weeks of
downstream planning. Probe-first.

### V2 RoPE — GQA packed-UB incompat

`aclnnApplyRotaryPosEmbV2` drop-in for two sequential v1 RoPEs. Parity
FAILED: 434 frames v1 vs 457 V2 (+5%), hidden-state drift from frame
0. Root cause: V2's `CopyInQK` packs Q+K into a single UB with shared
stride `dstRepSBr` — correct for MHA, broken for GQA (n_q=16 ≠ n_kv=8).

**Lesson — vendor docs say "HALF/NEOX" but hide MHA-only assumptions
in packed-UB internals**. Header contract doesn't document GQA
non-support; breakage surfaces only on GQA models. Frame-count-identity
gate + deterministic per-head harness is the only reliable validator.

### M1.B FFNV3 — A16W8 rejected at runtime

Scaffolding (`9aada3e6`) gated RED: FFNV3's no-expert branch rejects
`weight1=INT8` with `EZ9999 161002: weight1 only support dtype float16
without expert tokens`. Vendor allows INT8 weight only when
`expertTokens != null` (MoE). Qwen3-TTS is no-MoE — F16-only.

**Lesson — docstring vs runtime mismatch is a vendor pattern, not an
exception**. Third fused-op in a row (CannFusion validator, V2 RoPE
GQA, FFNV3 no-MoE) advertised A16W8 support and rejected at runtime.
**Generic Ascend fused-op APIs are A16W8-patchy**; reliable A16W8
remains exclusively via `aclnnWeightQuantBatchMatmulV3`.

### TALKER_CANN_GRAPH scaffold — net loss

`talker_cann_engine.cpp:1106-1136` scaffold wired in M4 era, gated by
`TALKER_CANN_GRAPH=1`. **Default OFF because measured runs drop
throughput ~2.5×** — per-pos CaptureBegin/End overhead swamps savings
since each `pos` is touched exactly once per utterance. Current
implementation does **lazy capture on first-touch** — the exact
anti-pattern.

**Lesson — pre-existing env gates can silently regress; always probe
stock scaffolds before trusting them**. A 10-minute A/B would have
caught this before ossifying as "default off".

### M3 group-collapse — RVQ-chained, NO-COLLAPSE

GD-audit confirmed Qwen3-TTS is strict 15-step RVQ depth transformer
across four independent sources. Each group consumes the embedding of
the previous group's sampled integer token; sampling is hard argmax.
**Group-collapse is structurally impossible.**

**Lesson — architectural levers need dependency audit before fund**.
40FPS contract had M3 as the load-bearing +5–15 fps lever; without
GD-audit, weeks of work would have produced wrong audio.

## Part 3 — Further optimization opportunities on Qwen-TTS (ranked by ROI)

Applying the measured 3–10× optimism discount (we project at the
optimistic end and deliver at the pessimistic end).

### 1. W4 quant (A16W4) — **+2–5 fps projected, +0.5–1.5 fps realistic**

Per-group INT4 via `aclnnWeightQuantBatchMatmulV3`. The header
reportedly supports W4 (`antiquantGroupSize` parameter documented), so
verify via a 30-minute probe before scoping. Calibration (AWQ or
similar) + wiring is ~1 week.

**Why this is #1**: it's the only remaining single-stream lever that
actually attacks HBM bandwidth (weight bandwidth halves again, 1.5 GB →
0.75 GB of W-HBM per frame). Ear-gate risk is real — every W4 TTS
landing requires careful coverage across speaker/language. On the
optimism discount, realistic **+1 fps** delivered, with tail risk of
quality regression forcing rollback.

### 2. M2 Talker aclGraph rewire — **+1–2 fps projected, +0.4–0.8 fps realistic**

Per `talker_aclgraph_feasibility.md`. The aclGraph pattern applies;
memory footprint is the new problem (75-pos cache ~1.5 GB; 500-pos cache
~10 GB). Cap at MAX=128. Requires D2D memcpy refactor at ~84 sites per
forward (vs CP's 15) and replacement of lazy-capture with
capture-at-init.

Realistic: +0.4–0.8 fps delivered. **~1 week agent-wall**.

### 3. Device-side n_tokens=2 batched prefill — **+0.5–1 fps realistic**

Extension of M3'new'. Today's +0.3 fps landing was a host-side batch;
true device-side S=2 batched Mm + causal mask + n_tokens=2 FIAv2
dispatch would recover the remaining ~1 ms/frame. Multi-day rewrite
with aclGraph capture re-inventory risk.

### 4. aclGraph for lm_head sub-graph — **verify if already captured**

W1 runs lm_head on NPU via `aclnnMm`. If the W1 dispatches are issued on
the same stream as `forward_one_token_launch` and the G2 capture scope
includes them, this is already done — needs a 10-minute verification
against `cp_cann_engine.cpp`'s capture boundary (`forward_one_token_capturable_`
at :1854). Likely already captured via G2, but not explicitly audited.

### 5. Post-CannFusion-A16W8 fusion — **+0.6–2 fps contingent on upstream**

GitCode #26 upstream PR (CannFusion A16W8 lane) is the blocker. If it
lands, the FFN sublayer collapses from 5 dispatches to 1 with HBM
intermediate savings. Per `cannfusion_upstream_ask.md` math: ~0.25
ms/forward saved = ~5–8% fps = +1.5–2.5 fps projected, +0.6–1 fps
realistic after discount. **Contingent, multi-week wait**.

### 6. Speculative decoding for Talker — **2–3× optimistic, weeks of work**

`TALKER_SPECULATIVE` was removed (commit `84af6590`). Re-audit the
stack first (W2.2 never ran). Needs a small draft Talker model to
exist; training or distilling one is its own multi-week project.
Quality risk on TTS is distinct from LLM speculative decoding —
accept-rate variance at codec-frame granularity may introduce audible
prosody artefacts. **Deprioritise until W4 quant lands**.

### 7. Cluster TP — **1.5–2× tensor-parallel for multi-stream serving**

Not a single-utterance fps lever. For serving-workload scale (batch=N
utterances), tensor-parallel across ≥ 2 Ascend cards with HCCL
collectives can amortise weight-HBM across devices. This is the
structural path to 40+ fps **at serving scale**, not at single-utterance.
Needs its own contract covering weight sharding, pipeline stages, and
FFI bridge evolution (current `ASCEND_API_BRIDGE` B1-B3/5 is
single-device).

### Honest ceiling projection

- **Single-card single-stream, no W4**: ~33–34 fps realistic
- **Single-card single-stream, with W4 success**: ~35–37 fps
- **Cluster TP for batch=N serving**: 40+ fps at serving scale
- **Single-card 40 fps**: requires something we haven't identified —
  quite likely a model-architecture change (parallel codebook heads,
  which is a retrain)

## Part 4 — Transferable learnings to other Ascend workloads

### 4a. Qwen-Image-Edit-2512 (ominix_diffusion)

`ominix_diffusion` tree is structurally ported (`common/`, `src/`,
`server/`, `cli/`) but not-yet-optimised.

- **Phase 0 (30 min)**: probe fps vs MLX M3 Max baseline. Hypothesis
  (not verified): starting gap is 5–10×.
- **Phase 1 (1 week)**: escape `ggml-cann` → native `DiffusionCannEngine`
  equivalent of `CpCannEngine`. Engine-owned dev buffers per GFORWARD.
- **Phase 2 (half-day probe + 1 week)**: audit `aclnnop/` for `Conv`,
  `Attention2D`, `GroupNorm`, `SiLUInplace`, `UNet*`, `CrossAttention`.
  Attention will hit `aclnnFusedInferAttentionScoreV2` same as Talker.
- **Phase 3 (3–5 days)**: aclGraph per-denoising-step. Unlike TTS
  (shape varies with pos), diffusion is **shape-stable per step count
  — capture once per step, replay across batch**. Should be a cleaner
  aclGraph win than TTS G2; no pos-keyed cache needed.
- **Quality gate**: user-eye FID on reference suite, not drift-count.
  Diffusion tolerates more numerical variation than autoregressive TTS.
- **Realistic timeline**: 2–3 weeks agent-wall to match MLX; +1–2
  weeks to exceed. TTS debug paths (D2D memcpy non-capturability,
  stream-barrier ordering, env-gate mechanics) already proven.

### 4b. Qwen3-VL-32B cluster (Slide 10 target)

- **Phase 1 (1–2 weeks)**: single-card first. Same W1/G2/A.1 moves
  on VL transformer. 32B F16 (~64 GB) doesn't fit 910B4's 32 GB —
  **W8 is mandatory**, W4 likely needed for KV at long context.
- **Phase 2 (2–3 weeks)**: TP wiring. HCCL collectives, per-column Mm
  for FFN, per-head for attention, pipeline stages.
- **FFI bridge already half-built**: B1-B3/B5 done; C-ABI surface
  generalises to any `load/synthesize/free`-pattern model. B4 remaining.
- **Expected end state**: 32B at 4-card TP, O(10–20 tok/s) batch=1
  plausible; batch=8 serving is where ROI lives. Single-stream
  bottlenecks on HBM-per-card, not compute.

### 4c. Generic pattern insights

- **Probe-first is cheap (~30 min) and saves weeks**. Every RED in
  Part 2 had a cheap probe that would have caught it: CannFusion
  validator grep, V2 RoPE GQA harness, FFNV3 dtype audit,
  TALKER_CANN_GRAPH A/B. Codify as Gate 0.
- **Offline diff gate is insufficient**. W4.1 passed 1e-3 offline and
  failed live at 1949. Live token-drift mandatory for hot-path kernels.
- **Vendor catalog grep is a FIRST move**. FO-audit found three unused
  fused ops in 30 min; should be contract-start, not after Path C closed.
- **Projection math needs 3–10× discount**. G0 projected +6–10, delivered
  +1.15. W3b +3.5 → +0.66. W4.1 +2 → regression. **Agent estimate ÷ 5.**
- **Noise band characterisation before sub-fps claims**. N=10 stock
  trials with std reported is cheap insurance.

## Part 5 — Meta-process improvements

Codify into the next contract template:

1. **Mandatory noise-band measurement at contract start**. N=10 stock
   runs on the canonical benchmark, report mean + std. All subsequent
   fps claims are stated as deltas vs this measured band.
2. **Mandatory scaffold-probe step** for any pre-existing env var the
   contract touches. If `TALKER_CANN_GRAPH` or similar scaffold exists,
   a 15-minute A/B with default-off vs default-on is Gate 0.
3. **Mandatory vendor-op catalog grep as contract Gate 0**. Grep the
   `aclnnop/` headers for the fusion patterns the contract proposes
   before agent dispatch. If vendor ops exist, evaluate them before
   funding custom kernels.
4. **Realistic projection = "agent estimate ÷ 5" as the planning
   number**. Use the optimistic estimate to scope ceiling, the
   discounted number to justify spend.
5. **Frame-count-identity gate** (learned from V2 RoPE failure)
   promoted to universal gate. Any change to attention/RoPE/softmax
   must produce frame-count-identical output on the canonical
   benchmark before any fps measurement is meaningful.
6. **Patch-file push mechanism documented as a reusable policy**, not
   per-contract. `git format-patch` on ac0N + scp to Mac + PM pushes
   is the working pattern across W1, W3b, G1, G2, Path C, CannFusion.
   Standardise the patch naming and the target branch review flow.
7. **Offline-diff budget is a soft gate, not a hard gate**. Offline
   fixture parity is necessary but not sufficient; require a live
   10+ frame token-drift measurement before declaring correctness.
8. **PM-gate cadence**: one agent per milestone, PM gate at each
   quantitative threshold, no silent continuation past a failing
   gate. This worked. Keep it.

## Part 6 — Vendor-side asks (ecosystem moves)

In order of leverage for our workloads:

1. **A16W8 no-expert branch in FFNV3 + QKV-fused Mm**. Unlocks at
   least +1–2 fps on TTS, probably more on VL. This is the single
   highest-leverage vendor change. Can be raised through the CANN
   engineering channel (we already have an escalation precedent via
   CannFusion §1.12).
2. **Open-source `libruntime.so`**. Agents currently read error-code
   names from headers but cannot decode the internal error-chain from
   the runtime. Full open-source unlocks faster iteration on every RED
   we see (EZ9999 161002 debugging took days when it could have been
   minutes).
3. **English-first op documentation**. Current `aclnnop/*.h` docstrings
   are mixed; some ops have only Chinese or only English, some have
   inconsistent parameter descriptions (V2 RoPE's packed-UB GQA
   behaviour was undocumented). A single source of truth, English +
   Chinese parity, with capability tags (training / inference / MoE /
   non-MoE / MHA / GQA / MLA).
4. **Local CANN SDK for macOS + Linux dev**. Currently any header
   audit, dlsym exploration, or ABI validation requires ssh to ac01 or
   a ModelArts container. A static-link-only SDK (no kernel runtime,
   headers + bindgen-compatible) for Mac dev would kill the ac01
   dependency for iteration. Every PM-gate conversation that needed
   "let me check what headers are available" would collapse to a
   local grep.
5. **Better aclnn op index — curated catalog with capability tags**.
   Today the catalog is 717 raw headers. A curated index (like
   cuDNN's op index) with per-op capability matrix
   (training/inference/dtype-matrix/MoE-or-not/MHA-or-GQA-or-MLA)
   would collapse every fused-op audit from 2–3 hours to 10 minutes.
6. **Community-contributed CannFusion A16W8 support**. GitCode #26
   already moving per the upstream ask. Low-priority vendor, but the
   community path is unblocked; if Huawei co-signs it, this lands
   faster than a direct vendor track.

## Closing

**PM takeaway**: the 32× lift from 1 fps to 32.2 fps happened because
we funded many parallel tracks, killed the unpromising ones fast, and
landed the two big wins (W1 cross-layer port, G2 aclGraph) cleanly. The
meta-process worked when gates were numerical and probes preceded
dispatch. It over-spent when projections were taken at face value.
Single-card Qwen-TTS is now at a realistic ceiling of ~33 fps without
W4 quant; reaching 40 fps at single-utterance requires either W4 (risky
quality) or a model architecture retrain (parallel codebooks). The
honest 40+ fps path is cluster TP for serving workloads.

**Future agent-swarm inheritance defaults**:

- Gate 0 for every contract: noise-band + scaffold-probe + vendor-op
  catalog grep (≤ 1 hour combined).
- Projection math: report optimistic + discounted estimates; budget
  against the discounted number.
- Live token-drift + user-ear/eye gates are mandatory, not advisory.
- One agent per milestone, PM gate at each threshold, no silent
  continuation past failure.
- Validator whitelists and vendor docs are **designed limitations**,
  not oversights — read them as intentional before assuming otherwise.
- Patch-file push from device hosts to Mac PM is the standard workflow;
  codify as policy, not per-contract.
