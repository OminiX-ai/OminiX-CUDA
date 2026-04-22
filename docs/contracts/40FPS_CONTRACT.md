# 40 fps on Ascend 910B4 — Single-Card Parity with M4 Pro MLX

## 1. Status & mandate

**Status (2026-04-21 FINAL)**: **RECALIBRATED — M3 DEAD**.

Agent GD-audit (2026-04-21) confirmed Qwen3-TTS is **strict 15-step
RVQ depth transformer**. Four independent sources (MLX reference at
`qwen3-tts-mlx/src/talker.rs:372-389`, Ascend native CANN at
`talker.cpp:1681-1734`, llama.cpp path at `talker.cpp:1813-1837`, CPU
path at `talker.cpp:1879-1896`) confirm each group's forward pass
receives the embedding of the previous group's sampled integer token.
**Group-collapse is structurally impossible.** See
`docs/cp_group_dependency_audit.md` for the citations.

**Revised realistic single-card ceiling**:

| Lever | Δ | Cumulative |
|---|---|---|
| Current (aclGraph on, LONG) | baseline | 31.6 |
| M1 Phase A+B (vendor fused ops) | +1.5 | ~33 |
| M1-extra (lower-ranked FO-audit) | +0.5 | ~33.5 |
| M3'new' — position 0+1 batching (tiny) | +1.0 | ~34.5 |
| M2 Talker aclGraph | +1-2 | ~35.5-36.5 |
| M4 W8 → W4 quant (ear-gate risky) | +2-5 | ~37.5-41.5 |

**40 fps is only reachable with M4 W4 quant** — the quality risk is
real. Without W4, realistic single-card ceiling is **~36 fps**
(≈M4 Pro parity). For a dependable 40+, cluster tensor-parallel is
the structural path.

Umbrella contract subsumes `FUSED_OP_LANDING_CONTRACT.md` as M1. M3
(group-collapse) is CLOSED without dispatch. M3'new' (pos-batching)
added as small micro-opt.

**Target**: **≥ 40 fps** on canonical xvec mayun zh LONG text, aclGraph
on, sampling on (ship config), byte-identical user-ear verdict.

**Reference points**:
- Today: **31.6 fps** (aclGraph on, LONG), validated by G4-ext
- M4 Pro Mini MLX: **37 fps** (Agent benchmark)
- M3 Max MLX: **46 fps** (same benchmark)
- **40 fps = parity with M4 Pro**, stretch toward M3 Max

**Why 40 and not 32**: the old ≥32 fps gate was aspirational with no
anchor. **40 fps = "don't lose to a laptop"** — the gate that matters
for customer credibility. Below 40 fps, Ascend 910B4's 2× HBM
bandwidth advantage over Apple Silicon (800 vs 400 GB/s) looks
wasted.

**PM role**: supervise contract milestones; gate each on drift +
perf + user-ear; arbitrate trade-offs between stacked levers.

## 2. Background — what closes the 8-fps gap

Single lever analysis (on 31.6 fps baseline, 33 ms/frame):

| Lever | Projected Δ | Cumulative | Cost | Risk |
|---|---|---|---|---|
| M1: vendor fused ops (Phase A+B) | +1.5 fps | 33 fps | 2-3 days | low |
| M1-stretch: lower-ranked audit candidates | +0.5 fps | 33.5 | 1-2 days | low |
| M2: Talker (LLM) aclGraph | +1-2 fps | 34.5-35.5 | ~1 week | med |
| **M3: CP group-collapse (algorithmic)** | **+5-15 fps** | **40-50** | 1-2 weeks | high |
| M4: W8 → W4 quant | +2-5 fps | 42-55 | 1 week | med (quality risk) |
| M5: Decoder (VAE) fusion / AscendC | RTF only | — | separate | — |

M3 is the structural lever that closes the gap. The CP generates 15
codebook-group tokens per frame autoregressively today; if the
dependency graph permits batching (parallel codebooks, shared hidden
state, or RVQ with inter-frame-only deps), 15 sequential GEMVs
collapse into 1+ GEMMs. That's the 4-10× CP speedup that makes 40 fps
reachable.

**M3 feasibility probe is currently in flight** (Agent GD-audit).

## 3. Scope

**In scope**:
- M1 vendor fused ops (Phase A: RoPE V2 + InplaceAddRmsNorm; Phase B: FFNV3)
- M2 Talker aclGraph (NEW track; uses aclGraph playbook from G0-G4)
- M3 CP group-collapse (conditional on GD-audit verdict)
- M4 W8 → W4 quant (conditional; only if M1+M2+M3 don't hit 40)
- Stacked correctness gates: token drift ≤ 1 per lever + composed
  drift ≤ 1 overall + ear gate each time

**Out of scope**:
- Decoder/VAE fusion (separate RTF-improvement track)
- Cluster (tensor/pipeline parallel) — different contract
- Multi-utterance session amortisation — deferred

## 4. Host plan

- ac01: primary M1 (Phase A in flight), M2 (Talker aclGraph)
- ac02: M3 prototyping (if GD-audit returns GREEN/YELLOW)
- ac03: M4 W4 quant calibration harness
- Patch-file push mechanism per convention

## 5. Milestones with gates

### M1 — Vendor fused ops (→ ~33 fps)

See `FUSED_OP_LANDING_CONTRACT.md` §4 Phase A + Phase B. Already
dispatched.

- [ ] M1.A.1 `aclnnInplaceAddRmsNorm` wired, env-gated
- [ ] M1.A.2 `aclnnApplyRotaryPosEmbV2` wired, env-gated
- [ ] M1.A.3 Combined + aclGraph, ≥ 32 fps LONG, ear clean
- [ ] M1.B.1 Offline W8 gate∥up re-pack script
- [ ] M1.B.2 `aclnnFFNV3` wired, env-gated
- [ ] M1.B.3 Combined, ≥ 33 fps LONG, ear clean

**M1 gate**: fps ≥ 33 on LONG canonical + ear identical to baseline.

### M2 — Talker aclGraph (→ ~34.5-35.5 fps)

Talker is the 28-layer LLM that predicts the next codec group-0 token.
Current cost ~7.8 ms/frame at LONG. aclGraph pattern from G-track
should apply analogously.

- [ ] M2.1 Audit Talker forward op chain (analog of G0 feasibility)
- [ ] M2.2 One-layer Talker smoke (analog of G1)
- [ ] M2.3 Full-forward capture with pos-keyed cache OR TaskUpdate
      (whichever G1-style probe recommends)
- [ ] M2.4 Drift gate on token sequences + ear gate
- [ ] M2.5 Combined with M1 + CP aclGraph, measure fps delta

**M2 gate**: fps ≥ 34.5 LONG + drift ≤ 1 + ear clean.

### M3 — CP group-collapse — **CLOSED (NO-COLLAPSE verdict)**

GD-audit (`docs/cp_group_dependency_audit.md`, 2026-04-21): Qwen3-TTS
is strict RVQ. Group *i*'s forward pass consumes group *i-1*'s
sampled integer token via `codec_embeddings[i-1]` → embed → project
→ new 5-layer CP transformer pass. 15 separate codec_embeddings and
15 separate lm_heads are classic RVQ-depth-transformer architecture
(Qwen2.5-Omni / MoonCast lineage). Dispatch-free dependency from
group *i-1* to *i* makes batch=N impossible.

M3 not dispatched. Fps ceiling without this lever = M1 + M2 + M3'new'
= ~35-36 fps.

### M1.B — aclnnFFNV3 — **CLOSED (A16W8 rejected at runtime)**

Agent PB landed scaffolding (fork `9aada3e6`, env-gated off) but gates
RED at runtime: `aclnnFFNV3`'s no-expert branch rejects `weight1=INT8`
with error `EZ9999 161002: weight1 only support dtype float16 without
expert tokens`. Vendor allows INT8 weight only when `expertTokens !=
null` (MoE case). Qwen3-TTS CP is single-token, no MoE, so the
no-expert branch is F16-only.

Third vendor-op in a row to advertise A16W8 support and reject it at
runtime (pattern: CannFusion validator whitelist, V2 RoPE packed-UB
GQA incompat, FFNV3 no-MoE INT8 rejection). Generic Ascend fused-op
APIs are A16W8-patchy; reliable A16W8 still only through
`aclnnWeightQuantBatchMatmulV3`.

Scaffolding preserved in-tree for future CANN-release unblock.

### M3'new' — Position 0+1 batching micro-opt — **LANDED**

**Verified-by Agent M3N (2026-04-21, fork `a3c9ebf1`)**.

- [x] M3'new'.1 `forward_two_tokens_launch` method added; chains two
      pos-keyed `forward_one_token_launch` calls with async H2D for
      pos 1 and aggregated event record. Existing callers unchanged.
- [x] M3'new'.2 Drift + ear gates: frame-count identical, output wav
      md5 byte-identical to pre-POS_BATCH stock. Drift = 0.
- [x] M3'new'.3 Perf gate: **+0.3 fps median** (31.9 → 32.2 on LONG).
      Clears ≥32 fps.

**Honest delta analysis**: projected +1 fps, delivered +0.3. Most of
the ~1 ms/frame estimate was host-side only; actual device-side
serialization on the two launches was smaller than GD-audit assumed.
A true device-side batched prefill (S=2 batched Mm, causal mask,
n_tokens=2 FIAv2) would recover the rest but is multi-day with
re-capture risk. Shipped the safe host-side slice.

Env gate: `TALKER_CP_POS_BATCH=1`, default off. Stock path untouched.
aclGraph compatibility preserved — no capture re-inventory needed.

### M4 — W8 → W4 quant (conditional, → potentially 42-55)

Only if M1+M2+M3 don't reach 40 fps.

- [ ] M4.1 Per-channel W4 quant calibration (AWQ or similar)
- [ ] M4.2 Validate via `aclnnWeightQuantBatchMatmulV3` W4 lane (if
      supported on 8.3.RC1) or fallback to `aclnnGroupedMatmul`
- [ ] M4.3 Full quality pass (ear gate mandatory; W4 risk is real)

Skip to M4 only if M3 closes the gap.

## 6. Acceptance criteria (summary)

- [ ] M1 lands: ≥ 33 fps LONG + ear clean
- [ ] M2 lands: ≥ 34.5 fps LONG + ear clean
- [ ] M3 GD-audit verdict known; if GREEN/YELLOW, M3 lands ≥ 40 fps LONG + ear clean
- [ ] Total ≥ 40 fps LONG cumulative with all env gates on
- [ ] Token drift ≤ 1 per milestone and composed
- [ ] Coverage wavs delivered; user ear verdict CLEAN
- [ ] Contract stamped with commit SHAs + Verified-by per milestone

## 7. Risks

1. **M3 NO-COLLAPSE**: the model may be strict RVQ. Then M3 is dead
   and ceiling is ~35-36 fps. Mitigation: GD-audit is in flight;
   verdict gates M3 dispatch.
2. **M2 Talker aclGraph may duplicate G1 3e-3 multi-op drift**:
   Talker op count and shape class differs from CP; re-verify
   TaskUpdate semantic on Talker's specific aclOpExecutors. Mitigation:
   apply G2's pos-keyed fallback pattern.
3. **M4 W4 quality regression**: W4 in TTS can introduce audible
   artefacts. Mitigation: ear gate is hard-kill; W4 only dispatches if
   M1+M2+M3 fall short.
4. **Composed drift explosion**: 4 stacked fused ops + group-collapse
   + W4 = many places drift can accumulate. Mitigation: drift gate is
   composed, not just per-milestone.
5. **Env-gate combinatorial space**: 5-6 env vars would create 32-64
   possible configs. Mitigation: ship with `TALKER_CP_40FPS=1`
   umbrella env that enables the default production stack in one flag.

## 8. Host rules

- Each milestone commits locally on host, patches to PM, PM pushes fork
- No Claude coauthor
- HARD KILL at each perf/ear gate
- User-ear outranks fps numerics
- M3 is the load-bearing milestone; M3 failure cuts the contract
  target from 40 to 34-35 realistic

## 9. Timeline (agent-wall)

- M1 (in flight): 2-3 days
- M2: ~1 week
- M3 (conditional): 1-2 weeks
- M4 (conditional on M3 miss): 1 week
- Total if M3 passes: **2-3 weeks agent-wall**
- Total if M3 fails: **1-1.5 weeks**, ceiling ~35 fps
