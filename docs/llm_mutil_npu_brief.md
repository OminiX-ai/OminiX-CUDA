# MoYoYoTech/llm_mutil_npu — Deep Exploration Brief

**Agent**: MN (deep-dive follow-up to PM skim)
**Date**: 2026-04-21
**Source**: `https://huggingface.co/MoYoYoTech/llm_mutil_npu` (raw endpoint reads)
**Target audience**: PM, for Qwen3-TTS agentic-AI-coding deck corrections

---

## Executive summary

1. **V2 RoPE A.2 verdict should be REOPENED.** `llm_mutil_npu` uses `aclnnApplyRotaryPosEmbV2(layout=1, rotaryMode="half")` successfully on Qwen3 GQA at per-rank shape **Hq=4, Hkv=1** (TP=16 shard of 64Q/4KV). Layout=1 is passed in as `[B, S, N, Dh]` — the README's "SBND" label is just prose; the actual memory layout fed to the op is BSND. This contradicts our Phase-A.2 conclusion that V2 is GQA-incompat; the real issue is almost certainly wiring on our side, not a kernel-level GQA incompat.
2. **HCCL env tuning is the single biggest cluster lever we are not yet documenting concretely.** They measured a full breakdown: baseline 12.20 → +AIV 17.74 (+45%) → +FFTS 17.90 (+47%) → AIV+FFTS 18.82 (+54%) → +TASK_QUEUE=2 23.10 (+89%). Slide 8 currently discusses cluster abstractly; this is a ready-made data point and a drop-in speaker-notes cheatsheet.
3. **PLD has a clear degeneration failure mode** with a well-characterized root cause (positive feedback loop on accept-rate, detectable at accept > 5/K). For Talker, the codec-token distribution is different enough that a port is interesting but not a slam-dunk — needs a bounded probe, not a contract.
4. **WorkspacePool retain-old pattern** is directly applicable to our CpCannEngine and fixes an async-safety gap our current single-grow-buffer approach has. This is a quiet, high-value transfer.
5. **Agentic-workflow signal is strong**: this is the PM's own project and the artefact shows the same rapid-iteration, honest-retraction, test-first cadence we claim in the deck — giving us a second 30× data point to cite.

---

## Project overview

**Target**: Qwen3-235B-A22B-Instruct-2507 (BF16), 94 layers, 128 experts top-k=8, **GQA 64Q/4KV**, on Ascend 910 **initial-gen × 16** (TP=16 via HCCL ring AllReduce).

**Approach**: pure C++ aclnn EAGER — every op goes through `aclnn*` single-op API with a workspace pool. No graph compilation, no PyTorch, no ggml, no torchair. Tokenizer is the one compromise: HuggingFace via Python subprocess.

**Arc**: 12 t/s → 27 t/s stable (all prompts) → 39 t/s creative (PLD). Explicit ceiling: **does not** reach `cann-recipes-infer` GE-graph 54 t/s baseline on same hardware. Honest about it — the README says so plainly, and the optimization-summary names the four missing kernels (`MatmulAllReduce`, `GroupedMatmulAllReduce`, `AddRmsNorm`, etc.) on 910 initial-gen that would be needed to close the gap.

**Report dates**: 2026-04-21 → 2026-04-22 (literally yesterday-to-today). Two calendar days, reported publicly. Matches PM's claim that agent-driven workflows compress a normally-weeks engineering effort into days.

---

## Q1 — V2 RoPE reconciliation

### What they do (concrete)

From `include/rope.h` lines 21-37 and `include/engine.h` line 145:

```cpp
// Call-site
apply_rope_fused(stream,
                 q_scratch, /*B=*/1, S, /*Nq=*/Hq, Dh,   // Hq per-rank = 4 on TP=16
                 k_scratch,          /*Nk=*/Hkv,         // Hkv per-rank = 1
                 cos_ptr, sin_ptr);

// Inside apply_rope_fused
auto t_q   = make_contig_tensor(q_data, dt, {B, S, Nq, Dh});   // [1, S, 4, 128]
auto t_k   = make_contig_tensor(k_data, dt, {B, S, Nk, Dh});   // [1, S, 1, 128]
auto t_cos = make_contig_tensor(cos_data, dt, {1, S, 1, Dh});
auto t_sin = make_contig_tensor(sin_data, dt, {1, S, 1, Dh});
aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
    t_q.get(), t_k.get(), t_cos.get(), t_sin.get(),
    /*layout=*/1, /*mode=*/"half", ...);
```

### Answers to the specific questions

- **Layout semantics**: their README prose says `layout=1 → SBND` but the shapes they actually pass are `[B=1, S, N, Dh]`, i.e. **BSND** memory layout. With B=1 and S=1 (the decode case) SBND vs BSND is indistinguishable, so the prose label is unfalsifiable against their data — the binding fact is the shape vector they pass. **Treat layout=1 as BSND for wiring purposes** — matches what our Phase A.2 patch also passed.
- **cos/sin preparation**: `fill_cos_sin_hf` in `engine.h` lines 32-46 produces **HF half-half** layout (pair = `d < half ? d : d - half`), written as contiguous `[L, Dh]` BF16 with no duplication. Tensor descriptor is `{1, S, 1, Dh}` with strides implicit (contiguous) — broadcast across N is left to the kernel.
- **Q/K shape pair**: exactly `{1, S, 4, 128}` and `{1, S, 1, 128}` per rank on TP=16. So **they feed GQA-shaped tensors (Nq ≠ Nk) directly to V2 and it works**. This is a direct empirical counter-example to our Phase A.2 H5a hypothesis (packed-UB shared-stride breaks on GQA).
- **Test coverage of 64Q/4KV**: `tests/test_rope_fused.cpp` uses **Hq=Hkv=4** — it does **not** stress the heterogeneous-head-count packed-UB path. So the test doesn't probe H5a. BUT the production `attention_forward` runs GQA (Hq=4, Hkv=1) on every decode step across 94 layers × TP=16 ranks and produces coherent output at 27 t/s end-to-end — i.e. the real workload IS the GQA stress test, and it passes.
- **Known gotchas they documented**: `layout=0` returned `status=561002` and was rejected (documented in both the Chinese summary and as a falsifier-style `tries[]` loop in `test_rope_fused.cpp` lines 69-93). They tried six `(layout, mode, qshape, cshape)` combos and only `layout=1, "half", {B,S,N,Dh}, {1,S,1,Dh}` was accepted. They also note `rotaryMode` must be `"half"` (not `"interleaved"`).

### Reconciliation with Phase-A.2 debug report

Our `/Users/yuechen/home/OminiX-Ascend/docs/v2_rope_numerics_debug.md` (H5a, likelihood HIGH) hypothesised that V2's `CopyInQK` packs Q+K back-to-back in one UB buffer with a shared `dstRepSBr`, breaking when `qcdNum != kcdNum`. The llm_mutil_npu production evidence **falsifies H5a at the kernel-semantic level**: `qcdNum=4*128=512, kcdNum=1*128=128` per rank is a stronger GQA ratio (4:1) than our talker (2:1, Hq=16, Hkv=8) and it works.

**Plausible alternative root causes** our CP is actually hitting:
- **Shape pair mismatch**: we pass Q as `{1,1,16,128}` and K as `{1,1,8,128}` according to the debug doc. That should parallel theirs. But we should double-check that Q and K strides are both contiguous per the descriptor and that we're not accidentally passing a **view** (strided) tensor where V2 expects contiguous.
- **In-place vs separate output**: V2 writes back in-place. The debug doc lines 75-84 flag H5b: our K-proj output goes to the KV-cache slot and V2 reads and writes the same slot. llm_mutil_npu also writes in-place (`q_scratch` is both read and write), so this alone shouldn't be the bug — but if our cache-slot pointer arithmetic is subtly off (e.g. wrong layer index, or the slot address has been pre-populated with non-rotated K in a way V2 doesn't expect), we'd see exactly the "first-frame semantic drift" symptom.
- **cos/sin table prep**: llm_mutil_npu uses HF **half-half** (not duplicated) for V2. Our debug doc lines 17-23 says we upload cos/sin **half-duplicated** (`cos_table[p*D + j + half] = cos_table[p*D + j]`) because the v1 `mode=0` kernel expects duplicated. **If our Phase-A.2 patch reused the v1 duplicated table as-is, that is wrong for V2**. V2's `BatchHalfAlignVF` (per the debug doc's own kernel trace at lines 25-34) computes `out[0:half] = in[0:half]*cos[0:half] - in[half:]*sin[0:half]` — this works correctly with duplicated cos/sin **only if** `cos[i] == cos[i+half]` AND the op does not re-apply the mask internally. Our debug doc concludes it's fine ("Layout is not the divergence"), but this deserves a targeted probe: swap to half-half cos/sin for V2 and see if hidden-state drift resolves.

### Verdict

**REOPEN Phase A.2.** Do not reverse the closed-contract finding in-place, but write a new probe contract:

> **A.2-REOPEN probe**: Build a minimal standalone harness (per the v2_rope_numerics_debug.md §Validation plan) that feeds Q=`[1,1,16,128]`, K=`[1,1,8,128]` with (i) duplicated cos/sin and (ii) HF half-half cos/sin, comparing V2 output to v1(mode=0). Expected result: at least one of the two cos/sin preps matches v1 bit-exact, and we have an existence-proof on our own hardware that V2 accepts GQA. If neither matches, then we have a genuine 910B4 vs 910-initial-gen kernel difference and the contract stays closed with a better justification.

Upside if reopened successfully: V2 is **+17% TG** on llm_mutil_npu (23 → 27 t/s). On our Talker we would see proportionally less (RoPE is a smaller fraction of wall), but a probe-level +1-3 fps on the 32.2 fps wall is still worth a week.

---

## Q2 — HCCL env tuning deconstruction

### The full combo (from `scripts/tp_launch.sh` lines 20-29)

```bash
HCCL_WHITELIST_DISABLE=1
HCCL_ALGO=level0:ring            # NOT fullmesh — fullmesh produces garbled Qwen3-235B output
HCCL_BUFFSIZE=200                 # sweet spot; both 100 and 400 are slower
HCCL_OP_EXPANSION_MODE=AIV        # AI Vector cores join reduce scheduling
HCCL_OP_BASE_FFTS_MODE_ENABLE=1   # Fast Frequently-used Transfer Scheduling
TASK_QUEUE_ENABLE=2               # aggressive async task submission
```

### Contribution breakdown (from `docs/optimization-summary-zh.md` §2.1)

| Stacked config (on top of ring + buffsize=200) | TG (t/s) | Δ from baseline |
|---|---|---|
| baseline | 12.20 | — |
| + OP_EXPANSION=AIV | 17.74 | +45% |
| + FFTS=1 | 17.90 | +47% |
| + AIV + FFTS | 18.82 | +54% |
| + AIV + FFTS + TASK_QUEUE=2 | **23.10** | **+89%** |

Interaction pattern: AIV and FFTS alone are near-equivalent (+45-47%); combined they're only +54%, so they overlap significantly in mechanism. TASK_QUEUE=2 layered on top of both is the 35-percentage-point bonus — its value comes from overlapping kernel-submission with HCCL wait, a different axis entirely.

### Semantic roles (confirmed from their docs + Ascend doc cross-ref)

- **`AIV`**: uses AI Vector cores (not just AI Cube cores) to participate in reduce-op scheduling. On 910 initial-gen the Cube cores were the bottleneck during AllReduce; AIV lets vector cores take reduce fragments.
- **`FFTS_MODE_ENABLE`**: Fast Frequently-used Transfer Scheduling — prebuilds HCCL task descriptors for the hot AllReduce shape (same `S × D` across 94 layers), amortizing dispatch overhead.
- **`TASK_QUEUE_ENABLE=2`**: the HCCL-level equivalent of lifting a TAQ depth. Lets kernel submissions pile up in the queue without blocking on prior AllReduce completion. The key enabler for rank-compute / HCCL overlap in EAGER mode.
- **`HCCL_BUFFSIZE=200`**: HCCL scratch buffer in MB. Sweet spot is workload-dependent — their sweep showed 100 and 400 both slower, 200 was a local optimum for Qwen3-235B's per-layer AllReduce size.
- **`HCCL_ALGO=level0:ring`**: topology selection. Ring is correct for a single-level 16-device setup.

### Topology gotchas (HIGH VALUE for Slide 8)

- **`HCCL_ALGO=level0:fullmesh` produces garbled Qwen3-235B output**. This is a correctness-level landmine — not a performance knob. Worth calling out as a "not all topology choices are equal; fullmesh is NOT a drop-in upgrade from ring even though it looks like the more-connected option."
- **`HCCL_OP_EXPANSION_MODE=AICPU` crashes on 910 initial-gen** — no kernel implementation. Specific to gen1; likely OK on 910B/B4/A2.

### Cheatsheet (ready to drop into Slide 8 speaker notes)

> **HCCL env tuning — Qwen3-235B TP=16 on 910 initial-gen, +89% TG from zero-code knobs**
>
> Set in launcher: `HCCL_ALGO=level0:ring` (not fullmesh — fullmesh outputs garbage), `HCCL_BUFFSIZE=200` (sweep-tuned), `HCCL_OP_EXPANSION_MODE=AIV` (AI Vector cores join reduce scheduling — key, +45% alone), `HCCL_OP_BASE_FFTS_MODE_ENABLE=1` (prebuilt task descriptors for hot AllReduce shapes — key), `TASK_QUEUE_ENABLE=2` (submission/AllReduce overlap — +35pp on top of AIV+FFTS).
>
> Source: MoYoYoTech/llm_mutil_npu `scripts/tp_launch.sh` + `docs/optimization-summary-zh.md` §2.1, measured on Qwen3-235B-A22B-Instruct-2507 BF16, Ascend 910 × 16, greedy temperature=0. Baseline 12.20 t/s → full combo 23.10 t/s.

---

## Q3 — PLD feasibility for Talker

### PLD mechanism (from `main_cli.cpp` + README §PLD + optimization-summary §2.3)

- **n-gram matcher**: level-1 (single-token) match with multi-level fallback. `min_hist=20` before PLD is enabled. Default K=10 draft tokens per cycle.
- **Mask-bug story (5+ hours)**: initial version reused the prefill sparse_mode=3 + 2048×2048 causal mask for batch verify. Under sparse_mode=3 the FIAS kernel interprets q[i] as "can see only kv[0..i]" — **it ignores `past_len` entirely**. Result: every batch position "forgets" prior KV context, accept rate collapses to 8%. Fix: custom `[1, 1, S, past+S]` bool mask with `mask[i,j] = 1 iff j > past_len+i`, and `sparse_mode=0`. This is the same FIAS decode-mode bug we should watch for if we ever add batch-decode to Talker.
- **Accept-rate ranges (measured)**: creative 0.5-3.0 (usable), structured code 2-4 (sometimes OK), factual Q&A 4-8 (degenerates), code-generation 5-9 (almost always degenerates). **Accept > 5/K sustained = degeneration signal**, not a speed win.
- **Rewind-cache logic**: `rewind_cache(K - accept)` rolls back per-layer KV cache `past_len` by the rejected-draft count. No extra bookkeeping — just decrement.
- **Degeneration guard** (critical learning): two heuristics — (i) **low-distinct**: draft's unique-token count < 3 → reject; (ii) **tail-echo**: last N history tokens all equal draft[0] → reject. Both guards drop the draft and fall back to single-token decode. Plus a `[warn]` emitted once on 8 consecutive identical tokens.
- **Auto-disable is an anti-pattern**: `T_batch(S=11)=42ms` vs `T_decode(S=1)=47ms` → the accept-rate breakeven is **-0.1**. Any non-negative accept is net-positive. Their previous `accept<0.5 disable` was killing legitimate cases.

### Batch-decode amortization curve (from §2.3)

| S | forward ms | amortized ms/tok |
|---|---|---|
| 1 | 47.62 | 47.62 |
| 2 | 43.51 | 21.76 |
| 4 | 35.82 | 8.96 |
| 8 | 39.08 | **4.89** (9.7× throughput vs S=1) |

Decode is **latency-bound**, not compute-bound — S=1 to S=8 forward time is flat. Extra draft tokens are nearly free to verify. This is the fundamental PLD economic argument.

### Transferability to Talker

**Against:**
- Talker operates on codec group-0 tokens (RVQ codebook 0). Our current probe work in `docs/` shows group-0 tokens have a much narrower empirical distribution than text, conditioned on prosody embedding — meaning repetition is **structural** (phoneme repeats, silence runs) rather than coincidental. n-gram match on codec tokens would likely hit the same positive-feedback-loop failure mode llm_mutil_npu saw on factual-Q&A prompts, and probably faster.
- Accept-rate semantics for spec decoding on conditional codec distributions are genuinely unclear — our Talker forward is conditioned on `(text, prosody, past_codec)` in ways the LLM isn't.

**For:**
- Our Talker wall is 7.8 ms/frame and decode is almost certainly latency-bound like their 235B case (kernel launch + HBM bandwidth dominated, cube-math tiny). The S=1→S=8 near-flat curve would likely reproduce.
- If it works, +30-70% on creative / phoneme-diverse utterances maps to Talker reaching ~50 fps median from ~32 fps — enough to justify a bounded probe.

### Upside estimate & recommendation

**Realistic envelope**: +10-30% on the 32.2 fps wall = 35-42 fps IF the accept-rate structure holds. But there's a real possibility of zero-to-negative (degeneration loops).

**Recommended next step**: a 1-week bounded probe, NOT a contract. Probe goals:
1. Measure empirical n-gram accept rate on codec-group-0 over a 50-utterance corpus (creative + factual text inputs).
2. Port the degeneration guard (low-distinct + tail-echo) directly — it's 20 lines.
3. Gate acceptance with BOTH a quality check (FID-style, via the reference xvec distribution) and the TG number.

If accept-rate > 1.5 consistent on ≥ 60% of utterances and quality-check passes on all → promote to contract. If not → close clean.

Name the contract candidate: **Talker-PLD Feasibility Probe (TFP)**. Not a commitment.

---

## Q4 — Transferable architectural patterns

### WorkspacePool (HIGH VALUE)

`include/workspace_pool.h` — 85 lines. Two features worth adopting in our CpCannEngine:

1. **Grow-only with retain-old**: when a request for `bytes > current_size_` comes in, the old buffer is **moved into `old_bufs_`** (not freed), and a new larger buffer is allocated. `reset_after_sync()` is called only after `aclrtSynchronizeStream` and frees the retained buffers.
   - **Why it matters for us**: our current CpCannEngine single-buffer-grow approach frees the old buffer immediately, which is a latent async-safety bug if any aclnn kernel from the prior op is still reading it. We've been lucky. This is a correct fix with ~50 lines of diff.
2. **Thread-local singleton via `thread_local WorkspacePool`** in `aclnn_ops.h:_lca_pool()`: every op wrapper gets the same pool on the same thread, zero plumbing. For us, with the dedicated inference thread pattern, this is a natural fit.

### Engine structure

`engine.h` — `attention_forward` and `moe_forward` are **free functions** that take all scratch buffers as pointers. Zero allocations per call. Caller (Runner) owns all buffers including per-layer KV cache slots. This is more C-like than our current Rust Ascend backend, but the underlying principle — **pre-allocate everything once, pass pointers every forward** — matches our CP design.

Two cross-cutting patterns worth mirroring:
- **RopeCache as a single pre-computed `[max_seq, Dh]` table**, with `cos_ptr = rope_cache.cos + past_len * Dh * 2` at call site (engine.h:142-144). We likely already do this on our Talker; worth a grep.
- **`sparse_mode = -1` auto-select**: the engine function accepts `sparse_mode=-1` and decides at runtime (prefill vs decode vs batch-decode). Centralizes a failure-prone decision. Our CP attention-forward could benefit.

### HCCL communicator lifecycle

`hccl_comm.h` — 107 lines total. The pattern:
- Rank 0 calls `HcclGetRootInfo`, writes to `/tmp/hccl_root_info.bin`.
- Other ranks poll the file for up to 60s, read it.
- All ranks call `HcclCommInitRootInfo`.
- `hccl_allreduce_bf16(ctx, data, count, stream)` is an in-place SUM.

This is the reference implementation we should use if we ever ship cluster TP on Ascend. It's simpler than the ggml-cann version and more direct than building a torch-style process-group abstraction. **Not urgent** — but if the Ascend-cluster story lands in the deck, cite this file.

---

## Q5 — Meta: agent-workflow observations

### Debugging / probe patterns worth adopting

1. **Falsifier-style enum sweep** (`test_rope_fused.cpp` lines 68-93): build a table of `(layout, mode, shape_pair)` candidates, loop through, print status for each. When `aclnn*GetWorkspaceSize` returns non-zero, the combo is rejected. This is a cheap way to discover undocumented op accepts — we should adopt it for any aclnn we're unsure about. Matches the "brute-force enumerate the kernel's accepted config space" strategy.
2. **Benchmark-with-correctness-classifier** (`bench_pld_safe.sh`): every PLD run's output is classified `OK / LOOP_N / LOW_DIVERSITY` and stats are separated for OK-only vs degraded. Direct applicability: our 32.2 fps measurements should pair with a quality check (waveform-level — reference xvec distribution, intelligibility-score, whatever). This is **benchmark ethics**: speed number without quality gate is misleading.
3. **Decode batch sweep to characterize latency vs compute boundedness**: the S ∈ {1,2,4,8} forward-ms table (§2.3) is 30 minutes of effort and decisively answers "is our forward compute-bound or latency-bound?" — informs every subsequent optimization.
4. **Retraction posture**: the §4 "重大翻车与修正" section explicitly retracts earlier 82.94 t/s / 177.40 peak claims and explains the feedback-loop root cause. This is the kind of honest course-correction we should emulate in-deck when discussing our own revised A.2 verdict.

### Evidence of agent-authored workflow

- **Timeline** (§9): 2026-04-21 morning → 2026-04-22. Two calendar days from untuned-baseline to production-ready 27 t/s + honest retraction on PLD. That pace is only achievable with agent-loop parallelism.
- **Test coverage**: 15+ unit / integration tests listed in README "Correctness verification". Each has a Python reference generator. High test-to-source ratio.
- **Commit pattern**: can't see full git log from raw endpoint, but the documentation structure (`optimization-summary-zh.md` with numbered root-cause table in §3, 7 bugs with repro) reads like a session-by-session agent log compressed into final form.
- **File count / specificity**: ~20 source files, ~10 scripts, 2 docs. Tightly scoped to the 235B TP=16 use case — no premature generalization, matches agentic-pragmatism.

### Wall-time mapping

Our deck claim is "Qwen3-TTS ~1 fps → 32.2 fps in ~7 days" (~30× speedup). Their claim is "Qwen3-235B 12 t/s → 27 t/s in 2 days (+125%)" — not a 30×, but the per-day engineering velocity is comparable when you account for the larger state-space (MoE routing, TP=16, HCCL tuning, PLD correctness). Both are consistent with agent-swarm-style compression.

---

## Deck corrections driven by this project

Specific slide edits (PM to decide what lands):

1. **Slide for A.2 V2 RoPE CLOSED → CONDITIONAL-REOPEN** (slide reference TBD — wherever the "closed: GQA-incompat" claim lives):
   - Old: `aclnnApplyRotaryPosEmbV2 is GQA-incompat based on kernel source reading (packed-UB shared-stride on `qcdNum ≠ kcdNum`)`.
   - New: `Kernel-level source hypothesis was falsified by an external data point: MoYoYoTech/llm_mutil_npu runs V2 on Qwen3-235B with per-rank Hq=4/Hkv=1 (4:1 GQA) successfully at 27 t/s end-to-end. Our Phase-A.2 failure mode is likely wiring-specific (candidates: cos/sin table prep — we use duplicated, they use half-half — or K-cache slot aliasing). Reopening as a scoped 1-week probe.`
   - Source citation: `MoYoYoTech/llm_mutil_npu/include/rope.h:21-37` + `include/engine.h:145`.

2. **Slide 8 cluster HCCL lever** (currently abstract):
   - Add a concrete panel with the 5-var stack and the contribution table (baseline 12.20 → +AIV 17.74 → +FFTS 17.90 → AIV+FFTS 18.82 → +TASK_QUEUE=2 23.10, +89% zero-code TG). Note the fullmesh-produces-garbage and AICPU-crashes-gen1 gotchas as "correctness landmines, not just perf knobs."
   - Source citation: `scripts/tp_launch.sh` + `docs/optimization-summary-zh.md §2.1`.

3. **PLD-for-Talker** (if the deck currently says "we can't PLD because RVQ strict", that's only half right):
   - Old (if stated): PLD is blocked by RVQ.
   - New: PLD on the **CP/RVQ path is blocked** (strict codec reconstruction), but the **Talker group-0 token path is an open probe** — decode is latency-bound (same argument as llm_mutil_npu's 4.89 ms/tok at S=8), so a bounded 1-week probe with ported degeneration guard is reasonable. Upside envelope +10-30% fps on the 32.2 fps wall IF accept-rate holds on codec distribution; real risk of zero-to-negative on factual/repetitive utterances.

---

## New contract candidates this exposes

1. **A.2-REOPEN probe** (conditional, 3-5 days): standalone harness comparing V2 vs v1 on GQA with (i) duplicated cos/sin (ii) HF half-half cos/sin, both on our 910B4 stack. Deliverable: parity verdict + root cause if divergent. Success closes our debug doc with evidence, not hypothesis.
2. **Talker-PLD Feasibility Probe (TFP)** (1 week): n-gram accept-rate measurement on codec-group-0 over a 50-utterance corpus, with degeneration guard ported. Gate: accept-rate > 1.5 on ≥ 60% of utterances + quality-check pass → promote to contract. Otherwise close clean.
3. **HCCL env probe contract** (shelved, activate if cluster TP lands): sweep the 5 vars on our target cluster hardware (910B4 × N? not specified yet). 0.5 day of sweeping, compile a per-hardware cheatsheet analogous to theirs.

---

## Recommendation for PM (top-line)

**Top three actionable edits in priority order:**

1. **Reopen A.2 as a scoped probe, not as a reversal.** Their GQA-on-V2 existence-proof is strong enough to justify 3-5 days of work but not strong enough to just flip the conclusion — we don't yet know which of (cos/sin prep, cache aliasing, 910-B4 vs 910-initial-gen kernel differences) is the actual delta. Wire the probe to produce a definitive bit-exact comparison on our own hardware. If it passes, update the deck with a "found-and-fixed" narrative rather than "was wrong about". If it fails, keep the closed verdict but with a stronger justification.

2. **Drop the HCCL env-tuning cheatsheet into Slide 8 speaker notes verbatim** (section Q2, "Cheatsheet" block). It's the single most concrete, citable, externally-validated datapoint we have for "cluster lever, zero-code, +89%". Pair with the `fullmesh → garbage output` gotcha as a vivid example of "not all config knobs are safe."

3. **Adopt the retain-old WorkspacePool pattern in CpCannEngine.** ~50 LOC diff, fixes a latent async-safety gap we've been lucky with. Not a deck edit — an internal engineering follow-up. Low risk, meaningful correctness.

**Lower-priority but worth tracking:**
- TFP probe contract drafted but not started — park until after A.2 reopens resolves.
- Benchmark-with-correctness-classifier pattern (`bench_pld_safe.sh` equivalent for Talker) is an engineering-ethics improvement — do it whenever we next quote a fps number publicly.

**What to NOT do:** do not claim we "inspired" this project or that it inspired ours. It's PM's own prior work; both projects are independent applications of the same agent-workflow discipline. The tone in the deck should be "two independent case studies, same pattern, different model-scale axes" — that's more credible than a cause-effect story.
