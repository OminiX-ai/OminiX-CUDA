# G0 — aclGraph feasibility probe for Qwen3-TTS CP forward

**Date**: 2026-04-21.
**Scope**: feasibility only — go/no-go verdict, no production code.
**Target**: replace the ~80-op aclnn dispatch chain of `CpCannEngine::forward_one_token_launch` with a one-time capture + per-frame replay, on **CANN 8.3.RC1** (ac01 toolkit).

## Verdict: **CONDITIONAL-GO**

The API is present, the semantic model fits our shape-stable / pointer-updating decode, a production reference (vllm-ascend) already does this on CANN, and the expected upside is in the right order of magnitude once the per-frame math is corrected. The conditionality is on the **task-update** sub-API (`aclmdlRICaptureTaskUpdateBegin/End`) actually accepting rebinding of our two dynamic pointers (cos/sin slice, KV-slot write). That is the single remaining unknown and it should be answered by a 1-day G1 smoke (one-layer capture + 2-pos replay).

## G0.1 — API presence (HARD GATE, PASS)

CANN 8.3.RC1 ships the full `aclmdlRI*` family in `<acl_rt.h>`. All symbols resolve from `/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so`:

```
nm -D libascendcl.so | grep -c aclmdlRI  →  38
```

The header (`include/acl/acl_rt.h` lines 269–309, 3210–3393) exposes:
- **Stream-capture family**: `aclmdlRICaptureBegin/End/GetInfo/ThreadExchangeMode`, with three modes (`GLOBAL`, `THREAD_LOCAL`, `RELAXED`).
- **Replay family**: `aclmdlRIExecuteAsync` (stream replay), `aclmdlRIExecute` (blocking).
- **Parameter-update family** (critical): `aclmdlRICaptureTaskGrpBegin/End` to wrap a sub-range of the capture as an updatable task group, and `aclmdlRICaptureTaskUpdateBegin/End` to re-bind the tensors of a captured task group on a live replay stream. This is the semantic we need.
- **Manual-build family**: `aclmdlRIBuildBegin/End/BindStream/EndTask` for offline model construction (not our path).

Our `cp_cann_symbols.cpp` already `dlsym`s 4 of these optionally (lines 166–173): `aclmdlRICaptureBegin/End`, `aclmdlRIExecuteAsync`, `aclmdlRIDestroy`. Extending that list to the task-group + task-update symbols is one-line additions.

There is no separate `aclGraphCapture*` family; on CANN 8.3 the "aclGraph" is the `aclmdlRI` opaque type. 8.5.0 headers (side-installed at `/home/ma-user/Ascend/cann-8.5.0/`) keep the same API.

## G0.2 — Semantic match (PASS with one open question)

- **Two-phase aclnn capture**: vllm-ascend [ACL Graph guide](https://docs.vllm.ai/projects/ascend/en/latest/developer_guide/feature_guide/ACL_Graph.html) captures `torch.compile`-emitted aclnn ops that go through `GetWorkspaceSize + Execute`. This proves the capture records the `Execute` launch with its workspace/executor bindings; the preceding `GetWorkspaceSize` call is host-side planning and is simply re-executed outside the captured region. No special handling required — our existing `CANN_OP` macro already separates the two phases.
- **Parameter rebinding across replays**: vllm-ascend's `update_graph_params` mutates `seq_lens` and `block_table` on paged-attention nodes between replays without re-capture; they use `aclmdlRICaptureTaskUpdateBegin/End` on a separate update stream to rewrite the captured task arguments. This is **exactly** the primitive we need for CP: on every frame, only three sets of pointers change — the KV write slot, the RoPE cos/sin slice, and the seq-length scalar consumed by `aclnnFusedInferAttentionScoreV2`. All else (weight pointers, residual/normed/cur buffers, workspace) is constant for the life of the engine.
- **Dynamic shapes**: not required. `forward_one_token` shape set = `{ pos ∈ [0, MAX_SEQ) }` and only the `seq_len` passed into FIAv2 and the KV-slot offset vary. Either (a) we capture once with `pos=0` and use task-update to rebind both, or (b) we keep a small cache of per-`pos` captured graphs (~17 graphs, one per position up to our 120-frame budget — trivial RAM). (a) is strictly better if task-update accepts FIAv2 int64 scalars; (b) is the always-works fallback.
- **Stream/event semantics**: capture records the exact stream ordering, and replay preserves it (header spec: replay reissues the same task DAG on the bound stream). Our `forward_done_event_` recording at the tail of `forward_one_token_launch` is a plain `aclrtRecordEvent` — capturable. The `aclrtStreamWaitEvent` at the head consumes an external event (the W1 lm-head dispatch's completion); this is capturable if recorded in `GLOBAL` capture mode, otherwise `RELAXED` is the safe setting.

**Open question**: whether `aclmdlRICaptureTaskUpdateBegin/End` will accept re-binding the FIAv2 `seq_len` scalar and the `cos_pos/sin_pos` strided tensor pointers. The header doesn't specify; vllm-ascend's use case rebinds `seq_lens` on paged attention, which is conceptually identical. Confidence: high, but the G1 smoke test below pins it down.

## G0.3 — Engine fit

Audit of `forward_one_token_launch` (W8 + fusion enabled — the production path at `cp_cann_engine.cpp:1479`). Per-forward op inventory, 5 layers:

| Op                                     | Count/forward | Category                                                       |
|----------------------------------------|---------------|----------------------------------------------------------------|
| `aclnnMm` (proj_w · x)                 | 1             | pure (fixed weights, fixed staging buffer)                     |
| `aclnnInplaceAdd` (+ proj_b)           | 1             | pure                                                           |
| `aclnnCast` (F32→F16 entry)            | 1             | pure                                                           |
| `aclrtMemcpy` H2D (input embed)        | 1             | **param-update**: host ptr changes per frame                   |
| `aclrtMemcpyAsync` D2D (residual=cur)  | 1–5           | pure (device ptrs fixed)                                       |
| `aclnnRmsNorm` (input_ln, layer 0)     | 1             | pure                                                           |
| `aclnnWeightQuantBatchMatmulV3` (Q/K/V)| 15            | pure                                                           |
| `aclnnRmsNorm` (q_norm, k_norm)        | 10            | pure                                                           |
| `aclnnRotaryPositionEmbedding` (Q,K)   | 10            | **param-update**: cos/sin slice ptrs change per `pos`          |
| `aclrtMemcpyAsync` D2D (V→cache slot)  | 5             | **param-update**: dst offset `k_cache_dev_ + pos * kv_dim`     |
| `aclnnFusedInferAttentionScoreV2`      | 5             | **param-update**: K/V tensor stride over `seq_len`, scalar `seq_len` |
| `aclnnWeightQuantBatchMatmulV3` (O)    | 5             | pure                                                           |
| `aclnnAddRmsNorm` (post-attn, fused)   | 5             | pure                                                           |
| `aclrtMemcpyAsync` D2D (residual=cur)  | 5             | pure                                                           |
| `aclnnWeightQuantBatchMatmulV3` (G/U)  | 10            | pure                                                           |
| `aclnnSilu`                            | 5             | pure                                                           |
| `aclnnInplaceMul` (gate *= up)         | 5             | pure                                                           |
| `aclnnWeightQuantBatchMatmulV3` (Down) | 5             | pure                                                           |
| `aclnnAddRmsNorm` (post-FFN, fused)    | 5             | pure                                                           |
| `aclnnCast` (F16→F32 output)           | 1             | pure                                                           |
| `aclrtRecordEvent` (tail)              | 1             | pure (event handle is fixed at engine init)                    |
| **Total aclnn + memcpy**               | **~96**       | 80 aclnn + 11–16 memcpy                                        |

All "param-update" entries are addressable:

- **KV-slot write (V-copy memcpy)**: replace the per-frame dst-offset calculation with a pre-allocated slotted address table. Capture a memcpy whose dst is a *fixed* scratch slot; after capture, use `aclmdlRICaptureTaskUpdateBegin` to swap the dst to `k_cache_dev_[il] + pos * kv_dim`. Workaround if task-update rejects scalar dst updates: **pre-slotted cache** — capture one graph per `pos` (up to MAX_POS=17 for a 120-frame cp_groups=15 run amortized across groups, actually per-position within a group, so only 17 graphs).
- **RoPE cos/sin pointers**: identical pattern — capture with `pos=0`, rebind stride slice on update. Fallback = pos-keyed graph cache.
- **FIAv2 seq_len + K/V strides over seq_len**: this is the riskiest rebind. The `seq_len` is passed as a plain `int64_t` into `GetWorkspaceSize`, which produces a cached `aclOpExecutor`; the captured graph holds that executor. If FIAv2's executor is seq_len-templated, we must capture one graph per `pos`. vllm-ascend handles exactly this for paged attention via `update_graph_params`, so the primitive exists; whether FIAv2 specifically is in scope is the G1 smoke test.
- **H2D memcpy of talker input**: trivially addressable — pre-stage the input into a fixed dev buffer via `aclrtMemcpy` *outside* the captured region, then let the captured region consume that buffer. Zero cost.

**Not-capturable ops**: none identified. All host-side branches (`w8_applied_`, `cp_fusion_applied_`, `cp_ascendc_applied_`) are build/init-time config and don't vary across frames.

## G0.4 — Verdict, effort, expected upside, risks

### Expected fps math (prompt-challenge reconciled)

Contract thesis was "~80 aclnn × 0.1 ms = 8 ms dispatch/forward × 17 forwards/frame = 136 ms/frame". That 17-forward figure is **wrong**: one CP frame = 15 `forward_one_token` (one per cp_group), not 17. Also, 0.1 ms/dispatch is the **eager-mode** figure; production runs with `TASK_QUEUE_ENABLE=2` (two-phase submit), which already amortizes most host-side launch overhead. The W3b commit-message data point is explicit: "each saved launch costs ~2-3 μs on TQE=2 rather than ~40 μs eager" — a **~15x** reduction already baked in.

Realistic model for CP forward wall-time on TQE=2:
- 80 aclnn × ~3 μs/dispatch host = ~0.24 ms host overhead → **masked by kernel exec** because TQE=2 overlaps queue-tile with prior kernel.
- Reported per-forward wall: 17–18 ms (W3b ON, cp_cann_engine measurements). The bulk is actual cube/vector time on small matmuls + FIAv2, not dispatch.

**So the kill-the-dispatch-floor thesis has already been 90% paid for by TQE=2.** aclGraph would recover the remaining host overhead and, more interestingly, the *inter-op scheduling gaps* that TQE=2 cannot collapse (dependencies that force queue drain). Expected win per forward: **0.5–1.5 ms**, so **7–22 ms/frame** (×15 forwards). At the 30.5 fps (33 ms/frame) baseline: **+6 to +10 fps**, landing at 36–40 fps. This is the best realistic band; it assumes task-update is not a deal-breaker.

### Effort to a runnable first-cut

**4–6 engineering days to `TALKER_CP_ACLGRAPH=1` behind an env gate, single-utt.**

- Day 1: extend `cp_cann_symbols.cpp` with task-group / task-update symbols; add `has_aclgraph_update()` gate. Smoke-capture a trivial 3-op sequence and replay 10× on ac01.
- Day 2: G1 one-layer smoke — capture one transformer layer at `pos=0`, replay with task-update rebinding cos/sin + KV-slot. Go/No-Go on the task-update primitive. If No-Go, pivot to pos-keyed graph cache (adds ~1 day).
- Day 3–4: full `forward_one_token` capture, replay loop with per-pos rebind or per-pos graph cache, parity check against the current path (dump + diff CP tokens, must be byte-identical at `--cp_greedy`).
- Day 5: fps measurement against the W1+W3b baseline on canonical mayun xvec; land env-gated, default off.
- Day 6 (buffer): KV-cache-append semantics on fresh `QwenTtsCtx` (re-capture cost vs. session reuse — this aligns with the deferred W3a session-mode contract).

### Risks

1. **Task-update rejects FIAv2 dynamic seq_len**: mitigated by pos-keyed graph cache (~17 graphs, ~MB of structural memory). Additive ~1 day.
2. **Shape stability**: none detected — CP is pure decode at `seq_len=pos+1`, `batch=1`, `head_dim=64`. KV cache layout is static. No risk.
3. **Bisheng / aclnnW8 kernel capturability**: W8 matmul dispatches the aclnn executor the same way as plain Mm; vllm-ascend captures paged attention which is a similarly compiled kernel, so high confidence. Smoke verifies.
4. **Stream/event ordering with W1 lm_head on the same stream**: our design shares `stream_` between CP forward and the W1 dispatches, so capture must open with `ACL_MODEL_RI_CAPTURE_MODE_GLOBAL` (or RELAXED) to tolerate the cross-stream W1 wait. Trivial fix.
5. **Re-capture cost on fresh `QwenTtsCtx`**: the reason W2.3 was deferred in the first place. One-time capture at engine init (pre-computed during the warmup path already in `QwenTTS::load`) makes this a per-process cost, not per-utt. No session-API dependency for *this* milestone as long as capture happens at init, not per-utt.

### Recommended next contract milestone (if GO)

**G1 — one-layer CP aclGraph smoke (2 days, ac01)**:

1. Extend `cp_cann_symbols.cpp` to resolve `aclmdlRICaptureTaskGrpBegin/End` + `aclmdlRICaptureTaskUpdateBegin/End` (optional, gated by `has_aclgraph_task_update()`).
2. Isolate one transformer layer (`il=0`) into a standalone capture harness inside `cp_cann_engine.cpp` behind `#ifdef TALKER_CP_ACLGRAPH_SMOKE`. Capture at `pos=0`, execute 10× with task-update rebinding cos/sin and KV-slot; verify output bit-identical (or within F16 float epsilon) against the stock path for all 10 positions.
3. Report: per-replay wall time, rebinding success/failure, go/no-go for full-forward capture.

Exit criterion: **≤1.5 ms/replay for 1 layer** (extrapolated target 7.5 ms/forward, vs current ~18 ms). If replay is slower than eager, the approach is a bust — abandon and escalate the deferred W3a session-aclGraph thinking instead, or fund Path C kernel authoring properly.

## Summary

aclGraph (via `aclmdlRI*`) is present on CANN 8.3.RC1 with all needed symbols including the task-update primitive vllm-ascend uses in production. The CP `forward_one_token` op set is shape-stable and has exactly three parameter-update points, all well-matched to the task-update semantic. The original 8 ms/forward dispatch-floor claim is inflated by ~10x because TQE=2 already amortizes it, but the realistic upside is still **+6 to +10 fps** on the canonical 33 ms/frame baseline — enough to be worth the 4–6 day spike. **Recommend GO, gated on a 2-day G1 one-layer smoke that pins down the task-update semantic**; if G1 passes, fund the full-forward implementation behind `TALKER_CP_ACLGRAPH=1`.
