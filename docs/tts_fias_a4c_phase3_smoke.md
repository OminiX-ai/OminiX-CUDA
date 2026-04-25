# A4c Phase 3 — TTS prefill batched-FIAS transfer (NULL on user-facing axis)

**Author**: Agent (task #71)
**Date**: 2026-04-25
**Host**: ac01 (port 30410), 910B4, CANN 8.3.RC1
**Repo**: `~/work/OminiX-Ascend-w1` @ `ec89a5b` + worktree edit
**Binary**: `build-w1/bin/qwen_tts`
**Source**: ASR A4c Phase 3 batched-FIAS landed at `31da2bbc` (ASR mayun
RTF -12.4%, parity GREEN). Mission: transfer the same recipe to TTS
prefill.

---

## TL;DR

**Mechanism transfer GREEN** — prefill ms drops 125 → 69 (-44.6%) and the
batched FIAS path is **byte-identical** to the per-row baseline (sha256
match across 18/18 wavs, three prompt lengths × three reps × two arms).

**User-facing axis NULL** — total wall +2.3% lift (28.87 s → 28.21 s),
total RTF +0.2% lift, both at or below noise floor. **Below the +3%
ship gate.** Transfer recipe is sound but TTS user-facing latency is
dominated by codec autoregressive decode (~28 s of decode vs ~125 ms of
prefill), so prefill optimization does not move the metric the contract
gates on.

**Decision**: do not push code. Document as null. Patch retained on
ac01 worktree under `TALKER_TTS_FIAS_BATCHED` gate (default-off, alias
of `TALKER_W8_FIAS_BATCHED`); harvest later if a future encoder-prefill
contract surfaces.

---

## Methodology

Driver: `~/tts_regression_a4c/run_tts_a4c.sh`. Three canonical prompts
× three reps × two arms (baseline / batched), seed=42, greedy decode,
ellen_ref voice clone. Shipping flag combo on both arms:

```
TALKER_W8_QUANT=1
TALKER_CP_ACLGRAPH=1
TALKER_CP_INPLACE_ADDRMSNORM=1
TALKER_CP_POS_BATCH=1
```

Batched arm adds `TALKER_TTS_FIAS_BATCHED=1`. The new env name is an
alias of the existing `TALKER_W8_FIAS_BATCHED` flag — the engine ORs
both env vars, so unset (or both `0`) preserves byte-identical per-row
FIAS as required by the 32.2 fps TTS gate.

Prompts:
- short: `"The quick brown fox jumps over the lazy dog."` → 2.68 s audio
- medium: `"Artificial intelligence has transformed..."` → 9.72 s audio
- long: `"It is a truth universally acknowledged..."` → 18.52 s audio

Receipts: `~/tts_regression_a4c/{baseline,batched}_{short,medium,long}_r{1,2,3}.{log,wav}` + `matrix.log`.

## Results

### Aggregate (n=9 per arm)

| Metric | Baseline | Batched | Lift | Gate |
|---|---|---|---|---|
| Prefill (ms) | 125.4 | 69.4 | **-44.6%** | (informational) |
| Total wall (s) | 28.87 | 28.21 | -2.3% | < -3% required → **MISS** |
| Inference RTF | 3.223 | 3.204 | -0.6% | < -3% required → **MISS** |
| Total RTF | 3.873 | 3.866 | -0.2% | < -3% required → **MISS** |

### Per prompt

| Prompt | Audio (s) | Baseline prefill (ms) | Batched prefill (ms) | Baseline total (s) | Batched total (s) |
|---|---|---|---|---|---|
| short | 2.68 | 123.7 | 69.0 | 17.50 | 17.69 |
| medium | 9.72 | 122.3 | 68.0 | 27.86 | 28.33 |
| long | 18.52 | 130.3 | 71.3 | 41.24 | 38.61 |

(Per-prompt total deltas are within run-to-run variance; the long prompt
batched mean is pulled down by an r2 outlier at 34.72 s vs baseline-long
median 41.23 s.)

## Parity (ear-check / sha256)

All 18 baseline / batched wav pairs are byte-identical at every prompt
× rep cell:

| Prompt | sha256 prefix | size (B) | Pairs identical |
|---|---|---|---|
| short × r1-r3 | `29ae3f49a4af14f0` | 128684 | 6/6 |
| medium × r1-r3 | `21be5abfae28d04e` | 466604 | 6/6 |
| long × r1-r3 | `d47215cc9830f9b0` | 889004 | 6/6 |

`fias_batched=1` is confirmed in batched logs (each run hits "A4c Phase
3 ENABLED: batched prefill FIAS"). Parity gate: **GREEN**.

## Why TTS doesn't lift like ASR did

ASR A4c Phase 3 lifted mayun_ref RTF -12.4% because ASR end-to-end is
~140 ms of which prefill FIAS was ~103 ms — collapsing the per-row loop
freed ~88 ms, which is ~63% of total. On the same architecture (28L ×
2048 × 16Q/8KV GQA) at TTS shapes the prefill saving is the same ~55 ms
of absolute time, but TTS total is ~28 s dominated by the codec head's
14-deep autoregressive loop per audio frame. ~55 ms of ~28000 ms = 0.2%.
The mechanism is correct; the lever is in the wrong place for the user-
facing metric.

This matches the bottleneck breakdown already in
`docs/qwen_tts_optimization_writeup.md` (Code Predictor 46 ms/frame =
60% of frame budget; talker LLM decode 17 ms/frame = 22%; prefill is
amortised across the entire utterance in a single one-shot dispatch).

## Disposition

- Patch (env-alias only, ~25 LoC at one site in `talker_cann_engine.cpp`)
  is retained as a worktree edit on `ac01:~/work/OminiX-Ascend-w1`
  rather than committed. If later contract work needs a TTS-named
  prefill knob (e.g. encoder-prefill streaming), the alias re-lands as
  one commit; until then there is no ship benefit.
- The `TALKER_W8_FIAS_BATCHED` flag (already on the fork) covers the
  codepath; the alias rename was cosmetic.
- No follow-up gate. ASR A4c Phase 3 ship verdict in `asr_a4c_gate.md`
  remains the canonical use of this mechanism.

## References

- ASR A4c Phase 3 receipt: `docs/asr_a4c_gate.md`
- Phase 3 commit: `31da2bbc`
- Bottleneck breakdown: `docs/qwen_tts_optimization_learnings.md`,
  `docs/qwen_tts_optimization_writeup.md`
- Driver + raw logs: `ac01:~/tts_regression_a4c/`
