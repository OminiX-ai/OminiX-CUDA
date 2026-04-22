# ASR Tier-1 Gate — TTS → ASR Self-Consistency Harness (Design Spec)

**Author**: Agent A0
**Status**: DRAFT — design only, no implementation yet
**Context**: `QWEN3_ASR_CONTRACT.md` §A0.5 — fast iteration loop regression gate

## 1. Purpose

Every agent commit on the ASR optimization track must prove it does not regress
transcription quality. A real-dataset gate (LibriSpeech / AISHELL) is too heavy
to run per-commit. This harness produces a deterministic, offline, zero-network
regression signal in < 60 seconds per iteration by looping TTS synthesis →
ASR transcription and measuring divergence from the *known* source text.

Principle: if WER/CER on TTS-synthesised audio drifts by > 0.5% abs between
two ASR builds, the newer build regressed (or the hypothesis needs a
Tier-3 human spot-check per §A0.7).

## 2. Canonical sentence set (20 utterances)

Versioned in-repo as `tools/asr/harness/canonical_sentences.yaml`. Rotating
through 4 domains × 5 sentence-length buckets.

**Chinese (10)**:

| # | Domain | Text | Target length |
|---|--------|------|---------------|
| 1 | News | "国务院今天召开常务会议，部署优化营商环境措施。" | short |
| 2 | News | "据中国气象局最新预报，未来三天北方地区将出现大范围降雪天气。" | medium |
| 3 | Conversation | "你好，请问最近的地铁站怎么走？" | short |
| 4 | Conversation | "我昨天晚上看了一部特别好看的电影，讲的是人工智能的故事。" | medium |
| 5 | Technical | "该模型采用多头注意力机制，隐藏层维度为二千零四十八。" | medium |
| 6 | Technical | "梯度下降算法收敛速度取决于学习率和批大小的选择。" | long |
| 7 | Numerics | "订单编号是八四三六二七，总金额一万两千三百五十元。" | medium |
| 8 | Code-switch | "我在GitHub上提了一个pull request，希望reviewer尽快approve。" | medium |
| 9 | Long-form | "深度学习模型在计算机视觉、自然语言处理和语音识别领域都取得了突破性进展，但对算力的需求也随之增长。" | long |
| 10 | Dialect cue | "今朝外头落雨伐，要勿要带伞出门？" (Wu dialect phrasing) | short |

**English (10)**:

| # | Domain | Text | Target length |
|---|--------|------|---------------|
| 11 | News | "The Federal Reserve announced a quarter-point rate cut today." | short |
| 12 | News | "Scientists at CERN have detected a new signal consistent with beyond-standard-model physics." | long |
| 13 | Conversation | "Hey, could you pass me the salt please?" | short |
| 14 | Conversation | "I was thinking we could grab coffee after the meeting if you have time." | medium |
| 15 | Technical | "The transformer uses grouped query attention with sixteen heads and eight kv-heads." | medium |
| 16 | Technical | "Gradient accumulation across eight steps effectively quadruples the batch size." | medium |
| 17 | Numerics | "The account balance is two thousand four hundred and eighty-seven dollars and sixty-three cents." | medium |
| 18 | Code-switch | "Let's merge this PR after we fix the 反向传播 计算 bug." | short |
| 19 | Long-form | "Large language models are now routinely deployed for summarisation, translation, retrieval-augmented generation, and multi-agent orchestration across enterprise systems." | long |
| 20 | Acronyms | "Please SSH into the ASCEND NPU and check the HBM utilisation via npu-smi." | short |

Rationale: covers short (< 2 sec audio), medium (3-6 sec), long (7-12 sec);
3 domains (news, conversation, technical); numerics, code-switch, acronyms,
one Wu-dialect edge. Total synth audio ~ 90-120 sec.

## 3. Voice rotation

Rotate through `tools/qwen_tts/data/ref_audios/` — 14 refs available:
`mayun_ref`, `maple_ref`, `trump_ref`, `ellen_ref`, `cove_ref`, `juniper_ref`,
`shenyi_ref`, `yangmi_ref`, `doubao_ref`, `luoxiang_ref`, `bys_ref`,
`zhoujielun_ref`, `mabaoguo_ref`, `test_ref`.

Assignment: deterministic voice = voices[i % 14] for sentence i (1..20), so
diff between two runs is reproducible.

## 4. Pipeline

```
for i in 1..20:
    text_i = canonical_sentences[i]
    voice_i = voices[(i-1) % 14]
    wav_i = TTS(text_i, voice_i)                # on TTS host
    # scp wav_i to ASR host if different
    hyp_i = ASR(wav_i)                          # on ac03
    metric_i = WER(text_i, hyp_i) if EN else CER(text_i, hyp_i)

aggregate:
    avg_wer_en  = mean(metric for EN sentences)
    avg_cer_cn  = mean(metric for CN sentences)
    max_per_sent = max(metric)                  # catches single catastrophic
    per_domain = groupby(domain).mean(metric)
```

## 5. Host topology

Constraint: ac03 is the ASR host. ac01 is TTS-primary but agent A0 is forbidden
to ssh ac01. Options:

**Option A (preferred)**: use ac01 TTS via a patch-file + PM-run. PM kicks off
`qwen_tts` on ac01, scp the 20 wavs to `ac03:~/asr_a0/harness/wavs/` once.
Thereafter the harness is self-contained on ac03 (wavs are static input).

**Option B (fallback)**: if ac03 has sufficient HBM headroom after ASR load
(910B4 32 GB; ASR is ~5 GB BF16 or ~2.5 GB Q8, TTS model is ~3 GB), run TTS
on ac03 too. Requires building `qwen_tts` binary on ac03, non-trivial.

**Decision**: go Option A. Wavs are static — we don't re-synthesize per
iteration. Only ASR runs per-commit.

## 6. Metrics

**English**: word error rate (WER) using `jiwer` library (already Python),
case-insensitive, punctuation-stripped, whitespace-normalised.

**Chinese**: character error rate (CER) after Unicode NFKC normalisation,
full-width/half-width punctuation folded, whitespace removed.

**Code-switch sentences**: computed both ways, flagged separately; use WER
for the English tokens + CER for the Chinese tokens.

**Threshold**: baseline_wer ± 0.5% absolute = green. Δ > 0.5% abs = RED,
halt commit.

## 7. Output artefacts

Per run:
```
reports/asr_tier1_<git-sha>_<timestamp>/
  summary.md             # avg wer/cer, per-sentence table, per-domain
  per_sentence.csv       # i, voice, domain, ref, hyp, metric
  diffs.json             # character-level diffs for inspection
```

`summary.md` diffs against the previous recorded baseline and emits
`GATE: PASS|FAIL`.

## 8. Runtime budget

- 20 sentences × avg 6 sec audio = 120 sec audio to transcribe
- Target RTF < 0.5 on NPU → ASR wall < 60 sec total
- Overhead (load, tokenise, metrics): < 30 sec
- **Total: < 90 sec per iteration**

## 9. Implementation dependencies (NOT delivered by A0)

- A0.3 confirms: `tools/qwen_asr/` scaffolding exists (llama.cpp + ggml-cann
  path); README documents Q8_0 + gpu_layers=28 = RTF 0.15 on 910B2
- A0.4 confirms: baseline binary builds on ac03 (CANN 8.3.RC1 matches ac01
  config; gcc 10.3.1 + cmake 3.22 sufficient; stub-lib workaround documented)
- Synth wavs come from ac01 (per host rule) — A1 agent must request PM kick
  to regenerate if canonical sentences change

## 10. Open questions (for PM decision at A0 review)

1. Do we commit the 20 wavs into the repo (LFS; ~3-5 MB)? Or regenerate each
   time the canonical sentence set changes?
2. Jiwer vs sclite for WER — jiwer is already installed; sclite gives more
   trusted numbers but requires C build. Start with jiwer, upgrade if
   audited.
3. Chinese CER: do we strip tone marks, standardise traditional→simplified?
   Recommend: stop at NFKC + whitespace + punctuation. Simplified-only for
   now (matches training).
4. Failure policy when ASR produces empty string (silence / catastrophic
   failure): count as 100% WER vs flag separately? Recommend flag.

## 11. Deliverable for A1 agent

- Implement `tools/asr/harness/` with:
  - `canonical_sentences.yaml`
  - `run_tier1.py` (ingests static wavs, runs ASR binary, computes metrics)
  - `baseline.json` (checked in; current reference WER per sentence)
- CI-style hook: `make asr-tier1` invokes `run_tier1.py`, exits non-zero on
  gate FAIL
