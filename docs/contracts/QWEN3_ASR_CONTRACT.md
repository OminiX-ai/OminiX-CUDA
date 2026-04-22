# Qwen3-ASR Optimization on Ascend

## 1. Status & mandate

**Status**: NEW (drafted 2026-04-22, PM pending sign).
**Target**: port Qwen3-ASR to OminiX-Ascend native engine and drive RTF (real-time factor) down to production-serving thresholds with WER parity to the llama.cpp / reference baseline.

**Goal (subject to A0 discovery calibration)**:
- **A) First-landing arc**: conventional stack (llama.cpp + ggml-cann) baseline → native `AsrCannEngine` operating at clean-quality WER parity
- **B) RTF < 0.5** on canonical Chinese + English clips at single-card 910B4 (bounded by discovery; may calibrate to 0.3 or 0.2 depending on model size class)
- **C) Full arc ≈ 10-30× from conventional stack** (extrapolating Qwen3-TTS playbook: ~1 t/s → 32 t/s = ~32×; ASR structurally similar decode pattern if autoregressive)

**PM role**: supervise, gate, arbitrate. PM does not write kernel code. Every milestone below is an agent deliverable with numeric gates.

## 2. Background

Qwen3-ASR is Qwen team's speech recognition model family. Exact architecture / size class / tokenizer specifics need A0 discovery (model not yet local on dev host per the Q0-style audit pattern). Likely structural assumptions pending confirmation:

- **Encoder**: Conformer or transformer-based acoustic encoder (ingests mel-spectrogram, emits hidden states)
- **Decoder**: autoregressive transformer decoder with cross-attention to encoder hidden states (emits text tokens)
- **Quantization**: likely has BF16 reference + Q8_0/Q4_K_M quant variants on HF (matches Qwen family pattern)
- **Metric**: **WER** (word / character error rate vs reference transcript) for quality, **RTF** = wall_time / audio_duration for speed. RTF < 1.0 = faster than realtime. Production streaming typically targets RTF < 0.3 on 910B.

**Prior art in OminiX ecosystem**:
- `funasr-qwen4b-mlx` (OminiX-MLX) — Qwen4B ASR on Apple Silicon. Related family, different stack.
- `qwen3-tts-mlx` (OminiX-MLX) — TTS counterpart, decode pattern transferable.
- Qwen3-TTS on Ascend (this project) — autoregressive decode playbook proven (~1 fps → 32.2 fps).

**Why ASR is different from TTS structurally**:
- TTS output is fixed-rate (codec frames); ASR output is variable-rate (stops at punctuation / EOS)
- TTS has encoder-once-per-utterance + decoder-per-frame; ASR has same pattern (encoder-once, decoder-per-token)
- ASR encoder does large mel-spectrogram → hidden states transform (one-shot, heavy but amortized)
- ASR decoder is autoregressive transformer like TTS Talker (28 layers pattern may recur)
- ASR quality gate is WER, not user-ear — **automatable quality metric** (huge improvement vs TTS's subjective gate)

## 3. Scope

**In scope**:
- A0: discovery probe — model size, architecture, weight availability, current ggml-cann baseline
- A1: native `AsrCannEngine` bring-up (escape llama.cpp hot path)
- A2: standard kernel fusion playbook — AddRmsNorm (W3b analog), aclGraph pos-keyed replay
- A3: NPU port for any CPU-resident op (lm_head analog / output projection)
- A4: W8 quantization of decoder matmuls
- A5: encoder optimization (mel-spec prep + encoder transformer)
- A6: beam search / sampling on NPU (if ASR uses beam decode)
- A7: streaming support (if target workload is live transcription)
- A8: quality gate (WER ≤ reference) + RTF gate (< target)

**Out of scope**:
- Multi-language tokenizer work (use Qwen's shipped tokenizer)
- Domain-specific fine-tuning
- Cluster TP (future contract; single-card first)
- Punctuation restoration / post-processing (use stock)

## 4. Host plan

- **Primary**: ac01 (port 31984), 910B4, CANN 8.3.RC1. Existing native build infrastructure + WSPOOL / aclGraph / W8 patterns from Qwen3-TTS transferable.
- **Secondary**: ac02 / ac03 (both 910B4) available for parallel agent work and benchmark variance runs.
- **Fork**: `ymote/OminiX-Ascend` `main`. Patch-file push mechanism per convention.
- **Reference dataset**: A0 discovery defines — likely LibriSpeech test-clean (English) + AISHELL-1 or WenetSpeech test (Chinese). Standard public ASR benchmarks for WER comparison.

## 5. Workstreams with gates

### A0 — Discovery probe (1-2 days, HARD GATE before A1 funding)

Same pattern as Q0 (QIE discovery):

- [ ] A0.1 Model discovery: Qwen/Qwen3-ASR-* on HF. Size class (1B / 4B / 14B?), architecture config (n_layers, hidden, heads, vocab), tokenizer.
- [ ] A0.2 HF cache state on ac01: is the model + weights already downloaded? If not, what's the bandwidth cost?
- [ ] A0.3 Existing tooling: does OminiX-Ascend `tools/` have any ASR scaffolding? Is there a `tools/asr/` or similar?
- [ ] A0.4 Baseline path: does ggml-cann support ASR? Can the current llama.cpp path on ac01 run Qwen3-ASR at all? If yes, measure RTF. If no, note what's missing.
- [ ] A0.5 Reference dataset: smallest standard set (e.g. 10 clips of LibriSpeech test-clean + 10 from AISHELL-1) that gives stable WER measurement.

**Gate**: A0 deliverables → discovery report at `docs/asr_a0_discovery.md`. PM reviews, confirms or adjusts:
- Target RTF (calibrated to model size class)
- First-landing scope (full engine rewrite vs incremental stages)
- Reference dataset

### A1 — Native `AsrCannEngine` bring-up (2-3 weeks, gated on A0 green)

Mirror `CpCannEngine` / `TalkerCannEngine` patterns. Build the decoder forward pass dispatching aclnn directly, bypassing ggml-cann.

- [ ] A1.1 Scaffold `tools/asr/asr_cann_engine.{h,cpp}` + `asr_cann_symbols.{h,cpp}` (dlsym pattern per CP convention)
- [ ] A1.2 Encoder forward (likely already mature in MLX counterpart; port with minimal changes)
- [ ] A1.3 Decoder `forward_one_token_launch` equivalent (Talker-analog — autoregressive transformer w/ cross-attn to encoder hidden)
- [ ] A1.4 Weight loader from GGUF (reuse llama.cpp GGUF parser; don't reinvent)
- [ ] A1.5 End-to-end smoke: transcribe 1 clip, compare to llama.cpp reference transcript
- [ ] A1.6 Quality gate: exact-match transcript on 5 canonical clips (or ≤ 1-token drift at greedy decode)

**Gate**: end-to-end transcription works with native engine. WER within ±1% of llama.cpp baseline on 20-clip reference set.

### A2 — Standard playbook — fusion + capture (1-2 weeks)

Transfer the Qwen3-TTS pattern:

- [ ] A2.1 `aclnnAddRmsNorm` fusion at decoder tail (W3b analog)
- [ ] A2.2 `aclnnApplyRotaryPosEmbV2` if architecture has RoPE (probe first; Qwen3 family typically does)
- [ ] A2.3 aclGraph pos-keyed capture (G2-analog): decoder pos 0..MAX_TOKENS, capture-at-init
- [ ] A2.4 WSPOOL async-safe workspace (already in fork; reuse pattern)
- [ ] A2.5 `aclnnInplaceAddRmsNorm` (A.1-analog) if A2.1's op variant exists

Each sub-gate: frame-count identity equivalent (for ASR: **token-count identity** at greedy decode vs stock), WER ≤ baseline + 0.1%, RTF improvement measured.

### A3 — NPU port for CPU-resident ops (1 week)

Equivalent of W1 (Qwen3-TTS lm_head):

- [ ] A3.1 Audit ASR decoder's output projection (vocab head) — is it currently CPU matvec? If yes, port to NPU.
- [ ] A3.2 Audit any text-prep / tokenizer-adjacent ops on CPU that could move to NPU
- [ ] A3.3 Any encoder-output post-processing on CPU

Gate: RTF improvement proportional to frame-fraction of op moved; WER unchanged.

### A4 — W8 quantization (1-2 weeks)

- [ ] A4.1 Audit decoder matmuls for W8 eligibility (same pattern as TTS CP — `aclnnWeightQuantBatchMatmulV3`)
- [ ] A4.2 Per-channel INT8 weight calibration (reuse Qwen3-TTS calibration script if compatible)
- [ ] A4.3 Encoder matmuls — W8 often impacts encoder audio-feature extraction more than decoder; probe carefully
- [ ] A4.4 WER gate: within ±0.3% absolute of F16 baseline

### A5 — Encoder optimization (1-2 weeks)

ASR encoders are typically 12-24 transformer layers processing ~5-50 sec of mel-spec. One-shot per utterance so less dispatch-bound than decoder but potentially more compute-heavy.

- [ ] A5.1 Profile encoder: which ops dominate?
- [ ] A5.2 aclGraph for encoder (shape-stable per utterance within a mel-length bucket)
- [ ] A5.3 Flash-attention variants for encoder self-attention (vendor aclnnFIA V2/V3/V4)
- [ ] A5.4 Mel-spec compute on NPU (often on CPU today) — audit worth-it

### A6 — Beam search / sampling (1 week, conditional)

Only if Qwen3-ASR uses beam decoding (CTC+attention joint decode or pure beam attention). If greedy is production-enough, skip.

- [ ] A6.1 Move beam-search state management to NPU (tensor ops instead of host loops)
- [ ] A6.2 Score-aggregation fusion

### A7 — Streaming support (2-3 weeks, conditional)

If live-transcription target. Otherwise skip.

- [ ] A7.1 Chunked encoder (hopping-window) with cross-chunk attention state
- [ ] A7.2 Incremental decoder (stream tokens as encoder advances)
- [ ] A7.3 Latency gates: first-token-latency < 500 ms, streaming RTF < 0.5

### A8 — Final gates (HARD KILL)

- [ ] A8.1 **WER gate**: ≤ baseline + 0.3% absolute on reference dataset, over min 20 clips
- [ ] A8.2 **RTF gate**: ≤ target (A0-calibrated, likely 0.3-0.5) on same dataset
- [ ] A8.3 **Character-level drift** (Chinese): ≤ 1 character diff per 100 characters vs reference on AISHELL or equivalent
- [ ] A8.4 **Edge cases**: noisy audio, code-switch (English+Chinese), speaker accents — WER doesn't explode
- [ ] A8.5 Tag fork at `qwen3-asr-rtf-<N>-landed`

## 6. Acceptance criteria (summary)

- [ ] A0 discovery report; target RTF calibrated
- [ ] A1 native engine handles end-to-end transcription; WER parity with baseline
- [ ] A2+A3+A4 stacked: each lands with token-count identity + WER gate
- [ ] A5 encoder optimized if encoder is ≥ 30% of total wall
- [ ] A8 final gates cleared
- [ ] Contract stamped with commit SHAs + Verified-by per milestone

## 7. Risks

1. **A0 may reveal ASR is less dispatch-bound than TTS** — encoder's one-shot compute may dominate. RTF improvements from dispatch-kill patterns (aclGraph, fusion) may be smaller. Mitigation: A0's profile breakdown. If encoder dominates, pivot emphasis to encoder optimizations (A5) rather than decoder patterns.
2. **WER regression under W8 or V2 RoPE** — ASR is famously quant-sensitive (especially for low-resource languages / acoustic edge cases). Mitigation: WER gate at each stage; fallback to F16 if drift > 0.3% abs.
3. **Tokenizer coverage** — Qwen3-ASR's tokenizer must handle the benchmark datasets' label vocabulary. If tokenizer mismatches, WER number lies.
4. **Streaming scope creep** — streaming (A7) adds 2-3 weeks; easy to demand and hard to deliver. Mitigation: keep A7 conditional, phase separately from batch A1-A8.
5. **Encoder optimization regression** — mel-spec NPU port or encoder aclGraph may regress accuracy if mel-spec floats drift. WER gate mandatory.
6. **Beam-search complexity** — if required, beam search on NPU is its own multi-week track. Mitigation: verify greedy decode is production-enough; only scope A6 if not.

## 8. Timeline (agent-wall)

- A0: 1-2 days
- A1: 2-3 weeks (biggest single task; first-native port)
- A2: 1-2 weeks (playbook reuse)
- A3: ~1 week
- A4: 1-2 weeks (calibration + validation)
- A5: 1-2 weeks
- A6: 1 week (conditional)
- A7: 2-3 weeks (conditional — streaming)
- A8: 1 week gates

**Total**: **~6-10 weeks agent-wall** for full A0-A8 batch (skipping A6/A7). With streaming: **10-14 weeks**.

**First-value milestone** (A0 + A1 + A2): ~4-5 weeks — demonstrable native engine with standard-playbook perf lift.

## 9. Host rules

- Patch-file push mechanism (ac01 no fork push creds)
- No Claude coauthor
- HARD KILL at WER regression + RTF regression gates
- WER-gate outranks RTF — quality regression never ships (same discipline as user-ear for TTS but now automated via WER)
- Probe-first on any vendor fused op after FFNV3 / V2 RoPE saga: 30-60 min header + dtype probe before wiring, per Slide 9.5 orchestration mode #1
- Reuse Qwen3-TTS learnings: frame-count identity gate → **token-count identity gate**, universal and mandatory
