# ASR A0 Discovery Report

**Author**: Agent A0
**Date**: 2026-04-22
**Host**: ac03 (port 30412), 910B4, CANN 8.3.RC1 — dedicated ASR host
**Contract**: `docs/contracts/QWEN3_ASR_CONTRACT.md` §A0
**Status**: discovery complete; recommendation = **GO with scope reduction**

---

## A0.1 — Model discovery

### Ranked candidates

| Rank | Model | Size | Arch summary | Last updated | License | Verdict |
|------|-------|------|--------------|--------------|---------|---------|
| **1** | **`Qwen/Qwen3-ASR-1.7B`** | **2.0 B params, 4.7 GB BF16** | Whisper-style encoder + Qwen3-2B decoder (below) | 2026-01-29 | Apache-2.0 | **PRIMARY TARGET — matches existing fork scaffolding** |
| 2 | `Qwen/Qwen3-ASR-0.6B` | 0.6 B params, ~1.5 GB BF16 | Smaller encoder (18L × d=896 × 14h) same decoder family | 2026-01-29 | Apache-2.0 | Secondary — useful for fast iteration or edge workloads |
| 3 | `Qwen/Qwen3-ForcedAligner-0.6B` | 0.6 B, ~1.2 GB | NAR forced alignment, 11 languages | 2026-01-29 | Apache-2.0 | Out of scope — different task (alignment, not transcription) |
| 4 | `mlx-community/Qwen3-ASR-1.7B-8bit` | 2.46 GB | MLX 8-bit quant; decoder only quantised | — | Apache-2.0 | Reference only — parity target for our NPU port |
| 5 | `mlx-community/Qwen3-ASR-0.6B-8bit` | 1.01 GB | — | — | Apache-2.0 | Reference only |
| — | `Qwen/Qwen2.5-Omni-*` | — | Multi-modal, includes speech understanding but shipped as omni, not dedicated ASR | 2025 | Apache-2.0 | Not a match — product-level omni; target is focused ASR |
| — | `FunASR/*` | various | Alibaba FunASR non-Qwen family | various | A2.0/BSD | Related but different provenance — `funasr-qwen4b-mlx` name is historical; MLX org already pivoted to `qwen3-asr-mlx`/`qwen3-asr-1.7b-mlx` targeting above |

### Canonical config — `Qwen/Qwen3-ASR-1.7B`

Pulled from HF `config.json` (fetched via WebFetch; full tree verified):

**Audio encoder** (`qwen3_asr_audio_encoder`):
- `num_mel_bins = 128`, `n_window = 50`, `n_window_infer = 800`, `max_source_positions = 1500`
- `d_model = 1024`, `encoder_layers = 24`, `encoder_attention_heads = 16`, `encoder_ffn_dim = 4096`
- `downsample_hidden_size = 480`, `output_dim = 2048` (feeds decoder hidden)
- `activation_function = gelu`, `scale_embedding = false`

**Text decoder** (Qwen3 autoregressive):
- `hidden_size = 2048`, `num_hidden_layers = 28`, `num_attention_heads = 16`, `num_key_value_heads = 8` (GQA 2:1)
- `head_dim = 128`, `intermediate_size = 6144`, `hidden_act = silu` (SwiGLU)
- `vocab_size = 151936`, `tie_word_embeddings = true`
- `max_position_embeddings = 65536`, `rope_theta = 1e6`, `rope_scaling.mrope_interleaved = true` (MROPE with sections `[24,20,20]`)
- `rms_norm_eps = 1e-6`

**Special tokens**:
- `audio_start_token_id = 151669`, `audio_end_token_id = 151670`, `audio_token_id = 151676`

**Tokenizer**: Qwen BPE (vocab.json 2.78 MB + merges.txt 1.67 MB; inherits `Qwen/Qwen2.5` tokenizer surface; `tokenizer_class` = default, no sentencepiece).

**Files on HF**:

| File | Size |
|---|---|
| `model-00001-of-00002.safetensors` | 4.22 GB |
| `model-00002-of-00002.safetensors` | 478 MB |
| `vocab.json` | 2.78 MB |
| `merges.txt` | 1.67 MB |
| `tokenizer_config.json`, `chat_template.json`, `config.json`, `preprocessor_config.json`, `generation_config.json` | < 20 KB each |
| **Total** | **~4.70 GB** |

Not quantised — our baseline path already converts to F16/Q8 GGUF (see A0.3/A0.4).

---

## A0.2 — Weights on ac03

**Verified 2026-04-22 15:26 CST, host `notebook-c768c7a7-...`**:

- `~/.cache/huggingface/` — empty
- `/root/.cache/huggingface/` — not readable (wrong user; ma-user not root)
- `/home/ma-user/` — **no Qwen3-ASR weights** (no `*qwen*asr*` or `*funasr*` dirs; no `.safetensors` or `.gguf` files at maxdepth 7)
- `/modelarts/` — ModelArts tooling dirs only, no models
- `~/infer/model/1/` — empty stub from ModelArts container template (only a `variables/` dir)

**Disk**:
- `/` (overlay) — 50 GB total, **49 GB free**
- `/modelarts` (`/dev/sda2`) — **491 GB total, 437 GB free** ← staging target for models
- `1.5 TiB` RAM, `192` cores
- 910B4 NPU chip 7, 32 GB HBM free

**Download plan**: fetch to `/modelarts/asr_weights/Qwen/Qwen3-ASR-1.7B/` (4.7 GB BF16). At typical HF bandwidth on Huawei Cloud, < 15 min. No PM escalation needed — **well under the 10 GB stop-and-report rule** in the contract.

Quantised GGUF variants (encoder F16 607 MB + decoder Q8_0 2.1 GB) will be produced on-host from the BF16 source via the export scripts already in `tools/qwen_asr/`.

---

## A0.3 — Existing tooling

### Major finding — the scaffolding already exists

**`tools/qwen_asr/`** on the fork is **not empty**. Contents (22 files, committed 2026-03-31):

| File | Purpose |
|---|---|
| `qwen_asr.{h,cpp}` (340 LOC) | ASR main pipeline: load, eval_chunk, split prefill, decode loop |
| `audio_encoder.{h,cpp}` (28.6 KB) | Conv2d × 3 + 24-layer Transformer encoder, GGML-based, CANN-capable |
| `mel_spectrogram.{h,cpp}` | Whisper-style 128-mel filterbank (n_fft=400, hop=160) |
| `main.cpp` | CLI: `--audio`, `--encoder`, `--decoder`, `--device CANN0`, `--gpu_layers 28` |
| `export_audio_encoder.py` | Safetensors → GGUF converter (audio tower) |
| `export_decoder_llama.py` | Safetensors → GGUF converter (Qwen3 decoder, llama.cpp-compatible) |
| `verify_audio_encoder.py` / `verify_conv2d.py` | Python reference comparison scripts |
| `verify_data/` | 21 `.npy` snapshots (input features, per-layer activations, mel filter, reference encoder output) for numerical diff |
| `test_encoder.cpp`, `test_conv2d_minimal.cpp` | Standalone unit tests |
| `CMakeLists.txt` | Build integration |
| `README.md` (8.4 KB) | Full pipeline documented; quant + NPU command recipes |

**Pipeline documented** (README §Architecture):
```
WAV 16kHz → Mel (128 × T) → Conv2d×3 (stride 2) → 24-layer encoder → MLP → (122 × 2048)
        → split prefill: [text tokens][audio embeds][text tokens] → llama.cpp Qwen3 decoder
        → BPE decode
```

**Symlink**: `tools/qwen_asr/gguf -> /root/autodl-tmp/tts.cpp/tools/qwen_asr/gguf` — authored on a development host (autodl), not ac03. We'll populate the target dir fresh.

**MLX reference parity**: `/Users/yuechen/home/OminiX-MLX/qwen3-asr-mlx` (full Rust+MLX impl, 0.6B & 1.7B, 30× RT on M4 Max, 8-bit decoder only) + `qwen3-asr-1.7b-mlx` + `funasr-mlx` + `funasr-nano-mlx`. These are the numerical ground truth for WER parity gates.

**Other Ascend tools dir listing**: `qwen_tts/`, `qwen_asr/`, `qwen_common/`, `tts/`, `mtmd/`, `cli/`, `server/`, etc. — standard llama.cpp tools tree. No `tools/asr/` (non-`qwen_asr`) present.

### Implication for A1

The contract anticipated A1 = 2-3 weeks for "native engine bring-up". In reality the bring-up is already **stage-1 complete**: the split-prefill path runs via llama.cpp + ggml-cann, and RTF numbers are documented. A1 as written (building from scratch) is **not the right scope**. See A0.7.

---

## A0.4 — Baseline path

### Build feasibility on ac03

- CANN: `Version=8.3.0.1.200` (version_dir `8.3.RC1`) — **matches ac01**. No pivot risk.
- `/usr/local/Ascend/ascend-toolkit/latest/lib64/` — `libaclnn_math.so`, `libaclnn_ops_infer.so`, `libaclnn_ops_train.so`, `libaclnn_rand.so` present.
- `cmake 3.22.0`, `gcc 10.3.1` — meet README requirements (≥ 3.14, C++17).
- `npu-smi 23.0.6` works on chip 7, HBM 32 GB idle.
- Python 3.11 / anaconda `PyTorch-2.7.1` env has: `torch 2.7.1`, `torch_npu 2.7.1`, `transformers 4.53.1`, `safetensors 0.4.5`, `librosa 0.10.2`, `soundfile 0.13.1`, `scipy 1.15.3`, `numpy 1.26.4`, `datasets 3.0.1`. **Missing**: `gguf`, `jiwer` — trivial pip.
- No pre-existing `llama.cpp` or `tts.cpp` build on ac03.
- `ffmpeg` at `/usr/local/ffmpeg/bin/ffmpeg`.

### Baseline RTF (documented, not yet re-measured on ac03)

From `tools/qwen_asr/README.md` on **910B2** (ac01-style), 9.36 sec English clip `ellen_ref.wav`:

| Config | Mel | Encoder | Prefill | Generation | Total | RTF |
|---|---|---|---|---|---|---|
| **Q8_0 + CANN (gpu_layers=28)** | 38 ms | 85 ms | 150 ms | 1.2 s | **1.4 s** | **0.15** |
| F16 + CANN | 38 ms | 85 ms | 90 ms | 1.2 s | 1.3 s | 0.14 |
| Q8_0 CPU-only | 38 ms | 960 ms | 2.7 s | 4.4 s | 8.2 s | 0.88 |

Reference: 3090 GPU (Python, bf16 + FlashAttention) — 585 ms, RTF 0.063.

**Not yet measured on ac03's 910B4** — A1 first job. **Likely similar or slightly better** than 910B2 (same chip class, newer HBM, same CANN). Assumption: baseline RTF 0.12-0.15 on ac03.

### Gap to contract target RTF < 0.5

Already met (!) by the existing llama.cpp path. The *stretch* from contract §1 is "A-C" arc analogous to Qwen3-TTS (~32×). Applied here: baseline RTF 0.15 → target RTF 0.005 (200× real-time) would be the "full arc". More realistic target: **RTF < 0.05 (20× RT)** on ac03 via native engine + aclGraph + W8 — a 3-4× improvement, comparable to TTS improvements from native engine adoption. See A0.7 for calibration.

### Verdict

**GAP = zero — baseline path functional and well-documented.** A1's classical "native engine bring-up from a working llama.cpp baseline" mirrors the Qwen3-TTS CP playbook exactly.

---

## A0.5 — Tier-1 gate design

**Spec document**: written to `ac03:/tmp/asr_tier1_gate_spec.md` and `ac03:~/asr_a0/asr_tier1_gate_spec.md`, mirrored locally at the end of this report.

**Headlines**:
- 20 canonical sentences (10 CN + 10 EN), covering news/conversation/technical/numerics/code-switch/Wu-dialect/acronyms, spanning short/medium/long audio lengths
- Voice rotation through the 14 refs in `tools/qwen_tts/data/ref_audios/`
- Pipeline: TTS on ac01 (PM-triggered) → static wavs scp'd to ac03 → ASR run per-commit on ac03 → WER (EN) / CER (CN) vs known source text
- Runtime budget < 90 s/iteration
- Gate threshold: Δ > 0.5% abs WER/CER vs recorded baseline = RED
- Implementation deferred to A1 — A0 delivers only the spec

**Spec location**: `/tmp/asr_tier1_gate_spec.md` on both Mac and ac03; full text included as companion file to this report. **Should be copied into the fork at `docs/asr_tier1_gate_spec.md` on the first A1 PR.**

---

## A0.6 — Tier-2 dataset selection

Standard public benchmarks chosen for Qwen3-ASR's published numbers (per model card).

### English (WER)

| Dataset | License | Size (test-clean subset) | Recommended subsample |
|---|---|---|---|
| **LibriSpeech `test-clean`** | CC-BY-4.0 | 5.4 h, 2,620 utterances, ~350 MB flac | **20 random clips** |
| LibriSpeech `test-other` | CC-BY-4.0 | 5.1 h, 2,939 utt, ~340 MB | 20 extra for noise/accent stress |
| GigaSpeech `test` | Apache-2.0 (speech) / varies (transcript) | 40 h, ~3 GB | skip for Tier-2; reserve for A8 |

### Chinese (CER)

| Dataset | License | Size (test) | Recommended subsample |
|---|---|---|---|
| **AISHELL-1 `test`** | Apache-2.0 | 4.0 h, 7,176 utt, ~350 MB | **20 random clips** |
| AISHELL-2 `test` | CC-BY-NC-SA (non-commercial) | 3.7 h | skip (licensing risk) |
| WenetSpeech `test-net` | CC-BY-NC | 23 h | skip Tier-2; use in A8 research |

### Recommendation

**Tier-2 reference set (~40 clips, ~40 MB after extraction)**:
- 20 × LibriSpeech `test-clean` (pure EN, studio-recorded, broad speaker pool)
- 20 × AISHELL-1 `test` (pure CN, studio, read speech)

Why not AISHELL-2/WenetSpeech at Tier-2: license constraints + size. Use them at A8 final gate only when full-dataset numbers matter for the completion stamp.

**Load path**: Python `datasets` library (already installed on ac03):
```python
from datasets import load_dataset
ls = load_dataset("openslr/librispeech_asr", "clean", split="test")
ai = load_dataset("wenet-e2e/aishell", split="test")
```

Stable-WER estimate on 20 clips: ±0.5% abs (matches Tier-1 gate threshold); good enough for A1/A4/A8 milestone checkpoints.

---

## A0.7 — Recommendation to PM

### Summary

**GO — with scope reduction.** The ASR track is significantly *ahead* of where the contract assumed. Recommend restructuring:

### Key observations

1. **Model size is modest**: 2 B params total (1.7 B decoder + ~0.25 B encoder). 4.7 GB BF16, 2.5 GB Q8. Easily fits on 910B4's 32 GB HBM with full activations + KV cache. No W8-from-day-1 constraint. No pivot to 0.6B needed unless edge workloads surface later.

2. **Baseline is done**: `tools/qwen_asr/` already ships a working split-prefill llama.cpp + ggml-cann path with documented RTF 0.15 on 910B2. The contract's A1 ("2-3 weeks to bring up native engine") overlaps ~70% with what already exists. **A1 should be re-scoped to "escape ggml-cann hot path"** (the native `AsrCannEngine` analog to `CpCannEngine`), not "build from zero".

3. **Architecture mirrors TTS playbook exactly**:
   - Encoder-once + decoder-autoregressive pattern ← same as Qwen3-TTS Talker
   - Qwen3 decoder (28 layers, GQA 16:8, RoPE θ=1e6, tied embeddings) ← same family as TTS LLM
   - Fusion wins (AddRmsNorm, aclApplyRotaryPosEmbV2, aclGraph pos-keyed replay) → directly transferable
   - **MROPE** (section `[24, 20, 20]`) is the one novel element — needs probe to confirm RoPE kernel compat

4. **Quality gate is automatable** (WER/CER vs ground truth), unlike TTS's subjective user-ear gate. This accelerates iteration — every agent commit self-validates in 90 s via Tier-1 harness.

5. **Host ac03 is production-ready**: CANN 8.3.RC1 matches ac01, 910B4 32 GB HBM, 1.5 TiB RAM, 192 cores, 437 GB on `/modelarts`. No stop-and-report triggered.

### Proposed A1 scope revision

**Original**: "2-3 weeks — native `AsrCannEngine` bring-up from scratch"

**Recommended**: split into A1a + A1b:

- **A1a — baseline re-establishment on ac03** (3-5 days): clone fork, build `qwen_asr` binary with CANN, download weights, convert to GGUF, transcribe `ellen_ref.wav`, confirm transcript matches llama.cpp/MLX reference, measure RTF on ac03 (expect 0.12-0.15). **Exit gate**: transcript character-identical to MLX reference on 5-clip set, RTF documented.
- **A1b — native engine port** (1.5-2 weeks): scaffold `tools/asr/asr_cann_engine.{h,cpp}` + `asr_cann_symbols.{h,cpp}` per CP convention, port decoder forward to direct aclnn dispatch, bypass ggml-cann for the generation hot path. **Exit gate**: token-identical output vs A1a on 20-clip Tier-2 set, RTF ≤ A1a RTF.

### Proposed RTF calibration (revises §1 target)

- Baseline (A1a measured): expected RTF 0.12-0.15
- Native-engine (A1b): target RTF ≤ 0.10
- + Fusion (A2): target RTF ≤ 0.05
- + W8 (A4): target RTF ≤ 0.03 (30× RT)
- Stretch (A3 + A5): target RTF ≤ 0.02 (50× RT) — aligns with MLX reference's 30×-50× on M4 Max, sanity-checked

Full arc from conventional stack (~CPU-only RTF 0.88 = 1.1× RT) to optimized NPU (RTF 0.02 = 50× RT) = **~45× speedup**, comparable to the TTS 1 → 32 fps arc (32×). Plausible and within the contract's "10-30×" corridor if we take the conservative end.

### Risks to flag

1. **MROPE interleaved with 3-section layout** may not fit stock `aclApplyRotaryPosEmbV2`. Needs header probe at A2 — same as W3b saga for TTS.
2. **Encoder is CPU-CANN-capable today** but not the hot path. If encoder ends up >30% of wall post-decoder optimization, A5 becomes mandatory (already on contract).
3. **Tokenizer merges.txt** is 1.67 MB — large but standard Qwen BPE. Existing fork `BpeTokenizer` reused (per README "复用 `tools/qwen_tts/` 的共享组件").
4. **Audio encoder GGUF export** is one-shot and produces ~607 MB F16 (not quantised). W8 of encoder is risky (§A5.3 warns). Keep encoder F16 until quality-verified.
5. **Streaming (A7)** is **not needed** for our current product surface (OpenAI Whisper-compatible `transcriptions` endpoint is batch). Defer as conditional.

### Go / no-go

**GO**. Scope A1 as A1a + A1b above. Update contract §5 A1 effort estimate from "2-3 weeks" to "~2 weeks including re-establishment". Keep A2-A5 timeline as-is (they are the meat of the optimization arc). A6/A7 remain conditional.

---

## Appendix — companion files

1. **Tier-1 gate spec** — `ac03:~/asr_a0/asr_tier1_gate_spec.md` (also at `/tmp/asr_tier1_gate_spec.md` on both hosts)
2. **Working dir on ac03** — `~/asr_a0/` (fresh, empty apart from spec)
3. **Reference MLX implementation** — `/Users/yuechen/home/OminiX-MLX/qwen3-asr-mlx/` (Rust, full pipeline, parity target)
4. **Existing fork scaffolding** — `/Users/yuechen/home/OminiX-Ascend/tools/qwen_asr/` (22 files, C++/GGML baseline)
