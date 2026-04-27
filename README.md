# ominix-cuda: Pure C++ Qwen3 inference on NVIDIA Blackwell

`ominix-cuda` is the CUDA fork of OminiX-Ascend. The production stack is native C++ on ggml/ggml-cuda for NVIDIA GB10-class Blackwell systems.

No Python, PyTorch, diffusers, transformers, vLLM, or sglang is used in the production inference path. Dev-only conversion and inspection scripts may exist, but the runtime target is C++/CUDA direct dispatch.

## What works today

Three authentic CUDA-native deliverables are in hand:

- **QIE-Edit cat PNG** — `make the cat smile` at 1024² / 20-step Euler on GB10 #2: **2:45 wall (165 s)** via `ominix-diffusion-cli --diffusion-fa` (7.4× over the no-FA baseline). Cossim 1.0000 vs Ascend at n=1.
- **TTS audio** — text → 24 kHz WAV end-to-end via TalkerCudaEngine + CodePredictor + SpeechTokenizerDecoder vocoder. **RTF 0.62 steady-state** (1.74 cold). Sampling defeats greedy mode collapse; cross-prompt and cross-seed audio decorrelate.
- **ASR transcript** — Ellen audiobook 9.36 s WAV → 43-token plausible audiobook intro. **RTF 0.37 wall**. Audio encoder cossim 1.000000 across all 11 stages vs HF Python reference.

## Phase Status

| Phase | Scope | Status | Receipt |
|---|---|---|---|
| Phase 0 | Repo bootstrap, CANN strip, ggml-cuda build | ✓ | `1aaa75d2` |
| Phase 1 | sd.cpp CUDA baseline, CFG batching, RoPE pre-compute | ✓ ⚠ | `ad5ef19c` (165 s wall vs 140 s gate; closes when Phase 3.7 lands) |
| Phase 2 | Native TalkerCudaEngine + Predictor + vocoder + E2E | ✓ | 2.1–2.5 (`d60452a7`→`f92503b8`), 2.7–2.9 (`3fd4253d`→`7680e727`); 2.6 ⏸ FP8/INT8 deferred |
| Phase 3 | Native ImageDiffusionCudaEngine + production-CLI FA | ✓ | 3.1–3.4d (`fc629955`→`07fb6b6d`), 3.6 (`cfb31930`) |
| Phase 4 | Native ASR CUDA path | ✓ | 4.1–4.6 (`f50488e6`→`a8858f86`) |
| Phase 5 | Docs + ship | ✓ | this README + `SHIP_SUMMARY.md` |

Headline numbers:

- **TalkerCudaEngine**: 53.85 TPS autoreg (Phase 2.3); 50.55 TPS with CUDA Graphs (28L Qwen3 body, F16, GEMM-bound)
- **CodePredictor**: 944 TPS hot loop (Phase 2.4); 10.3× wall reduction at Phase 2.9 via device F16 LM-head + CUDA Graphs
- **Native QIE-Edit DiT**: cossim 1.0000 vs Ascend at n=1, 1024² (byte-parity)
- **Production CLI QIE-Edit**: 7.53 s/step with `--diffusion-fa` (was 60.0 s/step naive F32)

## Quick start

The three demo scripts wrap the production smoke commands. They require SSH access to the GB10 hosts (`zgx-3675`, `zgx-5b44`) and the model assets in place — see `SHIP_SUMMARY.md` §4–§6 for build commands, model paths, and SSH config.

```bash
# Image edit (1024² / 20-step Euler, --diffusion-fa enabled, ~2:45 wall):
./scripts/demos/run_qie_edit.sh ./inputs/cat.jpg "make the cat smile"

# TTS (text → 24 kHz mono WAV, RTF 0.62 steady-state):
./scripts/demos/run_tts.sh "Hello world, this is the ominix CUDA TTS."

# ASR (WAV → transcript, RTF 0.37):
./scripts/demos/run_asr.sh ./samples/ellen_ref.wav

# Run all three back-to-back:
./scripts/demos/run_all.sh
```

## What is in this repo

- `ggml/src/ggml-cuda/` — CUDA backend vendored from upstream `ggml-org/llama.cpp` tag `b8532`. Local `CONV_1D` and `FLIP` CUDA extensions preserved.
- `tools/ominix_diffusion/` — stable-diffusion.cpp based image generation and image-edit CLI (`ominix-diffusion-cli`).
- `tools/qwen_tts/` — TalkerCudaEngine (28L Qwen3), CodePredictor (5L Qwen3), SpeechTokenizerDecoder vocoder (RVQ + 2× upsample + 4 vocoder blocks + tanh).
- `tools/qwen_asr/` — AsrCudaEngine + AudioEncoderCudaEngine; reuses TalkerCudaEngine verbatim for the 28L decoder.
- `tools/qwen_image_edit/` — native ImageDiffusionCudaEngine (60-block DiT, F32-widened residual chain).
- `docs/contracts/OMINIX_CUDA_CONTRACT.md` — project contract and acceptance gates.
- `SHIP_SUMMARY.md` — full per-phase receipts, perf tables, build commands per host, run commands, model paths, known gaps.

## GB10 build

Target hosts: GB10 #1 `zgx-3675` (port 6222) for TTS/ASR, GB10 #2 `zgx-5b44` (port 6022) for QIE-Edit. Ubuntu 24.04 aarch64 + CUDA 13.0.88 + sm_121a (Blackwell-A).

```bash
cmake -B build \
  -DGGML_CUDA=ON -DGGML_CANN=OFF \
  -DQWEN_TTS_CUDA=ON -DQWEN_IMAGE_EDIT_CUDA=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake auto-promotes `CMAKE_CUDA_ARCHITECTURES=121` to `121a`. Expected linkage: `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcuda.so.1`. No `libascend*`.

## Runtime direction

Phase 1 uses `ominix-diffusion-cli` as the CUDA sd.cpp baseline; production cat PNGs run through this CLI with `--diffusion-fa` enabled. Phases 2–4 added CUDA-native direct-dispatch engines:

- Phase 2 — `TalkerCudaEngine` + `CodePredictor` + `SpeechTokenizerDecoderCudaEngine` (text → 24 kHz speech)
- Phase 3 — `ImageDiffusionCudaEngine` (cossim 1.0000 vs Ascend at n=1; native E2E gated on cuDNN FMHA)
- Phase 4 — `AsrCudaEngine` + `AudioEncoderCudaEngine` (WAV → text, audio encoder byte-parity vs HF Python)

## Known gaps

- **Phase 2.6** (FP8/INT8 cuBLASLt) — deferred. Path to the 80–150 fps Talker contract bracket; CUDA Graph plumbing already in place.
- **Phase 3.7** (text encoder + VAE FlashAttention) — next perf lever to clear the 140 s Phase 1 gate; ~50 s of the current 165 s wall is text encode + VAE encode/decode.
- **Phase 4.6** (CPU mel parity) — currently 0.80 cossim vs HF Python (window/log-scale divergence); non-blocking, bypassed via `OMINIX_ASR_USE_MEL_BIN` when bit-parity matters.
- **Native QIE-Edit E2E** — naive F32 attention (~960 ms/block); gated on a future cuDNN FMHA pass for native-engine production runs.

See [SHIP_SUMMARY.md](SHIP_SUMMARY.md) for full per-phase receipts, perf numbers, build commands per host, and operational notes.
