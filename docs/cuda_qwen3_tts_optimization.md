# Qwen3-TTS CUDA/GB10 Optimization Log

Date: 2026-04-24 America/Los_Angeles
Remote: `user1@163.192.33.32:6222`
Work dir: `/home/user1/qwen3_tts_cuda`

## Mission

Optimize Qwen3-TTS on NVIDIA GB10 toward an 80+ fps shipping target, using
the Ascend 910B 32.2 fps arc as the reference for architecture and gates.

## Phase 0 - Environment Baseline

Status: **YELLOW**

The GB10 environment is ready for Phase 1 llama.cpp CUDA benchmarking and
PyTorch CUDA experiments. TensorRT-LLM is installed, but its Python import is
blocked by a PyTorch internal ABI mismatch; this is the one Phase 0 item that
needs a follow-up before Phase 2 compile work can start.

### Remote Hardware / OS

- Host: `zgx-3675`
- OS: Ubuntu 24.04.4 LTS, kernel `6.17.0-1014-nvidia`
- Arch: `aarch64`
- GPU: NVIDIA GB10, driver `580.142`, CUDA driver API `13.0`
- CUDA toolkit: `/usr/local/cuda-13.0`, `nvcc 13.0.88`
- Memory: 119 GiB unified CPU/GPU memory visible to Linux
- Disk: 916 GiB root volume, ~834 GiB free at start

### Access

- SSH key auth installed with the existing local Ed25519 public key.
- Subsequent commands work with:
  `ssh -p 6222 -o BatchMode=yes user1@163.192.33.32 ...`

### Python / CUDA Environments

Main CUDA PyTorch env:

- Path: `/home/user1/qwen3_tts_cuda/.venv`
- Python: 3.12.3
- `torch==2.13.0.dev20260424+cu130`
- `torchaudio==2.11.0.dev20260424+cu130`
- `torchvision==0.27.0.dev20260424+cu130`
- CUDA smoke: `torch.cuda.is_available() == True`
- Device: `NVIDIA GB10`, capability `(12, 1)`
- BF16 GEMM smoke: 4096x4096, 20 iterations, 1.764 ms/iter
- FP8 dtype smoke: `torch.float8_e4m3fn` available

vLLM env:

- Path: `/home/user1/qwen3_tts_cuda/.venv_vllm`
- `vllm==0.19.1`
- `torch==2.10.0+cu130`
- CUDA smoke: BF16 2048x2048 GEMM, 0.192 ms/iter
- Note: PyTorch warns that this build advertises support through SM 12.0
  while GB10 is SM 12.1, but CUDA operations succeeded.

TensorRT-LLM env:

- Path: `/home/user1/qwen3_tts_cuda/.venv_trtllm_pypi`
- `tensorrt==10.14.1.48.post1`
- `tensorrt-llm==1.2.1`
- User-space OpenMPI installed via PyPI package `openmpi==5.0.10`
- Local `libpython3.12.so` symlink added under
  `/home/user1/qwen3_tts_cuda/lib`
- Current blocker:
  `ImportError: .../tensorrt_llm/libs/libth_common.so: undefined symbol: _ZN3c104impl12PyObjectSlotD1Ev`
- Attempts:
  - `torch==2.9.1+cu130`: failed on
    `_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb`
  - `torch==2.10.0+cu130`: fixed the CUDA-check symbol, failed on
    `c10::impl::PyObjectSlot`
  - `torch==2.11.0+cu130`: same `PyObjectSlot` failure
- Interpretation: public aarch64 CUDA Torch wheels do not match the
  PyTorch internal ABI expected by the NVIDIA TensorRT-LLM 1.2.1 wheel.
  Phase 2 likely needs either an NVIDIA NGC-matched stack, a source build
  against the selected Torch wheel, or a pinned TRT-LLM version with a
  matching public aarch64 Torch ABI.

### llama.cpp CUDA

- Repo: `/home/user1/qwen3_tts_cuda/llama.cpp`
- Commit: `0adede8`
- Build: `/home/user1/qwen3_tts_cuda/llama.cpp/build-cuda`
- CMake flags:
  - `-DGGML_CUDA=ON`
  - `-DCMAKE_CUDA_ARCHITECTURES=121`
  - `-DCMAKE_BUILD_TYPE=Release`
  - `-DLLAMA_BUILD_TESTS=OFF`
- Built targets: `llama-cli`, `llama-bench`
- Device enumeration:
  `NVIDIA GB10, compute capability 12.1, VMM: yes, VRAM: 122502 MiB`

Small smoke on Q8_0 talker GGUF with Flash Attention:

```text
model: gguf_q8_0/qwen3_tts_talker.gguf
model_type: qwen3vl 1.7B Q8_0
ngl: 99
flash_attn: true
n_prompt=16: 1169.78 tok/s
n_gen=16: 132.916 tok/s
```

This is only a load/execution smoke, not the canonical TTS fps benchmark.
Phase 1 must run the actual talker generation path and report frame/audio
wall-clock.

### Weights / Assets

Downloaded from `cgisky/qwen3-tts-custom-gguf`:

- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_talker.gguf`
  - 1,511,314,656 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_predictor.gguf`
  - 151,124,320 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_assets.gguf`
  - 406,374,528 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/onnx/qwen3_tts_decoder.onnx`
  - 456,760,558 bytes
- `/home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/tokenizer/tokenizer.json`
  - 11,423,986 bytes
- Preset speaker JSONs under `preset_speakers/`

Note: the model card mentions `onnx_int8`, but the repo file list currently
contains only `onnx/` files.

## Next Phase Scope

Phase 1 should produce the first real CUDA number to beat:

1. Identify the lightest working Qwen3-TTS C++/Python runner compatible with
   the downloaded GGUF package.
2. Run canonical prompts matching the Ascend reports as closely as possible.
3. Report audio frames/sec, token wall-clock, generated frame count, and an
   eye/ear-check verdict.
4. Keep llama.cpp measurements separate from full TTS pipeline measurements,
   because the smoke above only validates decoder-token execution.

TensorRT-LLM follow-up before Phase 2:

1. Check whether an NVIDIA NGC PyTorch/TensorRT-LLM wheel stack is available
   without Docker, or whether the user-space machine allows container runtime.
2. If not, source-build TensorRT-LLM against the selected CUDA Torch aarch64
   wheel.
3. Re-run import smoke and a minimal Qwen/Qwen2 builder smoke before investing
   in Qwen3-TTS graph conversion.

## Phase 2 - vLLM Pivot

Date: 2026-04-25 America/Los_Angeles

Status: **YELLOW**

The TRT-LLM path remains blocked on PyTorch internal ABI mismatches, so Phase 2
pivoted to vLLM 0.19.1. vLLM can load and serve the Talker GGUF through a
derived Qwen3 HF config, and CUDA graph mode reaches llama.cpp-class raw decode
throughput on GB10. Full end-to-end TTS fps is still pending because the
current local asset bundle has the decoder ONNX and GGUF assets, but no ready
Python runner or complete ONNX encoder/speaker-encoder path to wire Talker
tokens into audio.

### Setup Fixes

- `vllm==0.19.1` imported only after adding CUDA 12 runtime libraries inside
  `.venv_vllm`:
  `pip install nvidia-cuda-runtime-cu12`
- Runtime library path used for vLLM:
  `LD_LIBRARY_PATH=.venv_vllm/.../torch/lib:.venv_vllm/.../nvidia/cuda_runtime/lib:.venv_vllm/.../nvidia/cu13/lib`
- Torch/Triton helper compilation needed Python development headers. `sudo`
  was not available non-interactively, so headers were extracted into:
  `/home/user1/qwen3_tts_cuda/sysroot/python312-dev`
- Compile path uses:
  `CPATH=/home/user1/qwen3_tts_cuda/sysroot/python312-dev/usr/include:/home/user1/qwen3_tts_cuda/sysroot/python312-dev/usr/include/python3.12`

### Model Loading Path

Direct GGUF loading failed because the Talker GGUF reports architecture
`qwen3vl`, which vLLM/Transformers did not accept from GGUF metadata:

```text
ValueError: GGUF model with architecture qwen3vl is not supported yet.
```

The official HF config for `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` uses a
Qwen3-TTS model type rather than a plain vLLM-supported Qwen3 CausalLM config,
so the run used a derived HF config for the GGUF Talker:

- Config path:
  `/home/user1/qwen3_tts_cuda/vllm_hf/qwen3_tts_talker_gguf_config/config.json`
- Shape:
  `vocab_size=3072`, `hidden_size=2048`, `intermediate_size=6144`,
  `num_hidden_layers=28`, `num_attention_heads=16`,
  `num_key_value_heads=8`, `head_dim=128`,
  `max_position_embeddings=32768`, `rope_theta=1000000.0`
- Architecture override: `Qwen3ForCausalLM`

vLLM's early speculator probe still tried to parse the GGUF before
`--hf-config-path` was applied, so the CLI entrypoint is wrapped by:

```text
/home/user1/qwen3_tts_cuda/scripts/vllm_qwen3_tts_cli.py
```

This wrapper skips the early speculator probe and then delegates to vLLM's
normal CLI. The model must be driven with raw Talker token IDs. Text tokenizer
IDs are invalid for this 3072-token audio-code vocabulary; for example,
`"Hello"` encodes to token ID `9707`, which is out of range for the Talker
embedding table.

### Serve Commands

CUDA graph FP16 server:

```bash
V=/home/user1/qwen3_tts_cuda/.venv_vllm
SYS=/home/user1/qwen3_tts_cuda/sysroot/python312-dev/usr/include
. "$V/bin/activate"
export LD_LIBRARY_PATH="$V/lib/python3.12/site-packages/torch/lib:$V/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$V/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$SYS:$SYS/python3.12:${CPATH:-}"
export VLLM_NO_USAGE_STATS=1

python /home/user1/qwen3_tts_cuda/scripts/vllm_qwen3_tts_cli.py serve \
  /home/user1/qwen3_tts_cuda/models/cgisky-qwen3-tts-custom-gguf/gguf_q8_0/qwen3_tts_talker.gguf \
  --served-model-name qwen3-tts-talker \
  --host 127.0.0.1 \
  --port 18000 \
  --hf-config-path /home/user1/qwen3_tts_cuda/vllm_hf/qwen3_tts_talker_gguf_config \
  --skip-tokenizer-init \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 512 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 512 \
  --gpu-memory-utilization 0.50
```

FP8 KV cache variant adds:

```bash
--kv-cache-dtype fp8
```

Eager control adds:

```bash
--enforce-eager
```

API smoke request:

```json
{
  "model": "qwen3-tts-talker",
  "prompt": [1],
  "max_tokens": 16,
  "temperature": 0,
  "stop_token_ids": [],
  "return_token_ids": true
}
```

Smoke output token IDs:

```text
[2157, 2150, 498, 498, 498, 498, 1311, 1489, 1489, 1489, 1489, 1247, 999, 613, 613, 613]
```

### Talker Microbenchmarks

Benchmark script:

```text
/home/user1/qwen3_tts_cuda/scripts/vllm_qwen3_tts_api_bench.py
```

Raw-token benchmark settings:

- Prompt token IDs: `[1]`
- Decode length: 128 tokens for throughput
- Streaming probe: 64 tokens
- `temperature=0`
- `ignore_eos=True`
- `stop_token_ids=[]`
- `return_token_ids=True`
- `max_num_seqs=1`
- `max_num_batched_tokens=512`

These are Talker-token API measurements, not complete audio synthesis fps.

| Path | Mean decode | Warm stream TTFT | Notes |
| --- | ---: | ---: | --- |
| llama.cpp Q8_0 CUDA baseline | 132.916 tok/s | n/a | Phase 1 smoke, `n_gen=16` |
| vLLM eager FP16 GGUF | 110.03 tok/s | 49.99 ms | Control path, no CUDA graphs |
| vLLM CUDA graph FP16 GGUF | 131.78 tok/s | 31.69 ms | 99.1% of llama.cpp baseline |
| vLLM CUDA graph + FP8 KV | 130.95 tok/s | 11.80 ms | 98.5% of llama.cpp baseline |
| Ascend shipping reference | 32.2 fps | n/a | Full TTS reference, not token-only |
| MLX equivalent | not measured | n/a | No equivalent run available |

Best current vLLM Talker result:

```text
vLLM graph FP16: 131.78 tok/s
vs llama.cpp baseline: 132.916 tok/s
vs vLLM eager: +19.8%
vs Ascend 32.2 fps reference: about 4.1x if compared only as raw Talker token rate
```

The Ascend comparison is directional only until the codec path is wired and
measured as real synthesized-audio fps.

### Memory / Compile Receipts

Eager FP16 server log:

```text
/home/user1/qwen3_tts_cuda/logs/vllm_serve_talker_eager_18000.log
```

- Model loading: 1.43 GiB, 0.805 s
- Available KV cache memory: 56.77 GiB
- GPU KV cache size: 531,488 tokens
- Engine init: 2.98 s

CUDA graph FP16 server log:

```text
/home/user1/qwen3_tts_cuda/logs/vllm_serve_talker_graph_18000.log
```

- Model loading: 1.43 GiB, 0.840 s
- `torch.compile`: 13.18 s
- Estimated CUDA graph memory: 0.07 GiB
- Available KV cache memory: 56.41 GiB
- GPU KV cache size: 528,144 tokens
- Engine init: 17.98 s

CUDA graph + FP8 KV server log:

```text
/home/user1/qwen3_tts_cuda/logs/vllm_serve_talker_graph_kvfp8_18000.log
```

- Attention backend: FLASHINFER
- Model loading: 1.43 GiB, 0.849 s
- `torch.compile`: 4.61 s, with compile-cache reuse
- Estimated CUDA graph memory: 2.35 GiB
- Available KV cache memory: 54.78 GiB
- GPU KV cache size: 1,025,680 tokens
- Engine init: 23.17 s
- Saved benchmark JSON:
  `/home/user1/qwen3_tts_cuda/logs/vllm_talker_graph_kvfp8_api_bench.json`

FP8 KV nearly doubles KV capacity versus FP16 KV at this sequence cap
(`1,025,680 / 528,144 = 1.94x`) but did not improve single-stream throughput
for the 128-token decode probe.

### Lever Delta

| Lever | Result |
| --- | --- |
| CUDA 12 runtime in vLLM venv | Unblocked vLLM import against `libcudart.so.12` |
| User-space Python headers | Unblocked Triton/Inductor helper builds |
| Derived Qwen3 HF config | Unblocked GGUF model load despite `qwen3vl` GGUF architecture tag |
| Raw prompt token IDs | Avoided out-of-range tokenizer IDs for Talker vocab |
| CUDA graphs / vLLM compile | 110.03 -> 131.78 tok/s, +19.8% |
| FP8 KV cache | 528k -> 1.026M KV tokens, no speed gain in single stream |
| FP8 weights | Not applied; current checkpoint is GGUF Q8_0 and vLLM reports `quantization=gguf` |
| Single-stream batch tuning | `max_num_seqs=1`, `max_num_batched_tokens=512`, `max_model_len=512` |

### Codec Wiring Status

The intended low-latency path is:

```text
vLLM token stream -> audio-code token buffer -> codec decoder on GPU -> audio
```

The vLLM side can accept list-of-int prompts and stream generated token IDs,
but the raw-token path is not enough to drive the codec. Qwen3-TTS Talker
generation is not a plain CausalLM loop:

1. The prompt is built as `inputs_embeds` from text projection, speaker/ref
   embeddings, language tags, and codec prefill IDs.
2. Each generated step produces the first codec group through the Talker head.
3. The code predictor then consumes the Talker hidden state plus the first
   codec ID and generates the remaining 15 codec groups.
4. The summed 16-codebook embedding, plus trailing text hidden state, becomes
   the next Talker step input.

The current vLLM OpenAI path returns only token IDs from the first-codebook
head. It does not return per-step Talker hidden states, does not run
`qwen3_tts_predictor.gguf`, and therefore cannot directly produce the
`[num_frames, 16]` codec code tensor required by the decoder. vLLM 0.19.1 does
support `--enable-prompt-embeds`, so initial prompt embedding is solvable, but
a real vLLM TTS bridge still needs either a custom vLLM model/runner that
implements the code-predictor feedback loop or an API path that exposes hidden
states for an external predictor.

### Codec Asset Reconstruction

Completed on 2026-04-25:

- Installed official `qwen-tts==0.1.1` into
  `/home/user1/qwen3_tts_cuda/.venv` without replacing the CUDA 13 PyTorch
  wheel. Dependencies added: `transformers==4.57.3`, `accelerate==1.12.0`,
  `librosa`, `soundfile`, `sox`, `einops`, and `onnxruntime==1.25.0`.
- Cloned Qwen reference source at:
  `/home/user1/qwen3_tts_cuda/vendor/Qwen3-TTS`
  (`022e286b98fbec7e1e916cb940cdf532cd9f488e`).
- Downloaded official Base model for reference-audio cloning:
  `/home/user1/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base` (4.3G).
  The official HF repos contain PyTorch `speech_tokenizer/` weights, not split
  ONNX encoder/speaker-encoder files.
- Downloaded a split 1.7B ONNX shared codec stack from
  `xkos/Qwen3-TTS-12Hz-1.7B-ONNX`:
  `/home/user1/qwen3_tts_cuda/models/xkos-qwen3-tts-12hz-1.7b-onnx`.
  ONNX contracts:
  - `speaker_encoder.onnx`: `mel_spectrogram [batch,time,128] -> speaker_embedding [batch,2048]`
  - `speech_tokenizer_encoder.onnx`: `input_values [batch,1,audio_len] -> audio_codes [batch,16,codes_len]`
  - `speech_tokenizer_decoder.onnx`: `codes [batch,seq_len,16] -> audio [batch,audio_len]`
- Confirmed the existing cgisky decoder is a streaming decoder, not the simple
  stateless wrapper:
  `audio_codes [1,num_frames,16]` plus `is_last`, conv history, latent buffer,
  and 8 layers of KV cache state -> `final_wav`, `valid_samples`, next caches.

Harness:

```text
/home/user1/qwen3_tts_cuda/scripts/tts_e2e_bench.py
OminiX-Ascend/scripts/tts_e2e_bench.py
```

The harness loads the official Base model, creates a voice-clone prompt from
reference audio/text, runs Talker generation, decodes codec frames to 24 kHz
audio, writes WAV, and records timing/memory JSON.

### End-to-End Audio Measurement

Canonical prompt:

```text
The quick brown fox jumps over the lazy dog.
```

Reference audio:

```text
/home/user1/qwen3_tts_cuda/inputs/qwen3_tts_clone_ref.wav
```

Reference text:

```text
Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.
```

Hot official Base run, one warmup + three measured runs in one loaded model:

```text
metrics: /home/user1/qwen3_tts_cuda/logs/qwen3_tts_e2e_base_hot3_metrics.json
wav:     /home/user1/qwen3_tts_cuda/outputs/qwen3_tts_e2e_quick_brown_fox_hot.wav
```

| Metric | Result |
| --- | ---: |
| Model load wall | 27.08 s |
| Mean total synthesis wall | 3.103 s |
| Median total synthesis wall | 3.205 s |
| Mean end-to-end audio fps | 11.53 fps |
| Median end-to-end audio fps | 11.55 fps |
| Mean Talker decode | 12.11 tok/s |
| Median Talker decode | 12.12 tok/s |
| Mean codec-only decode | 0.0963 s |
| Mean codec-only fps | 368.66 fps |
| Reference prompt build | 0.0332 s mean after warmup |
| Realtime factor | 0.922x mean |
| Peak CUDA allocated | 4.62 GB |
| First-token latency | not measured in official non-streaming path |
| First-audio latency | not measured in official non-streaming path |

One-shot runs:

| Run | Total wall | Audio fps | Talker tok/s | Codec fps | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| URL ref audio | 6.297 s | 5.56 | 11.11 | 236.92 | Includes cold-ish reference processing path |
| Local ref audio | 9.754 s | 3.59 | 11.42 | 229.68 | One-shot; slower prompt encode/compile variance |

Interpretation:

- The audio loop is now closed and produces valid 24 kHz WAV output.
- Codec decode is not the bottleneck on the official path; it is already
  hundreds of codec frames/sec after warmup.
- The official PyTorch Talker loop is the bottleneck at ~12 tok/s, far below
  the raw vLLM first-codebook microbench of 131.78 tok/s.
- The current end-to-end official Base result is below the Ascend shipping
  reference of 32.2 fps. It is a correctness/measurement closure, not the
  target CUDA result.
- The previous vLLM 131.78 tok/s number remains a raw first-codebook ceiling,
  not an audio fps claim, until the code-predictor hidden-state loop is wired
  inside or alongside vLLM.

### Next Measurements / Work

1. Implement a vLLM-compatible Qwen3-TTS runner that preserves the official
   code-predictor feedback loop:
   `prompt_embeds -> Talker hidden -> first codec ID -> predictor 15 IDs -> next embed`.
2. Reuse the split ONNX codec stack or the official PyTorch speech tokenizer
   for decode once vLLM produces complete `[T,16]` codec frames.
3. Re-measure first-token latency and first-audio latency on the real streaming
   path. The official path above cannot report these because it is
   non-streaming at the API level.
4. After complete-code vLLM generation works, re-test CUDA graphs, FP8 KV, and
   ONNX Runtime CUDA/TensorRT codec execution. Current evidence says the codec
   is already fast enough; the missing lever is Talker/code-predictor coupling.
