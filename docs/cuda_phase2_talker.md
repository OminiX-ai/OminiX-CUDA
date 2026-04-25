# OminiX-CUDA Phase 2 — Native TalkerCudaEngine

Direct port of `tools/qwen_tts/talker_cann_engine.{h,cpp}` (Ascend, 32.2 fps
ship state) onto NVIDIA Blackwell (GB10, sm_121). Replaces aclnn dispatch
with cuBLAS / cuDNN / custom CUDA kernels. Reference contract:
`/Users/yuechen/home/OminiX-Ascend/docs/contracts/OMINIX_CUDA_CONTRACT.md`
§ Phase 2.

Headline target (Gate 2): **≥ 80 fps end-to-end audio on GB10 #1**, no Python
in process tree.

| Reference | fps |
|---|---|
| codex vLLM end-to-end | 11.5 |
| Ascend native ship | 32.2 |
| Ascend stretch (ACLGRAPH on) | 31.6 |
| llama.cpp tok/s on Talker (raw) | 132 |
| **GB10 target** | **80–150** |

## Per-phase status

### Phase 2.1 — Scaffold + GGUF parse  *(LANDED)*

- New directory: `tools/qwen_tts/native/`
- New files:
  - `talker_cuda_engine.h` — full class surface (mirrors TalkerCannEngine
    ABI: `init_from_gguf`, `forward_decode`, `forward_prefill`,
    `reset_kv_cache`, `set_rope_speed_factor`, MRoPE xvec toggle, INT8/FP8
    quant toggles for Phase 2.6, CUDA Graph toggle for Phase 2.5,
    multi-stream pipelining accessors for Phase 2.3 lookahead).
  - `talker_cuda_engine.cpp` — Phase 2.1 init body:
    - cuBLAS + (optional) cuDNN handle creation
    - primary/secondary CUDA streams + decode-done event
    - hparam cache from `TalkerConfig`
    - GGUF open via vendored ggml/gguf API
    - mrope_section parse (B6.1 parity with Ascend)
    - cudaMalloc for all S=1 + prefill activation scratch
    - per-layer KV cache `[MAX_SEQ, kv_dim]` F16 (28 × ~16.4 MB ≈ 460 MB)
    - F32 I/O staging buffers (`input_stage_f32`, `output_stage_f32`)
    - precomputed NEOX-mode RoPE cos/sin tables (host F32 → device F16,
      sector-aware xvec layout supported via `set_use_mrope_xvec_layout`)
    - `decode_graph_execs_` vector pre-sized to `MAX_SEQ` (Phase 2.5 fills)
  - `test_talker_cuda_init.cpp` — minimal smoke harness; invokes
    `init_from_gguf` and prints PASS/FAIL.
- `tools/qwen_tts/CMakeLists.txt`:
  - new `option(QWEN_TTS_CUDA "Build native CUDA Talker engine (Phase 2)" OFF)`
  - links `CUDA::cudart` + `CUDA::cublas` (and `CUDA::cudnn` when
    `OMINIX_CUDA_USE_CUDNN=ON`) onto both `qwen_tts` executable and
    `qwen_tts_api` shared lib when `QWEN_TTS_CUDA=ON`.
  - new target `test_talker_cuda_init` for the Phase 2.1 smoke.

Stub-only (intentional, lands in Phase 2.2 / 2.6):
- `forward_decode`, `forward_prefill`, `run_decode_ops_` — stubs that
  `std::abort()` so any accidental call surfaces immediately.
- `int8_calibrate_weight_` — Phase 2.6 stub.

**Phase 2.1 gate**: `cmake -B build -DGGML_CUDA=ON -DQWEN_TTS_CUDA=ON &&
cmake --build build -j --target test_talker_cuda_init` succeeds on GB10 #1,
binary runs and prints `[talker_cuda] Phase 2.1 scaffold init OK …` for the
canonical Talker GGUF.

### Phase 2.2 — Per-token forward (PENDING)
### Phase 2.3 — KV cache + autoregressive loop (PENDING)
### Phase 2.4 — Codec C++ via cuDNN (PENDING)
### Phase 2.5 — CUDA Graphs at per-pos capture (PENDING)
### Phase 2.6 — FP8/INT8 quant via cuBLAS (PENDING)

### Gate 2 ship measurement (PENDING)

Will populate with: total wall, fps, ear-PASS check, `ps -ef | grep python`
during inference (must be empty).
