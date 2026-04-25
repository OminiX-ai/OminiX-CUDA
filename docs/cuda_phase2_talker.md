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

**Phase 2.1 gate result — 2026-04-22 GB10 #1 (zgx-3675, sm_121a, CUDA 13.0.88)**:

```
$ ./bin/test_talker_cuda_init ~/ominix-cuda/models/ggml-vocab-llama-bpe.gguf
[talker_cuda] Phase 2.1 scaffold init OK  device=0  n_embd=2048  n_heads=16  \
    n_kv=8  head_dim=128  inter=6144  n_layers=28  MAX_SEQ=4096  MAX_PREFILL=512
[smoke] Phase 2.1 scaffold init PASS  ready=1  use_cuda_graphs=0  \
    use_int8=0  use_fp8=0
exit=0
```

Process tree during run contains only `test_talker_cuda_init` — **no Python**.

CMake auto-promoted `CMAKE_CUDA_ARCHITECTURES=121` to `121a` (Blackwell-A).
nvcc options applied: `-O3 -use_fast_math -extended-lambda -compress-mode=size`.
`ldd` resolves cleanly against `libcudart.so.13`, `libcublas.so.13`, libgcc/libc.

Build host: GB10 #1 (zgx-3675 aarch64 Grace + Blackwell, 119 GB unified mem).
Build wall: ~3 min from cold cmake (most spent compiling ggml-cuda's
`fattn-tile-instance-*` template instances; subsequent reconfigure re-link
< 5s thanks to cached object files).

Gate: PASS. Phase 2.2 cleared to start.

### Phase 2.2 — Per-token forward (CODE LANDED, GB10 SMOKE PENDING)

Per-token autoregressive `forward_decode` body wired up on the Mac side.

Files added:
- `tools/qwen_tts/native/cuda_kernels/cuda_kernels.h` — C-style launcher
  signatures for the six custom kernels listed below.
- `tools/qwen_tts/native/cuda_kernels/elementwise.cu` — F32<->F16 casts
  and an F16 elementwise add (residual). Add does sum-in-F32 to match the
  Ascend `aclnnAdd` precision contract.
- `tools/qwen_tts/native/cuda_kernels/rmsnorm.cu` — block-per-row RmsNorm
  with F16 input/output, F32 gamma. Single-pass mean-of-squares -> rstd ->
  scaled write.
- `tools/qwen_tts/native/cuda_kernels/rope_neox.cu` — NEOX-mode RoPE on a
  `[n_heads, head_dim]` tile against a precomputed `cos[head_dim]` /
  `sin[head_dim]` row. One block per head; one thread per pair-index.
- `tools/qwen_tts/native/cuda_kernels/swiglu.cu` — fused SwiGLU
  `y[i] = silu(gate[i]) * up[i]` in F32 compute.
- `tools/qwen_tts/native/cuda_kernels/attn_gqa.cu` — single-token GQA
  attention. One block per Q head (16 blocks for Talker). Loads
  `q[h]` into shmem, dot-product against the live KV cache rows
  (`pos+1` of them) with warp-stride dim accumulation, online softmax
  with max-subtract numerical stabilization, then weighted sum into
  the output. GQA ratio `n_heads / n_kv = 2` handled via
  `h_kv = h / group`.
- `tools/qwen_tts/native/test_talker_cuda_decode.cpp` — Phase 2.2 smoke
  harness. Drives `forward_decode` for 10 sequential positions against a
  deterministic sin-wave embedding, checks no NaN/inf and non-zero
  magnitude per call, prints avg wall-clock per step.

Files modified:
- `tools/qwen_tts/native/talker_cuda_engine.cpp`:
  - Added `load_gguf_tensor_f32` / `upload_tensor_f16` / `upload_tensor_f32`
    helpers (mirror the Ascend reference).
  - Wired the per-layer Q/K/V/O/gate/up/down F16 + attn_norm/ffn_norm/
    q_norm/k_norm F32 + final `output_norm` upload into `init_from_gguf`.
  - Replaced the `run_decode_ops_` and `forward_decode` stubs with the
    real 28-layer body. Six `cublasGemmEx` calls per layer (op_A=T to
    consume row-major `[out, in]` storage), interleaved with the custom
    kernel launches above. Compute type FP32 / I/O FP16. KV cache slot
    write happens by passing the cache slot pointer directly as the RoPE
    output for K (avoids an extra D2D), and a tiny D2D for V.
  - `forward_prefill` still aborts (Phase 2.3 deliverable).
- `tools/qwen_tts/CMakeLists.txt`:
  - New `qwen_tts_cuda_kernels` STATIC library bundling the five `.cu`
    files; PIC ON; `CMAKE_CUDA_ARCHITECTURES` defaults to `121` (GB10
    Blackwell sm_121a).
  - Linked into `qwen_tts` exe, `qwen_tts_api` shared lib,
    `test_talker_cuda_init`, and `test_talker_cuda_decode`.

GB10 build/run is blocked on this dispatch: the GB10 #1 host
(`zgx-3675:6222`) is not reachable from the Mac development environment
(no `Host` entry in `~/.ssh/config`, hostname does not resolve, IP
192.222.56.72 returns "Connection refused" on 6222). Code is ready; the
rsync + `cmake --build build-phase21 --target test_talker_cuda_decode`
+ smoke-binary run is the only step gated on access. No git commit
made — Mac local only, awaiting GB10 reachability.

### Phase 2.3 — KV cache + autoregressive loop (PENDING)
### Phase 2.4 — Codec C++ via cuDNN (PENDING)
### Phase 2.5 — CUDA Graphs at per-pos capture (PENDING)
### Phase 2.6 — FP8/INT8 quant via cuBLAS (PENDING)

### Gate 2 ship measurement (PENDING)

Will populate with: total wall, fps, ear-PASS check, `ps -ef | grep python`
during inference (must be empty).
