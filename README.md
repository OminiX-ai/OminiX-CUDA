# ominix-cuda: Pure C++ Qwen3 inference on NVIDIA Blackwell

`ominix-cuda` is the CUDA fork of OminiX-Ascend. The production stack is native C++ on ggml/ggml-cuda for NVIDIA GB10-class Blackwell systems.

No Python, PyTorch, diffusers, transformers, vLLM, or sglang is used in the production inference path. Dev-only conversion and inspection scripts may exist, but the runtime target is C++/CUDA direct dispatch.

## Phase Status

| Phase | Scope | Status | Receipt |
|---|---|---|---|
| Phase 0 | Repo bootstrap, CANN strip, ggml-cuda build |  |  |
| Phase 1 | sd.cpp CUDA baseline, CFG batching, RoPE pre-compute |  |  |
| Phase 2 | Native TalkerCudaEngine |  |  |
| Phase 3 | Native ImageDiffusionCudaEngine |  |  |
| Phase 4 | Native ASR CUDA path |  |  |
| Phase 5 | CUDA optimization docs and ship prep |  |  |

## What Is In This Repo

- `ggml/src/ggml-cuda/`: CUDA backend vendored from upstream `ggml-org/llama.cpp` tag `b8532`, the latest compatible point before upstream added ggml core types this fork does not yet carry. This fork's local `CONV_1D` and `FLIP` CUDA extensions are preserved.
- `tools/ominix_diffusion/`: stable-diffusion.cpp based image generation and image-edit baseline for CUDA.
- `tools/qwen_tts/`, `tools/qwen_asr/`, `tools/qwen_image_edit/`: shared scaffolding retained; Ascend native engines are stripped and will be re-ported as CUDA-native engines in later phases.
- `docs/contracts/OMINIX_CUDA_CONTRACT.md`: project contract and acceptance gates.

## GB10 Build

Target host: GB10 #2 at `user1@163.192.33.32`, SSH port `6022`.

```bash
cmake -B build -DGGML_CUDA=ON -DGGML_CANN=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

The Phase 0 smoke target is:

```bash
build/bin/ominix-diffusion-cli
```

Expected CUDA linkage includes CUDA runtime and cuBLAS:

```bash
ldd build/bin/ominix-diffusion-cli | grep -E 'cuda|cublas|cudnn|ascend'
```

No `libascend*` dependencies should appear.

## Remote Smoke Command

```bash
ssh -p 6022 user1@163.192.33.32 \
  "cd ~/ominix-cuda && cmake -B build -DGGML_CUDA=ON -DGGML_CANN=OFF -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc) 2>&1 | tail -30"
```

## Runtime Direction

Phase 1 uses `ominix-diffusion-cli` as the CUDA sd.cpp baseline. Later phases replace Ascend-specific native engines with CUDA-native direct dispatch:

- Phase 2: `TalkerCudaEngine`
- Phase 3: `ImageDiffusionCudaEngine`
- Phase 4: ASR CUDA text decoder path

See [docs/contracts/OMINIX_CUDA_CONTRACT.md](docs/contracts/OMINIX_CUDA_CONTRACT.md) for the full contract.
