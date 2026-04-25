# ominix-cuda Phase 1 Baseline

Date: 2026-04-25

Host: GB10 #2 (`user1@163.192.33.32:6022`)

Binary: `~/ominix-cuda/build/bin/ominix-diffusion-cli`

## Assets

- Diffusion: `/home/user1/dev/sd-cpp/models/Qwen-Image-Edit-2509-Q4_0.gguf`
- LLM: `/home/user1/dev/sd-cpp/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf`
- Vision projector: `/home/user1/dev/sd-cpp/models/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf`
- VAE: `/home/user1/dev/sd-cpp/models/qwen_image_vae.safetensors`
- Ref image: `/home/user1/qie_cuda/inputs/cat.jpg`

GGUFs were available, so no safetensors conversion was needed for this receipt.

## Command Shape

All runs used the canonical Qwen-Image-Edit-2509 cat edit at 1024x1024, 20 Euler steps, seed 42:

```bash
GGML_CUDA_VISIBLE_DEVICES=0 \
~/ominix-cuda/build/bin/ominix-diffusion-cli \
  -M img_gen \
  --diffusion-model /home/user1/dev/sd-cpp/models/Qwen-Image-Edit-2509-Q4_0.gguf \
  --llm /home/user1/dev/sd-cpp/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf \
  --llm_vision /home/user1/dev/sd-cpp/models/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf \
  --vae /home/user1/dev/sd-cpp/models/qwen_image_vae.safetensors \
  -r /home/user1/qie_cuda/inputs/cat.jpg \
  -p "convert to black and white" \
  --steps 20 --cfg-scale 4.0 --sampling-method euler --flow-shift 3 --diffusion-fa \
  -W 1024 -H 1024 --seed 42
```

## Results

| Run | Env | Output | Sampling | Per step | Wall | Delta vs baseline |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Baseline | none | `/tmp/ominix_cuda_phase1_baseline.png` | 301.91s | 15.10s | 310.17s | - |
| CFG batching | `OMINIX_CFG_BATCHED=1` | `/tmp/ominix_cuda_phase1_cfg_batched.png` | 302.99s | 15.15s | 311.21s | +1.04s |
| CFG + RoPE precompute | `OMINIX_CFG_BATCHED=1 OMINIX_QIE_ROPE_PRECOMPUTE=1` | `/tmp/ominix_cuda_phase1_rope_cfg.png` | 300.62s | 15.03s | 308.82s | -1.35s |

All three commands exited 0 and produced 1024x1024 PNGs.

## Gate Notes

- Eye check: PASS. Output is a recognizable black-and-white cat.
- Latency: FAIL. Best canonical run is 308.82s wall, above the <140s Phase 1 gate.
- CFG env gate: PASS for activation and fallback safety. The run logs `CFG batching ACTIVE`, then falls back because the exact padded-context path would require a 12935.3 MiB attention mask at 1024x1024, above the 256 MiB mask budget.
- RoPE env gate: PASS. The CUDA fallback path originally thrashed a one-entry cache between cond/uncond context lengths. Phase 1 updates this to a small shape-keyed cache; the receipt shows two misses followed by hits (`HIT count=30 (miss=2)`).

## Memory

`nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits` returned `N/A` on this GB10 driver (`580.142`), and `nvidia-smi -q -d MEMORY` reports FB memory Total/Used/Free as `N/A`.

Log-reported backend allocations:

- Params: 19596.72 MB VRAM total
- VAE encode compute buffer: 7702.65 MB VRAM
- Qwen2.5-VL compute buffer: 61.24 MB VRAM
- Qwen image compute buffer: 1076.11 MB VRAM
- VAE decode compute buffer: 7493.50 MB VRAM

Largest log-visible footprint is params plus VAE encode buffer, approximately 27.3 GB VRAM.

## Build/Link Receipt

Remote rebuild:

```bash
cd ~/ominix-cuda
cmake --build build --target ominix-diffusion-cli -j$(nproc)
```

`ldd build/bin/ominix-diffusion-cli` links CUDA libraries (`libggml-cuda.so.0`, `libcudart.so.13`, `libcublas.so.13`, `libcublasLt.so.13`, `libcuda.so.1`) and no `libascend*`. It does not link `libcudnn`; this ggml-cuda path does not use cuDNN.

## Follow-up

The next latency blocker is exact CFG batching for unequal cond/uncond context lengths at 1024. Current code correctly avoids the crash by preflighting the mask allocation, but a compact per-batch mask or equal-length conditioning strategy is needed before the canonical QIE edit can reach the <140s target.
