# Qwen-Image-Edit-2509 CUDA Reference on NVIDIA GB10

**Host**: `zgx-5b44` via `ssh -p 6022 user1@163.192.33.32`  
**Target**: 1024x1024, 20-step Qwen-Image-Edit-2509 cat edit, prompt `convert to black and white`  
**Primary output dir**: remote `~/qie_cuda/outputs/`, local receipts copied back as needed

## Phase 0 - Environment

### Receipts - 2026-04-24 PDT

| Item | Result |
|---|---|
| SSH key auth | Installed and verified; subsequent access works with `BatchMode=yes` |
| Host | `zgx-5b44`, Ubuntu 24.04.4 LTS, Linux 6.17.0-1014-nvidia, `aarch64` |
| Memory | 119 GiB unified system memory reported by `free -h` |
| GPU | `NVIDIA GB10`, Blackwell, compute capability 12.1, driver 580.142, CUDA runtime 13.0 |
| CUDA toolkit | `cuda-toolkit-13-0` already installed; `nvcc` at `/usr/local/cuda-13.0/bin/nvcc` |
| Workspace | remote `~/qie_cuda/{logs,outputs,src,models,inputs}` |
| `uv` | Installed user-local at `~/.local/bin/uv`, version 0.11.7 |
| Canonical input | Copied from `ac02:/home/ma-user/qie_q0v2/test/cat.jpg` to remote `~/qie_cuda/inputs/cat.jpg` |

### Package Plan

- PyTorch: official nightly CUDA 13.0 aarch64 wheels from `https://download.pytorch.org/whl/nightly/cu130`.
- Diffusers: latest `git+https://github.com/huggingface/diffusers`, matching the Hugging Face model card guidance for `QwenImageEditPlusPipeline`.
- Model id: `Qwen/Qwen-Image-Edit-2509`, confirmed by both the model card and Diffusers QwenImage docs.
- Baseline pipeline class: `diffusers.QwenImageEditPlusPipeline`, not the older single-image `QwenImageEditPipeline`, because 2509 is the Edit Plus variant.

Sources checked:
- Hugging Face model card: https://huggingface.co/Qwen/Qwen-Image-Edit-2509
- Diffusers QwenImage docs: https://huggingface.co/docs/diffusers/en/api/pipelines/qwenimage
- PyTorch nightly CUDA 13.0 wheel index: https://download.pytorch.org/whl/nightly/cu130/torch/

### In Progress

- Env install PID: `8604` complete. Earlier `8218` exited early because the first `uv venv` had no seeded `pip`; venv was recreated with `uv venv --clear --seed`.
- Env install log: `~/qie_cuda/logs/phase0_env_install.log`
- Model download PID: `10357` complete in 16m26s
- Model download log: `~/qie_cuda/logs/phase0_model_download.log`
- Command shape for long remote operations: `nohup setsid bash -lc ... < /dev/null > phase0_*.log 2>&1 &`

### Python Stack Receipt

| Package | Version / Result |
|---|---|
| `torch` | `2.13.0.dev20260424+cu130` |
| `torchvision` | `0.27.0.dev20260424+cu130` |
| `torch.version.cuda` | `13.0` |
| `torch.cuda.is_available()` | `True` |
| CUDA device | `NVIDIA GB10`, capability `(12, 1)` |
| `torch.cuda.mem_get_info()` | free `117592633344`, total `128453689344` bytes |
| `nvcc` | CUDA compilation tools release 13.0, `V13.0.88` |
| `diffusers` | `0.38.0.dev0` from git commit `f7fd76adcd288494a1a13c82d06e37579170aaf3` |
| `transformers` | `5.6.2` |
| `accelerate` | `1.13.0` |
| Pipeline import | `from diffusers import QwenImageEditPlusPipeline` succeeds |
| `xformers` binary wheel | Not available for this aarch64 environment via PyPI `--only-binary=:all:` |
| `flash-attn` binary wheel | Not available for this aarch64 environment via PyPI `--only-binary=:all:` |

### Load-Only Smoke

Command: `python src/bench_qie_diffusers.py --load-only --local-files-only ...`

| Measurement | Result |
|---|---|
| Pipeline component load | 0.836 s |
| `.to("cuda")` wall | 440.208 s |
| CUDA total via `torch.cuda.mem_get_info()` | 119.632 GiB |
| CUDA free after BF16 model load | 8.446 GiB |
| Verdict | PASS: full local `QwenImageEditPlusPipeline` loads onto GB10 CUDA |

Implication: the full BF16 2509 model leaves limited activation headroom when loaded fully resident. First generation smoke should start at 256x256 / 1-step before the 1024x1024 / 20-step baseline.

### Input Asset

- `~/qie_cuda/inputs/cat.jpg`: 256x256 progressive JPEG, 8.8 KB, same source path as Ascend Q1 baseline.
- ac03 also has a copy at `/home/ma-user/work/qie_test/cat.jpg`; the user-mentioned `/home/ma-user/qie_q0v2/test/cat.jpg` path was present on ac02, not ac03.

### Model Payload

- Model path: `~/qie_cuda/models/Qwen-Image-Edit-2509`
- Download wall: 16m26s via unauthenticated `hf download`
- Final local-dir size: 54 GiB
- Largest files:
  - Transformer: 5 safetensor shards, four near 9.3 GiB and one 937 MiB
  - Text encoder: 4 safetensor shards, three near 4.6 GiB and one 1.6 GiB
  - VAE: one 242 MiB safetensor

### Benchmark Harness

- Local source: `tools/qwen_image_edit/cuda/bench_qie_diffusers.py`
- Remote copy: `~/qie_cuda/src/bench_qie_diffusers.py`
- Captures: model load wall, `.to("cuda")` wall, total inference wall, transformer forward call times, grouped CFG step hot-loop times, pipeline callback step deltas, peak allocated/reserved CUDA memory, and output PNG path.
- Runtime patch: enabled by default for PyTorch `2.13.0.dev20260424+cu130` on GB10, where CUDA integer `Tensor.prod(dim)` fails with `CUDA driver error: invalid argument`. The harness routes only CUDA integer `.prod()` through CPU and copies the tiny result back. This unblocks Qwen2.5-VL `image_grid_thw.prod(-1)` in prompt/image encoding.

### Phase 0 Smoke Failure 1

256x256 / 1-step generation initially failed before denoising:

```
RuntimeError: CUDA driver error: invalid argument
at transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py:1180
split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
```

Independent repro:

| Tensor | Result |
|---|---|
| CUDA `int32` `.prod(-1)` | FAIL: `CUDA driver error: invalid argument` |
| CUDA `int64` `.prod(-1)` | FAIL: `CUDA driver error: invalid argument` |
| CUDA `float32` `.prod(-1)` | PASS |
| CUDA `bfloat16` `.prod(-1)` | PASS |

Workaround added to the harness and validated on the minimal repro.

### Phase 0 End-to-End Smoke

Command: `python src/bench_qie_diffusers.py --height 256 --width 256 --steps 2 --true-cfg-scale 4.0 ...`

| Measurement | Result |
|---|---|
| Output | `~/qie_cuda/outputs/phase0_smoke_256_2step.png` |
| Output file | 82 KB, 256x256, 8-bit RGB PNG |
| Pipeline load | 0.797 s |
| `.to("cuda")` wall | 368.092 s |
| Inference wall after load | 9.378 s |
| Transformer calls | 4 total, mean 1.808 s |
| Grouped CFG step hot-loop | step 1: 3.624 s, step 2: 3.607 s |
| Pipeline callback step deltas | step 1: 5.544 s, step 2: 3.609 s |
| Peak PyTorch allocated | 55.859 GiB |
| Peak PyTorch reserved | 56.422 GiB |
| Verdict | PASS for end-to-end CUDA execution. Quality not meaningful at 2 steps; output is textured/blue, not an eye-gate cat. |

The 1-step variant is not valid for this scheduler: it produced `timestep=nan` and a 270-byte PNG. Use at least 2 steps for smokes.

## Phase 1 - Diffusers Native Baseline

### 1024x1024 / 20-step Run

Started under `nohup setsid`:

- PID: `18976`
- Log: `~/qie_cuda/logs/phase1_baseline_1024_20step.log`
- Metrics: `~/qie_cuda/logs/phase1_baseline_1024_20step_metrics.json`
- Output: `~/qie_cuda/outputs/phase1_baseline_1024_20step.png`
- Prompt: `convert to black and white`
- Input: `~/qie_cuda/inputs/cat.jpg`
- Args: `--height 1024 --width 1024 --steps 20 --true-cfg-scale 4.0`

Completed successfully.

| Measurement | Result |
|---|---|
| Output | `~/qie_cuda/outputs/phase1_baseline_1024_20step.png` |
| Output file | 832 KB, 1024x1024, 8-bit RGB PNG |
| Eye check | PASS: recognizable cat converted to black and white |
| Pipeline component load | 0.800 s |
| `.to("cuda")` cold load | 429.288 s |
| Inference wall after load | 141.438 s |
| Total cold process wall, measured components | 571.526 s |
| Callback step total | 140.621 s |
| Callback step mean / median | 7.031 s / 6.936 s |
| DiT transformer calls | 40 total, mean 3.463 s, median 3.465 s |
| Grouped CFG DiT step total | 138.513 s |
| Grouped CFG DiT step mean / median | 6.926 s / 6.929 s |
| Peak PyTorch allocated | 57.970 GiB |
| Peak PyTorch reserved | 58.680 GiB |
| CUDA free after load | 8.232 GiB |
| CUDA free after inference | 1.365 GiB |

Step profile:

| Step | Callback delta (s) | Grouped DiT hot-loop (s) |
|---:|---:|---:|
| 1 | 8.997 | 6.896 |
| 2 | 6.872 | 6.871 |
| 3 | 6.887 | 6.887 |
| 4 | 6.895 | 6.894 |
| 5 | 6.892 | 6.892 |
| 6 | 6.900 | 6.900 |
| 7 | 6.905 | 6.905 |
| 8 | 6.915 | 6.915 |
| 9 | 6.928 | 6.928 |
| 10 | 6.916 | 6.915 |
| 11 | 6.930 | 6.930 |
| 12 | 6.945 | 6.944 |
| 13 | 6.942 | 6.941 |
| 14 | 6.946 | 6.945 |
| 15 | 6.954 | 6.954 |
| 16 | 6.949 | 6.949 |
| 17 | 6.958 | 6.958 |
| 18 | 6.965 | 6.964 |
| 19 | 6.960 | 6.960 |
| 20 | 6.963 | 6.963 |

Immediate interpretation:

- Native diffusers baseline is correct but far from the <=10 s target: post-load total is 141.4 s/image.
- Sequential CFG dominates exactly as expected: two transformer forwards per step, each ~3.46 s, and the grouped step hot-loop is ~6.93 s.
- Cold `.to("cuda")` is 429 s. Future measurements should use a persistent process for per-image latency; otherwise cold start dwarfs inference.
- Memory is tight but the full BF16 model and 1024x1024 activations fit on GB10 unified memory.

### Notes

- `nvidia-smi` on GB10 does not report FB memory totals (`N/A`), so Phase 1 memory receipts should rely primarily on `torch.cuda.max_memory_allocated()`, `torch.cuda.mem_get_info()`, and system memory deltas.
- Login PATH did not include `/usr/local/cuda-13.0/bin`; Phase scripts should export `PATH=$HOME/.local/bin:/usr/local/cuda-13.0/bin:$PATH` and `CUDA_HOME=/usr/local/cuda-13.0`.

## Phase 2 - Stacked Optimizations

### Step 1 - CFG Batching

Date: 2026-04-25 PDT

Harness changes:

- Added opt-in batched true-CFG path to `tools/qwen_image_edit/cuda/bench_qie_diffusers.py` / remote `~/qie_cuda/src/bench_qie_diffusers.py`.
- Flag: `--cfg-batching` stacks uncond + cond into one transformer batch, then chunks output and applies the same true-CFG combine and norm rescale.
- Diagnostic flag: `--cfg-batching-no-mask-padding` pads the shorter text condition but suppresses the padding mask to test whether masked SDPA caused the slow path.
- Kept the Phase 1 sequential path as default for reproducibility.

256x256 / 2-step smoke:

| Measurement | Result |
|---|---|
| Command mode | `--cfg-batching` |
| Output | `~/qie_cuda/outputs/phase2_cfgbatch_smoke_256_2step.png` |
| Metrics | `~/qie_cuda/logs/phase2_cfgbatch_smoke_256_2step_metrics.json` |
| Inference wall after load | 10.373 s |
| Transformer calls | 2 total for 2 steps |
| Transformer call mean | 4.081 s |
| Output file | 79 KB, 256x256, 8-bit RGB PNG |
| Verdict | PASS mechanical smoke: valid PNG and 1 transformer call per denoise step |

Canonical 1024x1024 / 20-step results:

| Variant | True CFG? | Calls | Inference wall | Step / grouped mean | Transformer call mean | Peak allocated | Eye check | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Phase 1 sequential baseline | Yes | 40 | 141.438 s | 6.926 s | 3.463 s | 57.970 GiB | PASS | Baseline |
| Batched CFG with padding mask | Yes | 20 | 184.471 s | 9.077 s | 9.077 s | 57.975 GiB | PASS | FAIL perf, slower than baseline |
| Batched CFG, no padding mask | Approx | 20 | 136.791 s | 6.713 s | 6.713 s | 57.975 GiB | PASS | Best Step 1 candidate, only -3.3% wall |
| No true CFG diagnostic | No | 20 | 72.052 s | 3.464 s | 3.464 s | 57.967 GiB | PASS | Diagnostic ceiling only, not accepted as CFG batching |

Receipts:

- Masked batched log: `~/qie_cuda/logs/phase2_cfgbatch_1024_20step.log`
- Masked batched metrics: `~/qie_cuda/logs/phase2_cfgbatch_1024_20step_metrics.json`
- Masked batched output: `~/qie_cuda/outputs/phase2_cfgbatch_1024_20step.png`
- No-mask batched log: `~/qie_cuda/logs/phase2_cfgbatch_nomask_1024_20step.log`
- No-mask batched metrics: `~/qie_cuda/logs/phase2_cfgbatch_nomask_1024_20step_metrics.json`
- No-mask batched output: `~/qie_cuda/outputs/phase2_cfgbatch_nomask_1024_20step.png`
- No-true-CFG diagnostic metrics: `~/qie_cuda/logs/phase2_no_true_cfg_1024_20step_metrics.json`

Visual gate:

- Batched masked output is visually the same black-and-white cat as Phase 1. Pixel diff versus Phase 1: mean abs diff `[0.2693, 0.2937, 0.2649]`, RMSE `[0.5247, 0.5509, 0.5196]`.
- Batched no-mask output also passes the eye check. Pixel diff versus Phase 1: mean abs diff `[0.6074, 0.6370, 0.6896]`, RMSE `[0.8289, 0.8591, 0.8945]`.
- No-true-CFG output is still a recognizable black-and-white cat but is only a diagnostic because it removes true CFG rather than batching it. Pixel diff versus Phase 1: mean abs diff `[4.9392, 6.2403, 6.7140]`, RMSE `[5.2301, 6.5712, 7.0675]`.

Interpretation:

- The exact batched CFG path reduced transformer calls from 40 to 20, but it introduced a padding mask because uncond and cond text lengths differ: uncond shape `(1, 209, 3584)`, cond shape `(1, 213, 3584)`, combined shape `(2, 213, 3584)`.
- That mask appears to force a much slower attention backend on this PyTorch nightly / GB10 stack: call mean regressed from 3.463 s sequential to 9.077 s batched.
- Suppressing the padding mask restores the faster attention path, but a batch=2 transformer call still costs 6.713 s, close to two sequential 3.463 s calls. Net improvement is only 4.647 s/image, or 3.3% versus Phase 1.
- Step 1 does not deliver the expected 40-50% win on this stack. Use `--cfg-batching --cfg-batching-no-mask-padding` only as the current best visual-pass candidate; keep the default sequential path for exact baseline comparisons.

Next:

- Proceed to FP8 quantization and/or `torch.compile`; CFG batching alone is not enough to reverse the MLX M4 Max comparison.

### Step 2 - FP8 Quantization

Date: 2026-04-25 PDT

Harness changes:

- Added opt-in transformer FP8 loading to `tools/qwen_image_edit/cuda/bench_qie_diffusers.py` / remote `~/qie_cuda/src/bench_qie_diffusers.py`.
- Flag: `--fp8-backend {none,torchao-float8wo,torchao-float8dq}`. Default remains BF16/no quantization.
- `torchao-float8dq` loads only the QwenImage transformer through Diffusers `TorchAoConfig(Float8DynamicActivationFloat8WeightConfig(... e4m3fn ...))`.
- `torchao-float8wo` loads only the QwenImage transformer through Diffusers `TorchAoConfig(Float8WeightOnlyConfig(weight_dtype=torch.float8_e4m3fn))`.
- Optional flag: `--fp8-modules-to-not-convert` for future selective quantization sweeps.

Environment/package findings:

| Option | Result |
|---|---|
| TransformerEngine | BLOCKED. `transformer-engine[pytorch]` had no prebuilt wheel for `torch==2.13.0.dev20260424+cu130` on aarch64. Source build first failed under isolated build, then on missing cuDNN/NCCL include paths. After explicit venv NVIDIA include/lib paths, build succeeded only by force-reinstalling `torch==2.11.0`, which invalidated the Phase 1 comparison and still failed import after restoring nightly Torch due `libtransformer_engine.so: undefined symbol: cublasLtGroupedMatrixLayoutInit_internal`. TE was uninstalled from the remote venv after receipts. |
| bitsandbytes `0.49.2` | Installed/imports, but Diffusers `BitsAndBytesConfig` only exposes int8 plus FP4/NF4 4-bit paths; FP8 kwargs are ignored. Not a Step 2 FP8 candidate on this stack. |
| Diffusers + TorchAO `0.17.0` | Works mechanically with `QwenImageTransformer2DModel.from_pretrained(..., quantization_config=TorchAoConfig(...))` and keeps Phase 1 Torch nightly intact. |

Install/check receipts:

- TE isolated install log: `~/qie_cuda/logs/phase2_fp8_te_install.log`
- TE no-isolation install log: `~/qie_cuda/logs/phase2_fp8_te_install_noiso.log`
- TE explicit cuDNN path install log: `~/qie_cuda/logs/phase2_fp8_te_install_cudnnpath.log`
- TE explicit NVIDIA paths install log: `~/qie_cuda/logs/phase2_fp8_te_install_nvidia_paths.log`
- Torch nightly restore log: `~/qie_cuda/logs/phase2_restore_torch_nightly.log`
- bitsandbytes install log: `~/qie_cuda/logs/phase2_fp8_bnb_install.log`
- TorchAO install log: `~/qie_cuda/logs/phase2_fp8_torchao_install.log`
- Post-clean venv: `torch==2.13.0.dev20260424+cu130`, `torchvision==0.27.0.dev20260424+cu130`, `torchao==0.17.0`, `bitsandbytes==0.49.2`; TransformerEngine removed.

256x256 / 2-step smokes:

| Variant | Inference wall | Transformer calls | Transformer call mean | Grouped step mean | Peak allocated | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Phase 0 BF16 smoke | 9.378 s | 4 | 1.808 s | 3.616 s | 55.859 GiB | Baseline smoke |
| TorchAO FP8 dynamic act+weight | 10.835 s | 4 | 2.255 s | 4.509 s | 36.834 GiB | PASS mechanical, FAIL perf |
| TorchAO FP8 weight-only | 15.507 s | 4 | 3.425 s | 6.849 s | 36.854 GiB | PASS mechanical, FAIL perf; canonical skipped |

Canonical 1024x1024 / 20-step results:

| Variant | True CFG? | Calls | Inference wall | Step / grouped mean | Transformer call mean | Peak allocated | Eye check | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|---|
| Phase 1 BF16 baseline | Yes | 40 | 141.438 s | 6.926 s | 3.463 s | 57.970 GiB | PASS | Baseline |
| TorchAO FP8 dynamic act+weight | Yes | 40 | 178.516 s | 8.809 s | 4.404 s | 38.943 GiB | PASS | FAIL perf, +26.2% slower |

Receipts:

- Dynamic FP8 smoke log: `~/qie_cuda/logs/phase2_fp8dq_smoke_256_2step.log`
- Dynamic FP8 smoke metrics: `~/qie_cuda/logs/phase2_fp8dq_smoke_256_2step_metrics.json`
- Dynamic FP8 smoke output: `~/qie_cuda/outputs/phase2_fp8dq_smoke_256_2step.png`
- Dynamic FP8 canonical log: `~/qie_cuda/logs/phase2_fp8dq_1024_20step.log`
- Dynamic FP8 canonical metrics: `~/qie_cuda/logs/phase2_fp8dq_1024_20step_metrics.json`
- Dynamic FP8 canonical output: `~/qie_cuda/outputs/phase2_fp8dq_1024_20step.png`
- Weight-only FP8 smoke log: `~/qie_cuda/logs/phase2_fp8wo_smoke_256_2step.log`
- Weight-only FP8 smoke metrics: `~/qie_cuda/logs/phase2_fp8wo_smoke_256_2step_metrics.json`
- Weight-only FP8 smoke output: `~/qie_cuda/outputs/phase2_fp8wo_smoke_256_2step.png`

Visual gate:

- TorchAO FP8 dynamic canonical output is a recognizable black-and-white cat and passes the eye check.
- Pixel diff versus Phase 1: mean abs diff `[1.8057, 1.7782, 1.8004]`, RMSE `[2.1684, 2.1320, 2.1523]`.

Interpretation:

- TorchAO FP8 materially reduces PyTorch peak allocation by about 19.0 GiB (`57.970 -> 38.943 GiB`) and shortens cold `.to("cuda")` (`429.288 -> 94.373 s`) because the transformer weights are quantized before the device transfer.
- The actual denoise loop is slower: transformer call mean regressed from `3.463 s` to `4.404 s`. Dynamic quant/dequant overhead dominates any Blackwell FP8 matmul win in this eager Diffusers path.
- Weight-only FP8 is worse in the 256x256 smoke and should not be used for canonical latency. It is only useful as a memory reduction diagnostic.
- Step 2 does not deliver the expected 30-50% wall reduction on this stack. Do not include FP8 in the performance stack unless a later `torch.compile` run flips the TorchAO dynamic path from slower to faster.

Next:

- Proceed to `torch.compile` with BF16 first. If BF16 compile is green, optionally retest `torchao-float8dq + torch.compile` as an interaction check, but FP8 alone is red.

### Step 3 - torch.compile

Date: 2026-04-25 PDT

Harness changes:

- Added opt-in compile flags to `tools/qwen_image_edit/cuda/bench_qie_diffusers.py` / remote `~/qie_cuda/src/bench_qie_diffusers.py`.
- Flags: `--torch-compile`, `--torch-compile-mode` defaulting to `max-autotune`, `--torch-compile-fullgraph`, and `--compile-warmup-runs`.
- Compile is applied after `.to("cuda")`: `pipe.transformer = torch.compile(pipe.transformer, mode=..., fullgraph=...)`.
- Because compile is lazy, warmup generations are run before the measured generation when `--compile-warmup-runs > 0`; only the post-warmup run is used for the primary latency metrics.
- True CFG calls the transformer twice per denoise step. The compiled CUDA Graph output was overwritten between cond/uncond calls until the harness added `torch.compiler.cudagraph_mark_step_begin()` and cloned tensor outputs at the wrapper boundary.

Troubleshooting receipts:

- Failed compile smoke v1 log: `~/qie_cuda/logs/phase2_compile_smoke_256_2step.log`
- Failed compile smoke v2 log: `~/qie_cuda/logs/phase2_compile_smoke_256_2step_v2.log`
- Failure mode: `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`.
- Final smoke log: `~/qie_cuda/logs/phase2_compile_smoke_256_2step_v3.log`
- Final smoke metrics: `~/qie_cuda/logs/phase2_compile_smoke_256_2step_v3_metrics.json`
- Final smoke output: `~/qie_cuda/outputs/phase2_compile_smoke_256_2step_v3.png`

256x256 / 2-step smoke:

| Variant | Warmup inference | Measured inference | Transformer call mean | Grouped step mean | Peak allocated | Verdict |
|---|---:|---:|---:|---:|---:|---|
| Phase 0 BF16 smoke | N/A | 9.378 s | 1.808 s | 3.616 s | 55.859 GiB | Baseline smoke |
| BF16 `torch.compile` | 27.382 s | 6.611 s | 1.447 s | 2.894 s | 55.860 GiB | PASS, -29.5% smoke wall |

Canonical 1024x1024 / 20-step results:

| Variant | True CFG? | Calls | Warmup inference | Measured inference | Step / grouped mean | Transformer call mean | Peak allocated | Eye check | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| Phase 1 BF16 baseline | Yes | 40 | N/A | 141.438 s | 6.926 s | 3.463 s | 57.970 GiB | PASS | Baseline |
| BF16 `torch.compile` | Yes | 40 | 221.813 s | 113.425 s | 5.601 s | 2.801 s | 57.961 GiB | PASS | PASS, -19.8% wall |

Canonical receipts:

- Log: `~/qie_cuda/logs/phase2_compile_1024_20step.log`
- Metrics: `~/qie_cuda/logs/phase2_compile_1024_20step_metrics.json`
- Output: `~/qie_cuda/outputs/phase2_compile_1024_20step.png`
- Compile warning: `Not enough SMs to use max_autotune_gemm mode`; despite this, Inductor still reduced post-warmup transformer call time.

Visual gate:

- `torch.compile` canonical output is a recognizable black-and-white cat and passes the eye check.
- Pixel diff versus Phase 1: mean abs diff `[1.1923, 1.0827, 0.8258]`, RMSE `[1.6259, 1.5059, 1.1604]`.

Interpretation:

- `torch.compile` is the first clearly useful CUDA lever in Phase 2, but the gain is modest rather than dramatic: post-warmup canonical wall improves by `28.013 s`, or `19.8%`.
- The denoise loop remains dominated by true-CFG sequential calls. Compile reduces each transformer call from `3.463 s` to `2.801 s`, but there are still 40 calls.
- Compile warmup is expensive on first same-shape generation: `221.813 s` total for canonical, with the first step paying about `115 s`. Any production path needs a persistent compiled process, not one process per image.

Next:

- Proceed to CacheDIT. For the final stack, benchmark both `torch.compile` alone and `torch.compile + --cfg-batching --cfg-batching-no-mask-padding`; Step 1 no-mask may compound with compile even though it was small in eager mode.
