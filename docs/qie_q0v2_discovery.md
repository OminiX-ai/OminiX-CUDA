# QIE Q0-v2 Discovery Report (ac02, 910B4)

## Status: YELLOW — binary builds + weights load, but pipeline crashes on two CANN op gaps. Dispatching A1 now would stall immediately.

## Host confirmation

`npu-smi` confirms ac02 as **Ascend 910B4 V1, 32 GB HBM, Board 0x34 / BOM 1, PCIe 0000:C1:00.0, CANN 8.3.RC1**. 2856 MB baseline HBM usage, 0 % AICore at probe time, no competing processes. Host storage `/home/ma-user/work` is a 1.3 TB SSD, 1.5 TB RAM, 192 CPU cores — plenty of headroom for download + build + offload.

Proxy note: ac02 routes HF through `proxy-notebook.modelarts.com:8083`. Direct `huggingface.co` fetches return intermittent 503 via that proxy; `HF_ENDPOINT=https://hf-mirror.com` worked reliably once the first probe failed.

## ominix_diffusion state

Clone at `~/work/OminiX-Ascend-w1` tracks `main @ 710f545`. The `tools/ominix_diffusion/` tree is present with the full stable-diffusion.cpp lineage (SD1/SD2/SDXL/SD3/Flux/Flux2/Qwen-Image/Wan2/Anima/Z-Image), and `SDVersion::VERSION_QWEN_IMAGE` is wired through `QwenImageModel`, `QwenImageRunner`, and the `QwenImageEditPlusPipeline` branch in `conditioner.hpp:1991`. Edit mode is auto-enabled when `!vae_decode_only && is_qwen_image(version)` — the CLI flag is `-r / --ref-image` (Flux-Kontext alias).

Prior to this probe the binary had been **configured but never built** (cmake cache present, `build-w1/bin/` contained only libs + `qwen_tts`). I built `ominix-diffusion-cli` in-place with the existing `GGML_CANN=ON`, `SD_CANN=ON`, `LLAMA_BUILD_TOOLS=ON` flags — clean build, ~2 min, no fork code changes. Binary at `~/work/OminiX-Ascend-w1/build-w1/bin/ominix-diffusion-cli`.

## Weight availability for Qwen-Image-Edit-2512-4bit

Key finding: **`Qwen-Image-Edit-2512` does not exist as a released artefact**. Qwen org's latest edit variant is `Qwen/Qwen-Image-Edit-2511` (Nov 2025, 175 k downloads). `Qwen-Image-2512` exists only as the base T2I model (not edit). The earlier Q0 brief and the deck's Appendix C refer to a model that shipped under a different SKU — the correct name is `Qwen-Image-Edit-2509` (previous edit) or `Qwen-Image-Edit-2511` (current). Community GGUFs exist for both:

| Source | 4-bit size | 8-bit size | VAE | mmproj |
|---|---|---|---|---|
| QuantStack/Qwen-Image-Edit-2509-GGUF | Q4_0 11.9, Q4_K_M 13.1 | 21.8 | external | external |
| QuantStack/Qwen-Image-Edit-GGUF (base) | Q4_0 11.9, Q4_K_M 13.1 | 21.8 | external | external |
| unsloth/Qwen-Image-Edit-2511-GGUF | Q4_0 11.9, Q4_K_M 13.2 | 21.8 | external | external |

Companion weights:
- `unsloth/Qwen2.5-VL-7B-Instruct-GGUF`: `Q4_0` 4.44 GB, `Q4_K_M` 4.68 GB, `Q8_0` 8.1 GB; `mmproj-BF16` 1.35 GB, `mmproj-F16` 1.35 GB.
- `Comfy-Org/Qwen-Image_ComfyUI/split_files/vae/qwen_image_vae.safetensors`: 254 MB (bf16).

Downloaded for this probe (on ac02, ~20 GB total, mirror): `Qwen-Image-Edit-2509-Q4_0.gguf` (11.9 GB), `Qwen2.5-VL-7B-Instruct-Q4_0.gguf` (4.44 GB), `mmproj-BF16.gguf` (1.35 GB), `qwen_image_vae.safetensors` (0.25 GB). Q4_K_M variants were also fetched and discarded after the CANN gap below.

Both edit variants are **full fine-tuned checkpoints**, not LoRA deltas — the 2509 → 2511 step is a fresh 20 B DiT, not a small adapter. Contract numbers can therefore be computed against any single-variant checkpoint.

## Baseline run result

All three runs **failed at inference time** — none produced an output image. Weight load succeeded in every case, with `total params memory size = 18.4 – 19.8 GB (VRAM only)` reported by stable-diffusion.cpp (text_encoders 5.8 – 7.1 GB, diffusion 11.4 – 12.5 GB, VAE 0.14 – 0.24 GB). That leaves ~12 – 14 GB of the 32 GB HBM free for activation / KV / workspace — **HBM is not the blocker on 910B4**.

### Failure 1 — Q4_K_M edit with ref-image

`-r cat.jpg -p "convert to black and white"`, 20 steps, 256×256, Q4_K_M weights. Runs to `QwenImageEditPlusPipeline` log, then aborts in Qwen2.5-VL vision encode with `ascendc/gather_v3/gather_v3_base.h:137` assertion, AIV cores 0 – 39 reporting indices like `-1085232137`, `1051041638` (classic float32-bit-pattern-being-read-as-int32 signature). Wall-time to crash: 16.9 s. Output: none.

### Failure 2 — Q4_K_M T2I (no ref image)

Same weights, no `-r`. Aborts earlier at `aclnn_ops.cpp:2670: Unsupported type for mul_mat`. Root cause: the CANN `ggml_cann_mul_mat` switch only supports `F32 / F16 / BF16 / Q4_0 / Q8_0` — **Q4_K / Q5_K / Q6_K are not wired in**. Q4_K_M is categorically unusable on this backend.

### Failure 3 — Q4_0 edit with ref-image

Re-downloaded Q4_0 weights (supported for mul_mat) and re-ran. T2I-only path now trips `aclnn_ops.cpp:2272: Unsupported tensor type for GGML_OP_GET_ROWS`. CANN `get_rows` only supports `F16 / F32 / BF16 / Q8_0` — **Q4_0 and Q4_1 are not wired in for embedding lookup**, which matters because Qwen-Image keeps 3 × Q4_1 tensors in the text-encoder embedding table even at Q4_0 export. With `-r`, the same vision-encoder `gather_v3` crash from Failure 1 fires before GET_ROWS is reached. Wall-time to crash: ~18 s.

### Interpretation

Two independent CANN op gaps block end-to-end inference at Q4:
1. **GET_ROWS does not support Q4_0 / Q4_1** — text encoder embedding lookup on a Q4 checkpoint aborts at first forward. Only Q8_0 (on the lookup table) is safe.
2. **Vision encoder on `gather_v3`** — the mmproj → gather index path receives float-bit-pattern inputs on AIV. This is likely a graph-builder bug in the Qwen2.5-VL wiring, independent of the quant format (same crash on Q4_K_M and Q4_0).

The README's "910B2 + CANN 8.5.0 — 20 steps / 32.04 s on Qwen-Image Q8_0" number was measured with **Q8_0 weights on a newer CANN**, against **non-edit T2I** (no vision encoder, no ref image). Neither precondition holds on ac02 today.

HBM + build + weight-plumbing are all fine. The blockers are backend-op coverage, not memory.

## Realistic optimization scoping for contract

Three paths forward, ranked:

**(a) Unblock existing ominix_diffusion on 910B4 at Q8_0 T2I first.** Download `Qwen-Image-Edit-2509-Q8_0.gguf` (21.8 GB) + `Qwen2.5-VL-7B-Q8_0.gguf` (8.1 GB) + mmproj + VAE ≈ 31.5 GB on disk, ~29 GB HBM at load — leaves ~3 GB for activation / KV, which is enough for 512×512 but tight for 1024×1024. Requires PM confirm for the >30 GB download gate. This validates the T2I path and gets a real s/iter number on 910B4. Vision-encoder bug still needs a separate fix before edit mode works.

**(b) Fix the two CANN op gaps in-fork.** GET_ROWS for Q4_0/Q4_1 is a 30 – 50 line addition mirroring the existing Q8_0 case in `aclnn_ops.cpp`. The vision-encoder gather_v3 bug is harder — will need to print the graph around the first gather to find the dtype mismatch. Both would land as patches against `ggml/src/ggml-cann/`. 1 – 2 weeks of A1-level work before any baseline exists.

**(c) Procure 910B2/B3 (64 GB HBM).** Unblocks Q8_0 at 1024×1024 with activation headroom and matches the README test path exactly. Does NOT fix the vision-encoder crash — that blocks edit mode on any Ascend variant until patched. But (c) + (b's vision-encoder half) would be the cleanest baseline.

MLX reference remains unchanged: Apple Silicon 4-bit Qwen-Image-2512 at ~30 fps (MLX-mx) is not reachable on 910B4 due to memory-bandwidth gap and mandatory Q8_0. Realistic 910B4 ceiling after TTS-playbook optimization: **~3 – 5 × lift over baseline** (aclGraph step-keyed + fusion), not 30 ×. TTS playbook parts that transfer: step-stable-shape aclGraph (diffusion steps are identical-shape), WSPOOL for workspace reuse. Parts that do NOT transfer: RmsNorm fusion (diffusion uses LayerNorm), byte-identity gates (Q8 + bf16 compute can't be byte-identical, only eye-check-identity against MLX reference).

## Artefacts

- Host + build: `~/work/OminiX-Ascend-w1/build-w1/bin/ominix-diffusion-cli` (built this probe)
- Weights: `~/qie_q0v2/weights/` on ac02
  - `Qwen-Image-Edit-2509-Q4_0.gguf` (11.9 GB)
  - `Qwen2.5-VL-7B-Instruct-Q4_0.gguf` (4.44 GB)
  - `mmproj-BF16.gguf` (1.35 GB)
  - `split_files/vae/qwen_image_vae.safetensors` (243 MB)
- Test image: `~/qie_q0v2/test/cat.jpg` (256×256 JPEG, 8.8 KB)
- CANN op gaps to patch (for reference):
  - `ggml/src/ggml-cann/aclnn_ops.cpp:2670` — `ggml_cann_mul_mat` switch, add Q4_K / Q5_K / Q6_K
  - `ggml/src/ggml-cann/aclnn_ops.cpp:2272` — `ggml_cann_get_rows` switch, add Q4_0 / Q4_1
  - `ascendc/gather_v3` dtype mismatch in mmproj path (needs graph print to localise)

## Verdict

**YELLOW for "dispatch QIE A1 now"**. The fork builds, weights load under the 32 GB budget (Q4: ~20 GB HBM, Q8 projected: ~29 GB), and the edit pipeline is correctly wired on the CPU side. But two CANN op gaps and one vision-encoder gather bug mean the first A1 task cannot be "baseline + optimize" — it has to be "unblock the backend first". Recommend: A1 scoped as op-coverage + vision-encoder fix (1 – 2 weeks), with baseline-plus-optimization moved to A2 once a green end-to-end run exists. PM should also decide whether to re-align deck Appendix C / contract text from "Qwen-Image-Edit-2512" to "Qwen-Image-Edit-2511" (current release) or "-2509" (the variant tested in this probe).
