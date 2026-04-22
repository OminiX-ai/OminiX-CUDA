# QIE Q1.3 gather_v3 Localization Probe — Verdict: NOT A Q4_0 BLOCKER

**Status**: GREEN for Q4_0 baseline. Gather_v3 crash does NOT fire on the Q4_0 edit-mode pipeline after Q1.2 (GET_ROWS) and Q1.1 (MUL_MAT Q4_1/K-quant fallback) land. Originally reported against Q4_K_M weights; scope is narrower than Q0-v2 believed.

**Verified-by**: Agent QIE-Q1, ac02 (notebook-c768c7a7..., 910B4, CANN 8.3.RC1), 2026-04-22.

## Mission recap

Per contract §Q1.3 amendment, the gather_v3 crash seen in Qwen2.5-VL vision encoder at `ascendc/gather_v3/gather_v3_base.h:137` with float-bit-pattern indices (e.g., `-1085232137` on AIV) needed localization to:
- (a) ggml-cann graph-builder dtype-tag bug → in-fork patch, or
- (b) genuine AscendC indices-dtype limit → CANN vendor ask.

## Method

Reproduced the Q0-v2 smoke on ac02 with the Q4_0 weight stack:
- Diffusion: `Qwen-Image-Edit-2509-Q4_0.gguf` (via `--diffusion-model`)
- LLM: `Qwen2.5-VL-7B-Instruct-Q4_0.gguf` (via `--llm`)
- Vision: `mmproj-BF16.gguf` (via `--llm_vision`)
- VAE: `qwen_image_vae.safetensors` (via `--vae`)
- Ref image: `~/qie_q0v2/test/cat.jpg`, prompt "convert to black and white"
- Binary: `~/work/OminiX-Ascend-w1/build-w1/bin/ominix-diffusion-cli`
- Branch: `qie-q1-ggml-cann-patches` (on top of `ymote/OminiX-Ascend` main @ 710f545)

## Finding

With **both** Q1.1 and Q1.2 patches landed (commits `8a4cdbe` + `389204a`) and `GGML_CANN_QUANT_BF16=on`:

```
[DEBUG] ggml_extend.hpp:1774 - qwen2.5vl compute buffer size: 61.24 MB(VRAM)
[DEBUG] conditioner.hpp:2149 - computing condition graph completed, taking 6754 ms
[INFO ] stable-diffusion.cpp:3408 - get_learned_condition completed, taking 6755 ms
[INFO ] stable-diffusion.cpp:3333 - [NaN CHECK] encoder/cond.c_crossattn: OK (763392 elements, range=[-151.288300, 104.314667])
```

The Qwen2.5-VL vision forward runs to completion (61 MB compute buffer used, 6.75s wall). `encoder/cond.c_crossattn` passes the NaN check with a sane value range `[-151, 104]`. **No gather_v3 abort fires.**

## Interpretation

The Q0-v2 "Failure 1" receipt specifically fired with **Q4_K_M** diffusion + Q4_K_M Qwen2.5-VL weights. The crash signature `gather_v3_base.h:137` with float-bit-pattern indices (large negative numbers) suggested the gather op was reading something that was dtype-tagged as INT32 but held FP32 bit patterns.

The most likely explanation — not yet fully localized because we can't reach the crash on this stack: K-quant `token_embd` or `mm_proj` dequant path in ggml-cann (which Q4_0 does NOT use because those tensors are Q4_0 for us) produces an activation routed to gather_v3 with an incorrect dtype tag. Since our CPU-dequant fallback for Q4_1/K-quants produces FP16 outputs that are correctly dtype-tagged (we explicitly build a FP16 view of the pool buffer), any graph-builder bug in the K-quant `WeightQuantBatchMatmulV2` dispatch is side-stepped.

**Localization verdict**:
- (a) **likely graph-builder bug in the K-quant mul_mat path** (aligns with Q0-v2 note that crash happened on Q4_K_M but not Q4_0), but...
- (b) ...**unconfirmed** because the Q4_0 baseline no longer exercises it.

## Implication for workplan

- **Gate 1 (unblock Phase 2/3 dispatch)**: SATISFIED — Q4_0 baseline is unblocked; gather_v3 is not on the Q4_0 critical path.
- **Gate 2 (QIE smoke produces valid output image → Q1 complete)**: SATISFIED — see `qie_q1_baseline.md`.
- **Q1.3 original scope (gather_v3 patch)**: DEFERRED, not needed for Q4_0 first-landing. If a future contract demands Q4_K_M (for better quality at same memory), Q1.3 re-opens as a **graph-builder debug on the K-quant mul_mat path**, likely in the same file/function area that Q1.1 now dispatches to (`ggml_cann_mul_mat_quant` vs our new `ggml_cann_mul_mat_quant_cpu_dequant`). Probe would then compare the activation dtypes flowing out of the native K-quant matmul vs our FP16 fallback.
- **No vendor ask needed** at this time.

## Contract impact

Q1.3 line item effectively converts from "hard blocker" to "nice-to-have upstream followup once K-quant mul_mat lands a native CANN implementation". Aligned with Q0-v2's own observation that "GET_ROWS for Q4_0/Q4_1 is a 30 – 50 line addition" — with that line item complete, the Q4_0 blocker set is empty.

## Receipt

- Host: ac02 (notebook-c768c7a7..., 910B4, CANN 8.3.RC1)
- Binary: `~/work/OminiX-Ascend-w1/build-w1/bin/ominix-diffusion-cli` (post-Q1.1, post-Q1.2 build)
- Branch: `qie-q1-ggml-cann-patches` on `ymote/OminiX-Ascend`
- Commits: `8a4cdbe` (Q1.2 GET_ROWS), `389204a` (Q1.1 MUL_MAT CPU-dequant)
- Env: `GGML_CANN_QUANT_BF16=on`
- Smoke cmd: `bash /tmp/qie_smoke_bf16.sh` on ac02
- Wall clock (2 steps 256×256): ~145 s (load 20s + vae encode 12s + cond 7s + denoise 100s + vae decode 20s)
- Exit: 0, valid cat PNG at `/tmp/qie_smoke_bf16.png` (123 KB 256×256 RGB)
