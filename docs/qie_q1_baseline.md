# QIE Q1 Baseline Receipt — Q4_0 first-landing on ac02 (910B4)

**Status**: Q1 GATE PASSES at the **256×256 2-step smoke** configuration. Longer sequences / more steps fail with downstream NaN (DiT activation overflow, NOT a crash — see §Known regression below). Per contract §Q1.4 "any quality, any steps/sec — just 'doesn't crash'", the 2-step 256×256 run clears the gate with a valid output image; longer configurations are a Q2-scope follow-up.

**Verified-by**: Agent QIE-Q1, ac02 (notebook-c768c7a7..., 910B4, CANN 8.3.RC1), 2026-04-22.

## Q1.4 HARD GATE — 2-step 256×256 edit smoke

Canonical task: "convert to black and white" (edit mode, cat.jpg ref image).

### Command

```bash
export GGML_CANN_QUANT_BF16=on
./build-w1/bin/ominix-diffusion-cli \
  -M img_gen \
  --diffusion-model ~/qie_q0v2/weights/Qwen-Image-Edit-2509-Q4_0.gguf \
  --llm           ~/qie_q0v2/weights/Qwen2.5-VL-7B-Instruct-Q4_0.gguf \
  --llm_vision    ~/qie_q0v2/weights/mmproj-BF16.gguf \
  --vae           ~/qie_q0v2/weights/split_files/vae/qwen_image_vae.safetensors \
  -r              ~/qie_q0v2/test/cat.jpg \
  -p 'convert to black and white' \
  --steps 2 --cfg-scale 1.0 -W 256 -H 256 \
  -o /tmp/qie_smoke_bf16.png -v
```

### Receipt (short smoke)

| Phase | Wall | Notes |
|---|---|---|
| Weight load | 22 s | text_encoders 5.1 GB, diffusion 11.4 GB, vae 242 MB = 16.7 GB total |
| VAE encode ref | 12.3 s | 256×256 input |
| Qwen2.5-VL text+vision encode | 6.8 s | 61 MB compute buffer, `encoder/cond.c_crossattn` OK range `[-151.29, 104.31]` |
| Denoise (2 steps) | ~100 s | ~50 s/step; `diffusion/x_0` OK range `[-1.45, 1.49]` |
| VAE decode | 20 s | 469 MB compute buffer; `vae/decoded_image` OK range `[0.04, 0.83]` |
| **Total wall** | **145 s** | exit 0, output is a recognisable cat photograph |

- **Output**: `/tmp/qie_smoke_bf16.png` on ac02, **123 KB**, **256×256 8-bit RGB PNG**, visually confirmed valid cat image (retrieved to Mac at `/tmp/qie_smoke_bf16.png`).
- **HBM peak**: weights 16.7 GB resident + 680 MB DiT compute buffer + 470 MB VAE decode buffer + ~2 GB CANN overhead → **~20 GB peak**, well under 32 GB ceiling and under the 30 GB budget requested by contract §Q8.3.
- **NaN checks**: encoder/cond OK, diffusion/x_0 OK, vae/decoded_image OK.
- **No crashes**: exit 0 every run; gather_v3 not hit (see `qie_q1_3_gather_v3_probe.md`).

This is the **Gate 2 PASS** per the contract: `baseline run produces a valid output image (any quality, any steps/sec — just "doesn't crash")`.

## Known regression — step-count / sequence-length NaN

Behaviour observed:

| W × H | Steps | Result |
|---|---|---|
| 256×256 | 2 | **OK (valid image)** |
| 256×256 | 3 | NaN at diffusion/x_0 (blank image output) |
| 256×256 | 4 | NaN |
| 256×256 | 20 | NaN |
| 512×512 | 2 | NaN |
| 512×512 | 20 | NaN |
| 256×256, batch=2, 2 steps each | — | OK for both (2 separate cat images) |

Two axes of regression:
1. **Sequence-length**: 512×512 (DiT seq ≈ 2048) fails even at 2 steps, while 256×256 (seq ≈ 512) is fine at 2 steps. Classic FP16 accumulation overflow signature in long-sequence attention or modulation add.
2. **Step-count**: 256×256 passes at 2 steps but fails from 3+ steps. Each step feeds its output as the next step's input; compound error or activation-range drift across timesteps drives some op over a precision threshold.

`GGML_CANN_QUANT_BF16=on` helps (without it, 256×256 2-step also NaNs), but it only widens the quant mul_mat accum dtype; full-precision ops (attention, RMSNorm, etc.) and cross-step state are still FP16/BF16.

Hypotheses, none yet verified (out of Q1 scope):
- **H1**: The CPU-dequant fallback in `ggml_cann_mul_mat_quant_cpu_dequant` is not bit-exact across calls — e.g., aclnnMm internal tiling is non-deterministic, and across 3+ steps the tiny delta blows past a threshold. Mitigation: pre-dequant Q4_1 / K-quant weights ONCE at buffer load time to a resident FP16 backend buffer, skip the per-call D2H/H2D round-trip entirely.
- **H2**: Attention softmax at sequence 2048+ overflows FP16 compute; vendor bug in `aclnnFusedInferAttentionScore` for long sequences with no output-scale or logit-clamp option.
- **H3**: Modulation / time-embedding broadcast compounds badly at certain timestep ranges.

**Debug handle for Q2**: add a per-step NaN check in `stable-diffusion.cpp::sample()` (currently only the final x_0 is checked) to bisect which step first introduces NaN. Also try the 20-step 256×256 with --diffusion-fa and --flash-attn flags to route attention through the FIA kernel and see if that changes the failure pattern.

**This is NOT blocking Q1** — the contract explicitly permits "any quality". It IS the first thing Q2 needs to tackle before meaningful optimization, because without a correct baseline at 20 steps, later perf lifts can't be measured.

## Q1.5 — upstream PR readiness

Two clean commits on branch `qie-q1-ggml-cann-patches` off `ymote/OminiX-Ascend` main @ 710f545:

```
389204a fix(ggml-cann): add Q4_1 / Q5_* / K-quant MUL_MAT fallback via CPU dequant
8a4cdbe fix(ggml-cann): add Q4_0/Q4_1 GET_ROWS dispatch for embedding lookup
```

Properties:
- Atomic: each commit addresses one dispatch gap (get_rows vs mul_mat) in isolation.
- No "Co-Authored-By" attribution.
- Well-documented commit messages with context (why CPU dequant, why 2D-only scope, what follow-up is needed).
- Patch file exported to Mac at `/tmp/qie_q1_patches.patch` (git-format-patch, 379 lines, 2 patches).
- Rebased cleanly against `ggerganov/llama.cpp/ggml/src/ggml-cann/` scope (the ggml-cann tree is vendored into OminiX-Ascend but identical to llama.cpp upstream).

**Upstream submission**: per mission scope, NOT submitted yet. Commits are ready for PM review before upstream push.

## Remaining gaps (not Q1 scope)

- **Q1.3 gather_v3**: DEFERRED — not hit on Q4_0 path. See `qie_q1_3_gather_v3_probe.md`. Re-opens if Q4_K_M becomes a quality requirement.
- **K-quant native mul_mat**: all K-quants (Q2_K..Q6_K) go through CPU-dequant fallback. Fine for correctness; **128 / 813 diffusion weights hit this per step** — dominates wall cost at Q4. Native aclnnWeightQuantBatchMatmulV3 support for 256-super-block K-quant layout is the right fix but needs CANN vendor cooperation.
- **Q4_1 via CANN transform**: currently no `ggml_backend_cann_transform_q4_1` so Q4_1 takes the same CPU-dequant path as K-quants. Adding a Q4_1 transform + aclnnWeightQuantBatchMatmulV2 dispatch (with 2 scales per block for `d` + `m`) is doable in the same shape as existing Q4_0.
- **Session-lifetime weight residency**: per hypothesis H1 above, pre-dequanting at buffer load time would eliminate the per-call D2H/H2D cost AND likely fix the step-count NaN regression. Sits in the Q2 native-engine brief naturally.

## Host rules honoured

- **ac02 only**: yes, no cross-host work touched ac01 / ac03.
- **HBM budget**: well under 30 GB during run (peak ~20 GB).
- **No libruntime.so EZ9999 errors**: none observed.
- **No hidden --no-verify / --no-gpg-sign**: commits are atomic.
- **Patch-file push mechanism**: patch at `/tmp/qie_q1_patches.patch` on Mac for PM to apply to fork.

## Summary

Q1 (ggml-cann backend unblock) **lands GREEN** for the minimum-bar gate defined in the contract. All three originally-identified bugs are resolved or scoped out:
- **Q1.1 mul_mat Q4_K/Q5_K/Q6_K** → landed as a broader CPU-dequant fallback covering Q4_1/Q5_0/Q5_1/Q2_K..Q6_K.
- **Q1.2 get_rows Q4_0/Q4_1** → landed with inlined CANN-transform reverse + CPU dequant round-trip.
- **Q1.3 gather_v3** → not a Q4_0 blocker; scope closes per separate probe doc.

QIE pipeline produces valid output end-to-end at 256×256 / 2-step. The longer-run NaN regression is a clear Q2 starting problem but does not violate Q1 gate terms.
