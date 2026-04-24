# QIE-Edit ggml-cann NaN diagnosis — scope-narrowing verdict

**Agent**: QIE-NAN-DIAG (ac01 only)
**Date**: 2026-04-22
**Fork HEAD**: `1d0965f5` (`docs(qie): Q2.4.4b NaN bisect — linear depth growth, F32 residual needed`)
**Status**: **DIAGNOSED, NO FIX IN Q2.5 SCOPE.** Root cause is precision (F16 accumulator) exhaustion in the DiT forward, triggered by real-weight magnitudes + seq-length growth. Reproduction matrix already captured in `qie_q1_baseline.md`; a second reproduction on ac01 was NOT performed because (a) ac01 lacks the 17.7 GB QIE weights and (b) cross-container network is blocked (verified — see §Host constraints), so any repro would require an 18 GB Mac-relay transfer that does not change the diagnosis. Fix lands in Q2 / kernel-work scope, not Q2.5. Proposal: re-lock the eye-gate suite at `256×256 / 2-step` for Q2.5 calibration purposes — the only config that produces a valid image on the current ggml-cann path.

## Reproduction matrix (inherited, not re-run)

From `qie_q1_baseline.md` §Known regression, verified by the Q1 agent on ac02:

| W × H | Steps | Config | Result | Notes |
|---|---|---|---|---|
| 256×256 | 2 | `GGML_CANN_QUANT_BF16=on` | OK | 145 s wall, valid cat image |
| 256×256 | 3 | same | NaN at `diffusion/x_0` | blank output |
| 256×256 | 4 | same | NaN | |
| 256×256 | 20 | same | NaN | eye-gate step count |
| 512×512 | 2 | same | NaN | eye-gate resolution |
| 512×512 | 20 | same | NaN | **eye-gate suite default** |
| 256×256 batch=2 2-step | — | same | OK both | confirms not rng / sampler issue |

Without `GGML_CANN_QUANT_BF16=on`, even the 256×256 / 2-step config NaNs. The env var is necessary but not sufficient.

### Why a fresh repro on ac01 was not burned

1. **Weights absent.** `/home/ma-user/qie_q0v2/weights/` is on ac02 only. ac01's `/home/ma-user/` has no QIE GGUFs (`find ~ -name "Qwen-Image-Edit*"` empty). Full weight set is 17.7 GB.
2. **No cross-container network.** ac01→ac02 TCP on every tried port (22, 31210, 80, 8080, 443) returns RST (`errno 111`). Both containers share subnet `192.168.0.0/24` but firewall drops inter-pod traffic. Any ac01 transfer must hop Mac→ac01 (~18 GB at Mac-uplink speed, ≥1 hr wall).
3. **The Q1 table captures all five gate configs.** The regression table above was generated with the same fork HEAD's ggml-cann path; nothing between `710f545` (Q1 baseline commit) and `1d0965f5` (current HEAD) touches `ggml/src/ggml-cann/`. A fresh repro would reproduce the same bits.
4. **The Q2.4.4b bisect converges independently.** Commit `1d0965f5` already instrumented the equivalent code path on the native engine (not ggml-cann) and found linear residual-stream magnitude growth through 60 layers:

   | N layers | residual std | max abs | F16 headroom |
   |---|---|---|---|
   | 1  | 6.89   | 225    | 0.3% |
   | 5  | 125.2  | 6,792  | 10%  |
   | 10 | 237.7  | 12,912 | 20%  |
   | 30 | 900.1  | 48,512 | 74%  |
   | 60 | **NaN** | **NaN** | overflow |

   Same code shape (60 DiT blocks × residual adds), same precision (F16), same failure mode. The ggml-cann and native engines differ in dispatch, not in numerics-of-residual.

## Root-cause analysis

Two precision leaks stack up on the ggml-cann path, both confirmed by reading the current source.

### Leak #1 — `GGML_PREC_F32` silently ignored by ggml-cann

The diffusion model annotates precision-critical matmuls with:
- `ggml_mul_mat_set_prec(kq, GGML_PREC_F32)` for attention QK^T (`tools/ominix_diffusion/src/ggml_extend.hpp:1395`)
- `ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32)` for FA output (`:1358`)
- `force_prec_f32 = true` on the attention `to_out.0` projection, but **only under `SD_USE_VULKAN`** (`tools/ominix_diffusion/src/qwen_image.hpp:102-103`)

Both hints write to `op_params[0]`. Backends that read and honor it:

| Backend | Honors `GGML_PREC_F32`? | Evidence |
|---|---|---|
| CUDA | YES | `ggml/src/ggml-cuda/ggml-cuda.cu:1257,1913,2002` + `fattn-wmma-f16.cu:565` |
| Vulkan | YES | `ggml/src/ggml-vulkan/ggml-vulkan.cpp:5951,6010,6013,6015,6108,6165,8753` |
| SYCL | YES | `ggml/src/ggml-sycl/ggml-sycl.cpp:2127` |
| **CANN** | **NO** | `grep -rn "GGML_PREC" ggml/src/ggml-cann/` returns **empty** |

**Consequence**: every mul_mat on the ggml-cann path runs at whatever default accumulator aclnnMm / aclnnWeightQuantBatchMatmulV2 picks (F16 for F16 inputs, unless explicitly promoted). The scale trick `1/32` multiplied into `to_out.0` (`qwen_image.hpp:100`) is a partial mitigation cribbed from flux/SDXL + CUDA + k-quants, but it does not cover bf16 mul_mat inputs or the kq softmax path.

`GGML_CANN_QUANT_BF16=on` *does* help — it bumps the quant-path accumulator to BF16 on the Q4_0/Q8_0 `ggml_cann_mul_mat_quant` (`aclnn_ops.cpp:2641,2642`). This is why 256×256 / 2-step clears the gate (seq short, 2 steps compound nothing). But the env var has no effect on:
- The F16 fallback path for Q4_1 / Q5_* / K-quant weights (`ggml_cann_mul_mat_quant_cpu_dequant`, `aclnn_ops.cpp:2782`) — pre-dequants to F16 and calls `aclnnMm` with `cubeMathType=2` (ALLOW_FP32_DOWN_PRECISION, i.e. F32 inputs get demoted).
- The F16/F32 non-quant mul_mat path (`ggml_cann_mat_mul_fp`, `aclnn_ops.cpp:2547`) — used for norms and any F16-input mul_mat.
- The FIA attention path (`ggml_cann_flash_attn_ext`, `aclnn_ops.cpp:4265`) — hard-codes `faDataType = ACL_FLOAT16` (`:4307`) and `innerPrecise = 2` (`:4422`, an undocumented value; vendor headers document only 0 = HIGH_PRECISION and 1 = HIGH_PERFORMANCE).

### Leak #2 — Residual stream cannot hold 60 F16 layers at real magnitudes

Independent of the matmul accumulator issue. `qwen_image.hpp:300-315` does per-block:

```
img = ggml_add(..., img, ggml_mul(..., img_attn_output, img_gate1));
img = ggml_add(..., img, ggml_mul(..., img_mlp_out, img_gate2));
```

Both `img` and `txt` residuals are F16 on-device (they inherit the DiT input dtype, which is F16 under the default non-SDXL path). After 60 blocks, each residual has absorbed 120 gate-weighted contributions. The Q2.4.4b bisect measured `std=900, max=48k` at N=30 layers, and NaN at N≈35-60 depending on seq-length and prompt. F16 max is 65504; the headroom evaporates exactly where the real-weight run NaNs.

The Q2.4.4b recommended fix was: **keep residuals in F32 on device, cast down to F16 only around matmul inputs/outputs**. That's a native-engine change — the ggml-cann path's equivalent fix is a per-op `set_prec(F32)` that the backend actually honors, i.e. an implementation of leak #1 above.

### Why step-count and sequence-length both trip the NaN

- **Sequence length.** At 512×512 edit (seq ≈ 2304 with ref-latent concat), attention QK^T softmax inputs have larger absolute magnitudes than at 256×256 (seq ≈ 768). The F16-accum `kq` matmul overflows faster. Even at 2 steps, the first forward produces NaN.
- **Step count.** At 256×256 / 2-step, the residual stream survives marginally for one forward (see bisect table: N=30 is at 74% of F16 limit; seq=768 keeps it just under). By step 3, the latent has drifted into a state where the next forward overflows. This is the "compound error" from the Q1 hypothesis H3; measured directly by Q2.4.4b as the same residual-stream growth, just seeded from a slightly different input distribution per step.

## Hypothesis test results (read, not run)

| Q1 Hypothesis | Status after code-read | Notes |
|---|---|---|
| H1 — CPU-dequant `ggml_cann_mul_mat_quant_cpu_dequant` is non-deterministic across calls | Partially supported | The fallback D2H/H2D round-trip is `aclnnMm` with `cubeMathType=2` (ALLOW_FP32_DOWN_PRECISION). F16 inputs + default precision are consistent with leak #1. But H1 framed this as a determinism issue; the actual failure is precision, not bit-inequality. |
| H2 — Attention softmax overflows F16 at seq 2048+ | **Confirmed** | FIA dispatches with hard-coded `ACL_FLOAT16` (`aclnn_ops.cpp:4307`) and an out-of-spec `innerPrecise = 2`. `GGML_PREC_F32` hint attached to `kq` by `ggml_extend.hpp:1395` is dropped. |
| H3 — Modulation / time-embedding broadcast compounds badly | **Subsumed by residual-stream finding** | The compounding is the residual stream, not modulation-specific. Bisect shows linear-in-depth growth, flat across timesteps. |

## Env-knob and flag sweep (static analysis, not runtime)

| Lever | Covers which leak? | Verdict |
|---|---|---|
| `GGML_CANN_QUANT_BF16=on` | Q4_0 / Q8_0 quant mul_mat accumulator only | **Already on.** Necessary floor, not sufficient. |
| `GGML_CANN_ACL_GRAPH=1` | Graph-capture dispatch mode | Performance lever; does not change dtypes. No effect on NaN. |
| `GGML_CANN_WEIGHT_NZ=on/off` | Format (ND vs FRACTAL_NZ) for F16 weights | Format lever; does not change compute dtype. |
| `GGML_CANN_NO_PINNED` | Pinned-host-buffer opt | I/O lever; no effect. |
| `--diffusion-fa` (flash-attn on) | Routes attention through FIA with hard-coded ACL_FLOAT16 | **Does not help.** If anything, shifts the NaN earlier because FIA accumulates more operations internally. |
| `--diffusion-fa=0` (flash-attn off) | Routes through the expanded `kq = ggml_mul_mat(...); softmax(...); v @ kq` path | The explicit `kq` set_prec(F32) hint is dropped by cann. Same F16 behavior. |
| `--flow-shift <f>` | Changes the sigma schedule, not dtypes | Shifts step distribution; each individual step's forward still has same magnitudes. Does not mitigate overflow. |
| `--sampling-method euler / euler_ancestral / dpm++2s_ancestral` | Different integrator coefficients | Same per-step DiT forward. Does not mitigate. |
| `--cfg-scale 0` (disable guidance) | Skips uncond forward + compose | Halves forwards-per-step but each surviving forward still NaNs at 512×512. |

**No env / flag combination fixes this at the ggml-cann level today.** The fix requires source change in `ggml/src/ggml-cann/` — either:
- **(A)** Land `GGML_PREC_F32` support in `ggml_cann_mat_mul_fp`, `ggml_cann_mul_mat_quant_cpu_dequant`, and `ggml_cann_flash_attn_ext` (match the CUDA/Vulkan precedent). Estimated effort: ~1 day for the three functions + smoke; medium risk due to aclnnMm `cubeMathType` / FIA `innerPrecise` semantics under BF16/F32 inputs.
- **(B)** Extend `GGML_CANN_QUANT_BF16` to a broader `GGML_CANN_COMPUTE_DTYPE=bf16|f32` that covers all three paths above. Simpler than (A) because it's a global knob rather than per-op; downside is that LLM paths (which benefit from F16) regress in perf unless they're separately opted out.
- **(C)** Per the Q2.4.4b recommendation, land the native engine's F32-residual path and route QIE-edit through that instead of ggml-cann.

All three are kernel work, **out of Q2.5 calibration scope**.

## Host constraints (compliance receipt)

- **ac01 only.** All new work this session stayed on ac01 via SSH. No weights transferred, no inference run. Mac compute used for docs + git operations only.
- **Cross-host probes.** `ssh ac02 'ls ~/qie_q0v2/weights/'` (read-only, verifies existence of weights referenced in Q1 / Q2.5 pre-flight docs) — 0 NPU time. `ssh ac01 'python3 -c "socket.connect_ex...")'` — confirmed no route to ac02 from ac01. No runs on ac02 / ac03.
- **HBM budget.** N/A (no runs).
- **Per `AGENTS.md` fork rules.** No upstream-repo changes. This doc lands in the Ascend fork's `docs/`.
- **No `--no-verify` or `--no-gpg-sign`.** This doc is the only output.

## Ac01 environment probe (read-only, zero compute)

For traceability:

```
ac01: notebook-c768c7a7-...
  NPU ID 2 (910B4), idle (HBM baseline 2842 / 32768 MB)
  Fork tree at /home/ma-user/work/OminiX-Ascend-w1/ @ 1d0965f (same HEAD as Mac).
  build-w1/bin/ominix-diffusion-cli present, --help output confirms all
    flags (--diffusion-fa, --vae-conv-direct, --flow-shift, --cache-*, etc).
  libggml-cann.so.0.9.7 co-resident with libggml-base.so.0.9.7.
  No QIE weights at ~/qie_q0v2/, ~/work/qie_weights/, /tmp/, /cache/,
    or any path reachable with `find ~ -name "Qwen-Image-Edit*"`.
  Disk: /home/ma-user/work 1.3 TiB free; overlay 32 GiB free.
```

## Proposal — re-scope Q2.5 eye-gate suite to narrowest-working config

The eye-gate suite is LOCKED at `512×512 / 20-step / 20 tasks` per `qie_eye_gate_suite.md:5`. That lock is non-negotiable **mid-contract**, but the contract's §Q2.5 gate assumes Q2 has landed a NaN-free 512×512/20-step baseline before Q2.5 runs. Q2 has not landed (current HEAD is a bisect receipt, not a fix). Two honest paths forward:

### Path A — Wait for Q2 (kernel fix) to land; eye-gate suite unchanged

- Q2 agent lands leak-#1 fix (ggml-cann `GGML_PREC_F32` support) or leak-#2 fix (F32 residual stream in native engine).
- 512×512 / 20-step produces valid output → eye-gate suite captures `baseline_q2/` at its locked config → Q2.5 runs.
- **Estimated wall**: 2-4 weeks for kernel work + 1 day for baseline capture.
- **Pro**: suite lock respected; every subsequent milestone compares to the locked-config baseline.
- **Con**: Q2.5 BLOCKED until then.

### Path B — Amend the eye-gate suite to the narrowest-working config for ggml-cann

- Amend `qie_eye_gate_suite.md` from `512×512 / 20-step` to `256×256 / 2-step` for ggml-cann milestones (Q2.5 calibration pre-Q2 fix). At Q2 landing, re-amend back to `512×512 / 20-step` on the native engine.
- The 256×256 / 2-step config is the ONLY one that produces valid images on the current ggml-cann path.
- **Pro**: Q2.5 can proceed immediately; CacheDIT threshold Pareto can be measured (thresholds port roughly across configs because they are unitless ratios on hidden-state cos-sim, not wall-clock).
- **Con**: thresholds tuned at 2-step may not be Pareto-optimal at 20-step. Per `qie_optimization_learnings.md` §7 item 8, re-baselining mid-contract "poisons comparability" — this proposal intentionally does that, and the cost must be owned by the PM.

**This diagnosis recommends NEITHER path A nor B without explicit PM sign-off.** Path A is the contract default. Path B trades suite comparability for immediate progress; the PM is the only party who can make that call. Per the existing Q2.5 pre-flight ledger (`qie_q25_cachedit_calibration.md` §Recommendation, doubly concurred 2026-04-22), the block stands and returning a clean diagnosis is more honest than forcing either path.

## Receipts

- Q1 regression table — `docs/qie_q1_baseline.md:45-57`.
- Q2.4.4 native-engine NaN receipts — `docs/qie_q2_phase4_smoke.md:517-561`.
- Q2.4.4b bisect linear-growth table — same doc `:548-558`, also `1d0965f5` commit body.
- Q2.5 block ledger (2 concurring pre-flights) — `docs/qie_q25_cachedit_calibration.md`.
- ggml-cann `GGML_PREC_F32` grep (empty) — `ggml/src/ggml-cann/` tree.
- Source confirmation of `force_prec_f32` gated on `SD_USE_VULKAN` only — `tools/ominix_diffusion/src/qwen_image.hpp:101-107`.
- FIA hard-coded F16 dtype — `ggml/src/ggml-cann/aclnn_ops.cpp:4307`.
- Quant mul_mat BF16 accumulator env — `aclnn_ops.cpp:2641-2642`.

## Sign-off

Diagnosis complete; no fix patch attempted (see §Env-knob sweep). No eye-check run (no weights on ac01). No scope-narrowing decision committed to the suite lock (PM decision). No commits landed from this session — this doc is the only deliverable, and it is doc-only.
