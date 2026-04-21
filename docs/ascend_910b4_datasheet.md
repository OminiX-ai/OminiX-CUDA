# Ascend 910B4 Hardware Summary

Research date: 2026-04-21. Purpose: inform AscendC kernel tiling for Qwen3-TTS hot path on 910B4 (32 GB HBM).

## Source reliability

- [CSET — "Pushing the Limits"](https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/) — **primary**: Georgetown reverse-engineered per-variant (910B1–B4) cores/freq/HBM.
- [arXiv 2505.15112 — Parallel Scan on Ascend](https://arxiv.org/html/2505.15112v1) — **primary (academic)**: explicitly "910B4 contains 20 Cube Units and 40 Vector Units", 800 GB/s.
- [arXiv 2509.25224 — AMLA (FlashAttention)](https://arxiv.org/html/2509.25224) — **secondary (academic)**: memory hierarchy numbers (UB 192KB, L1 512KB, L0A/B 64KB, L0C 128KB).
- [arXiv 2507.23387 — SGEMM-cube / H2SGEMM](https://arxiv.org/html/2507.23387v3) — **secondary**: 910B3 = 20 AI cores @1.8 GHz, L1 "half" of 910A.
- [arXiv 2604.03298 — ENEC](https://arxiv.org/html/2604.03298) — **secondary**: 910B2 = 24 AIC + 48 AIV; decoupled AIC/AIV via GM/L2.
- [Hot Chips 31 DaVinci paper](https://pdfs.semanticscholar.org/78b6/d0b2a12de2e7c106e8b4a81a6b29cf5c47b7.pdf) — **primary vendor**: original DaVinci-Max spec (baseline).
- [deepwiki Ascend-NPU-IR](https://deepwiki.com/ascendmirror/AscendNPU-IR) — **community**: 32-byte alignment on all address spaces; MTE1/MTE2/MTE3 semantics.
- [Tom's Hardware — 910B die analysis](https://www.tomshardware.com/tech-industry/artificial-intelligence/huaweis-homegrown-ai-chip-examined-chinese-fab-smic-produced-ascend-910b-is-massively-different-from-the-tsmc-produced-ascend-910) — **community**: die size, SMIC N+1, 1 matrix + 2 vector per AI core.
- [Atlas 300T A2 product page](https://e.huawei.com/en/products/computing/ascend/atlas-300t-training-9000) — **primary vendor**: 20 AI cores, 280 TFLOPS FP16, 75 TFLOPS FP32 (uses 910B4 die).

## Key datapoints

| Item | Value | Source | Confidence |
|---|---|---|---|
| Architecture | Da Vinci V220 (2nd-gen) | Tom's HW, Hot Chips | high |
| AI Cores (910B4) | 20 **AIC + 40 AIV** (decoupled, 1:2 ratio) | arXiv 2505.15112, CSET | high |
| AIC↔AIV comms | GM / L2 only — no direct path | arXiv 2505.15112, 2604.03298 | high |
| Clock | 1.65 GHz (910B4); 1.8 GHz on B1/B2/B3 | CSET | high |
| Peak FP16 | **280 TFLOPS** | Atlas 300T A2, CSET | high |
| Peak BF16 | ~280 TFLOPS (same cube path) | inferred | medium |
| Peak INT8 | ~560 TOPS (2× FP16) | inferred from 910B base | medium |
| Peak FP32 | 75 TFLOPS (native on B3/B4) | Huawei product card | medium |
| HBM | **32 GB** HBM2e (2× 16 GB stacks) | CSET | high |
| HBM bandwidth | **800 GB/s** | arXiv 2505.15112, CSET | high |
| L2 (die-shared) | 192 MB | CSET | high |
| UB per AIV | **192 KB** | arXiv 2604.03298, 2509.25224 | medium-high |
| L1 per AIC | **512 KB** (half of 910A's 1 MB) | arXiv 2507.23387, 2509.25224 | medium-high |
| L0A per AIC | **64 KB** | arXiv 2509.25224, Hot Chips | medium |
| L0B per AIC | **64 KB** | arXiv 2509.25224 | medium |
| L0C per AIC | **128 KB** (FP32 accumulator) | arXiv 2509.25224 | medium |
| Cube dims | 16×16 · 16×16 FP16 = 4096 MACs/cycle; 16×32 · 32×16 INT8 = 8192 MACs/cycle | Hot Chips | high |
| Cube dtypes | FP16, BF16, INT8; FP32 native on B3/B4 | SGEMM-cube, product card | high |
| Vector SIMD | 128 FP16 lanes / AIV (2048-bit) | Hot Chips | medium |
| Vector dtypes | FP16, FP32, INT32, INT8 | Hot Chips, Atlas book | high |
| Vector ops | add/mul/FMA, exp, log, rsqrt, max/min, reduce-sum/max, cast, gather, scatter | AscendC API / Ascend-NPU-IR | high |
| MTE engines | 3: MTE1 (L1↔L0A/B), MTE2 (GM↔L1/UB), MTE3 (UB↔GM). Same-MTE serial, cross-MTE parallel. | deepwiki, AscendCraft | high |
| DMA alignment | 32 B on all address spaces | deepwiki | high |
| Die / process | 665.6 mm², SMIC N+1 (~7 nm) | Tom's HW | high |

## 910B4 vs 910B variants

| | 910B1 | 910B2 | 910B3 | **910B4** |
|---|---|---|---|---|
| AI Cores | 24 | 24 | 20 | **20** |
| Freq (GHz) | 1.8 | 1.8 | 1.8 | **1.65** |
| FP16 (TFLOPS) | ~400 | ~313 | 313 | **280** |
| HBM (GB) | 64 | 64 | 64 | **32** |
| HBM BW (GB/s) | 1600 | 1600 | 1600 | **800** |

Same die, binned/fused. 910B4 = lower-clock, half HBM config of B3. Source: [CSET](https://cset.georgetown.edu/publication/pushing-the-limits-huaweis-ai-chip-tests-u-s-export-controls/).

## Gaps (unknown)

- **MTE queue depth**: not published. AscendC samples use 2-deep ping-pong; some repos go to 4. Treat as profile-driven.
- **GM→UB vs GM→L1 latency cost model**: no official cycle table. arXiv 2505.15112 only warns AIC↔AIV handoff "may be expensive".
- **BT / FP scratch buffers inside AIC**: named in arXiv 2604.03298, sizes never published (likely <32 KB each).
- **Vector intrinsic cycle/throughput table**: AscendC docs list ops but no latencies. Needs on-device microbenchmark.
- **910B4 vs B3 architectural deltas**: none found — only clock + HBM binning.
- **L1 exact size**: "half of 910A" is the only quote; 910A-L1=1 MB is itself from community sources.
- **INT4**: no evidence of native cube path on 910B; roadmap for 910C+.

## Implications for Path C kernel authoring

1. **blockDim**: use `20` for AIC-only kernels, `40` for AIV-only, `20` for mixed (runtime spawns 20 AIC + 40 AIV cohorts). Never try to fuse AIC+AIV work in one block without a GM/L2 handoff — there is no direct path.

2. **UB budget (per AIV block) = 192 KB**, reserve ~32 KB for double-buffer overhead → ~160 KB working set. For Qwen3-TTS RmsNorm/softmax on `[batch, seq, 768]` FP16 tiles: one row = 1.5 KB, so ≤128 rows/tile with ping-pong headroom.

3. **Cube natural tile** derived from L0 budget: L0A 64 KB / 2 B = 32 K FP16 elems = 128×256; L0B symmetric 256×128; L0C 128 KB / 4 B = 128×128 FP32. **Default cube tile: M=128, N=128, K=256**. CannFusion codegen should emit this by default.

4. **L1 residency matters 2× more on 910B4** than on B3 — HBM is 800 GB/s vs 1.6 TB/s. Pre-stage full K-dim per M-tile into L1 when `K·elem_size ≤ 512 KB` (K ≤ 4096 for FP16 at 128 cols). Qwen3-TTS attention `head_dim=64` → full KV slab fits L1 easily; prefer one-shot L1 load.

5. **3-stage MTE pipeline**: MTE2 (GM→L1/UB), MTE1 (L1→L0), compute (Cube/Vector), MTE3 (UB→GM) — overlap with ping-pong queues of depth 2. Emit `SetFlag(MTEn)/WaitFlag(MTEn)` pairs; **never `PipeBarrier<PIPE_ALL>` in hot loops** (nukes the pipeline; see tilelang-ascend issue #110).

6. **Bandwidth ceiling**: 800 GB/s ÷ 280 TFLOPS = **2.86 B/FLOP**. Any kernel with arithmetic intensity <140 FLOP/B (FP16) is HBM-bound. RmsNorm (~1 FLOP/B) and softmax tail (~5 FLOP/B) are HBM-bound — **fuse them into the producing GEMM's UB path**, never round-trip activations through GM.

7. **32-byte alignment is mandatory**: every `DataCopy` offset/stride must be 32 B-aligned or MTE silently corrupts or stalls. FP16 `head_dim=64` = 128 B (aligned); BF16 `head_dim=80` = 160 B (aligned); pad odd sequence-tail residuals to 32 B.
