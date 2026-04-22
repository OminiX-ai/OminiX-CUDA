# aclnnApplyRotaryPosEmbV2 vs aclnnRotaryPositionEmbedding divergence

## Symptom

Phase A.2 wired the CANN 8.3 fused Q+K in-place RoPE operator
(`aclnnApplyRotaryPosEmbV2`, `layout=1 BSND`, `rotaryMode="half"`) as a
drop-in replacement for the two sequential `aclnnRotaryPositionEmbedding(mode=0)`
calls in the talker CP engine. Parity FAILS: the canonical LONG xvec mayun zh
runs 434 codec frames under the v1 path but 457 frames under V2 (+5%), EOS
fires at a different frame. Hidden-state drift starts from frame 0 — not a
rounding artefact, an op-level semantic mismatch.

## Hypothesis ranking + evidence

### H1 — cos/sin table layout expectation differs. **LIKELIHOOD: LOW (ruled out by source).**

Engine uploads cos/sin as **half-duplicated**: `cos_table[p*D + j + half] = cos_table[p*D + j]`
(cp_cann_engine.cpp:1410–1413; h:286–290 comment: "each row duplicates the half
so the HF-style 'rotate_half' formula maps to aclnnRotaryPositionEmbedding(mode=0, NEOX)").

Mode=0 kernel (`rotary_pos_emb_base.h:485-506 CalcRope`) builds a `negOne_`
mask `[-1,-1,...,+1,+1,...]` and computes `out = cos*orig + sin*(mask*rotated)`,
which yields the standard NEOX formula assuming cos/sin are duplicated. Matches.

V2 (arch35 half path) `BatchHalfAlignVF` in
`apply_rotary_pos_emb/arch35/apply_rotary_pos_emb_common.h:488-504`:
```
out[0:half]  = in[0:half]*cos[0:half] - in[half:]*sin[0:half]
out[half:]   = in[half:]*cos[half:]  + in[0:half]*sin[half:]
```
With duplicated cos/sin (`cos[i]==cos[i+half]`, same for sin) this is **exactly**
NEOX. The older `_compute_ab.h` path reaches the same result via
`Muls(sin, -1.0, halfNum, preCBatchB, stride=row)` negating the first half of
sin, then rotate-half of q, then `mul1*sin + mul2*cos`.

Semantically the two formulas agree. Layout is not the divergence.

### H2 — rotaryMode semantics. **LIKELIHOOD: LOW.**

V2 string `"half"` → `RotaryPosEmbeddingMode::HALF = 0`
(`rotary_position_embedding_grad/arch35/rotary_position_embedding_grad_rotary_x_vf.h:36-42`).
v1 `mode=0` header doc says "half". Same mode. Patch passes `"half"` correctly.

(Caveat: v1 enum order from header doc is `0=half,1=quarter,2=interleave,3=half-interleave`,
V2 arch35 enum is `HALF=0,INTERLEAVE=1,QUARTER=2,DEEPSEEK_INTERLEAVE=3`. Numeric
orders differ between op families. **Not** a bug here because we pass a string
for V2, not a number.)

### H3 — layout semantics. **LIKELIHOOD: LOW.**

`layout=1` is BSND (batch, seq, n_heads, head_dim). Our Q descriptor is
`shape={1,1,n_heads,head_dim}, strides={q_dim,q_dim,head_dim,1}`, K similarly
— valid BSND. For B=S=1 the stride choice is irrelevant (contiguous).

### H4 — Sin sign convention. **LIKELIHOOD: LOW.**

Already traced: V2 math matches NEOX with duplicated sin. No flipped sign.

### H5 — In-place semantics hazard AND Q/K head-count packing. **LIKELIHOOD: HIGH.**

Two closely related mechanical bugs:

**H5a**: V2's `CopyInQK` (`apply_rotary_pos_emb_small.h:113-117`,
`apply_rotary_pos_emb_compute_ab.h:CopyInQK`) loads **Q and K packed
back-to-back into a single UB**: `qUb[0..qcdNum]` = Q, `qUb[qcdNum..qcdNum+kcdNum]`
= K. It then issues ONE fused Mul pipeline with `qkcNum` repeats and
`src1RepStride=0` for cos/sin broadcast. This design assumes `cos/sin` can be
broadcast across **all** Q+K heads uniformly — works for our uniform per-token
rotation, **but** the kernel's `dstRepStride = dstRepSBr` field couples Q-stride
and K-stride into a single `dstRepSBr`. **Qwen3-TTS talker is GQA with
n_heads=16, n_kv=8** — different head counts. A single shared stride
(`dstRepSBr`) over a heterogeneous Q+K packed buffer will misalign the rotation
for one or both sides.

**H5b**: Patch redirects K-proj output to the KV cache slot (`k_cache_dev_[il]
+ pos*kv_dim`). V2 then reads `keyRef=slot` and writes back to the **same**
slot. If the arch35 kernel's `BatchHalfAlignVF` loads `inPart1/inPart2` and
writes to `currOutUb` with `in==out` aliased (which is our case — in-place),
we rely on the two halves being loaded before the first store. The kernel
code does load both halves before any store, so elementwise this is safe —
but combined with packed Q+K in a shared UB (H5a) the stride arithmetic for K
writes back into a **different** address than K's read origin only if
`qcdNum != kcdNum` (GQA). Exact miswriting of the second half of K (or
addresses beyond K's valid range) is plausible.

### H6 — Different internal precision. **LIKELIHOOD: MEDIUM but insufficient to explain +5%.**

Mode=0's FP16 kernel casts to F32 for cos/sin expansion but still does the
multiply in F16. V2's compute_ab fp16 path keeps everything in F16. Slight
rounding drift, but 5% frame-count divergence = non-rounding.

## Recommended fix (highest-confidence hypothesis)

**Root cause (most likely): V2's fused Q+K packing assumes uniform head count
and a shared stride (`dstRepSBr`), which breaks for GQA (16 Q heads vs 8 KV
heads) in the talker.**

Concrete mitigation (no code change needed to validate):

1. **Issue V2 with Q only**, then issue V2 again with K only (two calls,
   still fewer than v1's two calls because v1 also did them separately — so
   perf may regress but parity should hold). Per the header V2 accepts
   `queryRef + keyRef`; nothing prevents passing the same buffer as both,
   but the cleaner shape is to wire two separate V2 calls with Q=Q,K=Q and
   then Q=K,K=K. Confirm parity first; if it passes, we've isolated H5a.

2. Alternatively, **reshape Q to `[1,1,n_kv,head_dim*(n_heads/n_kv)]`**
   so Q and K share head count — but this changes head-dim semantics and is
   fragile.

3. **Fallback**: keep V2 disabled for GQA models. The header contract does
   not explicitly document GQA support; the arch35 kernel's packed UB
   suggests it is written for MHA (n_heads_q == n_heads_k).

## Validation plan (next-agent task)

1. Write a minimal standalone harness (`tools/qwen_tts/test_rope_v2_diff.cpp`)
   that:
   - Allocates Q `[1,1,16,128]`, K `[1,1,8,128]`, cos/sin `[1,1,1,128]`
     (head_dim=128 matches talker).
   - Uploads deterministic f16 payloads (e.g. `q[i] = i/1000.0f`,
     `cos = engine's table at pos=0`, `sin = same`).
   - Path A: two `aclnnRotaryPositionEmbedding(mode=0)` calls → dump
     q_out_a, k_out_a.
   - Path B: one `aclnnApplyRotaryPosEmbV2(layout=1, rotaryMode="half")` →
     dump q_out_b, k_out_b.
   - Element-wise diff. Locate first divergent head/dim.
2. Expected outcome per H5a: Q output matches for heads 0..7; Q heads 8..15
   drift OR K output is wrong.
3. If H5a confirmed: try Q-only-call and K-only-call separately via V2 to
   verify single-tensor V2 works; propose per-tensor V2 dispatch for the
   GQA case.
4. If H5a NOT confirmed: focus on H5b (in-place hazard at cache slot) —
   reproduce by passing a distinct K output buffer vs Q in-place.

## Additional read-through notes

- Patch `/tmp/phase_a_2.patch` correctly resolves optional symbols
  (`cp_cann_symbols.cpp:244-252`) and gates on `has_rope_v2()`.
- `cos/sin` tensor descriptors passed are identical to v1 path (shape
  `[1,1,1,head_dim]`, line 1648-1659) — this is acceptable for V2 via the
  `src1RepStride=0` broadcast trick.
- aclGraph is correctly disabled under `cp_rope_v2_applied_` — good, in-place
  ops would conflict with capture-time descriptor binding.
- FIA input/output swap and O-proj source redirect are consistent — Q lives
  rotated at `q_dev_` post-V2, FIA writes to `attn_out_dev_`, O-proj reads
  `attn_out_dev_`. The wiring itself is clean.
