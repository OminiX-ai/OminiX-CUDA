# Changelog

## 2026-03-26

### Refactor: Migrate SD library source into tools/ominix_diffusion/src

- Moved `src-ominix-diffusion/` → `tools/ominix_diffusion/src/` via `git mv`
- SD library (libstable-diffusion), cli, server, and common now all live under `tools/ominix_diffusion/`
- Rewrote `tools/ominix_diffusion/CMakeLists.txt` as top-level entry: `add_subdirectory(src)` + cli + server
- Removed redundant `add_subdirectory(tools/ominix_diffusion)` from src CMakeLists (now managed by parent)
- Updated root `CMakeLists.txt`: `add_subdirectory(src-ominix-diffusion)` → `add_subdirectory(tools/ominix_diffusion)`
- Updated directory structures in README.md and README_CANN_OPTIMIZATIONS.md
- Split SD optimizations from `README_CANN_OPTIMIZATIONS.md` into `tools/ominix_diffusion/README.md`

### Merge: Unified ggml backend (merge/unify-ggml → main)

- Merged `ggml-diffusion/` (~330K lines) into unified `ggml/` backend; LLM, SD, and ASR now share one ggml
- Renamed `src-diffusion/` → `src-ominix-diffusion/`, moved SD cli/server to `tools/ominix_diffusion/`
- Binary renames: `sd-cli` → `ominix-diffusion-cli`, `sd-server` → `ominix-diffusion-server`
- CANN new ops: IM2COL_3D, CONV_2D, CONV_3D for SD inference on Ascend NPU
- BF16/FP16 conditional compute dtype via `GGML_CANN_QUANT_BF16` env var (default FP16 for LLM, BF16 for SD)
- ACL Graph default changed to off (SD enables via `GGML_CANN_ACL_GRAPH=1`)
- aclnnExpand broadcast fix for Ascend 910B MTE bug
- Restored `examples/diffusion/` (Dream LLM) from main
- Deduplicated `GGML_MAX_NAME=128` definition in root CMakeLists.txt
- Cleaned up work files (output.png, outputs/, PROGRESS.md, claude_prompt.txt)
- Updated `README_CANN_OPTIMIZATIONS.md` with project structure, ASR section, and new binary names

## 2026-03-25

### Fix: Remove residual `tools/qwen_tts` CMakeLists.txt breaking cmake

- Removed `add_subdirectory(qwen_tts)` from `tools/CMakeLists.txt`
- Deleted leftover `tools/qwen_tts/CMakeLists.txt` (source files were already removed in `8e6a914d`)
- This fixes `cmake -B build` failing with "Cannot find source file: main.cpp" errors

### Refactor: Extract shared modules into `tools/qwen_common/`

- Moved 6 shared modules (`bpe_tokenizer`, `audio_io`, `utils`, `model_loader`, `ctx_manager`, `build_graph`) from `tools/qwen_tts/` to new `tools/qwen_common/` static library
- Updated `tools/qwen_asr/CMakeLists.txt` to link `qwen_common` instead of referencing `qwen_tts` files directly
- Updated `tools/qwen_tts/CMakeLists.txt` to link `qwen_common` instead of compiling shared sources inline
- Fixed `#include` paths in `tools/qwen_asr/audio_encoder.cpp` (removed `../qwen_tts/` prefix)
- Removed unnecessary `model_defs.cpp` and `stft.cpp` from ASR build (TTS-only modules)
- `qwen_asr` now has zero references to `qwen_tts` directory

### Fix: Warn when `tokenizer_config.json` is missing

- `bpe_tokenizer.cpp` now prints a WARNING when `tokenizer_config.json` is not found, instead of silently skipping special token loading
- Previously this caused `token_to_id()` to return -1 for special tokens (`<|im_start|>`, `<|audio_start|>`, etc.), leading to `llama_decode` failure
