# Changelog

## 2026-03-25

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
