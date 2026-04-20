#!/usr/bin/env bash
# Bake built-in voice caches (.bin) from reference wav + transcript pairs.
#
# Reads tools/qwen_tts/data/voices/voices.json and, for every entry, runs
# qwen_tts in encode-only mode to produce the corresponding .bin cache next
# to voices.json. Existing caches are skipped unless FORCE=1 is set.
#
# Usage:
#   tools/qwen_tts/scripts/bake_voices.sh [MODEL_DIR]
#
# Environment:
#   MODEL_DIR       GGUF model directory (default: tools/qwen_tts/gguf)
#   TALKER_MODEL    Talker GGUF file (default: qwen_tts_talker_llama_q8_0.gguf under MODEL_DIR)
#   CP_MODEL        CP GGUF file (default: qwen_tts_cp_llama.gguf under MODEL_DIR)
#   N_GPU_LAYERS    NPU/GPU offload (default: 29)
#   BIN             qwen_tts binary (default: build/bin/qwen_tts)
#   FORCE=1         Re-bake even if cache file already exists
#
set -euo pipefail

MODEL_DIR="${1:-${MODEL_DIR:-tools/qwen_tts/gguf}}"
TALKER_MODEL="${TALKER_MODEL:-${MODEL_DIR}/qwen_tts_talker_llama_q8_0.gguf}"
CP_MODEL="${CP_MODEL:-${MODEL_DIR}/qwen_tts_cp_llama.gguf}"
N_GPU_LAYERS="${N_GPU_LAYERS:-29}"
BIN="${BIN:-build/bin/qwen_tts}"
VOICES_DIR="tools/qwen_tts/data/voices"
VOICES_JSON="${VOICES_DIR}/voices.json"

if [[ ! -x "$BIN" ]]; then
  echo "Error: qwen_tts binary not found at $BIN. Build first (cmake --build build)." >&2
  exit 1
fi
if [[ ! -f "$VOICES_JSON" ]]; then
  echo "Error: $VOICES_JSON not found" >&2
  exit 1
fi

# Dummy text (baking only needs to exercise the encoder; we throw away the wav).
DUMMY_TEXT="Baking voice cache."
TMP_OUT="$(mktemp -t voice_bake_XXXXXX.wav)"
trap 'rm -f "$TMP_OUT"' EXIT

# Extract voice entries via python (already required for exporting models).
python3 - "$VOICES_JSON" <<'PY' | while IFS=$'\t' read -r id cache lang src_wav src_txt; do
import json, sys
with open(sys.argv[1]) as f:
    j = json.load(f)
for v in j.get("voices", []):
    print("\t".join([v["id"], v["cache"], v.get("lang","en"),
                     v.get("src_wav",""), v.get("src_txt","")]))
PY
  cache_path="${VOICES_DIR}/${cache}"
  if [[ -f "$cache_path" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] $id -> $cache_path (exists; set FORCE=1 to overwrite)"
    continue
  fi

  # Resolve src_wav / src_txt relative to VOICES_DIR.
  wav_path="${VOICES_DIR}/${src_wav}"
  txt_path="${VOICES_DIR}/${src_txt}"
  if [[ ! -f "$wav_path" ]]; then
    echo "[warn] $id: ref wav not found: $wav_path — skipped" >&2
    continue
  fi
  if [[ ! -f "$txt_path" ]]; then
    echo "[warn] $id: ref txt not found: $txt_path — skipped" >&2
    continue
  fi

  # ref txt files have format:
  #   line 1: English|Chinese  (language tag)
  #   line 2: actual transcript
  # Take line 2 if it exists, else fall back to concatenated file.
  ref_text="$(sed -n '2p' "$txt_path" | tr -d '\r')"
  if [[ -z "$ref_text" ]]; then
    ref_text="$(tr -d '\r\n' < "$txt_path")"
  fi
  ref_lang="English"
  if [[ "$lang" == "zh" ]]; then ref_lang="Chinese"; fi

  echo "[bake] $id ($ref_lang) -> $cache_path"
  "$BIN" \
    -m "$MODEL_DIR" --tokenizer_dir "$MODEL_DIR" \
    -r "$wav_path" --ref_text "$ref_text" --ref_lang "$ref_lang" \
    --ref_cache "$cache_path" \
    --talker_model "$TALKER_MODEL" \
    --cp_model "$CP_MODEL" \
    --n_gpu_layers "$N_GPU_LAYERS" \
    --max_tokens 8 \
    -t "$DUMMY_TEXT" -o "$TMP_OUT" >/dev/null
  echo "[ok]   $id"
done

echo "Done. Caches written under $VOICES_DIR/"
