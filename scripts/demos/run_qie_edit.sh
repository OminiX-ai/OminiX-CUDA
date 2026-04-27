#!/usr/bin/env bash
# ominix-cuda QIE-Edit demo runner
# Usage: ./run_qie_edit.sh [<ref_image>] [<prompt>] [<output_png>]
# Defaults: cat.jpg + "make the cat smile" → /tmp/qie_cuda.png
# Wall: ~2:45 wall on GB10 (zgx-5b44) at 1024² 20-step with --diffusion-fa
set -euo pipefail

REF=${1:-/home/user1/qie_cuda/inputs/cat.jpg}
PROMPT=${2:-"make the cat smile"}
OUT=${3:-/tmp/qie_cuda.png}

HOST=${OMINIX_QIE_HOST:-zgx-5b44}
BIN=${OMINIX_QIE_BIN:-/home/user1/ominix-cuda/build/bin/ominix-diffusion-cli}
DIT=${OMINIX_QIE_DIT:-/home/user1/dev/sd-cpp/models/Qwen-Image-Edit-2509-Q4_0.gguf}
LLM=${OMINIX_QIE_LLM:-/home/user1/dev/sd-cpp/models/Qwen2.5-VL-7B-Instruct-Q8_0.gguf}
LLM_VISION=${OMINIX_QIE_LLM_VISION:-/home/user1/dev/sd-cpp/models/Qwen2.5-VL-7B-Instruct.mmproj-Q8_0.gguf}
VAE=${OMINIX_QIE_VAE:-/home/user1/dev/sd-cpp/models/qwen_image_vae.safetensors}

REMOTE_OUT="/tmp/$(basename "$OUT")"
echo "[run_qie_edit] host=$HOST  ref=$REF  prompt='$PROMPT'  out=$OUT"

ssh -o BatchMode=yes "$HOST" "$BIN \
  -M img_gen \
  --diffusion-model $DIT \
  --llm $LLM \
  --llm_vision $LLM_VISION \
  --vae $VAE \
  -r '$REF' \
  -p '$PROMPT' \
  -W 1024 -H 1024 --steps 20 --cfg-scale 1.0 \
  --sampling-method euler --vae-tiling \
  --diffusion-fa \
  -o '$REMOTE_OUT' --seed 42 --color"

scp -o BatchMode=yes "$HOST:$REMOTE_OUT" "$OUT"
echo "[run_qie_edit] saved: $OUT"
