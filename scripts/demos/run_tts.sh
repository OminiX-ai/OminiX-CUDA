#!/usr/bin/env bash
# ominix-cuda TTS demo runner
# Usage: ./run_tts.sh [<text>] [<output_wav>]
# Defaults: "Hello world..." → /tmp/qwen_tts.wav
# Wall: ~17.8s cold (RTF 1.74) / 6.4s steady-state (RTF 0.62) for 10.24s of audio
set -euo pipefail

TEXT=${1:-"Hello world, this is the ominix CUDA TTS."}
OUT=${2:-/tmp/qwen_tts.wav}

HOST=${OMINIX_TTS_HOST:-zgx-3675}
BIN=${OMINIX_TTS_BIN:-/home/user1/ominix-cuda/build-phase21/bin/test_qwen_tts_e2e}

echo "[run_tts] host=$HOST  text='$TEXT'  out=$OUT"

# Phase 2.5 graphs cover both Talker and Predictor (Phase 2.9 wired predictor too)
# Phase 2.8 sampling defaults match Ascend talker.h:25-43
ssh -o BatchMode=yes "$HOST" "TALKER_USE_CUDA_GRAPHS=1 $BIN '$TEXT'"

scp -o BatchMode=yes "$HOST:/tmp/qwen_tts_e2e.wav" "$OUT"
echo "[run_tts] saved: $OUT"
