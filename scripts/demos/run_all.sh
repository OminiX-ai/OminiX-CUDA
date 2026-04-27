#!/usr/bin/env bash
# ominix-cuda — exercise all three production flows back-to-back
# Usage: ./run_all.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"

echo
echo "=== 1/3  TTS  (text → WAV) ==="
"$DIR/run_tts.sh" "Hello world, this is the ominix CUDA TTS." /tmp/qwen_tts_demo.wav

echo
echo "=== 2/3  ASR  (WAV → transcript) ==="
"$DIR/run_asr.sh" /tmp/qwen_tts_demo.wav

echo
echo "=== 3/3  QIE-Edit  (image + prompt → image) ==="
"$DIR/run_qie_edit.sh" /home/user1/qie_cuda/inputs/cat.jpg "make the cat smile" /tmp/qie_cuda_demo.png

echo
echo "=== done ==="
echo "  TTS WAV:    /tmp/qwen_tts_demo.wav"
echo "  QIE PNG:    /tmp/qie_cuda_demo.png"
