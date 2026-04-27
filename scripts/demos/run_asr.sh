#!/usr/bin/env bash
# ominix-cuda ASR demo runner
# Usage: ./run_asr.sh [<wav_path>]
# Defaults: ellen_ref.wav (audiobook 9.36s @ 32kHz)
# Wall: ~3.5s on GB10 (zgx-3675) → RTF 0.37
set -euo pipefail

WAV_IN=${1:-/home/user1/ominix-cuda/ellen_ref.wav}

HOST=${OMINIX_ASR_HOST:-zgx-3675}
BIN=${OMINIX_ASR_BIN:-/home/user1/ominix-cuda/build-phase21/bin/test_qwen_asr_cuda_e2e}
MEL_FILTERS=${OMINIX_ASR_MEL_FILTERS:-/home/user1/ominix-cuda/tools/qwen_asr/verify_data/mel_filters_whisper.npy}

# If user passes a Mac-local path, scp it to the host first.
REMOTE_WAV="$WAV_IN"
if [[ -f "$WAV_IN" ]] && [[ "$WAV_IN" != /home/* ]]; then
  REMOTE_WAV="/tmp/$(basename "$WAV_IN")"
  scp -o BatchMode=yes "$WAV_IN" "$HOST:$REMOTE_WAV"
fi

echo "[run_asr] host=$HOST  wav=$REMOTE_WAV"
ssh -o BatchMode=yes "$HOST" "OMINIX_ASR_MEL_FILTERS='$MEL_FILTERS' $BIN '$REMOTE_WAV'" 2>&1 | tail -20
