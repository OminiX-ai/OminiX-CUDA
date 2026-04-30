#!/usr/bin/env bash
# ominix-cuda TTS demo runner
# Usage:
#   ./run_tts.sh [<text>] [<output_wav>]
#   ./run_tts.sh --server [<text>] [<output_wav>]   # Phase 2.10 warm daemon
#   ./run_tts.sh --start-server                     # background-launch daemon
#   ./run_tts.sh --stop-server                      # kill daemon
#   ./run_tts.sh --server-status                    # check + tail log
#
# Cold (binary launch + weight load): ~17.8s / RTF 1.74 for 10.24s audio.
# Warm (Phase 2.10 daemon, models pre-loaded):  ~6.4s / RTF 0.62.
set -euo pipefail

HOST=${OMINIX_TTS_HOST:-zgx-3675}
BIN=${OMINIX_TTS_BIN:-/home/user1/ominix-cuda/build-phase21/bin/test_qwen_tts_e2e}
SERVER_BIN=${OMINIX_TTS_SERVER_BIN:-/home/user1/ominix-cuda/build-phase21/bin/tts_server}
PORT=${OMINIX_TTS_SERVER_PORT:-7777}
SERVER_LOG=${OMINIX_TTS_SERVER_LOG:-/tmp/tts_server.log}

mode="cold"
case "${1:-}" in
    --server)        mode="warm";        shift ;;
    --start-server)  mode="start";       shift ;;
    --stop-server)   mode="stop";        shift ;;
    --server-status) mode="status";      shift ;;
esac

case "$mode" in
    start)
        echo "[run_tts] starting daemon on $HOST:$PORT (log: $SERVER_LOG)"
        # Pass through FP8 toggle if set on caller side.
        FP8_FLAG=""
        if [ "${OMINIX_TTS_USE_FP8_LMHEAD:-0}" = "1" ]; then
            FP8_FLAG="OMINIX_TTS_USE_FP8_LMHEAD=1"
            echo "[run_tts] FP8 LM-head will be enabled (predictor)"
        fi
        ssh -o BatchMode=yes "$HOST" "pkill -9 -f tts_server; sleep 2; $FP8_FLAG TALKER_USE_CUDA_GRAPHS=1 nohup $SERVER_BIN > $SERVER_LOG 2>&1 < /dev/null & disown; echo started_pid=\$!"
        echo "[run_tts] waiting for LISTENING (cold init ~10s)..."
        for i in $(seq 1 30); do
            if ssh -o BatchMode=yes "$HOST" "grep -q 'LISTENING on' $SERVER_LOG 2>/dev/null"; then
                echo "[run_tts] daemon READY (after ${i}s)"
                ssh -o BatchMode=yes "$HOST" "tail -3 $SERVER_LOG"
                exit 0
            fi
            sleep 1
        done
        echo "[run_tts] timeout — daemon did not become ready" >&2
        ssh -o BatchMode=yes "$HOST" "tail -20 $SERVER_LOG" >&2
        exit 1
        ;;
    stop)
        ssh -o BatchMode=yes "$HOST" "pkill -f tts_server; sleep 1; pgrep -f tts_server || echo stopped"
        exit 0
        ;;
    status)
        ssh -o BatchMode=yes "$HOST" "pgrep -fl tts_server || echo 'no daemon running'; echo ---; tail -20 $SERVER_LOG 2>/dev/null"
        exit 0
        ;;
    cold)
        TEXT=${1:-"Hello world, this is the ominix CUDA TTS."}
        OUT=${2:-/tmp/qwen_tts.wav}
        echo "[run_tts] mode=cold host=$HOST  text='$TEXT'  out=$OUT"
        ssh -o BatchMode=yes "$HOST" "TALKER_USE_CUDA_GRAPHS=1 $BIN '$TEXT'"
        scp -o BatchMode=yes "$HOST:/tmp/qwen_tts_e2e.wav" "$OUT"
        echo "[run_tts] saved: $OUT"
        ;;
    warm)
        TEXT=${1:-"Hello world, this is the ominix CUDA TTS."}
        OUT=${2:-/tmp/qwen_tts.wav}
        REMOTE_OUT=${OMINIX_TTS_REMOTE_OUT:-/tmp/qwen_tts_warm.wav}
        echo "[run_tts] mode=warm host=$HOST:$PORT  text='$TEXT'  out=$OUT"
        # Auto-start the daemon if it's not running.
        if ! ssh -o BatchMode=yes "$HOST" "pgrep -f tts_server >/dev/null"; then
            echo "[run_tts] daemon not running — starting it now..."
            "$0" --start-server
        fi
        # Send request: '<text>\t<remote_out>\n' -> 'OK\twall_ms\trtf\taudio_sec\n'.
        reply=$(ssh -o BatchMode=yes "$HOST" "printf '%s\t%s\n' '$TEXT' '$REMOTE_OUT' | nc -q1 127.0.0.1 $PORT")
        echo "[run_tts] reply: $reply"
        case "$reply" in
            OK*)
                scp -o BatchMode=yes "$HOST:$REMOTE_OUT" "$OUT"
                echo "[run_tts] saved: $OUT"
                ;;
            ERR*)
                echo "[run_tts] daemon returned error" >&2
                exit 1
                ;;
            *)
                echo "[run_tts] unexpected reply: $reply" >&2
                exit 1
                ;;
        esac
        ;;
esac
