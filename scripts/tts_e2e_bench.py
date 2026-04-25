#!/usr/bin/env python3
"""Qwen3-TTS end-to-end synthesis benchmark harness.

This script intentionally benchmarks the official Qwen3-TTS Base Python path
first. It closes the reference-audio -> talker-codes -> codec -> waveform loop
and emits timing fields that can be compared with the vLLM Talker microbench.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


DEFAULT_REF_AUDIO = (
    "https://qianwen-res.oss-cn-beijing.aliyuncs.com/"
    "Qwen3-TTS-Repo/clone.wav"
)
DEFAULT_REF_TEXT = (
    "Okay. Yeah. I resent you. I love you. I respect you. "
    "But you know what? You blew it! And thanks to you."
)
DEFAULT_TARGET_TEXT = "The quick brown fox jumps over the lazy dog."


def now() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def dtype_from_name(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def cuda_memory() -> dict[str, int]:
    if not torch.cuda.is_available():
        return {}
    return {
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "memory_allocated_bytes": int(torch.cuda.memory_allocated()),
        "memory_reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def synthesize_once(args: argparse.Namespace, model: Qwen3TTSModel) -> dict[str, Any]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    total_start = now()

    prompt_start = now()
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=args.ref_audio,
        ref_text=args.ref_text,
        x_vector_only_mode=args.x_vector_only_mode,
    )
    prompt_end = now()

    voice_clone_prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)
    ref_texts_for_ids = [it.ref_text for it in prompt_items]

    input_ids = model._tokenize_texts([model._build_assistant_text(args.text)])
    ref_ids = []
    for ref_text in ref_texts_for_ids:
        if ref_text is None or ref_text == "":
            ref_ids.append(None)
        else:
            ref_ids.append(model._tokenize_texts([model._build_ref_text(ref_text)])[0])

    gen_kwargs = model._merge_generate_kwargs(
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        subtalker_dosample=args.subtalker_dosample,
        subtalker_top_k=args.subtalker_top_k,
        subtalker_top_p=args.subtalker_top_p,
        subtalker_temperature=args.subtalker_temperature,
        max_new_tokens=args.max_new_tokens,
    )

    talker_start = now()
    talker_codes_list, _ = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt,
        languages=[args.language],
        non_streaming_mode=args.non_streaming_mode,
        **gen_kwargs,
    )
    talker_end = now()

    codes_for_decode = []
    ref_code_list = voice_clone_prompt.get("ref_code")
    for idx, codes in enumerate(talker_codes_list):
        if ref_code_list is not None and ref_code_list[idx] is not None:
            codes_for_decode.append(
                torch.cat([ref_code_list[idx].to(codes.device), codes], dim=0)
            )
        else:
            codes_for_decode.append(codes)

    decode_start = now()
    wavs_all, sample_rate = model.model.speech_tokenizer.decode(
        [{"audio_codes": codes} for codes in codes_for_decode]
    )
    decode_end = now()

    wavs_out: list[np.ndarray] = []
    for idx, wav in enumerate(wavs_all):
        if ref_code_list is not None and ref_code_list[idx] is not None:
            ref_len = int(ref_code_list[idx].shape[0])
            total_len = int(codes_for_decode[idx].shape[0])
            prefix_samples = int(ref_len / max(total_len, 1) * wav.shape[0])
            wavs_out.append(wav[prefix_samples:])
        else:
            wavs_out.append(wav)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_start = now()
    sf.write(output_path, wavs_out[0], sample_rate)
    write_end = now()

    total_end = now()

    generated_frames = int(talker_codes_list[0].shape[0])
    generated_audio_seconds = float(len(wavs_out[0]) / sample_rate)
    total_wall_s = total_end - total_start
    talker_wall_s = talker_end - talker_start
    codec_wall_s = decode_end - decode_start
    prompt_wall_s = prompt_end - prompt_start
    write_wall_s = write_end - write_start

    return {
        "mode": "official_base",
        "model_dir": args.model,
        "text": args.text,
        "language": args.language,
        "ref_audio": args.ref_audio,
        "ref_text": args.ref_text,
        "output_wav": str(output_path),
        "sample_rate": int(sample_rate),
        "generated_audio_seconds": generated_audio_seconds,
        "generated_codec_frames": generated_frames,
        "total_synthesis_wall_s": total_wall_s,
        "reference_prompt_wall_s": prompt_wall_s,
        "talker_decode_wall_s": talker_wall_s,
        "codec_decode_wall_s": codec_wall_s,
        "wav_write_wall_s": write_wall_s,
        "talker_decode_tok_s": generated_frames / talker_wall_s if talker_wall_s > 0 else None,
        "total_audio_fps": generated_frames / total_wall_s if total_wall_s > 0 else None,
        "steady_state_audio_fps": generated_frames / codec_wall_s if codec_wall_s > 0 else None,
        "realtime_factor": generated_audio_seconds / total_wall_s if total_wall_s > 0 else None,
        "first_token_latency_s": None,
        "first_audio_latency_s": None,
        "cuda_memory": cuda_memory(),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "total_synthesis_wall_s",
        "reference_prompt_wall_s",
        "talker_decode_wall_s",
        "codec_decode_wall_s",
        "talker_decode_tok_s",
        "total_audio_fps",
        "steady_state_audio_fps",
        "realtime_factor",
    ]
    summary: dict[str, Any] = {"runs": len(runs)}
    for key in keys:
        vals = [run[key] for run in runs if isinstance(run.get(key), (int, float))]
        if not vals:
            continue
        summary[f"{key}_mean"] = statistics.mean(vals)
        summary[f"{key}_median"] = statistics.median(vals)
        summary[f"{key}_min"] = min(vals)
        summary[f"{key}_max"] = max(vals)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="/home/user1/qwen3_tts_cuda/models/Qwen3-TTS-12Hz-1.7B-Base",
        help="Official Qwen3-TTS Base model path or HF repo id.",
    )
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--ref-text", default=DEFAULT_REF_TEXT)
    parser.add_argument("--text", default=DEFAULT_TARGET_TEXT)
    parser.add_argument("--language", default="English")
    parser.add_argument(
        "--output",
        default="/home/user1/qwen3_tts_cuda/outputs/qwen3_tts_e2e.wav",
    )
    parser.add_argument(
        "--metrics-json",
        default="/home/user1/qwen3_tts_cuda/logs/qwen3_tts_e2e_metrics.json",
    )
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-top-k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", type=float, default=1.0)
    parser.add_argument("--subtalker-temperature", type=float, default=0.9)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--subtalker-dosample",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--x-vector-only-mode", action="store_true")
    parser.add_argument("--non-streaming-mode", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--runs", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    load_start = now()
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device_map,
        dtype=dtype_from_name(args.dtype),
        attn_implementation=args.attn_implementation,
    )
    load_end = now()

    for _ in range(args.warmups):
        synthesize_once(args, model)

    run_metrics = []
    for idx in range(args.runs):
        metrics = synthesize_once(args, model)
        metrics["run_index"] = idx
        run_metrics.append(metrics)

    common = {
        "model_load_wall_s": load_end - load_start,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }

    if len(run_metrics) == 1:
        metrics_out = {**run_metrics[0], **common}
    else:
        metrics_out = {
            **common,
            "mode": "official_base",
            "model_dir": args.model,
            "text": args.text,
            "language": args.language,
            "ref_audio": args.ref_audio,
            "ref_text": args.ref_text,
            "output_wav": args.output,
            "sample_rate": run_metrics[-1]["sample_rate"],
            "generated_audio_seconds": run_metrics[-1]["generated_audio_seconds"],
            "generated_codec_frames": run_metrics[-1]["generated_codec_frames"],
            "warmups": args.warmups,
            "runs": run_metrics,
            "summary": summarize_runs(run_metrics),
        }

    write_json(Path(args.metrics_json), metrics_out)
    print(json.dumps(metrics_out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
