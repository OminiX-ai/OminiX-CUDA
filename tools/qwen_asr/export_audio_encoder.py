"""
Export Qwen3-ASR audio encoder to GGUF format for C++ inference.

Audio Encoder architecture:
- 3x Conv2d layers (downsample_hidden_size=480, stride=2)
- Linear projection (conv_out: 7680 -> 1024)
- Sinusoidal positional embedding (computed, not stored)
- 24-layer Transformer encoder (d_model=1024, heads=16, ffn=4096)
- Output MLP: LayerNorm + Linear(1024,1024) + GELU + Linear(1024,2048)

Usage:
    python export_audio_encoder.py --model_path Qwen/Qwen3-ASR-1.7B --output_dir gguf/
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import torch
from safetensors.torch import load_file

if 'NO_LOCAL_GGUF' not in os.environ:
    gguf_py_path = Path(__file__).parent.parent.parent / 'gguf-py'
    if gguf_py_path.exists():
        sys.path.insert(1, str(gguf_py_path))

import gguf


def export_audio_encoder(model_path: str, output_dir: str, use_f32: bool = False):
    """Export audio encoder to GGUF."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_f32" if use_f32 else ""
    output_path = f"{output_dir}/qwen_asr_audio_encoder{suffix}.gguf"

    # Load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)

    audio_cfg = config["thinker_config"]["audio_config"]

    writer = gguf.GGUFWriter(output_path, "qwen_asr_audio_encoder")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("qwen3-asr-audio-encoder")
    if use_f32:
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    else:
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    # Audio encoder configuration
    d_model = audio_cfg["d_model"]                         # 1024
    encoder_layers = audio_cfg["encoder_layers"]           # 24
    encoder_attention_heads = audio_cfg["encoder_attention_heads"]  # 16
    encoder_ffn_dim = audio_cfg["encoder_ffn_dim"]         # 4096
    num_mel_bins = audio_cfg["num_mel_bins"]                # 128
    downsample_hidden_size = audio_cfg["downsample_hidden_size"]  # 480
    output_dim = audio_cfg["output_dim"]                   # 2048
    max_source_positions = audio_cfg["max_source_positions"]  # 1500
    n_window = audio_cfg["n_window"]                       # 50
    n_window_infer = audio_cfg.get("n_window_infer", 800)  # 800

    # Compute mel_dim after 3 conv layers with stride=2
    mel_reduced = ((((num_mel_bins + 1) // 2 + 1) // 2 + 1) // 2)  # 16
    conv_out_dim = downsample_hidden_size * mel_reduced    # 480 * 16 = 7680

    writer.add_uint32("d_model", d_model)
    writer.add_uint32("encoder_layers", encoder_layers)
    writer.add_uint32("encoder_attention_heads", encoder_attention_heads)
    writer.add_uint32("encoder_ffn_dim", encoder_ffn_dim)
    writer.add_uint32("num_mel_bins", num_mel_bins)
    writer.add_uint32("downsample_hidden_size", downsample_hidden_size)
    writer.add_uint32("output_dim", output_dim)
    writer.add_uint32("max_source_positions", max_source_positions)
    writer.add_uint32("n_window", n_window)
    writer.add_uint32("n_window_infer", n_window_infer)
    writer.add_uint32("mel_reduced", mel_reduced)
    writer.add_uint32("conv_out_dim", conv_out_dim)

    # Load safetensors
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    # Find files containing audio_tower weights
    files_to_load = set()
    for hf_name, shard_file in weight_map.items():
        if hf_name.startswith("thinker.audio_tower."):
            files_to_load.add(shard_file)

    # Load all audio encoder tensors
    all_tensors = {}
    for shard_file in sorted(files_to_load):
        shard_path = os.path.join(model_path, shard_file)
        print(f"  Loading {shard_file}...")
        tensors = load_file(shard_path)
        for name, tensor in tensors.items():
            if name.startswith("thinker.audio_tower."):
                # Strip prefix "thinker.audio_tower."
                short_name = name[len("thinker.audio_tower."):]
                all_tensors[short_name] = tensor

    # Skip positional_embedding (sinusoidal, computed at runtime)
    tensors_to_skip = {"positional_embedding.positional_embedding"}

    tensor_count = 0
    for name, param in sorted(all_tensors.items()):
        if name in tensors_to_skip:
            print(f"  [skip] {name} (computed at runtime)")
            continue

        # Bias and norm layers → float32
        if param.dim() <= 1:
            param = param.to(torch.float32)
        elif "layer_norm" in name or "ln_post" in name:
            param = param.to(torch.float32)
        elif use_f32:
            param = param.to(torch.float32)
        else:
            param = param.to(torch.float16)

        # Conv weights (4D for Conv2d) keep shape; others squeeze
        is_conv = name.endswith(".weight") and param.dim() >= 3
        if not is_conv:
            param = param.squeeze()

        writer.add_tensor(name, param.cpu().numpy())
        tensor_count += 1

    print(f"  Audio encoder tensors: {tensor_count}")
    print(f"  Writing {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-ASR audio encoder to GGUF format")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output_dir", type=str, default="gguf")
    parser.add_argument("--f32", action="store_true",
                        help="Export in float32 (for debugging)")
    args = parser.parse_args()

    export_audio_encoder(args.model_path, args.output_dir, use_f32=args.f32)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
