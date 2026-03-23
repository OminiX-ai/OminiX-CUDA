"""
Export Qwen3-ASR text decoder in llama.cpp-compatible GGUF format.

The text decoder is a standard Qwen3 model (28 layers, hidden=2048, GQA)
with Q/K norms. It generates text tokens autoregressively.

Usage:
    python export_decoder_llama.py --model_path Qwen/Qwen3-ASR-1.7B --output_dir gguf/
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


# HuggingFace → llama.cpp tensor name mapping for Qwen3-ASR text decoder
# HF prefix: thinker.model.
TENSOR_MAP = {
    "layers.{i}.self_attn.q_proj.weight":              "blk.{i}.attn_q.weight",
    "layers.{i}.self_attn.k_proj.weight":              "blk.{i}.attn_k.weight",
    "layers.{i}.self_attn.v_proj.weight":              "blk.{i}.attn_v.weight",
    "layers.{i}.self_attn.o_proj.weight":              "blk.{i}.attn_output.weight",
    "layers.{i}.self_attn.q_norm.weight":              "blk.{i}.attn_q_norm.weight",
    "layers.{i}.self_attn.k_norm.weight":              "blk.{i}.attn_k_norm.weight",
    "layers.{i}.mlp.gate_proj.weight":                 "blk.{i}.ffn_gate.weight",
    "layers.{i}.mlp.up_proj.weight":                   "blk.{i}.ffn_up.weight",
    "layers.{i}.mlp.down_proj.weight":                 "blk.{i}.ffn_down.weight",
    "layers.{i}.input_layernorm.weight":               "blk.{i}.attn_norm.weight",
    "layers.{i}.post_attention_layernorm.weight":      "blk.{i}.ffn_norm.weight",
}

# Non-layer tensors
STATIC_MAP = {
    "embed_tokens.weight":  "token_embd.weight",
    "norm.weight":          "output_norm.weight",
}


def export_decoder_llama(model_path: str, output_dir: str, use_f32: bool = False):
    """Export ASR text decoder as llama.cpp-compatible Qwen3 GGUF."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_f32" if use_f32 else ""
    output_path = f"{output_dir}/qwen_asr_decoder{suffix}.gguf"

    # Load config
    with open(os.path.join(model_path, "config.json"), "r") as f:
        config = json.load(f)

    text_cfg = config["thinker_config"]["text_config"]

    writer = gguf.GGUFWriter(output_path, "qwen3")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("qwen3-asr-decoder")
    if use_f32:
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    else:
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    # Architecture metadata
    vocab_size = text_cfg["vocab_size"]  # 151936
    writer.add_context_length(text_cfg["max_position_embeddings"])
    writer.add_embedding_length(text_cfg["hidden_size"])
    writer.add_block_count(text_cfg["num_hidden_layers"])
    writer.add_head_count(text_cfg["num_attention_heads"])
    writer.add_head_count_kv(text_cfg["num_key_value_heads"])
    writer.add_feed_forward_length(text_cfg["intermediate_size"])
    writer.add_layer_norm_rms_eps(float(text_cfg["rms_norm_eps"]))
    writer.add_rope_freq_base(float(text_cfg["rope_theta"]))
    writer.add_vocab_size(vocab_size)

    # Dummy tokenizer (required by llama.cpp, actual tokenization done separately)
    tokens = [f"<tok_{i}>".encode("utf-8") for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    token_types = [1] * vocab_size  # NORMAL
    writer.add_tokenizer_model("llama")  # Use "llama" to avoid needing merges
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(151645)  # <|im_end|>
    writer.add_pad_token_id(151643)  # <|endoftext|>

    # Load safetensors
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    # Group files to load
    files_to_load = set()
    for hf_name, shard_file in weight_map.items():
        if hf_name.startswith("thinker.model.") or hf_name == "thinker.lm_head.weight":
            files_to_load.add(shard_file)

    # Load all relevant tensors
    all_tensors = {}
    for shard_file in sorted(files_to_load):
        shard_path = os.path.join(model_path, shard_file)
        print(f"  Loading {shard_file}...")
        tensors = load_file(shard_path)
        for name, tensor in tensors.items():
            if name.startswith("thinker.model.") or name == "thinker.lm_head.weight":
                # Strip prefix
                if name.startswith("thinker.model."):
                    short_name = name[len("thinker.model."):]
                else:
                    short_name = name
                all_tensors[short_name] = tensor

    n_layers = text_cfg["num_hidden_layers"]
    tensor_count = 0

    # Map and add layer tensors
    for i in range(n_layers):
        for hf_pattern, llama_pattern in TENSOR_MAP.items():
            hf_name = hf_pattern.format(i=i)
            llama_name = llama_pattern.format(i=i)
            if hf_name in all_tensors:
                param = all_tensors[hf_name]
                if use_f32:
                    param = param.to(torch.float32)
                elif param.dim() <= 1:
                    param = param.to(torch.float32)
                else:
                    param = param.to(torch.float16)
                param = param.squeeze()
                writer.add_tensor(llama_name, param.cpu().numpy())
                tensor_count += 1

    # Map and add static tensors
    for hf_name, llama_name in STATIC_MAP.items():
        if hf_name in all_tensors:
            param = all_tensors[hf_name]
            if use_f32:
                param = param.to(torch.float32)
            elif param.dim() <= 1:
                param = param.to(torch.float32)
            else:
                param = param.to(torch.float16)
            param = param.squeeze()
            writer.add_tensor(llama_name, param.cpu().numpy())
            tensor_count += 1

    # lm_head
    lm_head_key = "thinker.lm_head.weight"
    if lm_head_key in all_tensors:
        param = all_tensors[lm_head_key]
        if use_f32:
            param = param.to(torch.float32)
        else:
            param = param.to(torch.float16)
        param = param.squeeze()
        writer.add_tensor("output.weight", param.cpu().numpy())
        tensor_count += 1

    print(f"  Mapped {tensor_count} tensors to llama.cpp format")
    print(f"  Writing {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-ASR text decoder in llama.cpp GGUF format")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output_dir", type=str, default="gguf")
    parser.add_argument("--f32", action="store_true",
                        help="Export in float32 (for debugging)")
    args = parser.parse_args()

    export_decoder_llama(args.model_path, args.output_dir, use_f32=args.f32)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
