"""
Export Qwen3-TTS Talker backbone in llama.cpp-compatible GGUF format.

This re-exports the Talker's 28-layer Qwen3 transformer with tensor names
and metadata that llama.cpp expects, enabling use of llama.cpp's optimized
inference engine for the backbone.

Custom components (text_embedding, text_projection, codec_head) remain in
the original Talker GGUF and are loaded separately by TalkerEmbeddingModel.

Usage:
    python export_talker_llama.py --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base
"""

import sys
import os
from pathlib import Path
import argparse
import json
import numpy as np
import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    gguf_py_path = Path(__file__).parent.parent.parent / 'gguf-py'
    if gguf_py_path.exists():
        sys.path.insert(1, str(gguf_py_path))

import gguf


# HuggingFace → llama.cpp tensor name mapping for Qwen3
TENSOR_MAP = {
    "model.layers.{i}.self_attn.q_proj.weight":              "blk.{i}.attn_q.weight",
    "model.layers.{i}.self_attn.k_proj.weight":              "blk.{i}.attn_k.weight",
    "model.layers.{i}.self_attn.v_proj.weight":              "blk.{i}.attn_v.weight",
    "model.layers.{i}.self_attn.o_proj.weight":              "blk.{i}.attn_output.weight",
    "model.layers.{i}.self_attn.q_norm.weight":              "blk.{i}.attn_q_norm.weight",
    "model.layers.{i}.self_attn.k_norm.weight":              "blk.{i}.attn_k_norm.weight",
    "model.layers.{i}.mlp.gate_proj.weight":                 "blk.{i}.ffn_gate.weight",
    "model.layers.{i}.mlp.up_proj.weight":                   "blk.{i}.ffn_up.weight",
    "model.layers.{i}.mlp.down_proj.weight":                 "blk.{i}.ffn_down.weight",
    "model.layers.{i}.input_layernorm.weight":               "blk.{i}.attn_norm.weight",
    "model.layers.{i}.post_attention_layernorm.weight":      "blk.{i}.ffn_norm.weight",
}

# Non-layer tensors
STATIC_MAP = {
    "model.norm.weight":            "output_norm.weight",
    "model.codec_embedding.weight": "token_embd.weight",
    "codec_head.weight":            "output.weight",
}


def export_talker_llama(model, output_dir: str, use_f32: bool = False):
    """Export Talker backbone as llama.cpp-compatible Qwen3 GGUF."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_f32" if use_f32 else ""
    output_path = f"{output_dir}/qwen_tts_talker_llama{suffix}.gguf"

    writer = gguf.GGUFWriter(output_path, "qwen3")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("qwen3-tts-talker")
    if use_f32:
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    else:
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    talker = model.talker
    cfg = talker.config

    # Architecture metadata (llama.cpp format)
    writer.add_context_length(cfg.max_position_embeddings)
    writer.add_embedding_length(cfg.hidden_size)
    writer.add_block_count(cfg.num_hidden_layers)
    writer.add_head_count(cfg.num_attention_heads)
    writer.add_head_count_kv(cfg.num_key_value_heads)
    writer.add_feed_forward_length(cfg.intermediate_size)
    writer.add_layer_norm_rms_eps(float(cfg.rms_norm_eps))
    writer.add_rope_freq_base(float(cfg.rope_theta))
    writer.add_vocab_size(cfg.vocab_size)  # 3072 (codec vocab)

    # Dummy tokenizer (required by llama.cpp, not actually used)
    vocab_size = cfg.vocab_size  # 3072
    tokens = [f"<tok_{i}>" for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    token_types = [1] * vocab_size  # NORMAL
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(cfg.codec_bos_id)
    writer.add_eos_token_id(cfg.codec_eos_token_id)
    writer.add_pad_token_id(cfg.codec_pad_id)

    # Extract state dict, excluding code_predictor
    full_sd = talker.state_dict()
    talker_sd = {k: v for k, v in full_sd.items()
                 if not k.startswith("code_predictor.")}

    n_layers = cfg.num_hidden_layers
    tensor_count = 0

    # Map and add layer tensors
    for i in range(n_layers):
        for hf_pattern, llama_pattern in TENSOR_MAP.items():
            hf_name = hf_pattern.format(i=i)
            llama_name = llama_pattern.format(i=i)
            if hf_name in talker_sd:
                param = talker_sd[hf_name]
                if use_f32:
                    param = param.to(torch.float32)
                elif param.dim() <= 1:
                    param = param.to(torch.float32)
                param = param.squeeze()
                writer.add_tensor(llama_name, param.cpu().numpy())
                tensor_count += 1

    # Map and add static tensors
    for hf_name, llama_name in STATIC_MAP.items():
        if hf_name in talker_sd:
            param = talker_sd[hf_name]
            if use_f32:
                param = param.to(torch.float32)
            elif param.dim() <= 1:
                param = param.to(torch.float32)
            param = param.squeeze()
            writer.add_tensor(llama_name, param.cpu().numpy())
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
        description="Export Qwen3-TTS Talker in llama.cpp GGUF format")
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_dir", type=str, default="gguf")
    parser.add_argument("--f32", action="store_true",
                        help="Export in float32 (for debugging)")
    args = parser.parse_args()

    # Load model
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    print(f"Loading model from {args.model_path}...")
    model = AutoModel.from_pretrained(
        args.model_path, device_map="cpu", dtype=torch.float16,
        trust_remote_code=True,
    )
    print("Model loaded.")

    export_talker_llama(model, args.output_dir, use_f32=args.f32)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
