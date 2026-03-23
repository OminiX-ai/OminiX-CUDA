"""
Export Qwen3-TTS Code Predictor as llama.cpp-compatible Qwen3 GGUF.

The CP's 5-layer transformer is re-exported with llama.cpp tensor names
and Qwen3 metadata, enabling NPU acceleration via llama.cpp's CANN backend.

Only the transformer layers + final norm are exported. Components that remain
on CPU (input_proj, codec_embeddings, lm_heads) stay in the original CP GGUF.

Usage:
    python export_cp_llama.py --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base
"""

import sys
import os
from pathlib import Path
import argparse
import numpy as np
import torch

if 'NO_LOCAL_GGUF' not in os.environ:
    gguf_py_path = Path(__file__).parent.parent.parent / 'gguf-py'
    if gguf_py_path.exists():
        sys.path.insert(1, str(gguf_py_path))

import gguf


# HuggingFace CP → llama.cpp tensor name mapping
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


def export_cp_llama(model, output_dir: str, use_f32: bool = False):
    """Export Code Predictor as llama.cpp-compatible Qwen3 GGUF."""
    os.makedirs(output_dir, exist_ok=True)
    suffix = "_f32" if use_f32 else ""
    output_path = f"{output_dir}/qwen_tts_cp_llama{suffix}.gguf"

    writer = gguf.GGUFWriter(output_path, "qwen3")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("qwen3-tts-code-predictor")
    if use_f32:
        writer.add_file_type(gguf.LlamaFileType.ALL_F32)
    else:
        writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)

    cp = model.talker.code_predictor
    cfg = cp.config

    # Architecture metadata
    writer.add_context_length(32)           # CP max sequence = 17
    writer.add_embedding_length(cfg.hidden_size)           # 1024
    writer.add_block_count(cfg.num_hidden_layers)          # 5
    writer.add_head_count(cfg.num_attention_heads)         # 16
    writer.add_head_count_kv(cfg.num_key_value_heads)      # 8
    writer.add_key_length(cfg.head_dim)                    # 128 (override default 1024/16=64)
    writer.add_value_length(cfg.head_dim)                  # 128
    writer.add_feed_forward_length(cfg.intermediate_size)  # 3072
    writer.add_layer_norm_rms_eps(float(cfg.rms_norm_eps)) # 1e-6
    writer.add_rope_freq_base(float(cfg.rope_theta))       # 1000000.0
    writer.add_vocab_size(cfg.vocab_size)                  # 2048

    # Dummy tokenizer (required by llama.cpp, not actually used)
    vocab_size = cfg.vocab_size  # 2048
    tokens = [f"<tok_{i}>" for i in range(vocab_size)]
    scores = [0.0] * vocab_size
    token_types = [1] * vocab_size  # NORMAL
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(0)
    writer.add_eos_token_id(1)
    writer.add_pad_token_id(2)

    # Extract CP state dict
    cp_sd = cp.state_dict()

    n_layers = cfg.num_hidden_layers
    tensor_count = 0

    # Map and add layer tensors
    for i in range(n_layers):
        for hf_pattern, llama_pattern in TENSOR_MAP.items():
            hf_name = hf_pattern.format(i=i)
            llama_name = llama_pattern.format(i=i)
            if hf_name in cp_sd:
                param = cp_sd[hf_name]
                if use_f32:
                    param = param.to(torch.float32)
                elif param.dim() <= 1:
                    param = param.to(torch.float32)
                param = param.squeeze()
                writer.add_tensor(llama_name, param.cpu().numpy())
                tensor_count += 1

    # Final norm
    if "model.norm.weight" in cp_sd:
        param = cp_sd["model.norm.weight"].to(torch.float32).squeeze()
        writer.add_tensor("output_norm.weight", param.cpu().numpy())
        tensor_count += 1

    # Dummy token_embd.weight (required by llama.cpp but not used - we use batch.embd)
    # Must be F16 for CANN compatibility (CANN doesn't support F32 matmul weights)
    embd_dim = cfg.hidden_size  # 1024
    dummy_embd = np.zeros((vocab_size, embd_dim), dtype=np.float16)
    writer.add_tensor("token_embd.weight", dummy_embd)
    tensor_count += 1

    # Dummy output.weight (required but not used - we use embeddings mode)
    dummy_output = np.zeros((vocab_size, embd_dim), dtype=np.float16)
    writer.add_tensor("output.weight", dummy_output)
    tensor_count += 1

    print(f"  Mapped {tensor_count} tensors to llama.cpp format")
    print(f"  (including 2 dummy tensors: token_embd, output)")
    print(f"  Writing {output_path}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Done: {size_mb:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Export Qwen3-TTS Code Predictor in llama.cpp GGUF format")
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

    export_cp_llama(model, args.output_dir, use_f32=args.f32)
    print("\nExport complete!")


if __name__ == "__main__":
    main()
