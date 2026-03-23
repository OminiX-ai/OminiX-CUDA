#!/usr/bin/env python3
"""Compare C++ and Python hidden states by feeding identical prefill embeddings
to a Qwen3 model loaded from safetensors."""

import sys, os
os.chdir("/root/autodl-tmp")

import torch
import torch_npu
import numpy as np
import json
import struct
from safetensors.torch import load_file

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LOG_DIR = "/root/autodl-tmp/tts.cpp/logs"

device = "npu:0" if torch.npu.is_available() else "cpu"
print(f"Device: {device}")

# Load C++ prefill embeddings
cpp_embs_path = os.path.join(LOG_DIR, "cpp_prefill_embs.bin")
with open(cpp_embs_path, "rb") as f:
    seq_len, dim = struct.unpack("ii", f.read(8))
    data = np.frombuffer(f.read(seq_len * dim * 4), dtype=np.float32)
    cpp_embs = data.reshape(seq_len, dim)
print(f"C++ prefill: {seq_len} x {dim}")
print(f"  embs[0][:5]: {cpp_embs[0, :5]}")
print(f"  embs[-1][:5]: {cpp_embs[-1, :5]}")

# Load C++ hidden states
cpp_hidden_path = os.path.join(LOG_DIR, "cpp_prefill_hidden.bin")
cpp_hidden = np.fromfile(cpp_hidden_path, dtype=np.float32)
print(f"\nC++ post-prefill hidden[:5]: {cpp_hidden[:5]}")

# Load config and model weights
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)
talker_cfg = config["talker_config"]

print("\nLoading model weights...")
weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
weights = {}
for wf in sorted(weight_files):
    w = load_file(os.path.join(MODEL_PATH, wf))
    weights.update(w)

# Build Qwen3 model
from transformers import Qwen3ForCausalLM, Qwen3Config

qwen3_config = Qwen3Config(
    hidden_size=talker_cfg["hidden_size"],
    intermediate_size=talker_cfg["intermediate_size"],
    num_hidden_layers=talker_cfg["num_hidden_layers"],
    num_attention_heads=talker_cfg["num_attention_heads"],
    num_key_value_heads=talker_cfg["num_key_value_heads"],
    vocab_size=talker_cfg["vocab_size"],
    max_position_embeddings=talker_cfg.get("max_position_embeddings", 32768),
    rms_norm_eps=talker_cfg.get("rms_norm_eps", 1e-6),
    rope_theta=talker_cfg.get("rope_theta", 1000000.0),
    head_dim=talker_cfg.get("head_dim", 128),
)

model = Qwen3ForCausalLM(qwen3_config).to(device).eval()

# Map weights
talker_sd = {}
for k, v in weights.items():
    if k.startswith("talker.model."):
        new_k = k.replace("talker.model.", "model.")
        talker_sd[new_k] = v
    elif k.startswith("talker.codec_head."):
        new_k = k.replace("talker.codec_head.", "lm_head.")
        talker_sd[new_k] = v
talker_sd["model.embed_tokens.weight"] = weights["talker.model.codec_embedding.weight"]

missing, unexpected = model.load_state_dict(talker_sd, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

# Run with C++ embeddings
print("\n--- Forward pass with C++ prefill embeddings ---")
inputs_embeds = torch.tensor(cpp_embs, device=device, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    # Get hidden states from base model (post-norm)
    base_outputs = model.model(inputs_embeds=inputs_embeds, use_cache=False)
    py_hidden = base_outputs.last_hidden_state[0, -1, :].cpu().numpy()
    print(f"Python post-norm hidden[:5]: {py_hidden[:5]}")
    print(f"C++    post-norm hidden[:5]: {cpp_hidden[:5]}")

    # Compare
    diff = np.abs(py_hidden - cpp_hidden)
    print(f"\nMax diff: {diff.max():.6f}")
    print(f"Mean diff: {diff.mean():.6f}")
    print(f"RMS diff: {np.sqrt((diff**2).mean()):.6f}")

    # Get logits
    full_outputs = model(inputs_embeds=inputs_embeds, use_cache=False)
    logits = full_outputs.logits[0, -1, :]
    top5 = torch.topk(logits, 10)
    print(f"\nPython top-10 tokens:")
    for tok, val in zip(top5.indices.cpu().tolist(), top5.values.cpu().tolist()):
        print(f"  [{tok}] = {val:.4f}")

    # Apply codec_head manually for verification
    codec_head_w = weights["talker.codec_head.weight"].to(device).float()
    py_hidden_t = base_outputs.last_hidden_state[0, -1, :].to(device)
    manual_logits = py_hidden_t @ codec_head_w.T
    manual_top5 = torch.topk(manual_logits, 5)
    print(f"\nManual codec_head top-5:")
    for tok, val in zip(manual_top5.indices.cpu().tolist(), manual_top5.values.cpu().tolist()):
        print(f"  [{tok}] = {val:.4f}")

print("\nDone!")
