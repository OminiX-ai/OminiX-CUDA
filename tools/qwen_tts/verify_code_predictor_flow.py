#!/usr/bin/env python3
"""Verify Code Predictor autoregressive flow matches Python reference.
Tests the CP with a known talker hidden state + group 0 token to get reference
group 1-15 tokens."""

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

# Load config
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)
talker_cfg = config["talker_config"]
cp_cfg = talker_cfg["code_predictor_config"]

print(f"Talker hidden: {talker_cfg['hidden_size']}")
print(f"CP hidden: {cp_cfg['hidden_size']}, layers: {cp_cfg['num_hidden_layers']}")
print(f"CP vocab: {cp_cfg['vocab_size']}, groups: {cp_cfg['num_code_groups']}")

# Load weights
weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
weights = {}
for wf in sorted(weight_files):
    w = load_file(os.path.join(MODEL_PATH, wf))
    weights.update(w)

# Build CP model manually using Qwen3 transformer
from transformers import Qwen3ForCausalLM, Qwen3Config

# The CP is a small Qwen3 model with:
# - input_dim = talker_hidden = 2048 (projected to cp_hidden = 1024)
# - 5 transformer layers
# - 15 separate lm_heads and codec_embeddings

# For testing, we'll implement the forward pass manually
talker_hidden = talker_cfg["hidden_size"]  # 2048
cp_hidden = cp_cfg["hidden_size"]  # 1024
n_groups = cp_cfg["num_code_groups"] - 1  # 15

# Load projection weights
proj_w = weights["talker.code_predictor.small_to_mtp_projection.weight"].to(device).float()
proj_b = weights["talker.code_predictor.small_to_mtp_projection.bias"].to(device).float()
print(f"Projection: {proj_w.shape} + bias {proj_b.shape}")

# Load CP codec embeddings (15 tables, each [vocab_size, talker_hidden])
cp_codec_embs = []
for i in range(n_groups):
    key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
    cp_codec_embs.append(weights[key].to(device).float())
print(f"CP codec embeddings: {n_groups} x {cp_codec_embs[0].shape}")

# Load CP lm_heads (15 heads, each [vocab_size, cp_hidden])
cp_lm_heads = []
for i in range(n_groups):
    key = f"talker.code_predictor.lm_head.{i}.weight"
    cp_lm_heads.append(weights[key].to(device).float())
print(f"CP lm_heads: {n_groups} x {cp_lm_heads[0].shape}")

# Load talker codec embedding (for group 0)
talker_codec_emb = weights["talker.model.codec_embedding.weight"].to(device).float()
print(f"Talker codec embedding: {talker_codec_emb.shape}")

# Build Qwen3 model for CP transformer
qwen3_config = Qwen3Config(
    hidden_size=cp_hidden,
    intermediate_size=cp_cfg["intermediate_size"],
    num_hidden_layers=cp_cfg["num_hidden_layers"],
    num_attention_heads=cp_cfg["num_attention_heads"],
    num_key_value_heads=cp_cfg["num_key_value_heads"],
    vocab_size=cp_cfg["vocab_size"],
    max_position_embeddings=32768,
    rms_norm_eps=cp_cfg.get("rms_norm_eps", 1e-6),
    rope_theta=cp_cfg.get("rope_theta", 1000000.0),
    head_dim=cp_cfg.get("head_dim", 128),
)

cp_model = Qwen3ForCausalLM(qwen3_config).to(device).eval()

# Map CP weights
cp_sd = {}
for k, v in weights.items():
    if k.startswith("talker.code_predictor.model.layers."):
        new_k = k.replace("talker.code_predictor.model.layers.", "model.layers.")
        cp_sd[new_k] = v
    elif k == "talker.code_predictor.model.norm.weight":
        cp_sd["model.norm.weight"] = v

# Set embed_tokens to dummy (we'll use inputs_embeds directly)
cp_sd["model.embed_tokens.weight"] = torch.zeros(cp_cfg["vocab_size"], cp_hidden)
# Set lm_head to group 0's head as placeholder
cp_sd["lm_head.weight"] = cp_lm_heads[0]

missing, unexpected = cp_model.load_state_dict(cp_sd, strict=False)
print(f"CP model: missing={len(missing)}, unexpected={len(unexpected)}")

# Now load the C++ prefill hidden states if available
cpp_hidden_path = os.path.join(LOG_DIR, "cpp_prefill_hidden.bin")
if os.path.exists(cpp_hidden_path):
    cpp_hidden = np.fromfile(cpp_hidden_path, dtype=np.float32)
    talker_hs = torch.tensor(cpp_hidden[:talker_hidden], device=device).float()
    print(f"\nUsing C++ talker hidden state, norm={talker_hs.norm():.4f}")
else:
    # Use random hidden state for testing
    talker_hs = torch.randn(talker_hidden, device=device) * 0.1
    print(f"\nUsing random talker hidden state")

# Test with group 0 token = 302 (from previous debugging)
group0_token = 302

print(f"\n=== Code Predictor autoregressive generation ===")
print(f"Group 0 token: {group0_token}")

# Build prefill: [talker_hs, talker_codec_emb(g0)] as 2 positions
pos0 = talker_hs.unsqueeze(0)  # [1, 2048]
pos1 = talker_codec_emb[group0_token].unsqueeze(0)  # [1, 2048]
prefill_embs = torch.cat([pos0, pos1], dim=0)  # [2, 2048]

# Project through small_to_mtp_projection
prefill_projected = prefill_embs @ proj_w.T + proj_b  # [2, 1024]
prefill_projected = prefill_projected.unsqueeze(0)  # [1, 2, 1024]

print(f"Prefill projected shape: {prefill_projected.shape}")
print(f"  pos0 (talker hs projected)[:5]: {prefill_projected[0, 0, :5].cpu().numpy()}")
print(f"  pos1 (g0 emb projected)[:5]: {prefill_projected[0, 1, :5].cpu().numpy()}")

# Run prefill through CP transformer
with torch.no_grad():
    outputs = cp_model.model(inputs_embeds=prefill_projected, use_cache=True)
    past_kv = outputs.past_key_values
    last_hidden = outputs.last_hidden_state[0, -1, :]  # [1024]

    # Apply lm_head[0] for group 1
    logits = last_hidden @ cp_lm_heads[0].T  # [2048]
    g1_token = logits.argmax().item()
    top3 = torch.topk(logits, 3)
    print(f"\nGroup 1: token={g1_token}, top3={list(zip(top3.indices.cpu().tolist(), [f'{v:.2f}' for v in top3.values.cpu().tolist()]))}")

    group_tokens = [g1_token]

    # Generate groups 2-15
    for g in range(1, n_groups):
        # Embed previous group's token using CP codec_embedding[g-1]
        prev_emb = cp_codec_embs[g-1][group_tokens[-1]].unsqueeze(0).unsqueeze(0)  # [1, 1, 2048]

        # Project
        prev_projected = prev_emb @ proj_w.T + proj_b  # [1, 1, 1024]

        # Forward with KV cache
        outputs = cp_model.model(inputs_embeds=prev_projected,
                                  past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        last_hidden = outputs.last_hidden_state[0, -1, :]

        # Apply lm_head[g]
        logits = last_hidden @ cp_lm_heads[g].T
        token = logits.argmax().item()
        group_tokens.append(token)

        if g < 5:
            top3 = torch.topk(logits, 3)
            print(f"Group {g+1}: token={token}, top3={list(zip(top3.indices.cpu().tolist(), [f'{v:.2f}' for v in top3.values.cpu().tolist()]))}")

print(f"\nAll group tokens (1-15): {group_tokens}")

# Also test WITHOUT KV cache (growing window, matching C++ approach)
print(f"\n=== Growing window approach (no KV cache) ===")
seq_embs = [pos0, pos1]  # Start with [talker_hs, g0_emb]

group_tokens_no_kv = []
with torch.no_grad():
    for g in range(n_groups):
        # Build input sequence
        input_seq = torch.cat(seq_embs, dim=0)  # [g+2, 2048]
        # Project
        projected = input_seq @ proj_w.T + proj_b  # [g+2, 1024]
        projected = projected.unsqueeze(0)  # [1, g+2, 1024]

        # Forward (no cache)
        outputs = cp_model.model(inputs_embeds=projected, use_cache=False)
        last_hidden = outputs.last_hidden_state[0, -1, :]

        # Apply lm_head[g]
        logits = last_hidden @ cp_lm_heads[g].T
        token = logits.argmax().item()
        group_tokens_no_kv.append(token)

        if g < 5:
            top3 = torch.topk(logits, 3)
            print(f"Group {g+1}: token={token}, top3={list(zip(top3.indices.cpu().tolist(), [f'{v:.2f}' for v in top3.values.cpu().tolist()]))}")

        # Append this token's embedding for next iteration
        next_emb = cp_codec_embs[g][token].unsqueeze(0)  # [1, 2048]
        seq_embs.append(next_emb)

print(f"\nGrowing window tokens (1-15): {group_tokens_no_kv}")
print(f"KV cache tokens (1-15):       {group_tokens}")
print(f"Match: {group_tokens == group_tokens_no_kv}")

# Save reference outputs for C++ comparison
np.save(os.path.join(LOG_DIR, "py_cp_group_tokens.npy"), np.array(group_tokens_no_kv))
print(f"\nSaved reference to {LOG_DIR}/py_cp_group_tokens.npy")
print("Done!")
