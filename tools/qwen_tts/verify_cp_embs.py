#!/usr/bin/env python3
"""Verify Code Predictor group embeddings and next_emb computation."""

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

# Load config
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)
talker_cfg = config["talker_config"]

# Load weights
weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
weights = {}
for wf in sorted(weight_files):
    w = load_file(os.path.join(MODEL_PATH, wf))
    weights.update(w)

# Extract Code Predictor codec embeddings (groups 1-15 in Talker space)
cp_codec_embs = []
for i in range(15):
    key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
    if key in weights:
        cp_codec_embs.append(weights[key].to(device).float())
        if i < 3:
            print(f"CP group {i+1} embedding: {cp_codec_embs[-1].shape}")
    else:
        print(f"WARNING: missing {key}")
        break

# Talker codec embedding (group 0)
talker_codec_emb = weights["talker.model.codec_embedding.weight"].to(device).float()
print(f"Talker codec embedding: {talker_codec_emb.shape}")

# TTS pad embedding
text_emb_w = weights["talker.model.text_embedding.weight"].to(device).float()
text_proj_fc1_w = weights["talker.text_projection.linear_fc1.weight"].to(device).float()
text_proj_fc1_b = weights["talker.text_projection.linear_fc1.bias"].to(device).float()
text_proj_fc2_w = weights["talker.text_projection.linear_fc2.weight"].to(device).float()
text_proj_fc2_b = weights["talker.text_projection.linear_fc2.bias"].to(device).float()

def text_projection(x):
    h = x @ text_proj_fc1_w.T + text_proj_fc1_b
    h = torch.nn.functional.silu(h)
    return h @ text_proj_fc2_w.T + text_proj_fc2_b

tts_pad_emb = text_projection(text_emb_w[config["tts_pad_token_id"]].unsqueeze(0)).squeeze(0)

# Now compute next_emb for group0_token=302, groups_1_15=[2004,737,818,1859,...]
# Using the same tokens as C++
group0_token = 302
group_tokens_1_15 = [2004, 737, 818, 1859, 1859, 818, 818, 1859, 818, 818, 1859, 818, 1859, 818, 1859]
# Let me get the actual C++ group tokens by reading the log
# But for now, let's compute with the first 4 and see

print(f"\n--- next_emb computation for token 302 ---")
print(f"group0_token: {group0_token}")

# Group 0: Talker codec embedding
g0_emb = talker_codec_emb[group0_token]
print(f"g0 emb[:5]: {g0_emb[:5].cpu().numpy()}")

# Group 0 only + tts_pad
g0_only = g0_emb + tts_pad_emb
print(f"g0_only[:5]: {g0_only[:5].cpu().numpy()}")

# Now add CP groups
next_emb = g0_emb.clone()
for g in range(15):
    if g < len(group_tokens_1_15):
        token = group_tokens_1_15[g]
        cp_emb = cp_codec_embs[g][token]
        next_emb = next_emb + cp_emb
        if g < 3:
            print(f"  + group {g+1} (token {token}) emb[:5]: {cp_emb[:5].cpu().numpy()}")
            print(f"    running sum[:5]: {next_emb[:5].cpu().numpy()}")

# Add tts_pad
next_emb = next_emb + tts_pad_emb
print(f"\nnext_emb (all groups + tts_pad)[:5]: {next_emb[:5].cpu().numpy()}")
print(f"g0_only[:5]:                        {g0_only[:5].cpu().numpy()}")

# Check: is the sum magnitude reasonable?
g0_norm = g0_emb.norm().item()
next_norm = next_emb.norm().item()
g0_only_norm = g0_only.norm().item()
print(f"\nNorms: g0_emb={g0_norm:.4f}, g0_only={g0_only_norm:.4f}, next_emb={next_norm:.4f}")
print(f"Ratio next/g0_only = {next_norm/g0_only_norm:.4f}")

# Let me also check: in the Python model, does it actually sum all 16 embeddings?
# Or does it use a different approach?
# The Python model class Qwen3TTSTalkerForConditionalGeneration has a method
# that constructs the next input embedding.

# Let's also run multi-step with full 16-group embeddings
print("\n--- Multi-step with full 16-group embeddings ---")
from transformers import Qwen3ForCausalLM, Qwen3Config

qwen3_config = Qwen3Config(
    hidden_size=talker_cfg["hidden_size"],
    intermediate_size=talker_cfg["intermediate_size"],
    num_hidden_layers=talker_cfg["num_hidden_layers"],
    num_attention_heads=talker_cfg["num_attention_heads"],
    num_key_value_heads=talker_cfg["num_key_value_heads"],
    vocab_size=talker_cfg["vocab_size"],
    max_position_embeddings=32768,
    rms_norm_eps=talker_cfg.get("rms_norm_eps", 1e-6),
    rope_theta=talker_cfg.get("rope_theta", 1000000.0),
    head_dim=talker_cfg.get("head_dim", 128),
)
model = Qwen3ForCausalLM(qwen3_config).to(device).eval()

talker_sd = {}
for k, v in weights.items():
    if k.startswith("talker.model."):
        talker_sd[k.replace("talker.model.", "model.")] = v
    elif k.startswith("talker.codec_head."):
        talker_sd[k.replace("talker.codec_head.", "lm_head.")] = v
talker_sd["model.embed_tokens.weight"] = weights["talker.model.codec_embedding.weight"]
model.load_state_dict(talker_sd, strict=False)

# Load C++ prefill
cpp_embs_path = os.path.join(LOG_DIR, "cpp_prefill_embs.bin")
with open(cpp_embs_path, "rb") as f:
    seq_len, dim = struct.unpack("ii", f.read(8))
    data = np.frombuffer(f.read(seq_len * dim * 4), dtype=np.float32)
    cpp_embs = data.reshape(seq_len, dim)

inputs_embeds = torch.tensor(cpp_embs, device=device, dtype=torch.float32).unsqueeze(0)

# Code Predictor model (for predicting groups 1-15)
# We need to load and run the actual Code Predictor
# For this test, let's just hardcode the group tokens and compute the sum
# to verify the next_emb computation matches C++

print("Running with full 16-group next_emb...")
codec_eos_id = talker_cfg["codec_eos_token_id"]

with torch.no_grad():
    outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    past_kv = outputs.past_key_values
    logits = outputs.logits[0, -1, :]

    for step in range(10):
        token = logits.argmax().item()
        top3 = torch.topk(logits, 3)
        print(f"Step {step}: token={token}, top3={list(zip(top3.indices.cpu().tolist(), [f'{v:.2f}' for v in top3.values.cpu().tolist()]))}")

        if token == codec_eos_id:
            break

        # Compute full next_emb (all 16 groups)
        # Use hardcoded CP outputs (same as C++ for comparison)
        g0_emb = talker_codec_emb[token]
        next_emb_full = g0_emb.clone()

        # Add CP group embeddings (using fixed group tokens for this test)
        # In reality, CP would generate different tokens per step
        fixed_group_tokens = [2004, 737, 818, 1859, 1859, 818, 818, 1859, 818, 818, 1859, 818, 1859, 818, 1859]
        for g in range(15):
            cp_emb = cp_codec_embs[g][fixed_group_tokens[g]]
            next_emb_full = next_emb_full + cp_emb

        next_emb_full = next_emb_full + tts_pad_emb

        next_input = next_emb_full.unsqueeze(0).unsqueeze(0)
        outputs = model(inputs_embeds=next_input, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

print("Done!")
