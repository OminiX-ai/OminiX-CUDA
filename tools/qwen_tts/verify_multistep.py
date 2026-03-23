#!/usr/bin/env python3
"""Multi-step generation test: does the Python model also repeat token 302?"""

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

# Load weights
weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
weights = {}
for wf in sorted(weight_files):
    w = load_file(os.path.join(MODEL_PATH, wf))
    weights.update(w)

# Load C++ prefill embeddings
cpp_embs_path = os.path.join(LOG_DIR, "cpp_prefill_embs.bin")
with open(cpp_embs_path, "rb") as f:
    seq_len, dim = struct.unpack("ii", f.read(8))
    data = np.frombuffer(f.read(seq_len * dim * 4), dtype=np.float32)
    cpp_embs = data.reshape(seq_len, dim)
print(f"Prefill: {seq_len} x {dim}")

# Build Qwen3 model
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

# Map weights
talker_sd = {}
for k, v in weights.items():
    if k.startswith("talker.model."):
        talker_sd[k.replace("talker.model.", "model.")] = v
    elif k.startswith("talker.codec_head."):
        talker_sd[k.replace("talker.codec_head.", "lm_head.")] = v
talker_sd["model.embed_tokens.weight"] = weights["talker.model.codec_embedding.weight"]
model.load_state_dict(talker_sd, strict=False)

# Extract needed weights for manual embedding construction
text_emb_w = weights["talker.model.text_embedding.weight"].to(device).float()
codec_emb_w = weights["talker.model.codec_embedding.weight"].to(device).float()
text_proj_fc1_w = weights["talker.text_projection.linear_fc1.weight"].to(device).float()
text_proj_fc1_b = weights["talker.text_projection.linear_fc1.bias"].to(device).float()
text_proj_fc2_w = weights["talker.text_projection.linear_fc2.weight"].to(device).float()
text_proj_fc2_b = weights["talker.text_projection.linear_fc2.bias"].to(device).float()

# CP codec embeddings
cp_codec_embs = []
for i in range(15):
    key = f"talker.code_predictor.model.codec_embedding.{i}.weight"
    cp_codec_embs.append(weights[key].to(device).float())

tts_pad_id = config["tts_pad_token_id"]  # 151671

def text_projection(x):
    h = x @ text_proj_fc1_w.T + text_proj_fc1_b
    h = torch.nn.functional.silu(h)
    return h @ text_proj_fc2_w.T + text_proj_fc2_b

tts_pad_emb = text_projection(text_emb_w[tts_pad_id].unsqueeze(0)).squeeze(0)

print("\n--- Multi-step generation with C++ embeddings ---")
inputs_embeds = torch.tensor(cpp_embs, device=device, dtype=torch.float32).unsqueeze(0)

codec_eos_id = talker_cfg["codec_eos_token_id"]
print(f"codec_eos_id: {codec_eos_id}")

generated_tokens = []
with torch.no_grad():
    # Prefill
    outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    past_kv = outputs.past_key_values
    logits = outputs.logits[0, -1, :]

    for step in range(20):
        # Greedy sample
        token = logits.argmax().item()
        top3 = torch.topk(logits, 3)
        print(f"Step {step}: token={token}, top3={list(zip(top3.indices.cpu().tolist(), [f'{v:.2f}' for v in top3.values.cpu().tolist()]))}")

        generated_tokens.append(token)

        if token == codec_eos_id:
            print("EOS!")
            break

        # Compute next embedding: sum of all group embeddings + tts_pad
        # For simplicity, just use group 0 embedding + tts_pad
        # (This matches the C++ behavior where all groups produce the same tokens)
        next_emb = codec_emb_w[token].clone()

        # If token is a regular codec token (< 2048), we'd run Code Predictor
        # But for this test, just add tts_pad without CP groups
        # This won't match C++ exactly but shows if the main LLM behavior is similar
        next_emb = next_emb + tts_pad_emb

        # Feed to model
        next_input = next_emb.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]
        outputs = model(inputs_embeds=next_input, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

print(f"\nGenerated tokens: {generated_tokens}")
print("Done!")
