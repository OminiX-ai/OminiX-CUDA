#!/usr/bin/env python3
"""Multi-step generation with full CP: verify diverse token generation."""

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

with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)
talker_cfg = config["talker_config"]
cp_cfg = talker_cfg["code_predictor_config"]

weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
weights = {}
for wf in sorted(weight_files):
    w = load_file(os.path.join(MODEL_PATH, wf))
    weights.update(w)

# Build Talker model
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

# Build CP model
cp_qwen3_config = Qwen3Config(
    hidden_size=cp_cfg["hidden_size"],
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
cp_model = Qwen3ForCausalLM(cp_qwen3_config).to(device).eval()

cp_sd = {}
for k, v in weights.items():
    if k.startswith("talker.code_predictor.model.layers."):
        cp_sd[k.replace("talker.code_predictor.model.layers.", "model.layers.")] = v
    elif k == "talker.code_predictor.model.norm.weight":
        cp_sd["model.norm.weight"] = v
cp_sd["model.embed_tokens.weight"] = torch.zeros(cp_cfg["vocab_size"], cp_cfg["hidden_size"])
cp_sd["lm_head.weight"] = torch.zeros(cp_cfg["vocab_size"], cp_cfg["hidden_size"])
cp_model.load_state_dict(cp_sd, strict=False)

# Load weights for CP
talker_hidden = talker_cfg["hidden_size"]
cp_hidden = cp_cfg["hidden_size"]
n_groups = cp_cfg["num_code_groups"] - 1

proj_w = weights["talker.code_predictor.small_to_mtp_projection.weight"].to(device).float()
proj_b = weights["talker.code_predictor.small_to_mtp_projection.bias"].to(device).float()

cp_codec_embs = [weights[f"talker.code_predictor.model.codec_embedding.{i}.weight"].to(device).float() for i in range(n_groups)]
cp_lm_heads = [weights[f"talker.code_predictor.lm_head.{i}.weight"].to(device).float() for i in range(n_groups)]
talker_codec_emb = weights["talker.model.codec_embedding.weight"].to(device).float()

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


def predict_cp_groups(talker_hs, g0_token):
    """Run Code Predictor to predict groups 1-15 given talker hidden state."""
    pos0 = talker_hs.unsqueeze(0)
    pos1 = talker_codec_emb[g0_token].unsqueeze(0)
    seq_embs = [pos0, pos1]

    group_tokens = []
    with torch.no_grad():
        for g in range(n_groups):
            input_seq = torch.cat(seq_embs, dim=0)
            projected = input_seq @ proj_w.T + proj_b
            projected = projected.unsqueeze(0)

            outputs = cp_model.model(inputs_embeds=projected, use_cache=False)
            last_hidden = outputs.last_hidden_state[0, -1, :]
            logits = last_hidden @ cp_lm_heads[g].T
            token = logits.argmax().item()
            group_tokens.append(token)

            next_emb = cp_codec_embs[g][token].unsqueeze(0)
            seq_embs.append(next_emb)

    return group_tokens


# Load C++ prefill embeddings
cpp_embs_path = os.path.join(LOG_DIR, "cpp_prefill_embs.bin")
with open(cpp_embs_path, "rb") as f:
    seq_len, dim = struct.unpack("ii", f.read(8))
    data = np.frombuffer(f.read(seq_len * dim * 4), dtype=np.float32)
    cpp_embs = data.reshape(seq_len, dim)

inputs_embeds = torch.tensor(cpp_embs, device=device, dtype=torch.float32).unsqueeze(0)
codec_eos_id = talker_cfg["codec_eos_token_id"]

print(f"\n=== Multi-step generation with full CP (growing window) ===")
print(f"Prefill: {seq_len} tokens, codec_eos={codec_eos_id}")

generated_tokens = []
all_group_tokens = []
with torch.no_grad():
    # Prefill
    outputs = model(inputs_embeds=inputs_embeds, use_cache=True)
    past_kv = outputs.past_key_values
    logits = outputs.logits[0, -1, :]

    # Also get hidden state for CP
    base_outputs = model.model(inputs_embeds=inputs_embeds, use_cache=False)
    talker_hs = base_outputs.last_hidden_state[0, -1, :]

    for step in range(20):
        token = logits.argmax().item()
        top3 = torch.topk(logits, 3)

        generated_tokens.append(token)

        if token == codec_eos_id:
            print(f"Step {step}: EOS!")
            break

        # Run CP for this token
        group_tokens = predict_cp_groups(talker_hs, token)
        all_group_tokens.append(group_tokens)

        print(f"Step {step}: g0={token}, g1-3=[{group_tokens[0]},{group_tokens[1]},{group_tokens[2]}...], "
              f"top3={list(zip(top3.indices.cpu().tolist()[:3], [f'{v:.2f}' for v in top3.values.cpu().tolist()[:3]]))}")

        # Compute next embedding: sum all 16 groups + tts_pad
        next_emb = talker_codec_emb[token].clone()
        for g in range(n_groups):
            next_emb = next_emb + cp_codec_embs[g][group_tokens[g]]
        next_emb = next_emb + tts_pad_emb

        # Feed to talker
        next_input = next_emb.unsqueeze(0).unsqueeze(0)
        outputs = model(inputs_embeds=next_input, past_key_values=past_kv, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[0, -1, :]

        # Get hidden state for next CP call
        base_out = model.model(inputs_embeds=next_input, use_cache=False)
        talker_hs = base_out.last_hidden_state[0, -1, :]

print(f"\nGenerated group 0 tokens: {generated_tokens}")
print("Done!")
