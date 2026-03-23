#!/usr/bin/env python3
"""Verify C++ embedding construction against Python reference.
Uses raw PyTorch to avoid qwen_tts package dependency issues."""

import sys, os
os.chdir("/root/autodl-tmp")

import torch
import torch_npu
import numpy as np
import json

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUTPUT_DIR = "/root/autodl-tmp/tts.cpp/logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "npu:0" if torch.npu.is_available() else "cpu"
print(f"Device: {device}")

# Load config
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)

talker_cfg = config["talker_config"]
print(f"Talker hidden_size={talker_cfg['hidden_size']}, vocab_size={talker_cfg['vocab_size']}")
print(f"codec_think_id={talker_cfg['codec_think_id']}, codec_bos_id={talker_cfg['codec_bos_id']}")
print(f"codec_language_id: {talker_cfg.get('codec_language_id', {})}")

# Load model weights
from safetensors.torch import load_file
print("\nLoading model weights...")
weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
weights = {}
for wf in sorted(weight_files):
    w = load_file(os.path.join(MODEL_PATH, wf))
    weights.update(w)
    print(f"  Loaded {wf}: {len(w)} tensors")

# Extract embedding weights
text_emb_w = weights["talker.model.text_embedding.weight"].to(device).float()
codec_emb_w = weights["talker.model.codec_embedding.weight"].to(device).float()
text_proj_fc1_w = weights["talker.text_projection.linear_fc1.weight"].to(device).float()
text_proj_fc1_b = weights["talker.text_projection.linear_fc1.bias"].to(device).float()
text_proj_fc2_w = weights["talker.text_projection.linear_fc2.weight"].to(device).float()
text_proj_fc2_b = weights["talker.text_projection.linear_fc2.bias"].to(device).float()

print(f"\ntext_emb_w: {text_emb_w.shape}")
print(f"codec_emb_w: {codec_emb_w.shape}")
print(f"text_proj_fc1: {text_proj_fc1_w.shape} + {text_proj_fc1_b.shape}")
print(f"text_proj_fc2: {text_proj_fc2_w.shape} + {text_proj_fc2_b.shape}")

def text_projection(x):
    """Apply text projection: fc1 -> silu -> fc2"""
    h = x @ text_proj_fc1_w.T + text_proj_fc1_b
    h = torch.nn.functional.silu(h)
    return h @ text_proj_fc2_w.T + text_proj_fc2_b

def text_proj_embed(token_id):
    """Get text_proj(text_emb(token_id))"""
    return text_projection(text_emb_w[token_id].unsqueeze(0)).squeeze(0)

# Compute TTS special embeddings
tts_bos_id = config["tts_bos_token_id"]  # 151672
tts_eos_id = config["tts_eos_token_id"]  # 151673
tts_pad_id = config["tts_pad_token_id"]  # 151671
print(f"\nTTS token IDs: bos={tts_bos_id}, eos={tts_eos_id}, pad={tts_pad_id}")

tts_bos_emb = text_proj_embed(tts_bos_id)
tts_eos_emb = text_proj_embed(tts_eos_id)
tts_pad_emb = text_proj_embed(tts_pad_id)

print(f"tts_pad_emb[:5]: {tts_pad_emb[:5].cpu().numpy()}")
print(f"tts_bos_emb[:5]: {tts_bos_emb[:5].cpu().numpy()}")

# Save for C++ comparison
np.save(os.path.join(OUTPUT_DIR, "py_tts_pad_emb.npy"), tts_pad_emb.cpu().numpy())
np.save(os.path.join(OUTPUT_DIR, "py_tts_bos_emb.npy"), tts_bos_emb.cpu().numpy())
np.save(os.path.join(OUTPUT_DIR, "py_tts_eos_emb.npy"), tts_eos_emb.cpu().numpy())

# Build the exact prefill sequence for voice cloning
# Role prefix: <|im_start|>assistant\n -> [151644, 77091, 198]
role_ids = [151644, 77091, 198]
role_emb = text_projection(text_emb_w[role_ids])  # [3, hidden]
print(f"\nRole prefix embeddings[:5]: {role_emb[0, :5].cpu().numpy()}")

# Codec prefix for English (language specified → thinking mode)
language_id = talker_cfg["codec_language_id"]["english"]  # 2050
codec_prefix_ids = [
    talker_cfg["codec_think_id"],      # 2154
    talker_cfg["codec_think_bos_id"],  # 2156
    language_id,                        # 2050
    talker_cfg["codec_think_eos_id"],  # 2157
    # speaker embedding goes here (not a token)
    talker_cfg["codec_pad_id"],        # 2148
    talker_cfg["codec_bos_id"],        # 2149
]
print(f"Codec prefix IDs: {codec_prefix_ids}")

codec_prefix_emb = codec_emb_w[codec_prefix_ids]  # [6, hidden]
print(f"Codec prefix emb[0][:5]: {codec_prefix_emb[0, :5].cpu().numpy()}")

# Mixed prefix: N=7 (4 think + 1 spk + 2 pad/bos), N-1=6 positions
# text side: [tts_pad × 5, tts_bos]
# Without speaker: N=6, N-1=5 positions (for testing without spk)
# With speaker at index 4: we skip it here for pure embedding comparison

# Build mixed prefix (without speaker for now)
N = len(codec_prefix_ids)  # 6 (no speaker)
mixed_prefix = torch.zeros(N-1, talker_cfg["hidden_size"], device=device)
for i in range(N-1):
    text_component = tts_pad_emb if i < N-2 else tts_bos_emb
    codec_component = codec_prefix_emb[i]
    mixed_prefix[i] = text_component + codec_component

print(f"\nMixed prefix[0][:5] (tts_pad + think): {mixed_prefix[0, :5].cpu().numpy()}")
print(f"Mixed prefix[4][:5] (tts_bos + pad): {mixed_prefix[4, :5].cpu().numpy()}")

# Save all prefix embeddings
np.save(os.path.join(OUTPUT_DIR, "py_role_emb.npy"), role_emb.cpu().numpy())
np.save(os.path.join(OUTPUT_DIR, "py_mixed_prefix.npy"), mixed_prefix.cpu().numpy())

# Build ICL text embeddings
# ref_text = "Hello world" → tokenize
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

ref_ids = tokenizer.encode("<|im_start|>assistant\nHello world<|im_end|>\n", add_special_tokens=False)
target_ids = tokenizer.encode("<|im_start|>assistant\nThis is a test.<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)

ref_text_tokens = ref_ids[3:-2]
target_text_tokens = target_ids[3:-5]
print(f"\nRef text tokens: {ref_text_tokens}")
print(f"Target text tokens: {target_text_tokens}")

# ICL text part: text_proj(text_emb(ref + target)) + tts_eos, each + codec_pad
all_text_ids = ref_text_tokens + target_text_tokens
icl_text_embs = text_projection(text_emb_w[all_text_ids])  # [M, hidden]
# Append tts_eos
icl_text_embs = torch.cat([icl_text_embs, tts_eos_emb.unsqueeze(0)], dim=0)
# Add codec_pad
codec_pad_emb = codec_emb_w[talker_cfg["codec_pad_id"]]
icl_text_embs = icl_text_embs + codec_pad_emb.unsqueeze(0)

print(f"ICL text embeddings shape: {icl_text_embs.shape}")
print(f"ICL text[0][:5]: {icl_text_embs[0, :5].cpu().numpy()}")

np.save(os.path.join(OUTPUT_DIR, "py_icl_text_embs.npy"), icl_text_embs.cpu().numpy())

# Full prefill (without speaker and ref codec for now)
full_prefill = torch.cat([role_emb, mixed_prefix, icl_text_embs], dim=0)
print(f"\nFull prefill (partial) shape: {full_prefill.shape}")
np.save(os.path.join(OUTPUT_DIR, "py_prefill_partial.npy"), full_prefill.cpu().numpy())

print(f"\nSaved all reference embeddings to {OUTPUT_DIR}")
print("Done!")
