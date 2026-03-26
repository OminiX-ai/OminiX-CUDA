#!/usr/bin/env python3
"""Verify Talker LLM generation using raw PyTorch + transformers Qwen3 model.
Loads the Talker as a Qwen3 model (with codec vocab) and runs generation
to get reference first-token outputs."""

import sys, os
os.chdir("/root/autodl-tmp")

import torch
import torch_npu
import numpy as np
import json
from safetensors.torch import load_file

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUTPUT_DIR = "/root/autodl-tmp/tts.cpp/logs"

device = "npu:0" if torch.npu.is_available() else "cpu"
print(f"Device: {device}")

# Load config
with open(os.path.join(MODEL_PATH, "config.json")) as f:
    config = json.load(f)
talker_cfg = config["talker_config"]

# Load ALL model weights
print("Loading model weights...")
weight_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
weights = {}
for wf in sorted(weight_files):
    w = load_file(os.path.join(MODEL_PATH, wf))
    weights.update(w)
    print(f"  Loaded {wf}: {len(w)} tensors")

# Load the prefill embeddings from our previous verification
prefill = np.load(os.path.join(OUTPUT_DIR, "py_prefill_partial.npy"))
prefill_t = torch.tensor(prefill, device=device, dtype=torch.float32)
print(f"\nLoaded prefill embeddings: {prefill_t.shape}")

# We need to add the ICL codec section (codec_bos + ref frames).
# For this test, let's use a minimal prefill (the partial one without ref codec)
# The partial prefill has: role(3) + mixed_prefix(5) + icl_text(8) = 16 positions

# Now let's load the Talker as a Qwen3 model
# The Talker is basically a Qwen3 model with different vocab
from transformers import AutoConfig, AutoModelForCausalLM, Qwen3ForCausalLM, Qwen3Config

# Create Qwen3 config for the Talker
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
print(f"\nQwen3 config: {qwen3_config.hidden_size}d, {qwen3_config.num_hidden_layers}L, "
      f"heads={qwen3_config.num_attention_heads}/{qwen3_config.num_key_value_heads}")

# Build the model
model = Qwen3ForCausalLM(qwen3_config).to(device).eval()

# Map PyTorch weights to Qwen3 model
# The Talker's transformer weights are under "talker.model.layers.X.*"
# Qwen3ForCausalLM expects "model.layers.X.*"
talker_sd = {}
for k, v in weights.items():
    if k.startswith("talker.model."):
        new_k = k.replace("talker.model.", "model.")
        talker_sd[new_k] = v
    elif k.startswith("talker.codec_head."):
        new_k = k.replace("talker.codec_head.", "lm_head.")
        talker_sd[new_k] = v

# The model.embed_tokens is the codec embedding
talker_sd["model.embed_tokens.weight"] = weights["talker.model.codec_embedding.weight"]

print(f"Mapped {len(talker_sd)} tensors for Qwen3 model")

# Load weights
missing, unexpected = model.load_state_dict(talker_sd, strict=False)
print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")
if missing:
    print(f"  Missing: {missing[:5]}...")

# Run forward pass with prefill embeddings
print("\n--- Running forward pass ---")
with torch.no_grad():
    # The model expects inputs_embeds [batch, seq_len, hidden]
    inputs_embeds = prefill_t.unsqueeze(0)  # [1, seq_len, hidden]
    print(f"Input shape: {inputs_embeds.shape}")

    outputs = model(inputs_embeds=inputs_embeds, use_cache=True)

    # Get hidden states and logits from last position
    hidden_states = outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
    logits = outputs.logits  # [1, seq_len, vocab_size]

    last_logits = logits[0, -1, :]  # [vocab_size]
    top5_tokens = torch.topk(last_logits, 5)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Top-5 tokens: {list(zip(top5_tokens.indices.cpu().tolist(), top5_tokens.values.cpu().tolist()))}")

    # Get the hidden state at last position (pre-lm_head, post-norm)
    # Qwen3ForCausalLM applies: hidden -> layers -> norm -> lm_head
    # We want the post-norm hidden state
    # Use model.model() instead to get the hidden states directly
    base_outputs = model.model(inputs_embeds=inputs_embeds, use_cache=False)
    last_hidden = base_outputs.last_hidden_state[0, -1, :]  # [hidden]
    print(f"\nLast hidden (post-norm) [:5]: {last_hidden[:5].cpu().numpy()}")

    # Apply codec_head manually
    codec_head_w = weights["talker.codec_head.weight"].to(device).float()
    manual_logits = last_hidden @ codec_head_w.T
    manual_top5 = torch.topk(manual_logits, 5)
    print(f"Manual codec_head top-5: {list(zip(manual_top5.indices.cpu().tolist(), manual_top5.values.cpu().tolist()))}")

print("\nDone!")
