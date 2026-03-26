"""Generate reference data for Code Predictor verification.

Runs the Python Code Predictor on dummy hidden states and saves:
- data/ref_cp_hidden_states.bin: input hidden states [2048, seq_len] float32
- data/ref_cp_group_tokens.bin: input group tokens [seq_len] int32
- data/ref_cp_logits.bin: output logits [2048, seq_len] float32
"""
import sys
import os
import numpy as np
import torch

from pathlib import Path
gguf_py_path = Path(__file__).parent.parent.parent.parent / 'gguf-py'
if gguf_py_path.exists():
    sys.path.insert(1, str(gguf_py_path))


def main():
    model_path = os.environ.get(
        "QWEN_TTS_MODEL_PATH",
        "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )

    # Load model
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import (
        Qwen3TTSForConditionalGeneration,
    )
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path, device_map="cpu", dtype=torch.float32,
        trust_remote_code=True,
    )
    model.eval()
    print("Loaded.")

    talker = model.talker
    code_predictor = talker.code_predictor
    code_predictor.eval()

    # Generate random hidden states and group tokens
    seq_len = 10
    talker_hidden = 2048
    cp_hidden = 1024
    vocab_size = 2048

    np.random.seed(42)
    hidden_np = np.random.randn(1, seq_len, talker_hidden).astype(np.float32) * 0.1
    hidden_states = torch.from_numpy(hidden_np)

    # Random group 0 tokens
    group0_tokens = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)

    print(f"Hidden states shape: {hidden_states.shape}")
    print(f"Group 0 tokens shape: {group0_tokens.shape}")

    with torch.no_grad():
        # The actual CP flow:
        # 1. Prefill: project(talker_hidden) → transformer → KV cache
        # 2. For each group g: project(codec_emb[g-1](prev_tokens)) → transformer → lm_head[g-1]
        #
        # Simplified: process each group independently
        # Input = talker_hidden + codec_emb → project → transformer → lm_head

        # Step 1: Embed group 0 tokens in Talker space (2048-dim)
        codec_emb = code_predictor.model.codec_embedding[0](group0_tokens)
        print(f"Codec embedding shape: {codec_emb.shape}")  # [1, 10, 2048]

        # Step 2: Combine and project
        combined = hidden_states + codec_emb  # [1, 10, 2048]
        projected = code_predictor.small_to_mtp_projection(combined)
        print(f"Projected shape: {projected.shape}")  # [1, 10, 1024]
        print(f"Projected stats: mean={projected.mean():.6f}, std={projected.std():.6f}")

        # Step 3: Forward through transformer layers
        cur = projected
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf')), diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # Compute RoPE position embeddings
        position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        position_embeddings = code_predictor.model.rotary_emb(cur, position_ids)

        for i, layer in enumerate(code_predictor.model.layers):
            cur = layer(cur, attention_mask=causal_mask,
                        position_embeddings=position_embeddings)[0]
            if i == 0:
                layer0_out = cur.clone()
                print(f"Layer 0 output: mean={cur.mean():.6f}, std={cur.std():.6f}")

        cur = code_predictor.model.norm(cur)

        # Step 4: Apply lm_head for group 1
        logits = code_predictor.lm_head[0](cur)
        print(f"Logits shape: {logits.shape}")  # [1, 10, 2048]
        print(f"Logits stats: mean={logits.mean():.6f}, std={logits.std():.6f}")

        # Get predicted tokens
        predicted = logits[:, -1, :].argmax(dim=-1)
        print(f"Predicted group 1 token: {predicted.item()}")

    # Save reference data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # NOTE: ggml uses column-major layout. A ggml tensor [dim, seq_len] stores
    # data as: flat[d + s*dim]. numpy [seq_len, dim] row-major stores data as:
    # flat[s*dim + d]. These are identical! So we save WITHOUT transposing.

    # Save hidden states [seq_len, talker_hidden] (matches ggml [talker_hidden, seq_len])
    hs_np = hidden_np[0].astype(np.float32)  # [10, 2048]
    hs_np.tofile(os.path.join(data_dir, "ref_cp_hidden_states.bin"))
    print(f"Saved hidden states: {hs_np.shape}")

    # Save group 0 tokens
    g0_np = group0_tokens[0].numpy().astype(np.int32)
    g0_np.tofile(os.path.join(data_dir, "ref_cp_group_tokens.bin"))
    print(f"Saved group tokens: {g0_np.shape}")

    # Save projected output [seq_len, cp_hidden] (matches ggml [cp_hidden, seq_len])
    proj_np = projected[0].numpy().astype(np.float32)  # [10, 1024]
    proj_np.tofile(os.path.join(data_dir, "ref_cp_projected.bin"))
    print(f"Saved projected: {proj_np.shape}")

    # Save logits [seq_len, vocab_size] (matches ggml [vocab_size, seq_len])
    logits_np = logits[0].numpy().astype(np.float32)  # [10, 2048]
    logits_np.tofile(os.path.join(data_dir, "ref_cp_logits.bin"))
    print(f"Saved logits: {logits_np.shape}")

    # Save layer 0 output [seq_len, cp_hidden]
    l0_np = layer0_out[0].numpy().astype(np.float32)  # [10, 1024]
    l0_np.tofile(os.path.join(data_dir, "ref_cp_layer0.bin"))
    print(f"Saved layer0 output: {l0_np.shape}")

    print("\nDone!")


if __name__ == "__main__":
    main()
