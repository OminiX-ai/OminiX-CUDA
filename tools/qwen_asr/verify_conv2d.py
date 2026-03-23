"""
Step-by-step verification of Qwen3-ASR audio encoder for C++ comparison.

Saves intermediate results after each stage:
1. After each Conv2d layer
2. After flatten + linear projection (conv_out)
3. After positional embedding addition
4. After first transformer layer

Usage:
    python tools/qwen_asr/verify_conv2d.py
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn.functional as F

# Register Qwen3-ASR
from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
    _get_feat_extract_output_lengths,
    SinusoidsPositionEmbedding,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def main():
    model_path = "Qwen/Qwen3-ASR-1.7B"
    output_dir = "tools/qwen_asr/verify_data"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path, device_map="cpu", dtype=torch.float32,
    )
    model.eval()

    audio_tower = model.thinker.audio_tower
    print(f"Audio encoder config:")
    print(f"  d_model = {audio_tower.config.d_model}")
    print(f"  encoder_layers = {audio_tower.config.encoder_layers}")
    print(f"  encoder_attention_heads = {audio_tower.config.encoder_attention_heads}")
    print(f"  downsample_hidden_size = {audio_tower.config.downsample_hidden_size}")
    print(f"  n_window = {audio_tower.n_window}")
    print(f"  n_window_infer = {audio_tower.n_window_infer}")
    print(f"  attn_implementation = {audio_tower.config._attn_implementation}")

    # Load input mel
    mel_input = np.load(f"{output_dir}/input_features.npy")  # (128, T_total)
    feature_attention_mask = np.load(f"{output_dir}/feature_attention_mask.npy")  # (T_total,)

    # Use the feature_attention_mask to get actual length
    feature_len = int(feature_attention_mask.sum())
    mel_input_tensor = torch.tensor(mel_input[:, :feature_len], dtype=torch.float32)
    print(f"\nMel input shape: {mel_input_tensor.shape} (128, {feature_len})")

    # Compute chunking (same as Python forward)
    feature_len_t = torch.tensor(feature_len, dtype=torch.long)
    aftercnn_lens = _get_feat_extract_output_lengths(feature_len_t)
    print(f"aftercnn_lens: {aftercnn_lens}")

    chunk_num = torch.ceil(feature_len_t.float() / (audio_tower.n_window * 2)).long()
    chunk_lengths = torch.tensor(
        [audio_tower.n_window * 2] * chunk_num.item(),
        dtype=torch.long,
    )
    tail_chunk_index = chunk_num - 1
    remainder = feature_len_t % (audio_tower.n_window * 2)
    if remainder != 0:
        chunk_lengths[tail_chunk_index] = remainder
    else:
        chunk_lengths[tail_chunk_index] = audio_tower.n_window * 2

    print(f"chunk_num: {chunk_num.item()}")
    print(f"chunk_lengths: {chunk_lengths.tolist()}")

    # Split into chunks and pad
    chunk_list = mel_input_tensor.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
    padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
        [torch.ones(length, dtype=torch.bool) for length in feature_lens_after_cnn],
        batch_first=True,
    )
    padded_feature = padded_feature.unsqueeze(1)

    print(f"\nPadded mel batch shape: {padded_feature.shape}")  # (chunk_num, 1, 128, max_chunk_len)
    print(f"feature_lens_after_cnn: {feature_lens_after_cnn.tolist()}")
    print(f"padded_mask_after_cnn shape: {padded_mask_after_cnn.shape}")

    # Save padded mel for C++ comparison
    np.save(f"{output_dir}/padded_mel.npy", padded_feature.numpy())

    with torch.no_grad():
        # ==========================================
        # Stage 1: Conv2d layers
        # ==========================================
        x = padded_feature

        x = F.gelu(audio_tower.conv2d1(x))
        print(f"\nAfter conv2d1 + gelu: {x.shape}")
        np.save(f"{output_dir}/after_conv2d1.npy", x.numpy())

        x = F.gelu(audio_tower.conv2d2(x))
        print(f"After conv2d2 + gelu: {x.shape}")
        np.save(f"{output_dir}/after_conv2d2.npy", x.numpy())

        x = F.gelu(audio_tower.conv2d3(x))
        print(f"After conv2d3 + gelu: {x.shape}")
        np.save(f"{output_dir}/after_conv2d3.npy", x.numpy())

        # ==========================================
        # Stage 2: Flatten + linear projection
        # ==========================================
        b, c, f, t = x.size()
        print(f"\nConv output: batch={b}, channels={c}, freq={f}, time={t}")

        # PyTorch: permute(0,3,1,2) -> (b, t, c, f) -> view(b, t, c*f)
        x_flat = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        print(f"After permute+flatten: {x_flat.shape}")
        np.save(f"{output_dir}/after_flatten.npy", x_flat.numpy())

        x_proj = audio_tower.conv_out(x_flat)
        print(f"After conv_out linear: {x_proj.shape}")
        np.save(f"{output_dir}/after_conv_out.npy", x_proj.numpy())

        # ==========================================
        # Stage 3: Positional embedding
        # ==========================================
        pos_emb = audio_tower.positional_embedding.positional_embedding[:x_proj.shape[1], :]
        print(f"\nPositional embedding shape: {pos_emb.shape}")
        print(f"Pos emb first 10 values: {pos_emb[0, :10].tolist()}")
        np.save(f"{output_dir}/pos_emb.npy", pos_emb.numpy())

        x_with_pos = x_proj + pos_emb.unsqueeze(0)
        print(f"After pos emb addition: {x_with_pos.shape}")
        np.save(f"{output_dir}/after_pos_emb.npy", x_with_pos.numpy())

        # ==========================================
        # Stage 4: Mask and extract valid frames
        # ==========================================
        hidden_states = x_with_pos[padded_mask_after_cnn]
        print(f"\nHidden states (after mask extraction): {hidden_states.shape}")
        np.save(f"{output_dir}/hidden_states_input.npy", hidden_states.numpy())

        # ==========================================
        # Stage 5: Build cu_seqlens (same as Python forward)
        # ==========================================
        cu_chunk_lens = [0]
        window_aftercnn = padded_mask_after_cnn.shape[-1] * (audio_tower.n_window_infer // (audio_tower.n_window * 2))
        print(f"\nwindow_aftercnn: {window_aftercnn}")
        for cnn_len in [aftercnn_lens.item()]:
            cu_chunk_lens += [window_aftercnn] * (cnn_len // window_aftercnn)
            rem = cnn_len % window_aftercnn
            if rem != 0:
                cu_chunk_lens += [rem]
        cu_seqlens = torch.tensor(cu_chunk_lens).cumsum(-1, dtype=torch.int32)
        print(f"cu_seqlens: {cu_seqlens.tolist()}")

        # Build attention mask for eager attention (for comparison)
        seq_length = hidden_states.shape[0]
        attention_mask = torch.full(
            [1, 1, seq_length, seq_length],
            torch.finfo(hidden_states.dtype).min,
            dtype=hidden_states.dtype,
        )
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i], cu_seqlens[i-1]:cu_seqlens[i]] = 0
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Attention mask zero blocks: {(attention_mask == 0).sum().item()} out of {seq_length*seq_length}")

        # Save the attention mask
        np.save(f"{output_dir}/attention_mask_2d.npy", attention_mask[0, 0].numpy())

        # ==========================================
        # Stage 6: First transformer layer
        # ==========================================
        layer0 = audio_tower.layers[0]

        # Self-attention block
        residual = hidden_states
        hs = layer0.self_attn_layer_norm(hidden_states)
        print(f"\nAfter layer0 self_attn_layer_norm: {hs.shape}")
        np.save(f"{output_dir}/after_layer0_norm.npy", hs.numpy())

        # Run full attention
        hs_attn = layer0.self_attn(
            hidden_states=hs,
            cu_seqlens=cu_seqlens,
            attention_mask=None,  # eager mode: no mask
        )
        print(f"After layer0 self_attn: {hs_attn.shape}")
        np.save(f"{output_dir}/after_layer0_attn.npy", hs_attn.numpy())

        hs = residual + hs_attn
        print(f"After layer0 residual1: {hs.shape}")

        # FFN block
        residual = hs
        hs = layer0.final_layer_norm(hs)
        hs = layer0.fc1(hs)
        hs = layer0.activation_fn(hs)
        hs = layer0.fc2(hs)
        hs = residual + hs
        print(f"After layer0 (full): {hs.shape}")
        np.save(f"{output_dir}/after_layer0_full.npy", hs.numpy())

        # ==========================================
        # Stage 7: Run ALL transformer layers
        # ==========================================
        hs_all = hidden_states
        for il, layer in enumerate(audio_tower.layers):
            layer_out = layer(hs_all, cu_seqlens)
            hs_all = layer_out[0]

        print(f"\nAfter all {len(audio_tower.layers)} layers: {hs_all.shape}")

        # Output MLP
        hs_all = audio_tower.ln_post(hs_all)
        hs_all = audio_tower.proj1(hs_all)
        hs_all = audio_tower.act(hs_all)
        hs_all = audio_tower.proj2(hs_all)
        print(f"After output MLP: {hs_all.shape}")
        np.save(f"{output_dir}/encoder_output_verify.npy", hs_all.numpy())

        # Compare with saved reference
        ref = np.load(f"{output_dir}/audio_features.npy")
        print(f"\nReference audio features shape: {ref.shape}")
        diff = np.abs(hs_all.numpy() - ref)
        print(f"Max diff from reference: {diff.max():.6f}")
        print(f"Mean diff from reference: {diff.mean():.6f}")

        corr = np.corrcoef(hs_all.numpy().flatten(), ref.flatten())[0, 1]
        print(f"Correlation with reference: {corr:.6f}")

    print(f"\nAll verification data saved to {output_dir}/")


if __name__ == "__main__":
    main()
