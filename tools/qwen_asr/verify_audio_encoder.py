"""
Extract intermediate results from Qwen3-ASR for C++ verification.

Outputs:
- mel spectrogram
- audio encoder output features
- input token IDs
- expected text output
"""

import sys
import os
import json
import numpy as np
import torch
import librosa

from qwen_asr.core.transformers_backend import (
    Qwen3ASRConfig,
    Qwen3ASRForConditionalGeneration,
    Qwen3ASRProcessor,
)
from transformers import AutoConfig, AutoModel, AutoProcessor

AutoConfig.register("qwen3_asr", Qwen3ASRConfig)
AutoModel.register(Qwen3ASRConfig, Qwen3ASRForConditionalGeneration)
AutoProcessor.register(Qwen3ASRConfig, Qwen3ASRProcessor)


def main():
    model_path = "Qwen/Qwen3-ASR-1.7B"
    audio_path = "ellen_ref.wav"
    output_dir = "tools/qwen_asr/verify_data"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading model from {model_path}...")
    model = AutoModel.from_pretrained(
        model_path, device_map="cpu", dtype=torch.float32,
    )
    processor = AutoProcessor.from_pretrained(model_path, fix_mistral_regex=True)
    model.eval()

    # Build messages and process
    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_path}]},
    ]
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    print(f"Text prompt: {text_prompt!r}")

    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    print(f"Audio: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f}s")

    # Process through processor
    inputs = processor(text=text_prompt, audio=[audio], return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"]
    input_features = inputs["input_features"]
    feature_attention_mask = inputs["feature_attention_mask"]

    print(f"input_ids shape: {input_ids.shape}")
    print(f"input_ids: {input_ids[0].tolist()}")
    print(f"input_features shape: {input_features.shape}")
    print(f"feature_attention_mask shape: {feature_attention_mask.shape}")
    print(f"feature_attention_mask sum: {feature_attention_mask.sum()}")

    # Save input_ids
    np.save(f"{output_dir}/input_ids.npy", input_ids[0].numpy())

    # Save mel spectrogram (input_features)
    np.save(f"{output_dir}/input_features.npy", input_features[0].numpy())
    np.save(f"{output_dir}/feature_attention_mask.npy", feature_attention_mask[0].numpy())

    # Extract audio features
    with torch.no_grad():
        audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        feature_lens = audio_feature_lengths
        print(f"feature_lens: {feature_lens}")

        # Run audio encoder
        audio_features_list = []
        for input_feature, feature_len in zip(input_features, feature_lens):
            audio_output = model.thinker.audio_tower(
                input_feature[:, :feature_len],
                feature_lens=feature_len.unsqueeze(0),
            )
            audio_feature = audio_output.last_hidden_state
            audio_features_list.append(audio_feature)
        audio_features = torch.cat(audio_features_list, dim=0)

    print(f"Audio features shape: {audio_features.shape}")
    np.save(f"{output_dir}/audio_features.npy", audio_features.numpy())

    # Save audio encoder intermediate: after conv layers
    with torch.no_grad():
        audio_tower = model.thinker.audio_tower
        feature_len = feature_lens[0]
        inp = input_features[0][:, :feature_len]  # (128, T)

        # Compute chunk info
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import _get_feat_extract_output_lengths
        import torch.nn.functional as F

        aftercnn_lens = _get_feat_extract_output_lengths(feature_len.unsqueeze(0))
        chunk_num = torch.ceil(feature_len / (audio_tower.n_window * 2)).long()
        chunk_lengths = torch.tensor(
            [audio_tower.n_window * 2] * chunk_num.sum(),
            dtype=torch.long,
        )
        tail_chunk_index = F.pad(chunk_num.unsqueeze(0), (1, 0), value=-1).cumsum(0)[0][1:]
        chunk_lengths[tail_chunk_index] = feature_len % (audio_tower.n_window * 2)
        chunk_lengths[chunk_lengths == 0] = audio_tower.n_window * 2

        chunk_list_data = inp.T.split(chunk_lengths.tolist(), dim=0)
        padded_feature = torch.nn.utils.rnn.pad_sequence(chunk_list_data, batch_first=True).transpose(1, 2)
        feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)
        padded_mask_after_cnn = torch.nn.utils.rnn.pad_sequence(
            [torch.ones(length, dtype=torch.bool) for length in feature_lens_after_cnn],
            batch_first=True,
        )
        padded_feature = padded_feature.unsqueeze(1)

        # Run conv layers
        padded_embed = F.gelu(audio_tower.conv2d1(padded_feature))
        padded_embed = F.gelu(audio_tower.conv2d2(padded_embed))
        padded_embed = F.gelu(audio_tower.conv2d3(padded_embed))

        b, c, f, t = padded_embed.size()
        print(f"After conv: batch={b}, channels={c}, freq={f}, time={t}")
        conv_output = padded_embed.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)
        projected = audio_tower.conv_out(conv_output)
        print(f"After conv_out: {projected.shape}")

    # Save the mel input for single-sample verification
    np.save(f"{output_dir}/mel_input.npy", inp.numpy())

    # Run full inference to get expected text
    with torch.no_grad():
        inputs_for_gen = inputs.to(model.device).to(model.dtype)
        text_ids = model.generate(**inputs_for_gen, max_new_tokens=256)
        decoded = processor.batch_decode(
            text_ids.sequences[:, input_ids.shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    print(f"\nExpected output: {decoded[0]}")

    # Save metadata
    metadata = {
        "input_ids": input_ids[0].tolist(),
        "audio_feature_length": audio_features.shape[0],
        "expected_output": decoded[0],
        "mel_shape": list(input_features[0].shape),
        "feature_len": int(feature_lens[0]),
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nVerification data saved to {output_dir}/")
    print(f"Files: input_ids.npy, input_features.npy, audio_features.npy, mel_input.npy, metadata.json")


if __name__ == "__main__":
    main()
