"""
Verify Speech Tokenizer Encoder: run Python Mimi encoder and export reference data.
Exports:
  data/ref_conv_encoder_out.bin - conv encoder output before transformer [512, T] (float32)
  data/ref_encoder_hidden.bin   - encoder hidden states before RVQ [512, T] (float32)
  data/ref_encoder_codes.bin    - codec tokens [16, T] (int32)
  data/ref_encoder_audio.bin    - input audio [N] (float32)

Usage:
    python scripts/verify_encoder.py [--audio_path test.wav] [--model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base]
"""

import sys
import os
import argparse
import numpy as np
import torch

def load_tokenizer(model_path):
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Config,
    )
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
        Qwen3TTSTokenizerV2Model,
    )
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("qwen3_tts_tokenizer_12hz", Qwen3TTSTokenizerV2Config)
    AutoModel.register(Qwen3TTSTokenizerV2Config, Qwen3TTSTokenizerV2Model)

    tokenizer_path = os.path.join(model_path, "speech_tokenizer")
    print(f"Loading speech tokenizer from {tokenizer_path}...")
    tokenizer_model = AutoModel.from_pretrained(
        tokenizer_path, device_map="cpu", dtype=torch.float32,
        trust_remote_code=True,
    )
    tokenizer_model.eval()
    print("Speech tokenizer loaded.")
    return tokenizer_model


def generate_test_audio(duration_sec=2.0, sr=24000):
    """Generate a simple test audio signal."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), dtype=np.float32)
    # Mix of sine waves for a non-trivial signal
    audio = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 880 * t)
    audio += 0.1 * np.random.randn(len(t)).astype(np.float32)
    audio = np.clip(audio, -1.0, 1.0)
    return audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str,
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--audio_path", type=str, default=None,
                        help="Path to audio file (24kHz mono). If not provided, uses synthetic audio.")
    parser.add_argument("--output_dir", type=str, default="data")
    parser.add_argument("--duration", type=float, default=2.0,
                        help="Duration in seconds for synthetic audio")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate audio
    if args.audio_path:
        try:
            import torchaudio
            audio, sr = torchaudio.load(args.audio_path)
            if sr != 24000:
                audio = torchaudio.functional.resample(audio, sr, 24000)
            audio = audio[0].numpy()  # mono
        except ImportError:
            import soundfile as sf
            audio, sr = sf.read(args.audio_path)
            if sr != 24000:
                raise ValueError(f"Audio sample rate {sr} != 24000, please resample")
            audio = audio.astype(np.float32)
    else:
        audio = generate_test_audio(args.duration)

    print(f"Audio: {len(audio)} samples ({len(audio)/24000:.2f}s)")

    # Save audio
    audio.tofile(os.path.join(args.output_dir, "ref_encoder_audio.bin"))
    print(f"Saved ref_encoder_audio.bin: {audio.shape}")

    # Load tokenizer
    tokenizer = load_tokenizer(args.model_path)

    # Run encoder
    with torch.no_grad():
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)  # [1, 1, N]
        print(f"Input tensor: {audio_tensor.shape}")

        encoder = tokenizer.encoder
        mimi = encoder

        # Step 1: Conv encoder
        x = audio_tensor.float()
        for i, layer in enumerate(mimi.encoder.layers):
            x = layer(x)
            if i < 3 or i > len(mimi.encoder.layers) - 3:
                print(f"  encoder.layers.{i}: {x.shape}")
            elif i == 3:
                print(f"  ...")

        print(f"After conv encoder: {x.shape}")  # [1, 512, T]

        conv_out = x.squeeze(0).numpy()  # [512, T]
        conv_out.tofile(os.path.join(args.output_dir, "ref_conv_encoder_out.bin"))
        print(f"Saved ref_conv_encoder_out.bin: {conv_out.shape}")
        print(f"Conv encoder first 10 values: {conv_out.flatten()[:10]}")

        # Step 2: Transformer
        # encoder_transformer expects [batch, seq_len, hidden]
        x_transposed = x.transpose(1, 2)  # [1, T, 512]
        print(f"Transformer input: {x_transposed.shape}")
        tf_out = mimi.encoder_transformer(x_transposed)
        hidden = tf_out.last_hidden_state if hasattr(tf_out, 'last_hidden_state') else tf_out[0]
        print(f"After transformer: {hidden.shape}")  # [1, T, 512]
        hidden = hidden.transpose(1, 2)  # back to [1, 512, T]

        # Step 3: Downsample
        if hasattr(mimi, 'downsample') and mimi.downsample is not None:
            hidden = mimi.downsample(hidden)
            print(f"After downsample: {hidden.shape}")  # [1, 512, T/2]

        # Save hidden states (before RVQ) in [512, T] layout
        hidden_np = hidden.squeeze(0).numpy()  # [512, T]
        hidden_np.tofile(os.path.join(args.output_dir, "ref_encoder_hidden.bin"))
        print(f"Saved ref_encoder_hidden.bin: {hidden_np.shape}")

        # Step 4: RVQ quantize using the full tokenizer encode path
        print("\nRunning full tokenizer.encode()...")
        # tokenizer.encode expects [batch, audio_len] + padding_mask [batch, audio_len]
        audio_for_encode = torch.from_numpy(audio).unsqueeze(0).float()  # [1, N]
        padding_mask = torch.ones_like(audio_for_encode, dtype=torch.bool)  # all valid
        enc_output = tokenizer.encode(audio_for_encode, padding_mask=padding_mask)
        # enc_output is Qwen3TTSTokenizerV2EncoderOutput with .audio_codes: List[LongTensor]
        # Each element has shape (codes_length, num_quantizers) = (T, 16)
        codes_tensor = enc_output.audio_codes[0]  # first batch element: [T, 16]
        print(f"Raw codes shape: {codes_tensor.shape}")
        codes_np = codes_tensor.cpu().numpy().astype(np.int32)

        # Transpose to [16, T] layout (quantizers × time)
        # encode returns (T, 16), we want (16, T)
        if codes_np.ndim == 2 and codes_np.shape[1] == 16 and codes_np.shape[0] != 16:
            codes_np = codes_np.T
        print(f"Codes output shape (should be [16, T]): {codes_np.shape}")

        codes_np.tofile(os.path.join(args.output_dir, "ref_encoder_codes.bin"))
        print(f"Saved ref_encoder_codes.bin: {codes_np.shape}")

        # Print first few codes
        print("\nFirst 5 timesteps, first 4 codebooks:")
        for q in range(min(4, codes_np.shape[0])):
            print(f"  q{q}: {codes_np[q, :5]}")

        print(f"\nTotal frames: {codes_np.shape[1]}")
        print(f"Expected frames: {len(audio) // 1920}")


if __name__ == "__main__":
    main()
