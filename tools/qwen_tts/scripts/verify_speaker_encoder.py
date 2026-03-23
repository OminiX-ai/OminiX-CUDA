"""
Verify Speaker Encoder: generate reference data and compare with C++ output.

Usage:
    python scripts/verify_speaker_encoder.py --ref_audio ellen_ref.wav
    python scripts/verify_speaker_encoder.py --dump_mel  # dump mel for C++ testing
"""
import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def load_model():
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cpu",
        dtype=torch.float32,  # Use fp32 for reference
    )
    return model


def load_audio(path, sr=24000):
    """Load audio file at target sample rate."""
    import soundfile as sf
    audio, orig_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # mono
    if orig_sr != sr:
        # Simple resample using scipy
        from scipy.signal import resample
        audio = resample(audio, int(len(audio) * sr / orig_sr))
    return audio.astype(np.float32), sr


def test_speaker_embedding(args):
    """Extract speaker embedding and save reference."""
    model = load_model()
    audio, sr = load_audio(args.ref_audio)
    print(f"Audio: {len(audio)} samples, {sr} Hz, {len(audio)/sr:.2f}s")

    # Extract embedding using Python model
    with torch.no_grad():
        spk_emb = model.model.extract_speaker_embedding(audio, sr)

    emb = spk_emb.squeeze().numpy()
    print(f"Speaker embedding shape: {emb.shape}")
    print(f"  mean={emb.mean():.6f}, std={emb.std():.6f}")
    print(f"  min={emb.min():.6f}, max={emb.max():.6f}")
    print(f"  first 10: {emb[:10]}")

    # Save reference
    out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "ref_speaker_embedding.bin")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    emb.tofile(out_path)
    print(f"Saved reference embedding to {out_path}")

    # Also dump mel spectrogram for C++ testing
    if args.dump_mel:
        dump_mel(model, audio, sr)


def dump_mel(model, audio, sr):
    """Dump mel spectrogram for C++ comparison."""
    from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

    audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
    mel = mel_spectrogram(
        audio_tensor, n_fft=1024, num_mels=128,
        sampling_rate=24000, hop_size=256, win_size=1024,
        fmin=0, fmax=12000
    )
    mel_np = mel.squeeze(0).numpy()  # (128, T)
    print(f"Mel spectrogram shape: {mel_np.shape}")
    print(f"  mean={mel_np.mean():.6f}, std={mel_np.std():.6f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "data",
                            "ref_mel_spectrogram.bin")
    mel_np.tofile(out_path)
    print(f"Saved mel spectrogram to {out_path}")

    # Save shape info
    info_path = out_path.replace(".bin", ".txt")
    with open(info_path, "w") as f:
        f.write(f"{mel_np.shape[0]} {mel_np.shape[1]}\n")
    print(f"Saved shape info to {info_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref_audio", type=str, default="ellen_ref.wav")
    parser.add_argument("--dump_mel", action="store_true", default=True)
    args = parser.parse_args()
    test_speaker_embedding(args)


if __name__ == "__main__":
    main()
