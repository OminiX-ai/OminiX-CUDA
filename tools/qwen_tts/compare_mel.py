#!/usr/bin/env python3
"""Compare mel spectrogram between C++ and Python."""
import numpy as np
import struct

# Load audio the same way as C++
import soundfile as sf
audio, sr = sf.read("/root/autodl-tmp/tts.cpp/tools/qwen_tts/data/test_ref.wav")
if audio.ndim > 1:
    audio = audio[:, 0]
audio = audio.astype(np.float32)
print(f"Audio: {len(audio)} samples, sr={sr}")
print(f"Audio stats: min={audio.min():.6f}, max={audio.max():.6f}, rms={np.sqrt(np.mean(audio**2)):.6f}")

# Compute mel spectrogram matching Python model code
import torch

def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)

def mel_spectrogram_py(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax):
    from librosa.filters import mel as librosa_mel_fn
    mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    mel_basis = torch.from_numpy(mel).float()
    hann_window = torch.hann_window(win_size)

    padding = (n_fft - hop_size) // 2
    y = torch.nn.functional.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                       window=hann_window, center=False, pad_mode="reflect",
                       normalized=False, onesided=True, return_complex=True)
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)
    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = dynamic_range_compression_torch(mel_spec)
    return mel_spec

y = torch.from_numpy(audio).unsqueeze(0)
mel = mel_spectrogram_py(y, n_fft=1024, num_mels=128, sampling_rate=24000,
                          hop_size=256, win_size=1024, fmin=0, fmax=12000)
mel = mel.squeeze(0)  # (128, T)
print(f"\nPython mel: shape={mel.shape}")
print(f"  min={mel.min().item():.6f}, max={mel.max().item():.6f}")
print(f"  mean={mel.mean().item():.6f}, l2={torch.norm(mel).item():.4f}")
print(f"  n_frames={mel.shape[1]}")

# Save for C++ comparison
mel_np = mel.numpy()
np.save("/root/autodl-tmp/tts.cpp/logs/py_mel_spec.npy", mel_np)

# Print first few values for direct comparison
print(f"\nFirst 5 values of mel[0,:5]: {mel_np[0, :5].tolist()}")
print(f"First 5 values of mel[:,0]: {mel_np[:5, 0].tolist()}")

# Also compute the mel filterbank to compare
from librosa.filters import mel as librosa_mel_fn
fb = librosa_mel_fn(sr=24000, n_fft=1024, n_mels=128, fmin=0, fmax=12000)
print(f"\nMel filterbank shape: {fb.shape}")
print(f"  First filter (mel 0) nonzero range: indices {np.nonzero(fb[0])[0].tolist()[:5]}...")
print(f"  First filter max: {fb[0].max():.6f}")
np.save("/root/autodl-tmp/tts.cpp/logs/py_mel_filterbank.npy", fb)
