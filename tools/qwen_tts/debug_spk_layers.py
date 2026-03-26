#!/usr/bin/env python3
"""Debug speaker encoder layer by layer."""
import sys, os, types, importlib, importlib.machinery
os.chdir("/root/autodl-tmp")
new_path = [p for p in sys.path if '.local' not in p]
sys.path = new_path
torchaudio = types.ModuleType('torchaudio')
torchaudio.__spec__ = importlib.machinery.ModuleSpec('torchaudio', None)
torchaudio.__version__ = '0.0.0'
torchaudio.compliance = types.ModuleType('torchaudio.compliance')
torchaudio.compliance.__spec__ = importlib.machinery.ModuleSpec('torchaudio.compliance', None)
torchaudio.compliance.kaldi = types.ModuleType('torchaudio.compliance.kaldi')
torchaudio.compliance.kaldi.__spec__ = importlib.machinery.ModuleSpec('torchaudio.compliance.kaldi', None)
torchaudio.sox_effects = types.ModuleType('torchaudio.sox_effects')
torchaudio.sox_effects.__spec__ = importlib.machinery.ModuleSpec('torchaudio.sox_effects', None)
sys.modules['torchaudio'] = torchaudio
sys.modules['torchaudio.compliance'] = torchaudio.compliance
sys.modules['torchaudio.compliance.kaldi'] = torchaudio.compliance.kaldi
sys.modules['torchaudio.sox_effects'] = torchaudio.sox_effects
sys.path.insert(0, '/home/claude-temp/.local/lib/python3.10/site-packages')

import torch
import torch.nn.functional as F
import numpy as np

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"

def stats(name, x):
    x_flat = x.flatten().float()
    print(f"  {name}: shape={list(x.shape)}, min={x_flat.min():.6f}, max={x_flat.max():.6f}, "
          f"mean={x_flat.mean():.6f}, l2={torch.norm(x_flat):.4f}")

def main():
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram
    import json, safetensors.torch

    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config = Qwen3TTSConfig(**json.load(f))

    spk_enc = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)
    state_dict = {}
    for mf in os.listdir(MODEL_PATH):
        if mf.endswith('.safetensors'):
            sd = safetensors.torch.load_file(os.path.join(MODEL_PATH, mf))
            for k, v in sd.items():
                if k.startswith('speaker_encoder.'):
                    state_dict[k[len('speaker_encoder.'):]] = v
    spk_enc.load_state_dict(state_dict, strict=False)
    spk_enc.eval().float()

    # Compute mel spectrogram
    import soundfile as sf
    audio, sr = sf.read("/root/autodl-tmp/tts.cpp/tools/qwen_tts/data/test_ref.wav")
    if audio.ndim > 1: audio = audio[:, 0]
    y = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    mel = mel_spectrogram(y, n_fft=1024, num_mels=128, sampling_rate=24000,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000)
    print(f"Mel: {mel.shape}")

    # Step through forward manually
    with torch.no_grad():
        # Transpose: (1, 128, 187) → (1, 128, 187) — already (batch, channels, time)
        x = mel  # (1, 128, T)
        stats("input_mel", x)

        # blocks[0]: TDNN (Conv1d(128, 512, 5) + ReLU)
        x = spk_enc.blocks[0](x)
        stats("after_block0_tdnn", x)

        # blocks[1-3]: SE-Res2Net
        block_outs = []
        for i, blk in enumerate(spk_enc.blocks[1:], 1):
            x = blk(x)
            block_outs.append(x)
            stats(f"after_block{i}_res2net", x)

        # MFA: concatenate + Conv1d(1536, 1536, 1) + ReLU
        x = torch.cat(block_outs, dim=1)
        stats("mfa_cat", x)
        x = spk_enc.mfa(x)
        stats("after_mfa", x)

        # ASP
        x = spk_enc.asp(x)
        stats("after_asp", x)

        # FC
        x = spk_enc.fc(x)
        stats("after_fc", x)

        x = x.squeeze(-1)
        stats("final_embedding", x)

if __name__ == "__main__":
    main()
