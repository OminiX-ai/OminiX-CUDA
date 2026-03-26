#!/usr/bin/env python3
"""
Run speaker encoder on the same mel spectrogram to compare with C++.
Load only the speaker encoder (not the full model).
"""
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
import numpy as np

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LOG_DIR = "/root/autodl-tmp/tts.cpp/logs"

def main():
    # Load just the speaker encoder config and model
    from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
    from qwen_tts.core.models.modeling_qwen3_tts import Qwen3TTSSpeakerEncoder, mel_spectrogram

    import json
    with open(os.path.join(MODEL_PATH, "config.json")) as f:
        config_dict = json.load(f)
    config = Qwen3TTSConfig(**config_dict)

    # Load speaker encoder
    spk_enc = Qwen3TTSSpeakerEncoder(config.speaker_encoder_config)

    # Load weights
    import safetensors.torch
    model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.safetensors')]
    state_dict = {}
    for mf in model_files:
        sd = safetensors.torch.load_file(os.path.join(MODEL_PATH, mf))
        for k, v in sd.items():
            if k.startswith('speaker_encoder.'):
                new_k = k[len('speaker_encoder.'):]
                state_dict[new_k] = v
    print(f"Loaded {len(state_dict)} speaker encoder parameters")
    missing, unexpected = spk_enc.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing: {missing}")
    if unexpected:
        print(f"Unexpected: {unexpected}")

    spk_enc.eval()
    spk_enc = spk_enc.float()

    # Load the reference audio and compute mel spectrogram
    import soundfile as sf
    audio, sr = sf.read("/root/autodl-tmp/tts.cpp/tools/qwen_tts/data/test_ref.wav")
    if audio.ndim > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32)
    print(f"Audio: {len(audio)} samples, sr={sr}")

    y = torch.from_numpy(audio).unsqueeze(0).float()
    mel = mel_spectrogram(y, n_fft=1024, num_mels=128, sampling_rate=24000,
                           hop_size=256, win_size=1024, fmin=0, fmax=12000)
    print(f"Mel shape: {mel.shape}")  # (1, 128, T)

    # Transpose to (1, T, 128) as expected by speaker encoder
    mel_input = mel.transpose(1, 2)
    print(f"Mel input shape: {mel_input.shape}")

    # Run speaker encoder
    with torch.no_grad():
        embedding = spk_enc(mel_input)[0]  # (2048,)

    print(f"\nSpeaker embedding:")
    print(f"  shape: {embedding.shape}")
    print(f"  l2: {torch.norm(embedding).item():.4f}")
    print(f"  min: {embedding.min().item():.6f}")
    print(f"  max: {embedding.max().item():.6f}")
    print(f"  first5: {embedding[:5].tolist()}")

    # Save
    np.save(os.path.join(LOG_DIR, "py_spk_embedding_direct.npy"), embedding.numpy())

    # Also check: what does the saved py_spk_embedding look like?
    saved = np.load(os.path.join(LOG_DIR, "py_spk_embedding.npy"))
    print(f"\nPreviously saved spk_embedding:")
    print(f"  l2: {np.linalg.norm(saved):.4f}")
    print(f"  first5: {saved[:5].tolist()}")

    # Compare
    diff = np.abs(embedding.numpy() - saved)
    print(f"  max_diff from saved: {diff.max():.6f}")

if __name__ == "__main__":
    main()
