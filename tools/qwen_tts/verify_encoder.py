#!/usr/bin/env python3
"""Verify speech encoder output by running Python encoder on same audio."""
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
import soundfile as sf

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
LOG_DIR = "/root/autodl-tmp/tts.cpp/logs"

def main():
    # Load speech tokenizer
    import json

    # Load speech tokenizer config directly
    tokenizer_path = os.path.join(MODEL_PATH, "speech_tokenizer")
    from qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Config
    from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model
    with open(os.path.join(tokenizer_path, "config.json")) as f:
        cfg_dict = json.load(f)
    config = Qwen3TTSTokenizerV2Config(**cfg_dict)
    tokenizer = Qwen3TTSTokenizerV2Model(config)

    # Load weights
    import safetensors.torch
    # Speech tokenizer has its own model.safetensors
    st_path = os.path.join(tokenizer_path, "model.safetensors")
    state_dict = safetensors.torch.load_file(st_path)
    missing, unexpected = tokenizer.load_state_dict(state_dict, strict=False)
    print(f"Loaded {len(state_dict)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing keys (first 5): {missing[:5]}")
    tokenizer.eval().float()

    # Load audio
    audio, sr = sf.read("/root/autodl-tmp/tts.cpp/tools/qwen_tts/data/test_ref.wav")
    if audio.ndim > 1: audio = audio[:, 0]
    audio = audio.astype(np.float32)
    print(f"Audio: {len(audio)} samples, sr={sr}")

    # Encode
    with torch.no_grad():
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).float()
        codes = tokenizer.encode(audio_tensor)
        # codes shape: (n_quantizers, batch, T) or similar

    if isinstance(codes, torch.Tensor):
        codes_np = codes.cpu().numpy()
    elif isinstance(codes, list):
        codes_np = np.array([c.cpu().numpy() if isinstance(c, torch.Tensor) else c for c in codes])
    print(f"Codes shape: {codes_np.shape}")

    # Reshape to (T, groups)
    if codes_np.ndim == 3:
        codes_np = codes_np[:, 0, :]  # (Q, T)
    codes_np = codes_np.T  # (T, Q)
    print(f"Reshaped to: {codes_np.shape}")

    print(f"\nFrame 0 all groups: {codes_np[0].tolist()}")
    print(f"Frame 1 all groups: {codes_np[1].tolist()}")

    print(f"\nGroup 0 all frames: {codes_np[:, 0].tolist()}")

    # Compare with previously saved
    try:
        saved = np.load(os.path.join(LOG_DIR, "py_ref_codes.npy"))
        print(f"\nPreviously saved ref_codes: {saved.shape}")
        match = (codes_np == saved).all()
        print(f"Exact match with saved: {match}")
        if not match:
            for g in range(min(16, codes_np.shape[1])):
                g_match = (codes_np[:, g] == saved[:, g]).sum()
                print(f"  Group {g}: {g_match}/{codes_np.shape[0]} frames match")
    except:
        pass

    # Save
    np.save(os.path.join(LOG_DIR, "py_ref_codes_direct.npy"), codes_np)
    print(f"\nSaved to {LOG_DIR}/py_ref_codes_direct.npy")

if __name__ == "__main__":
    main()
