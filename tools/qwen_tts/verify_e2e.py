#!/usr/bin/env python3
"""Compare C++ TTS output with Python reference implementation."""

import sys
import os
import numpy as np

# Must run from non-tts.cpp dir to avoid torch path issue
os.chdir("/root/autodl-tmp")

import torch
import torch_npu
import soundfile as sf

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_AUDIO = "/root/autodl-tmp/tts.cpp/tools/qwen_tts/data/test_ref.wav"
REF_TEXT = "Hello world"
TARGET_TEXT = "This is a test."
LANGUAGE = "english"
OUTPUT_DIR = "/root/autodl-tmp/tts.cpp/logs"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading model...")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel(MODEL_PATH)
    print("Model loaded on", model.model.device)

    # Prepare voice clone prompt
    print("\nPreparing voice clone prompt...")
    ref_audio, sr = sf.read(REF_AUDIO)
    if ref_audio.ndim > 1:
        ref_audio = ref_audio[:, 0]
    ref_audio = ref_audio.astype(np.float32)

    voice_clone_prompt = model.prepare_voice_clone_prompt(
        ref_audio_list=[(ref_audio, sr)],
        ref_text_list=[REF_TEXT],
    )

    # Save ref_code for comparison
    ref_code = voice_clone_prompt["ref_code"][0]
    print(f"ref_code shape: {ref_code.shape}")
    ref_code_np = ref_code.cpu().numpy()
    np.save(os.path.join(OUTPUT_DIR, "py_ref_codes.npy"), ref_code_np)
    print(f"Saved ref codes: {ref_code_np.shape}")

    # Save speaker embedding
    spk_emb = voice_clone_prompt["ref_spk_embedding"][0]
    spk_np = spk_emb.cpu().float().numpy()
    np.save(os.path.join(OUTPUT_DIR, "py_spk_embedding.npy"), spk_np)
    print(f"Saved speaker embedding: {spk_np.shape}")

    # Generate
    print("\nGenerating with Python model...")
    print(f"  ref_text: {REF_TEXT}")
    print(f"  target_text: {TARGET_TEXT}")
    print(f"  language: {LANGUAGE}")

    results = model.synthesize(
        text=TARGET_TEXT,
        voice_clone_prompt=voice_clone_prompt,
        language=LANGUAGE,
        max_new_tokens=100,
        do_sample=False,  # greedy for deterministic comparison
        temperature=1.0,
    )

    # Extract results
    audio = results["audio"]
    codec_tokens = results.get("codes", None)

    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().float().numpy()
    if isinstance(audio, list):
        audio = np.array(audio)
    audio = audio.flatten()

    # Save audio
    sf.write(os.path.join(OUTPUT_DIR, "py_output.wav"), audio, 24000)
    print(f"Saved Python output: {len(audio)} samples ({len(audio)/24000:.2f} sec)")

    # Save codec tokens if available
    if codec_tokens is not None:
        if isinstance(codec_tokens, torch.Tensor):
            codec_np = codec_tokens.cpu().numpy()
        else:
            codec_np = np.array(codec_tokens)
        np.save(os.path.join(OUTPUT_DIR, "py_codec_tokens.npy"), codec_np)
        print(f"Saved codec tokens: {codec_np.shape}")

    print("\nDone! Files saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
