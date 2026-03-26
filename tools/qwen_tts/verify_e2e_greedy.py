#!/usr/bin/env python3
"""
End-to-end greedy generation using the Qwen3-TTS Python reference implementation.

Generates speech tokens with deterministic (greedy) decoding so we can compare
token-by-token against the C++ pipeline.

Strategy: Load full model on CPU for speech tokenizer operations,
then move the talker (transformer) to NPU for fast generation.

Outputs:
  - First 20 group-0 codec tokens
  - Generated codec token shape (all groups)
  - Audio saved to logs/python_ref_output.wav
"""

import sys
import os
import types
import importlib

# --- Environment setup ---
os.chdir("/root/autodl-tmp")

# Mock torchaudio to avoid .local torchaudio binary incompatibility
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

# --- Imports ---
import torch
import torch_npu
import numpy as np
import soundfile as sf

# --- Constants ---
MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_AUDIO = "/root/autodl-tmp/tts.cpp/tools/qwen_tts/data/test_ref.wav"
REF_TEXT = "Hello, this is a test."
TARGET_TEXT = "How are you today?"
LANGUAGE = "English"
OUTPUT_DIR = "/root/autodl-tmp/tts.cpp/logs"

NPU_DEVICE = "npu:0" if torch.npu.is_available() else "cpu"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"NPU available: {torch.npu.is_available()}, using: {NPU_DEVICE}")

    # --- Load model on CPU first ---
    print("Loading Qwen3TTSModel on CPU...")
    from qwen_tts import Qwen3TTSModel

    model = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        device_map="cpu",
        dtype=torch.float32,
    )
    print(f"Model loaded on device: {model.device}")

    # --- Step 1: Create voice clone prompt (CPU - uses speech tokenizer) ---
    print("\n=== Step 1: Creating voice clone prompt (CPU) ===")
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=REF_AUDIO,
        ref_text=REF_TEXT,
    )
    voice_clone_prompt_dict = model._prompt_items_to_voice_clone_prompt(prompt_items)

    ref_code = voice_clone_prompt_dict["ref_code"][0]
    print(f"ref_code shape: {ref_code.shape}")
    print(f"ref_code first 5 frames (group 0): {ref_code[:5, 0].tolist()}")
    spk_emb = voice_clone_prompt_dict["ref_spk_embedding"][0]
    print(f"spk_embedding shape: {spk_emb.shape}")

    # Save ref codes for comparison
    np.save(os.path.join(OUTPUT_DIR, "py_ref_codes.npy"), ref_code.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, "py_spk_embedding.npy"), spk_emb.cpu().float().numpy())

    # --- Step 2: Move talker to NPU for fast generation ---
    print(f"\n=== Step 2: Moving talker to {NPU_DEVICE} ===")
    model.model.talker = model.model.talker.to(NPU_DEVICE)
    # Update device tracking
    model.device = torch.device(NPU_DEVICE)
    print(f"Talker moved to {NPU_DEVICE}")

    # Move voice clone prompt tensors to NPU too
    voice_clone_prompt_dict["ref_code"] = [rc.to(NPU_DEVICE) if rc is not None else None
                                            for rc in voice_clone_prompt_dict["ref_code"]]
    voice_clone_prompt_dict["ref_spk_embedding"] = [se.to(NPU_DEVICE)
                                                     for se in voice_clone_prompt_dict["ref_spk_embedding"]]

    # --- Step 3: Prepare inputs ---
    print("\n=== Step 3: Preparing inputs ===")
    input_text = model._build_assistant_text(TARGET_TEXT)
    input_ids = model._tokenize_texts([input_text])
    # Move input IDs to NPU
    input_ids = [ids.to(NPU_DEVICE) for ids in input_ids]
    print(f"Input text: {repr(input_text)}")
    print(f"Input IDs shape: {input_ids[0].shape}")
    print(f"Input IDs: {input_ids[0][0].tolist()}")

    ref_ids = []
    for item in prompt_items:
        if item.ref_text:
            ref_tok = model._tokenize_texts([model._build_ref_text(item.ref_text)])[0]
            ref_ids.append(ref_tok.to(NPU_DEVICE))
        else:
            ref_ids.append(None)
    if ref_ids[0] is not None:
        print(f"Ref IDs shape: {ref_ids[0].shape}")
        print(f"Ref IDs: {ref_ids[0][0].tolist()}")

    # --- Step 4: Generate codec tokens (greedy) on NPU ---
    print("\n=== Step 4: Generating codec tokens (greedy) ===")
    print(f"  ref_audio: {REF_AUDIO}")
    print(f"  ref_text:  {REF_TEXT}")
    print(f"  target_text: {TARGET_TEXT}")
    print(f"  language: {LANGUAGE}")
    print(f"  do_sample: False (greedy)")
    print(f"  subtalker_dosample: False (greedy)")
    print(f"  repetition_penalty: 1.0 (disabled)")

    print("\nRunning model.generate()...")
    talker_codes_list, talker_hidden_states_list = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=voice_clone_prompt_dict,
        languages=[LANGUAGE],
        non_streaming_mode=False,
        max_new_tokens=2048,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
        repetition_penalty=1.0,
        subtalker_dosample=False,
        subtalker_top_k=1,
        subtalker_top_p=1.0,
        subtalker_temperature=1.0,
    )

    codes = talker_codes_list[0]  # shape: (T, num_code_groups)

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"=== Codec Token Results ===")
    print(f"{'='*60}")
    print(f"Generated codec token shape: {codes.shape}")
    print(f"  (T={codes.shape[0]} timesteps, G={codes.shape[1]} groups)")

    g0_tokens = codes[:, 0].cpu().tolist()
    n_show = min(20, len(g0_tokens))
    print(f"\nFirst {n_show} group-0 codec tokens:")
    print(f"  {g0_tokens[:n_show]}")

    n_detail = min(10, codes.shape[0])
    print(f"\nAll groups for first {n_detail} timesteps:")
    for t in range(n_detail):
        row = codes[t].cpu().tolist()
        print(f"  t={t:3d}: g0={row[0]:4d} | g1-15={row[1:]}")

    # --- Save codec tokens ---
    codes_np = codes.cpu().numpy()
    tokens_path = os.path.join(OUTPUT_DIR, "python_ref_codec_tokens.npy")
    np.save(tokens_path, codes_np)
    print(f"\nSaved codec tokens to: {tokens_path}")

    # --- Step 5: Decode to audio (move codes to CPU for speech tokenizer) ---
    print("\n=== Step 5: Decoding to audio (CPU) ===")
    codes_cpu = codes.cpu()
    ref_code_cpu = ref_code.cpu() if ref_code is not None else None

    if ref_code_cpu is not None:
        codes_for_decode = torch.cat([ref_code_cpu, codes_cpu], dim=0)
    else:
        codes_for_decode = codes_cpu

    wavs_all, fs = model.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])
    wav = wavs_all[0]

    if ref_code_cpu is not None:
        ref_len = int(ref_code_cpu.shape[0])
        total_len = int(codes_for_decode.shape[0])
        cut = int(ref_len / max(total_len, 1) * wav.shape[0])
        wav = wav[cut:]

    output_path = os.path.join(OUTPUT_DIR, "python_ref_output.wav")
    sf.write(output_path, wav, fs)
    print(f"Saved audio: {output_path}")
    print(f"  Sample rate: {fs}")
    print(f"  Audio length: {len(wav)} samples ({len(wav)/fs:.2f} sec)")

    print(f"\n{'='*60}")
    print("=== Done ===")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
