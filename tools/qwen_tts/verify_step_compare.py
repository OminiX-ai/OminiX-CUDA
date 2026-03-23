#!/usr/bin/env python3
"""
Step-by-step comparison of Python Talker generation vs C++.
Dumps hidden states, logits, and tokens for the first N steps.
Uses same inputs as C++ debug run: greedy, rep_penalty=1.0.
"""

import sys, os, types
import importlib
import importlib.machinery

# --- Environment setup (same as verify_e2e_greedy.py) ---
os.chdir("/root/autodl-tmp")

# Remove .local from path to avoid binary incompatibility
new_path = [p for p in sys.path if '.local' not in p]
sys.path = new_path

# Mock torchaudio to avoid .local torchaudio binary incompatibility
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

# Re-add .local for non-binary packages
sys.path.insert(0, '/home/claude-temp/.local/lib/python3.10/site-packages')

import torch
import torch_npu
import numpy as np
import soundfile as sf

MODEL_PATH = "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
REF_AUDIO = "/root/autodl-tmp/tts.cpp/tools/qwen_tts/data/test_ref.wav"
REF_TEXT = "Hello, this is a test."
TARGET_TEXT = "How are you today?"
LANGUAGE = "English"
OUTPUT_DIR = "/root/autodl-tmp/tts.cpp/logs"
N_DEBUG_STEPS = 5

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "npu:0" if torch.npu.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    from qwen_tts import Qwen3TTSModel
    model = Qwen3TTSModel.from_pretrained(MODEL_PATH, device_map="cpu", dtype=torch.float32)
    print(f"Model loaded")

    # Create voice clone prompt (CPU for speech tokenizer)
    print("\n=== Creating voice clone prompt ===")
    prompt_items = model.create_voice_clone_prompt(ref_audio=REF_AUDIO, ref_text=REF_TEXT)
    vc_prompt = model._prompt_items_to_voice_clone_prompt(prompt_items)

    ref_code = vc_prompt["ref_code"][0]
    spk_emb = vc_prompt["ref_spk_embedding"][0]
    print(f"ref_code shape: {ref_code.shape}")
    print(f"spk_embedding shape: {spk_emb.shape}")

    # Save for comparison
    np.save(os.path.join(OUTPUT_DIR, "py_ref_codes.npy"), ref_code.cpu().numpy())
    np.save(os.path.join(OUTPUT_DIR, "py_spk_embedding.npy"), spk_emb.cpu().float().numpy())

    # Move talker to NPU
    print(f"\nMoving talker to {device}...")
    model.model.talker = model.model.talker.to(device)
    model.device = torch.device(device)
    vc_prompt["ref_code"] = [rc.to(device) if rc is not None else None for rc in vc_prompt["ref_code"]]
    vc_prompt["ref_spk_embedding"] = [se.to(device) for se in vc_prompt["ref_spk_embedding"]]

    # Prepare inputs
    input_text = model._build_assistant_text(TARGET_TEXT)
    input_ids = model._tokenize_texts([input_text])
    input_ids = [ids.to(device) for ids in input_ids]

    ref_ids = []
    for item in prompt_items:
        if item.ref_text:
            ref_tok = model._tokenize_texts([model._build_ref_text(item.ref_text)])[0]
            ref_ids.append(ref_tok.to(device))
        else:
            ref_ids.append(None)

    print(f"Input IDs: {input_ids[0][0].tolist()}")
    if ref_ids[0] is not None:
        print(f"Ref IDs: {ref_ids[0][0].tolist()}")

    # ============================================================
    # Step-by-step generation with hooks to dump intermediate values
    # ============================================================
    print(f"\n=== Step-by-step greedy generation ===")

    talker = model.model.talker

    # Hook into the generate method to capture intermediate states
    # We need to access the talker's internal generation
    # Let's run the full generate and capture codes
    talker_codes_list, talker_hidden_states_list = model.model.generate(
        input_ids=input_ids,
        ref_ids=ref_ids,
        voice_clone_prompt=vc_prompt,
        languages=[LANGUAGE],
        non_streaming_mode=False,
        max_new_tokens=400,
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

    codes = talker_codes_list[0]  # (T, 16)
    codes_np = codes.cpu().numpy()

    print(f"\nGenerated {codes_np.shape[0]} frames")
    print(f"First 10 group-0 tokens: {codes_np[:10, 0].tolist()}")
    print(f"\nAll groups for first 5 timesteps:")
    for t in range(min(5, codes_np.shape[0])):
        print(f"  t={t:3d}: g0={codes_np[t,0]:4d} | g1-15={codes_np[t,1:].tolist()}")

    # Save
    np.save(os.path.join(OUTPUT_DIR, "python_ref_codec_tokens.npy"), codes_np)
    print(f"\nSaved codec tokens: {codes_np.shape}")

    # Decode to audio
    print("\n=== Decoding to audio ===")
    codes_cpu = codes.cpu()
    ref_code_cpu = ref_code.cpu()
    codes_for_decode = torch.cat([ref_code_cpu, codes_cpu], dim=0)
    wavs_all, fs = model.model.speech_tokenizer.decode([{"audio_codes": codes_for_decode}])
    wav = wavs_all[0]
    ref_len = int(ref_code_cpu.shape[0])
    total_len = int(codes_for_decode.shape[0])
    cut = int(ref_len / max(total_len, 1) * wav.shape[0])
    wav = wav[cut:]
    output_path = os.path.join(OUTPUT_DIR, "python_ref_output.wav")
    sf.write(output_path, wav, fs)
    print(f"Saved audio: {output_path} ({len(wav)} samples, {len(wav)/fs:.2f} sec)")

    # ============================================================
    # Now do a manual step-by-step to get hidden states and logits
    # ============================================================
    print(f"\n=== Manual step-by-step for debugging ===")

    # Access talker internals
    talker_model = talker
    # Build prefill embeddings manually to compare
    # This requires hooking into the model internals
    # For now, just compare the final tokens

    print("\n=== C++ vs Python Comparison ===")
    print("Python first 10 group-0 tokens:", codes_np[:10, 0].tolist())
    print("C++ first 3 group-0 tokens: [302, 1929, 780]  (from debug log)")
    print()
    print("Python groups at t=0:", codes_np[0].tolist())
    print("C++ groups at t=0:    [302, 395, 511, 1728, 1513, 1224, 772, 1094, 865, 1181, 44, 955, 438, 654, 569, 690]")

    print("\nDone!")


if __name__ == "__main__":
    main()
