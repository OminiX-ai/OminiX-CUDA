"""Generate reference data for Speech Tokenizer Decoder verification.

Runs the Python decoder on a small set of random codec tokens and saves:
- data/ref_decoder_codes.bin: input codes [16, T] as int32
- data/ref_decoder_audio.bin: output audio as float32
- data/ref_decoder_rvq_output.bin: intermediate RVQ output [512, T] as float32
"""
import sys
import os
import numpy as np
import torch

# Add project root for gguf-py
from pathlib import Path
gguf_py_path = Path(__file__).parent.parent.parent.parent / 'gguf-py'
if gguf_py_path.exists():
    sys.path.insert(1, str(gguf_py_path))

def main():
    model_path = os.environ.get(
        "QWEN_TTS_MODEL_PATH",
        "/root/autodl-tmp/weights/Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    )

    # Load speech tokenizer
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
    tokenizer = AutoModel.from_pretrained(
        tokenizer_path, device_map="cpu", dtype=torch.float32,
        trust_remote_code=True,
    )
    tokenizer.eval()
    print("Loaded.")

    # Generate random codec tokens: [1, 16, T]
    T = 20  # small T for testing
    n_q = 16
    codebook_size = 2048
    np.random.seed(42)
    codes_np = np.random.randint(0, codebook_size, size=(1, n_q, T), dtype=np.int64)
    codes = torch.from_numpy(codes_np)
    print(f"Input codes shape: {codes.shape}, range: [{codes.min()}, {codes.max()}]")

    # Run decoder
    decoder = tokenizer.decoder
    decoder.eval()

    with torch.no_grad():
        # Step 1: RVQ decode
        quantizer = decoder.quantizer
        # Decode rvq_first (semantic, codes[:, 0:1, :])
        first_codes = codes[:, 0:1, :]
        rest_codes = codes[:, 1:, :]

        # Manual RVQ decode to get intermediate
        def decode_rvq(rvq, input_codes):
            """Decode RVQ codes to embeddings."""
            quantized = None
            for i, layer_codes in enumerate(input_codes.transpose(0, 1)):
                # layer_codes: [batch, T]
                cb = rvq.vq.layers[i]._codebook
                usage = cb.cluster_usage.clamp(min=1e-5)
                embedding = cb.embedding_sum / usage[:, None]
                # embedding: [codebook_size, codebook_dim]
                q = torch.nn.functional.embedding(layer_codes, embedding)
                # q: [batch, T, codebook_dim]
                if quantized is None:
                    quantized = q
                else:
                    quantized = quantized + q
            return quantized  # [batch, T, codebook_dim]

        first_q = decode_rvq(quantizer.rvq_first, first_codes)
        rest_q = decode_rvq(quantizer.rvq_rest, rest_codes)

        # Apply output projections
        first_out = quantizer.rvq_first.output_proj(first_q.transpose(1, 2))
        rest_out = quantizer.rvq_rest.output_proj(rest_q.transpose(1, 2))
        rvq_output = first_out + rest_out  # [batch, 512, T]
        print(f"RVQ output shape: {rvq_output.shape}")
        print(f"RVQ output stats: mean={rvq_output.mean():.6f}, std={rvq_output.std():.6f}")

        # Full decode via forward()
        audio = decoder(codes)
        if isinstance(audio, tuple):
            audio = audio[0]
        print(f"Audio output shape: {audio.shape}")
        print(f"Audio stats: mean={audio.mean():.6f}, std={audio.std():.6f}, "
              f"min={audio.min():.6f}, max={audio.max():.6f}")

    # Save reference data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Save codes as int32 [n_q * T] flattened (q0_t0..q0_tT, q1_t0..q1_tT, ...)
    codes_flat = codes_np[0].astype(np.int32)  # [16, T]
    codes_flat.tofile(os.path.join(data_dir, "ref_decoder_codes.bin"))
    print(f"Saved codes: {codes_flat.shape} → ref_decoder_codes.bin")

    # Save RVQ output
    rvq_np = rvq_output[0].numpy().astype(np.float32)  # [512, T]
    rvq_np.tofile(os.path.join(data_dir, "ref_decoder_rvq_output.bin"))
    print(f"Saved RVQ output: {rvq_np.shape} → ref_decoder_rvq_output.bin")

    # Save audio
    audio_np = audio[0, 0].numpy().astype(np.float32)  # [samples]
    audio_np.tofile(os.path.join(data_dir, "ref_decoder_audio.bin"))
    print(f"Saved audio: {audio_np.shape} ({len(audio_np)} samples) → ref_decoder_audio.bin")

    print("\nDone! Reference data saved to data/")


if __name__ == "__main__":
    main()
