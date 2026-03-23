"""
Save Python mel spectrogram in a format that C++ can load directly.
This lets us test the audio encoder and decoder with correct mel input.
"""
import numpy as np
import struct

# Load the Python reference mel
mel = np.load('tools/qwen_asr/verify_data/mel_reference.npy')
# Only use the valid frames (936)
mel = mel[:, :936].astype(np.float32)
print(f"Mel shape: {mel.shape}")

# Save as raw binary: [n_mels (int32), n_frames (int32), data (float32)]
with open('tools/qwen_asr/verify_data/mel_python.bin', 'wb') as f:
    f.write(struct.pack('ii', mel.shape[0], mel.shape[1]))
    f.write(mel.tobytes())

print(f"Saved mel_python.bin ({mel.shape[0]} x {mel.shape[1]})")
