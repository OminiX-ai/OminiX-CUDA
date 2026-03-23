"""Save Python audio features in binary format for C++ testing."""
import numpy as np
import struct

# Load Python audio features
af = np.load('tools/qwen_asr/verify_data/audio_features.npy').astype(np.float32)
print(f"Audio features shape: {af.shape}")  # (122, 2048)

# Save as raw binary: [n_frames (int32), n_dim (int32), data (float32)]
with open('tools/qwen_asr/verify_data/audio_features_python.bin', 'wb') as f:
    f.write(struct.pack('ii', af.shape[0], af.shape[1]))
    f.write(af.tobytes())

print(f"Saved audio_features_python.bin ({af.shape[0]} x {af.shape[1]})")
