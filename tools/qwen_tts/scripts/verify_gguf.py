"""Verify exported GGUF files are readable and contain expected data."""
import sys
import os
from pathlib import Path

if 'NO_LOCAL_GGUF' not in os.environ:
    gguf_py_path = Path(__file__).parent.parent.parent / 'gguf-py'
    if gguf_py_path.exists():
        sys.path.insert(1, str(gguf_py_path))

import gguf

def verify_gguf(path: str):
    print(f"\n=== {os.path.basename(path)} ===")
    reader = gguf.GGUFReader(path)

    print(f"  KV pairs: {len(reader.fields)}")
    for key, field in reader.fields.items():
        data = field.data
        if isinstance(data, list):
            val = data
        elif hasattr(data, 'tolist'):
            val = data.tolist()
        else:
            val = str(data)
        if isinstance(val, list) and len(val) <= 10:
            print(f"    {key}: {val}")
        elif isinstance(val, list):
            print(f"    {key}: [{val[0]}...] (len={len(val)})")
        else:
            print(f"    {key}: {val}")

    print(f"  Tensors: {len(reader.tensors)}")
    total_bytes = 0
    for t in reader.tensors:
        total_bytes += t.n_bytes
    print(f"  Total tensor bytes: {total_bytes / 1024 / 1024:.2f} MB")

    # Print first 5 and last 5 tensor names
    for t in reader.tensors[:5]:
        print(f"    {t.name}: shape={list(t.shape)}, type={t.tensor_type}")
    if len(reader.tensors) > 10:
        print(f"    ... ({len(reader.tensors) - 10} more)")
    for t in reader.tensors[-5:]:
        print(f"    {t.name}: shape={list(t.shape)}, type={t.tensor_type}")

if __name__ == "__main__":
    gguf_dir = sys.argv[1] if len(sys.argv) > 1 else "gguf"
    for f in sorted(os.listdir(gguf_dir)):
        if f.endswith(".gguf"):
            verify_gguf(os.path.join(gguf_dir, f))
