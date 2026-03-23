# GPT-SoVITS C++ Inference

GGUF-based GPT-SoVITS text-to-speech C++ inference implementation.

## 1. Model Conversion

### 1.1 Convert chinese-roberta-wwm-ext-large

```bash
python convert_hf_to_gguf.py assets/models/tts/moyoyo/chinese-roberta-wwm-ext-large \
    --outfile gguf/chinese-roberta-wwm-ext-large.gguf \
    --outtype f32
```

### 1.2 Convert GPT-SoVITS Model

```bash
python export_gptsovits.py
```

Models will be saved in the `gguf/` folder under the current directory.

## 2. Build

```bash
cmake --preset x64-linux-clang-release
cmake --build build
```

## 3. Run

```bash
./build/bin/gptsovits [options]
```

### Command Line Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model_dir` | `-m` | GGUF model directory | `/home/wjr/mount/myy/VoiceDialogue/gguf/` |
| `--text` | `-t` | Target text to synthesize | `Today is a good day...` |
| `--target_lang` | | Target text language (en/zh) | `en` |
| `--ref_audio` | `-r` | Reference audio file path | - |
| `--ref_text` | | Reference audio transcript | - |
| `--ref_lang` | | Reference text language (en/zh) | `en` |
| `--output` | `-o` | Output audio file path | `o.wav` |
| `--device` | `-d` | Compute device (CUDA0/CPU) | `CUDA0` |
| `--n_threads` | `-n` | Number of compute threads | `4` |
| `--profiling` | `-p` | Enable profiling mode | `false` |

### Examples

```bash
# Basic usage
./build/bin/gptsovits -m /path/to/models -t "Hello, world!"

# Specify reference audio and output path
./build/bin/gptsovits \
    -m /path/to/models \
    -t "Hello, my name is Michael Jordan, nice to meet you." \
    --target_lang en \
    -r /path/to/ref.wav \
    --ref_text "This is reference text." \
    -o output.wav

# Enable profiling
./build/bin/gptsovits -m /path/to/models -t "Test text" -p
```

## 4. Performance Comparison

Test environment: NVIDIA RTX 3090

| Audio Duration | Python | C++ | Speedup |
|----------------|--------|-----|---------|
| 2 sec          | 390 ms | 152 ms | 2.6x |
| 8 sec          | 900 ms | 490 ms | 1.8x |
