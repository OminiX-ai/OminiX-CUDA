# Qwen3-TTS Voice Clone C++ Implementation

基于 GGML/llama.cpp 的 Qwen3-TTS-12Hz-1.7B 语音合成 C++ 实现，支持昇腾 NPU (CANN) 加速。

## 1. Model Conversion

```bash
python export_qwen_tts.py --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --output_dir gguf/
```

## 2. 设置 CANN 环境

编译和运行前需确保 `ASCEND_TOOLKIT_HOME` 指向正确的 CANN 工具包路径。如果使用了 conda 等虚拟环境，可能会覆盖系统默认值，请手动检查并设置：

```bash
# 检查当前路径
echo $ASCEND_TOOLKIT_HOME

# 若路径不正确（如指向 conda 环境内的不完整安装），手动指定系统 CANN：
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest

# 若驱动库不在默认 LD_LIBRARY_PATH 中（如容器环境），也需设置：
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/driver/lib64/common/:$LD_LIBRARY_PATH

# 验证
npu-smi info
```

## 3. Build

```bash
mkdir -p build && cd build

# CPU only
cmake .. -DLLAMA_CURL=OFF
make qwen_tts -j$(nproc)

# 昇腾 NPU (CANN)
cmake .. -DGGML_CANN=ON -DLLAMA_CURL=OFF \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal"
make qwen_tts -j$(nproc)
```

> **说明**：stub 链接参数用于解决编译期 `libascend_hal.so` 符号缺失问题。若 SOC 类型自动检测失败，可追加 `-DSOC_TYPE=Ascend910B`（根据实际芯片修改）。

## 4. Run

```bash
./build/bin/qwen_tts \
  -m tools/qwen_tts/gguf \
  --tokenizer_dir Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  -t "Hello, this is a test." \
  --target_lang English \
  -r ref_audio.wav \
  --ref_text "This is reference audio." \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 \
  -o output.wav
```

### CLI Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model_dir` | `-m` | GGUF model directory | - |
| `--tokenizer_dir` | | Tokenizer directory (vocab.json + merges.txt) | same as model_dir |
| `--text` | `-t` | Target text | - |
| `--target_lang` | | Target language (English/Chinese) | `English` |
| `--ref_audio` | `-r` | Reference audio path | - |
| `--ref_text` | | Reference transcript | - |
| `--ref_lang` | | Reference language | `English` |
| `--output` | `-o` | Output audio path | `output.wav` |
| `--talker_model` | | Override Talker GGUF (for quantized models) | - |
| `--cp_model` | | Override CP llama GGUF (for NPU acceleration) | - |
| `--device` | `-d` | Compute device (CPU/CANN0) | `CPU` |
| `--n_threads` | `-n` | Thread count | `8` |
| `--n_gpu_layers` | | Layers to offload to GPU/NPU (29=all) | `0` |
| `--max_tokens` | | Max generated codec frames | `2048` |
| `--temperature` | | Sampling temperature | `0.9` |
| `--top_k` | | Top-K sampling (0=disabled) | `50` |
| `--top_p` | | Top-P nucleus sampling | `1.0` |
| `--repetition_penalty` | | Repetition penalty | `1.05` |
| `--greedy` | | Disable sampling (greedy decoding) | `false` |
| `--seed` | | Random seed | `42` |
| `--profiling` | `-p` | Enable profiling | `false` |

## 5. Testing

```bash
bash test.sh
```

## 性能

测试环境：昇腾 910B2, CANN 8.1.RC1, Q8_0 量化, 29 层 NPU offload。

| 测试 | 音频时长 | 生成耗时 | RTF |
|------|---------|---------|-----|
| 短英文 | 6.2s | 12.1s | 1.95x |
| 长英文 | 12.6s | 23.0s | 1.83x |
| 中文 | 8.6s | 15.5s | 1.80x |
