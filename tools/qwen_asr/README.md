# Qwen3-ASR C++ Inference

基于 GGML/llama.cpp 的 Qwen3-ASR-1.7B 语音识别 C++ 实现，支持昇腾 NPU (CANN) 加速。

## 架构

```
WAV 音频 (16kHz)
    │
    ▼
Mel Spectrogram ─── Whisper 风格 (n_fft=400, hop=160, 128 mels)
    │
    ▼
Audio Encoder ───── 3×Conv2d + 24 层 Transformer + 输出 MLP (GGML)
    │                产出连续向量 embeddings (122 帧 × 2048 维)
    ▼
Split Prefill ───── [text tokens] → [audio embeddings] → [text tokens]
    │                通过 llama.cpp batch.token / batch.embd 分段注入
    ▼
Text Decoder ────── Qwen3 28 层 Transformer (llama.cpp, Q8_0/F16)
    │                自回归生成文本 token
    ▼
BPE Decode ──────── token IDs → 文本
```

## 依赖

- CMake >= 3.14
- C++17 编译器
- Python 3.10+ (导出 GGUF 用)
- 模型：[Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- (可选) 昇腾 CANN 8.1.RC1+ (NPU 加速，已验证 8.1.RC1)

## 快速开始

### 1. 下载模型

```bash
# HuggingFace
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir Qwen/Qwen3-ASR-1.7B

# 或 ModelScope
modelscope download Qwen/Qwen3-ASR-1.7B --local_dir Qwen/Qwen3-ASR-1.7B
```

### 2. 导出 GGUF

```bash
pip install safetensors gguf

# 导出音频编码器 (F16, 607MB)
python tools/qwen_asr/export_audio_encoder.py \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --output_dir tools/qwen_asr/gguf

# 导出文本解码器 (F16, 3.9GB)
python tools/qwen_asr/export_decoder_llama.py \
    --model_path Qwen/Qwen3-ASR-1.7B \
    --output_dir tools/qwen_asr/gguf

# 量化解码器为 Q8_0 (2.1GB)
./build/bin/llama-quantize \
    tools/qwen_asr/gguf/qwen_asr_decoder.gguf \
    tools/qwen_asr/gguf/qwen_asr_decoder_q8_0.gguf Q8_0
```

### 3. 导出 Mel 滤波器 (可选，提升精度)

```bash
pip install transformers
python -c "
import numpy as np
from transformers import WhisperFeatureExtractor
fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400)
np.save('tools/qwen_asr/gguf/mel_filters.npy', fe.mel_filters.astype(np.float32))
"
```

### 4. 设置 CANN 环境

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

### 5. 编译

```bash
mkdir -p build && cd build

# CPU only
cmake .. -DLLAMA_CURL=OFF
make qwen_asr -j$(nproc)

# 昇腾 NPU (CANN)
cmake .. -DGGML_CANN=ON -DLLAMA_CURL=OFF \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal"
make qwen_asr -j$(nproc)
```

> **说明**：stub 链接参数用于解决编译期 `libascend_hal.so` 符号缺失问题。若 SOC 类型自动检测失败，可追加 `-DSOC_TYPE=Ascend910B`（根据实际芯片修改）。

### 6. 运行

```bash
# CPU (Q8_0)
./build/bin/qwen_asr \
    --audio test.wav \
    --model_dir Qwen/Qwen3-ASR-1.7B \
    --encoder tools/qwen_asr/gguf/qwen_asr_audio_encoder.gguf \
    --decoder tools/qwen_asr/gguf/qwen_asr_decoder_q8_0.gguf \
    --threads 8

# 昇腾 NPU (推荐 Q8_0 + gpu_layers=28)
./build/bin/qwen_asr \
    --audio test.wav \
    --model_dir Qwen/Qwen3-ASR-1.7B \
    --encoder tools/qwen_asr/gguf/qwen_asr_audio_encoder.gguf \
    --decoder tools/qwen_asr/gguf/qwen_asr_decoder_q8_0.gguf \
    --gpu_layers 28 \
    --threads 8
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--audio` | `ellen_ref.wav` | 输入音频文件 (WAV) |
| `--model_dir` | `Qwen/Qwen3-ASR-1.7B` | 模型目录 (含 vocab.json, merges.txt) |
| `--encoder` | `tools/qwen_asr/gguf/qwen_asr_audio_encoder.gguf` | 音频编码器 GGUF |
| `--decoder` | `tools/qwen_asr/gguf/qwen_asr_decoder_q8_0.gguf` | 文本解码器 GGUF |
| `--vocab` | `{model_dir}/vocab.json` | 词表文件 |
| `--merges` | `{model_dir}/merges.txt` | BPE merges 文件 |
| `--device` | `CPU` | 音频编码器设备 (CPU/CANN0) |
| `--threads` | `4` | CPU 线程数 |
| `--gpu_layers` | `0` | 解码器卸载到 NPU 的层数 (28=全部) |
| `--max_tokens` | `256` | 最大生成 token 数 |

## 性能

测试音频：9.36 秒英语语音 (ellen_ref.wav)，生成 43 tokens。

### 昇腾 910B2

| 配置 | Mel | Encoder | Prefill | Generation | 总耗时 | RTF |
|------|-----|---------|---------|------------|--------|-----|
| **Q8_0 + CANN (gpu_layers=28)** | 38ms | 85ms | 150ms | 1.2s | **1.4s** | **0.15x** |
| F16 + CANN (gpu_layers=28) | 38ms | 85ms | 90ms | 1.2s | 1.3s | 0.14x |
| Q8_0 CPU only | 38ms | 960ms | 2.7s | 4.4s | 8.2s | 0.88x |

### 对比

| 平台 | 总耗时 | RTF | 备注 |
|------|--------|-----|------|
| 3090 GPU (Python, bfloat16) | 585ms | 0.063x | Flash Attention + CUDA |
| **910B2 NPU (C++, Q8_0)** | **1.4s** | **0.15x** | CANN + split prefill |
| 910B2 CPU (C++, Q8_0) | 8.2s | 0.88x | 8 threads |

## GGUF 文件

| 文件 | 大小 | 说明 |
|------|------|------|
| `qwen_asr_audio_encoder.gguf` | 607 MB | 音频编码器 (F16) |
| `qwen_asr_decoder.gguf` | 3.9 GB | 文本解码器 (F16) |
| `qwen_asr_decoder_q8_0.gguf` | 2.1 GB | 文本解码器 (Q8_0, 推荐) |

## 技术细节

### Split Prefill

多模态模型的音频特征是连续向量 (不是 token ID)，必须通过 `batch.embd` 注入 LLM。CANN 后端对 `batch.embd` 全量 prefill 有兼容性问题 (NZ 格式权重的异步拷贝断言)，因此采用分段 prefill：

```
Phase 1: [<|im_start|> system \n <|im_end|> \n <|im_start|> user \n <|audio_start|>]  → batch.token
Phase 2: [audio_feature_0, audio_feature_1, ..., audio_feature_121]                    → batch.embd
Phase 3: [<|audio_end|> <|im_end|> \n <|im_start|> assistant \n]                       → batch.token
Phase 4: 自回归生成                                                                      → batch.token
```

### 支持语言

继承 Qwen3-ASR 的 30+ 语言支持：中文、英文、粤语、日语、韩语、法语、德语、西班牙语等。

## 常见问题

### npu-smi 报错 `libdrvdsmi_host.so: cannot open shared object file`

容器环境中 NPU 驱动库路径未加入系统搜索路径。解决方法：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver/:/usr/local/Ascend/driver/lib64/common/:$LD_LIBRARY_PATH
npu-smi info  # 验证
```

### CMake 报错 `Auto-detech ascend soc type failed`

`npu-smi` 不可用导致无法自动检测芯片型号。通过 `-DSOC_TYPE=Ascend910B` 手动指定（根据实际芯片修改）。

### 链接报错 `undefined reference to drvGetDevIDs / drvHdcEpollCtl`

容器环境编译时缺少驱动符号。添加 stub 库路径解决：

```bash
cmake .. -DGGML_CANN=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal" \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal"
```

## 文件结构

```
tools/qwen_asr/
├── README.md                    # 本文件
├── CMakeLists.txt               # 构建配置
├── main.cpp                     # CLI 入口
├── qwen_asr.h/cpp               # ASR 主流程 (split prefill + 解码)
├── audio_encoder.h/cpp          # 音频编码器 (Conv2d + Transformer)
├── mel_spectrogram.h/cpp        # Whisper 风格 Mel 频谱图
├── export_audio_encoder.py      # 音频编码器 GGUF 导出
├── export_decoder_llama.py      # 文本解码器 llama.cpp GGUF 导出
├── verify_audio_encoder.py      # Python 中间结果验证
├── verify_conv2d.py             # Conv2d 逐层验证
└── gguf/                        # GGUF 模型文件
```

复用 `tools/qwen_tts/` 的共享组件：`bpe_tokenizer`, `audio_io`, `model_loader`, `ctx_manager`, `build_graph`, `infer_session`。
