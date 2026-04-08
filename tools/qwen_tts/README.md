# Qwen3-TTS Voice Clone C++ Implementation

基于 GGML/llama.cpp 的 Qwen3-TTS-12Hz-1.7B 语音合成 C++ 实现，支持昇腾 NPU (CANN) 加速。

## 1. Model Conversion

```bash
# 导出 encoder 时使用 f32 精度以获得最佳音质
# export_qwen_tts.py 中 load_speech_tokenizer 需使用 dtype=torch.float32
python export_qwen_tts.py --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base --output_dir gguf/
```

> **注意**: 参考音频建议预先 resample 到 24kHz 以获得最佳效果。

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
cmake .. -DGGML_CANN=ON -DLLAMA_CURL=OFF -DSOC_TYPE=Ascend910B \
    -DCMAKE_SHARED_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal" \
    -DCMAKE_EXE_LINKER_FLAGS="-L${ASCEND_TOOLKIT_HOME}/runtime/lib64/stub/ -lascend_hal"
make qwen_tts -j$(nproc)
```

## 4. Run

### 基本用法

```bash
./build/bin/qwen_tts \
  -m tools/qwen_tts/gguf \
  --tokenizer_dir tools/qwen_tts/gguf \
  -t "Hello, this is a test." \
  --target_lang English \
  -r ref_audio_24k.wav \
  --ref_text "This is reference audio." \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 \
  -o output.wav
```

### 使用参考音频缓存（推荐）

首次推理时保存缓存，后续推理直接加载，跳过 encoder 节省约 2 秒：

```bash
# 首次：encode + 保存缓存
./build/bin/qwen_tts \
  -m tools/qwen_tts/gguf --tokenizer_dir tools/qwen_tts/gguf \
  -r ref_audio_24k.wav --ref_text "Reference transcript." \
  --ref_cache speaker_cache.bin \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 \
  -t "First sentence." -o out1.wav

# 后续：只需缓存文件（无需 ref_audio 和 ref_text）
./build/bin/qwen_tts \
  -m tools/qwen_tts/gguf --tokenizer_dir tools/qwen_tts/gguf \
  --ref_cache speaker_cache.bin \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 \
  -t "Another sentence with same voice." -o out2.wav
```

### 使用内置语音（Built-in Voices）

项目自带一组预置音色，使用前先烘焙缓存（仅需一次）：

```bash
# 一次性烘焙所有内置音色 (依赖已构建的 build/bin/qwen_tts)
tools/qwen_tts/scripts/bake_voices.sh

# 查看可用音色
./build/bin/qwen_tts --list_voices

# 直接按音色 id 推理
./build/bin/qwen_tts \
  -m tools/qwen_tts/gguf --tokenizer_dir tools/qwen_tts/gguf \
  --voice ellen \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 \
  -t "Hello from a built-in voice." -o out.wav
```

音色清单由 `tools/qwen_tts/data/voices/voices.json` 维护，可通过 `--voices_dir` 指向其它目录。

### X-Vector 克隆模式

x-vector 模式只用一个 2048 维的 speaker embedding 做声音克隆，不需要 ref audio 的 codec
codes 也不需要 ref text，对应 Qwen3-TTS 原生的 `x_vector_only_mode=True`。
该模式 prefill 极短（~10 tokens），生成更快，且 .xvec 文件只有约 8 KB。

```bash
# 1) 从一段 wav 中提取 speaker embedding，写入 .xvec（一次即可）
./build/bin/qwen_tts \
  -m tools/qwen_tts/gguf --tokenizer_dir tools/qwen_tts/gguf \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 \
  --xvec_extract tools/qwen_tts/data/ref_audios/ellen_ref_24k.wav \
  --xvec_out ellen.xvec

# 2) 用 .xvec 直接合成，无需 ref_audio / ref_text / ref_cache
./build/bin/qwen_tts \
  -m tools/qwen_tts/gguf --tokenizer_dir tools/qwen_tts/gguf \
  --talker_model tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf \
  --cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf \
  --n_gpu_layers 29 \
  --xvec ellen.xvec \
  -t "Hello from x-vector cloning." -o out.wav
```

文件格式：`magic "QXVC" (4B) | version u32 | spk_dim u32 | float32[spk_dim]`。
`--xvec` 与 `--voice` / `--ref_cache` / `--ref_audio` 互斥（xvec 模式禁用 ICL）。

### CLI Arguments

| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model_dir` | `-m` | GGUF model directory | - |
| `--tokenizer_dir` | | Tokenizer directory (vocab.json + merges.txt) | same as model_dir |
| `--text` | `-t` | Target text to synthesize | - |
| `--target_lang` | | Target language (English/Chinese) | `English` |
| `--ref_audio` | `-r` | Reference audio path (24kHz recommended) | - |
| `--ref_text` | | Reference audio transcript | - |
| `--ref_cache` | | Pre-computed ref cache file (.bin) | - |
| `--voice` | | Built-in voice id (see `--list_voices`) | - |
| `--voices_dir` | | Directory containing `voices.json` | `tools/qwen_tts/data/voices` |
| `--list_voices` | | List built-in voices and exit | - |
| `--xvec` | | X-vector file (xvec-only inference, no ICL) | - |
| `--xvec_extract` | | Wav path for tool mode: extract spk embedding → `--xvec_out` | - |
| `--xvec_out` | | Output `.xvec` path for `--xvec_extract` | `voice.xvec` |
| `--output` | `-o` | Output audio path | `output.wav` |
| `--talker_model` | | Override Talker GGUF (for quantized models) | - |
| `--cp_model` | | Override CP llama GGUF (for NPU acceleration) | - |
| `--n_threads` | `-n` | Thread count | `8` |
| `--n_gpu_layers` | | Layers to offload to NPU (29=all) | `0` |
| `--max_tokens` | | Max generated codec frames | `2048` |
| `--temperature` | | Sampling temperature | `0.9` |
| `--top_k` | | Top-K sampling (0=disabled) | `50` |
| `--top_p` | | Top-P nucleus sampling | `1.0` |
| `--repetition_penalty` | | Repetition penalty | `1.05` |
| `--greedy` | | Disable sampling (greedy decoding) | `false` |
| `--seed` | | Random seed | `42` |
| `--profiling` | `-p` | Enable profiling + debug dumps | `false` |

## 5. Architecture

```
ref_audio --> [Speaker Encoder] ---------> spk_embedding --+
         \-> [Speech Encoder]  ---------> ref_codes -------+--> [Talker LLM] --> codec_tokens
ref_text --> [BPE Tokenizer]   --> ref_text_tokens --------+        |
target_text -> [BPE Tokenizer] --> target_text_tokens -----+    [Code Predictor]
                                                                    |
                                                ref_codes + codec_tokens
                                                                    |
                                                             [Decoder] --> audio
```

### Model Components

| Component | Architecture | Params | Precision | Device |
|-----------|-------------|--------|-----------|--------|
| Speaker Encoder | TDNN (Conv1d) | 23MB | f16/f32 | CPU |
| Speech Encoder | Conv + 8-layer Transformer (512-dim) | 215MB | f32 | CANN0 |
| Talker LLM | 28-layer Transformer (2048-dim, 16 heads) | 1.5GB | Q8_0 | CANN0 |
| Code Predictor | 5-layer Transformer (1024-dim, 16 heads) | 159MB | f16 | CANN0 |
| Decoder | RVQ + Transformer + Vocoder | 219MB×2 | f16 | CANN0 (chunked) |

## 6. Performance

测试环境：昇腾 910B2, CANN 8.1.RC1, Q8_0 Talker, 29 层 NPU offload。

### 无 cache 模式

| 文本长度 | 帧数 | 音频时长 | Prefill | Generate | Decode | Total | Inference RTF | Total RTF |
|---------|------|---------|---------|----------|--------|-------|---------------|-----------|
| 短 (2词) | 18 | 1.44s | 2.03s | 1.40s | 1.45s | 4.88s | 1.98x | 3.39x |
| 中 (20词) | 74 | 5.92s | 2.05s | 5.42s | 2.33s | 9.80s | 1.31x | 1.66x |
| 长 (40词) | 127 | 10.16s | 2.03s | 9.34s | 3.40s | 14.77s | 1.25x | **1.45x** |

### 有 cache 模式（跳过 encoder）

| 文本长度 | 帧数 | 音频时长 | Prefill | Generate | Decode | Total | Inference RTF | Total RTF |
|---------|------|---------|---------|----------|--------|-------|---------------|-----------|
| 短 (2词) | 18 | 1.44s | 0s | 1.38s | 1.44s | 2.82s | 1.96x | 1.96x |
| 中 (20词) | 74 | 5.92s | 0s | 5.85s | 2.33s | 8.18s | 1.38x | 1.38x |
| 长 (40词) | 127 | 10.16s | 0s | 9.32s | 3.31s | 12.63s | 1.24x | **1.24x** |

### Generate 耗时分布（长文本 127 帧）

| 子模块 | 耗时 | 占比 | Device |
|--------|------|------|--------|
| build_input_embeddings | 99ms | 1.1% | CPU (NEON+OMP) |
| Talker Prefill | ~30ms | 0.3% | CANN0 |
| Codec Head | 111ms | 1.2% | CPU (NEON+OMP) |
| Code Predictor | 6975ms | 74.8% | CANN0 |
| Talker LLM decode | 2027ms | 21.7% | CANN0 |
| Other (sample, EMB) | ~50ms | 0.5% | CPU |

## 7. Known Issues

- **背景音频 artifact**: 由 encoder 精度差异引起（ggml vs PyTorch f32 数值差异经 RVQ 残差链放大）。语义层 (q0) 100% 匹配 Python，声学层 (q1-q15) 有偏差。详见 `docs/qwen_tts_encoder_precision.md`。
- **Decoder 帧数限制**: CANN 单次最多处理 96 帧，长序列使用 chunked decode (overlap=72)。
