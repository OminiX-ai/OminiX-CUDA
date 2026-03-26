# OminiX-Ascend CANN 后端优化说明

基于 Ascend NPU (CANN) 后端，针对 LLM 推理的优化记录。

SD (Stable Diffusion) 相关优化请参考 [tools/ominix_diffusion/README.md](tools/ominix_diffusion/README.md)。

## 目录

- [项目结构](#项目结构)
- [构建](#构建)
- [LLM 优化](#llm-优化)
- [统一 ggml 合并](#统一-ggml-合并)
- [环境变量参考](#环境变量参考)
- [测试命令](#测试命令)
- [测试环境](#测试环境)

---

## 项目结构

```
OminiX-Ascend/
├── ggml/                       # 统一 ggml 后端（LLM + SD + ASR 共享）
├── src/                        # llama 核心库
├── include/
│   ├── llama.h
│   └── stable-diffusion.h      # SD 公共头文件
├── tools/
│   ├── qwen_asr/               # ASR 语音识别
│   ├── ominix_diffusion/       # SD 推理模块
│   │   ├── src/                # SD 推理库 (libstable-diffusion)
│   │   ├── cli/                # → ominix-diffusion-cli
│   │   ├── server/             # → ominix-diffusion-server
│   │   └── common/
│   └── ...                     # llama 原生工具
├── examples/
│   └── diffusion/              # Dream LLM diffusion 示例 (llama-diffusion-cli)
└── common/                     # llama 公共库
```

---

## 构建

```bash
# 标准构建（LLM + SD，启用 ACL Graph）
cmake -B build -DGGML_CANN=ON -DUSE_ACL_GRAPH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

构建产物：
- `build/bin/llama-cli` — LLM 交互式推理
- `build/bin/llama-bench` — LLM 性能基准测试
- `build/bin/llama-server` — LLM HTTP 服务
- `build/bin/ominix-diffusion-cli` — SD 图像生成（详见 [SD README](tools/ominix_diffusion/README.md)）
- `build/bin/ominix-diffusion-server` — SD HTTP 服务
- `build/bin/qwen_asr` — 语音识别 (ASR)
- `build/bin/llama-diffusion-cli` — Dream LLM diffusion 示例

---

## LLM 优化

以下优化均针对 Ascend 910B2 NPU，相比原生 llama.cpp CANN 后端。

### 1. BF16 模型全量 NPU 卸载

**问题**: 原生 CANN 后端的 `supports_op` 未声明 BF16 类型支持，导致 BF16 模型的所有算子回退到 CPU 执行。

**修复**: 为 MUL_MAT、MUL_MAT_ID、GET_ROWS、SET_ROWS、CPY、CONT 添加 BF16 类型支持。同时限制 FRACTAL_NZ 格式仅用于 F16 权重（Ascend 910B2 不支持 F32/BF16 的 FRACTAL_NZ）。

**效果** (Qwen3-1.7B-BF16):

| 指标 | 修复前 (CPU 回退) | 修复后 (NPU) | 提升 |
|------|-------------------|--------------|------|
| pp512 | ~7.8 t/s | 1482 t/s | 190x |
| tg128 | ~1.6 t/s | 41 t/s | 25x |

### 2. V cache SET_ROWS 重构

**问题**: 非 Flash Attention 模式下，V cache 使用转置布局 (v_trans=true)，SET_ROWS 将 2D 数据展平为 1D 后用 ScatterUpdate 逐元素写入。ScatterUpdate 形状 [524288, 1] 只能使用 1 个 NPU 核心，占总推理时间 42%。

**修复**: 通过 `view_src` 链追溯原始 V cache 形状，将逐元素 scatter 重构为列级 `InplaceIndexCopy`，充分利用 NPU 并行能力。

**效果** (Qwen3-8B-Q8_0):

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| pp512 | 1135 t/s | 5901 t/s | 5.2x |
| tg128 | 27.15 t/s | 37.70 t/s | 1.39x |

### 3. ADD + RMS_NORM 算子融合

**功能**: 将相邻的 ADD 和 RMS_NORM 操作融合为单次内核调用，减少内核启动开销和中间内存分配。

**启用**: 设置环境变量 `GGML_CANN_OPERATOR_FUSION=on`

### 4. ACL Graph 模式预热

**问题**: ACL Graph capture 期间调用 `aclrtMalloc` 会导致崩溃。

**修复**: 在 Graph capture 前执行预热（warmup），初始化内存池和算子缓存，确保 Graph 模式下不触发动态内存分配。

**启用**: 编译时 `-DUSE_ACL_GRAPH=ON`，运行时通过 `GGML_CANN_ACL_GRAPH` 控制

### 5. Flash Attention 支持

原生 CANN 后端已支持 Flash Attention，在 Ascend NPU 上可通过 `-fa 1` 启用。Flash Attention 让 V cache 使用非转置 2D 布局，从根本上避免 ScatterUpdate 瓶颈。

### LLM 综合性能 (Qwen3-8B-Q8_0)

| 配置 | pp512 | tg128 | 相比 Baseline |
|------|-------|-------|---------------|
| Baseline (原生) | 1135 t/s | 27.15 t/s | — |
| V cache 优化 + 算子融合 | 5856 t/s | 37.64 t/s | pp 5.16x, tg 1.39x |
| FA + 算子融合 (推荐) | 6220 t/s | 42.82 t/s | pp 5.48x, tg 1.58x |

---

## 统一 ggml 合并

将原先 SD 专属的 `ggml-diffusion/` 副本（~33 万行）合并到统一的 `ggml/` 后端。LLM 和 SD 现在共享同一个 ggml 后端，通过运行时环境变量区分行为。

**架构优势**:
- 消除重复代码维护负担
- 未来 ASR、TTS 等模型可直接复用同一后端
- Bug 修复和优化自动惠及所有模型类型

---

## 环境变量参考

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GGML_CANN_ACL_GRAPH` | `off` | ACL Graph 模式开关。SD 推理需要设置为 `1` 开启，LLM 推理不需要 |
| `GGML_CANN_QUANT_BF16` | `off` | 量化矩阵乘法使用 BF16 compute dtype。SD 推理需要开启以防止 NaN |
| `GGML_CANN_OPERATOR_FUSION` | `off` | ADD+RMS_NORM 算子融合。LLM 推理建议开启 |

---

## 测试命令

### LLM 推理

```bash
# 基础推理
./build/bin/llama-cli \
  -m <model>.gguf \
  -ngl 99 \
  -p "你好，请介绍一下你自己。" \
  -n 128

# 性能基准测试
./build/bin/llama-bench \
  -m <model>.gguf \
  -ngl 99

# 推荐配置：Flash Attention + 算子融合
GGML_CANN_OPERATOR_FUSION=on \
./build/bin/llama-bench \
  -m <model>.gguf \
  -ngl 99 -fa 1

# HTTP 服务
./build/bin/llama-server \
  -m <model>.gguf \
  -ngl 99 \
  --host 0.0.0.0 --port 8080
```

### ASR 语音识别

```bash
./build/bin/qwen_asr \
  --audio <audio_file>.wav \
  --model_dir <gguf_dir> \
  --encoder <gguf_dir>/qwen_asr_audio_encoder.gguf \
  --decoder <gguf_dir>/qwen_asr_decoder_q8_0.gguf \
  --gpu_layers 28 \
  --threads 8
```

---

## 测试环境

- 硬件: Ascend 910B2 (62GB HBM)
- CANN: 8.5.0
- 测试模型:
  - LLM: Qwen3-1.7B-BF16.gguf, Qwen3-8B-Q8_0.gguf
  - SD: Qwen-Image-Q8_0.gguf（详见 [SD README](tools/ominix_diffusion/README.md)）
