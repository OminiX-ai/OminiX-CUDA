# OminiX Diffusion

基于 Ascend NPU (CANN) 后端的 Stable Diffusion 图像生成模块，支持 SD1.5、SD2.1、SDXL、SD3、Flux、Qwen-Image 等模型。

## 目录结构

```
tools/ominix_diffusion/
├── src/                    # SD 推理库 (libstable-diffusion.a)
│   ├── stable-diffusion.cpp
│   ├── model.cpp
│   ├── thirdparty/         # zip, stb_image, json.hpp 等
│   ├── vocab/
│   └── *.hpp
├── cli/                    # → ominix-diffusion-cli
│   └── main.cpp
├── server/                 # → ominix-diffusion-server
│   └── main.cpp
└── common/
    └── common.hpp
```

## 构建

从项目根目录统一编译：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

cmake -B build -DGGML_CANN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

产物：
- `build/bin/ominix-diffusion-cli` — SD 图像生成
- `build/bin/ominix-diffusion-server` — SD HTTP 服务

## 使用

```bash
# 基础生图（1024x1024）
GGML_CANN_ACL_GRAPH=1 GGML_CANN_QUANT_BF16=on \
./build/bin/ominix-diffusion-cli \
  --diffusion-model <diffusion_model>.gguf \
  --vae <vae_model>.safetensors \
  --llm <llm_model>.gguf \
  -p "a lovely cat" \
  --cfg-scale 2.5 \
  --steps 20 \
  --sampling-method euler \
  -H 1024 -W 1024 \
  -o output.png

# 推荐配置：启用 diffusion flash attention + VAE 直接卷积
GGML_CANN_ACL_GRAPH=1 GGML_CANN_QUANT_BF16=on \
./build/bin/ominix-diffusion-cli \
  --diffusion-model <diffusion_model>.gguf \
  --vae <vae_model>.safetensors \
  --llm <llm_model>.gguf \
  -p "a lovely cat" \
  --cfg-scale 2.5 \
  --steps 20 \
  --sampling-method euler \
  --diffusion-fa \
  --flow-shift 3 \
  --vae-conv-direct \
  -H 1024 -W 1024 \
  -o output.png
```

## CANN 优化

### 1. CANN 原生卷积算子

**问题**: 原生 ggml 的 CONV_2D/CONV_3D 通过 im2col + matmul 实现，im2col_3d 在 CANN 后端无 NPU 实现，回退到 CPU（需要 device→host→CPU 计算→host→device 拷贝）。

**修复**:
- `CONV_2D`: 直接调用 `aclnnConvolution`，输入自动转换为 NCHW 格式
- `CONV_3D`: 将 4D ggml 张量重塑为 5D NCDHW 格式后调用 `aclnnConvolution`
- `IM2COL_3D`: 保留 CPU fallback 实现作为兼容路径

### 2. BF16 量化矩阵乘法

**问题**: SD 模型的量化矩阵乘法中，FP16 累加在大 K 维度（K=12288）下会溢出（activation ~131, sum > 65504），产生 NaN。

**修复**: 使用 BF16 作为 compute dtype（指数范围与 FP32 相同，max ~3.4e38），通过 `GGML_CANN_QUANT_BF16=on` 启用。

### 3. Repeat 广播修复

**问题**: Ascend 910B 上 `aclnnRepeat` 存在 MTE (Memory Transfer Engine) bug，特定形状下结果错误。

**修复**: 添加 `aclnnExpand` 路径作为替代广播实现。

### 4. ACL Graph 模式

SD 推理依赖 ACL Graph 模式进行算子图优化。通过 `GGML_CANN_ACL_GRAPH=1` 启用。

## 环境变量

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `GGML_CANN_ACL_GRAPH` | `off` | ACL Graph 模式，SD 推理需设为 `1` |
| `GGML_CANN_QUANT_BF16` | `off` | 量化矩阵乘法使用 BF16，SD 推理需开启以防止 NaN |

## 测试结果

硬件: Ascend 910B2 (62GB HBM), CANN 8.5.0

| 模型 | 关键指标 |
|------|----------|
| Qwen-Image-Q8_0 (1024x1024) | 20步采样 32.04s (1.59s/it), NaN 检查全部通过 |
