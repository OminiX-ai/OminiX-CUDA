# 统一 ggml 合并：消除 ggml-diffusion 重复代码

## 变更概述
将 `ggml-diffusion/`（SD 专属 ggml 副本）合并到统一的 `ggml/` 目标，消除约 33 万行重复代码。
LLM（llama-cli）和 SD（sd-cli）现在共享同一个 ggml 后端，通过运行时环境变量 `GGML_CANN_ACL_GRAPH` 控制 ACL Graph 模式。

## 合并策略
- **方案**: 单一 ggml target + 运行时切换（Scheme 2）
- **基准**: 以 ggml（LLM 侧，较新 llama.cpp）为基准，补充 SD 专属算子
- **架构目标**: 支持未来 ASR、TTS 等模型共用同一 ggml 后端

## 关键修复
1. **量化矩阵乘法**: 恢复 BF16 计算精度（防止 FP16 累加溢出导致 NaN）
2. **Repeat 广播**: 添加 aclnnExpand 路径（修复 Ascend 910B 上 aclnnRepeat 的 MTE bug）
3. **ACL Graph 默认值**: 改为 off（LLM 推理不使用 Graph 模式，SD 通过 `GGML_CANN_ACL_GRAPH=1` 显式启用）

## 构建命令
```bash
cmake -B build -DSD_CANN=ON -DUSE_ACL_GRAPH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
```

## 运行时控制
- LLM 推理（默认）: 不设置环境变量，ACL Graph 不激活，量化 matmul 使用 FP16
- SD 推理: `GGML_CANN_ACL_GRAPH=1` 启用 ACL Graph 模式，`GGML_CANN_QUANT_BF16=on` 启用 BF16 防止 NaN

## 测试结果
- LLM (Qwen3-8B-Q8_0): Prompt 187.9 t/s, Generation 37.5 t/s
- SD (Qwen-Image-Q8_0, 1024x1024): 20步采样 31.79s (1.58s/it), NaN检查全部通过

## 后续修复
4. **SD 算子实现恢复**: 重新添加 im2col_3d, conv_2d, conv_3d 三个 SD 专属 CANN 算子实现（git stash pop 冲突丢失）
5. **ggml-cann.cpp dispatch/supports_op**: 补充 IM2COL_3D, CONV_2D, CONV_3D 的计算分发和支持声明
6. **量化矩阵乘法 BF16/FP16 条件切换**: 合并时无条件使用 ACL_BF16 导致 LLM 性能回归（每次 matmul 多 3 个 cast 操作）。通过 `GGML_CANN_QUANT_BF16` 环境变量控制：默认 FP16（LLM 快速路径），`GGML_CANN_QUANT_BF16=on` 启用 BF16（SD 防 NaN）

## 删除（~33 万行）
- `ggml-diffusion/`, `include-diffusion/`, `examples/diffusion-ominix/`, `CMakeLists copy.txt`

---

# CANN 后端 BF16 支持变更记录

## 问题描述
基于 llama-cli 测试 Qwen3-1.7B-BF16.gguf 模型时，推理没有有效利用 NPU（Ascend 910B2）。
Generation 速度仅 1.6 t/s，与纯 CPU 推理速度（1.9 t/s）几乎相同。

## 根因分析
CANN 后端的 `ggml_backend_cann_supports_op` 函数中，以下关键操作没有对 BF16 类型的支持：

1. **MUL_MAT** (矩阵乘法) - 占模型计算量 99%+
2. **MUL_MAT_ID** (带 ID 的矩阵乘法)
3. **CPY** (张量复制)
4. **CONT** (张量连续化)
5. **GET_ROWS** (嵌入层)
6. **SET_ROWS** (嵌入层)

虽然底层 ACL 类型映射（`ggml_cann_type_mapping`）已支持 BF16 -> ACL_BF16，
但操作支持检查函数只包含 F32 和 F16，导致所有 BF16 操作回退到 CPU 执行。

此外，FRACTAL_NZ 格式（权重优化布局）在 Ascend 910B2 上只支持 F16 类型，
对 F32 和 BF16 权重使用 FRACTAL_NZ 会导致找不到对应的算子内核。

## 修改文件

### 1. ggml/src/ggml-cann/ggml-cann.cpp
- `GGML_OP_MUL_MAT`: 添加 `GGML_TYPE_BF16` 到支持类型列表
- `GGML_OP_MUL_MAT_ID`: 添加 `GGML_TYPE_BF16` 到支持类型列表
- `GGML_OP_GET_ROWS`: 添加 `GGML_TYPE_BF16` 到支持类型列表
- `GGML_OP_SET_ROWS`: 添加 `GGML_TYPE_BF16` 到支持类型列表
- `GGML_OP_CPY`: 添加 `GGML_TYPE_BF16` 到源和目标类型检查
- `GGML_OP_CONT`: 添加 `GGML_TYPE_BF16` 到支持类型列表（移除 TODO 注释）

### 2. ggml/src/ggml-cann/aclnn_ops.cpp
- `ggml_cann_mul_mat`: 在 switch 分发中添加 `GGML_TYPE_BF16` -> `ggml_cann_mat_mul_fp`
- `ggml_cann_mul_mat_id`: 在 switch 分发中添加 `GGML_TYPE_BF16` -> `ggml_cann_mul_mat_id_fp`
- `ggml_cann_get_rows`: 添加 `GGML_TYPE_BF16` 到 assert 和 switch 分支
- `ggml_cann_mat_mul_fp`: FRACTAL_NZ 格式仅用于 F16 权重（F32/BF16 使用 ND 格式）

## 性能对比

### llama-bench 标准测试 (pp512/tg128)

| 指标 | 修复前 (CPU回退) | 修复后 (NPU) | 提升倍数 |
|------|------------------|--------------|----------|
| pp512 (prompt) | ~7.8 t/s | **1482.52 t/s** | ~190x |
| tg128 (generation) | ~1.6 t/s | **41.05 t/s** | ~25x |

### llama-cli 交互测试

| 指标 | 修复前 | 修复后 | 提升倍数 |
|------|--------|--------|----------|
| Prompt | 7.8 t/s | 536.7 t/s | 68.8x |
| Generation | 1.6 t/s | 66.6 t/s | 41.6x |

## 测试环境
- 硬件: Ascend 910B2 (62GB HBM)
- CANN: 8.5.0
- 模型: Qwen3-1.7B-BF16.gguf (3.3GB)
- 参数: -ngl 99 (全部 28 层卸载到 NPU)

---

# Qwen3-8B-Q8_0 性能优化记录

## 基准 (Baseline)
- 模型: Qwen3-8B-Q8_0.gguf (8.11 GiB)
- pp512: **1135.32 t/s**
- tg128: **27.15 t/s**

## 性能分析发现
通过 msprof profiling 发现两个关键瓶颈：
1. **WeightQuantBatchMatmulV2** (43%): 量化 MatMul，正常开销
2. **ScatterUpdate** (42%!): V cache 写入异常缓慢

### V cache ScatterUpdate 问题根因
V cache 在非 Flash Attention 模式下使用转置布局 (v_trans=true)，导致 SET_ROWS：
- 将 2D 数据 [1024, 512] 展平为 1D [1, 524288]
- 使用 524288 个元素级索引进行 scatter
- ScatterUpdate 形状 [524288, 1] 只能使用 **1 个 NPU 核心**
- 每次调用耗时 ~10ms，占总时间 42%

## 优化措施

### 优化 1: V cache SET_ROWS 重构 (代码改动)
**文件**: `ggml/src/ggml-cann/aclnn_ops.cpp`

通过 `view_src` 链追溯原始 V cache 形状 [n_embd, kv_size]，将元素级 scatter 重构为列级 InplaceIndexCopy：
1. 将 src 从 [1, n_tokens*n_embd] 重塑为转置视图 [n_embd, n_tokens]
2. 从完整索引中每隔 n_embd 个元素提取位置索引
3. 在 V cache [n_embd, kv_size] 上沿 dim=1 执行 InplaceIndexCopy

| 测试 | Baseline | V cache 优化 | 加速 |
|------|----------|-------------|------|
| pp512 | 1135 t/s | **5901 t/s** | **5.2x** |
| tg128 | 27.15 t/s | **37.70 t/s** | **1.39x** |

### 优化 2: 启用 Flash Attention (运行参数 -fa 1)
Flash Attention 让 V cache 使用 2D 布局，完全避免 ScatterUpdate 问题。

| 测试 | Baseline | FA | 加速 |
|------|----------|-----|------|
| pp512 | 1135 t/s | **5722 t/s** | **5.04x** |
| tg128 | 27.15 t/s | **42.09 t/s** | **1.55x** |

### 优化 3: 算子融合 (环境变量 GGML_CANN_OPERATOR_FUSION=on)
启用 ADD+RMS_NORM 算子融合减少内核启动开销。

### 优化 4: ACL Graph 预热 (编译选项 USE_ACL_GRAPH=ON)
修复 ACL Graph capture 期间 aclrtMalloc 崩溃问题，通过预热执行初始化内存池和缓存。

**文件**: `ggml/src/ggml-cann/ggml-cann.cpp`

## 最终性能对比

### 非 Flash Attention 模式 (V cache 优化 + 算子融合)
| 测试 | Baseline | 优化后 | 加速 |
|------|----------|--------|------|
| pp512 | 1135 t/s | **5856 t/s** | **5.16x** |
| tg128 | 27.15 t/s | **37.64 t/s** | **1.39x** |

### Flash Attention 模式 (FA + 算子融合) — 推荐
| 测试 | Baseline | 优化后 | 加速 |
|------|----------|--------|------|
| pp512 | 1135 t/s | **6220 t/s** | **5.48x** |
| tg128 | 27.15 t/s | **42.82 t/s** | **1.58x** |

## 推荐运行命令
```bash
GGML_CANN_OPERATOR_FUSION=on \
build/bin/llama-bench -m model.gguf -ngl 99 -fa 1
```

## 测试环境
- 硬件: Ascend 910B2 (62GB HBM)
- CANN: 8.5.0
- 模型: Qwen3-8B-Q8_0.gguf (8.11 GiB, 8.19B params)
