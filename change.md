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

| 指标 | 修复前 | 修复后 | 提升倍数 |
|------|--------|--------|----------|
| Prompt 处理 | 7.8 t/s | 536.7 t/s | 68.8x |
| Generation | 1.6 t/s | 66.6 t/s | 41.6x |

## 测试环境
- 硬件: Ascend 910B2 (62GB HBM)
- CANN: 8.5.0
- 模型: Qwen3-1.7B-BF16.gguf (3.3GB)
- 参数: -ngl 99 (全部 28 层卸载到 NPU)
