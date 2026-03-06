# Qwen3.5-9B CANN Backend Segfault 修复报告

## 环境

| 项目 | 详情 |
|------|------|
| 模型 | Qwen3.5-9B-Q8_0.gguf |
| 硬件 | Ascend 910B2 |
| CANN SDK | 8.5.0 |
| llama.cpp | commit b4427a715 |

---

## 问题现象

在 Ascend 910B 上使用 CANN 后端推理 Qwen3.5-9B 时，程序在模型加载或推理阶段发生 segfault（信号 SIGSEGV），无法产出任何输出。

---

## 根因分析

最终定位到 **两个独立的 bug**，需要同时修复才能让推理正常运行。

### Bug 1：`GGML_OP_SET` 未在 CANN 后端实现

Qwen3.5 采用了 delta-net 架构，其分块注意力代码（`src/models/delta-net-base.cpp:262`）使用 `ggml_set_inplace` 将各 chunk 的输出拼装到目标 tensor 中。

CANN 后端的 `supports_op` 和 `compute_forward` 中均没有 `GGML_OP_SET` 的处理分支，因此：

1. 调度器发现 CANN 不支持 SET，将其路由到 CPU 后端执行。
2. 但 `ggml_set_inplace` 创建的是目标 tensor 的**视图（view）**，底层数据指针指向 CANN 设备内存。
3. CPU 端的 SET 实现直接解引用这个设备指针做 memcpy → **segfault**。

**证据**：
- CANN `supports_op` 函数中无 `GGML_OP_SET` case（默认返回 `false`）。
- CANN `compute_forward` 函数中无 `GGML_OP_SET` case。
- CANN buffer 的 `is_host = false`，CPU 无法直接访问设备内存。
- CUDA 和 Vulkan 后端均已实现 `GGML_OP_SET`。

### Bug 2：`aclnnRepeat` 内核在特定 tensor shape 下触发设备错误

修复 Bug 1 后，短 prompt 推理正常，但较长的 prompt（如 36 个 token）会触发 CANN 设备错误：

```
EZ9999: The write address of the MTE instruction is out of range.
```

通过在 `compute_forward` 中逐 op 同步（`aclrtSynchronizeStream`），精确定位到：

- **出错操作**：`REPEAT op=20 name=k_in-0 ne=[128,32,36,1]`
- **输入**：`src ne=[128,16,36,1]` → **输出**：`dst ne=[128,32,36,1]`，repeat 因子 `[1,1,2,1]`（沿 dim 1 重复 2 次）
- **根因**：CANN SDK 8.5.0 的 `aclnnRepeat` 底层 JIT 编译的 `CUSTOM_TILE_V3` 内核，在处理特定 tensor 形状时存在 MTE（Memory Transfer Engine）越界写入 bug。
- **规律**：短 prompt（如 2 token，ne[2]=2）不触发；长 prompt（如 36 token，ne[2]=36）触发。同样的 repeat 因子 `[1,1,2,1]`，只是第三维大小不同。

**证据**：
- 所有 SET op 的 debug 日志显示全部执行成功（打印了 "done"），排除 SET 导致。
- `aclrtSynchronizeStream` 在 REPEAT 之后、PAD 之前首次返回错误码 507035。
- 设备端错误信息明确指出 "MTE write address out of range"。

---

## 修复方案

### 修复 1：实现 `GGML_OP_SET`

**文件**：`ggml/src/ggml-cann/aclnn_ops.cpp`

新增 `ggml_cann_set()` 函数，模式参照已有的 `ggml_cann_acc()` 实现：

```cpp
void ggml_cann_set(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    // 1. 从 op_params 中解析 strides、offset、inplace 标志
    // 2. 若非 inplace，先将完整的 src0 拷贝到 dst（device-to-device memcpy）
    // 3. 创建 dst 的子区域视图（通过自定义 ne/nb/offset）
    // 4. 用 aclnnInplaceCopy 将 src1 拷贝到视图中
}
```

**文件**：`ggml/src/ggml-cann/ggml-cann.cpp`

- `compute_forward`：添加 `case GGML_OP_SET` 分支，调用 `ggml_cann_set()`。
- `supports_op`：添加 `case GGML_OP_SET`，类型检查与 CUDA 实现一致（仅支持 F32/I32，且 src0、src1、dst 类型必须相同）。

**注意事项**：`GGML_OP_SET` 的 case 必须放在 `GGML_OP_GROUP_NORM` **之后**，不能插在 `GGML_OP_ACC` 和 `GGML_OP_GROUP_NORM` 之间。因为 ACC 到 GROUP_NORM 之间有一条 fall-through 链（共享 `return true`），如果在中间插入带大括号的 SET case 块，会截断这条链路，导致 ACC 等操作走进 SET 的类型检查逻辑，解引用不存在的 `src[1]` → 空指针 segfault。

### 修复 2：绕过 `aclnnRepeat` 内核 bug

**文件**：`ggml/src/ggml-cann/aclnn_ops.cpp`

重写 `ggml_cann_repeat()` 函数，对**单维度 repeat** 使用 `aclnnInplaceCopy` + tensor 视图代替 `aclnnRepeat`：

```cpp
void ggml_cann_repeat(ggml_backend_cann_context & ctx, ggml_tensor * dst) {
    // 计算每个维度的 repeat 因子
    // 情况 1：仅一个维度需要 repeat（delta-net 的所有 case 均属此类）
    //   → 循环 R 次，每次创建 dst 的一个 slice 视图，用 InplaceCopy 从 src 拷入
    // 情况 2：无 repeat（identity）
    //   → 直接 device-to-device memcpy
    // 情况 3：多维度 repeat
    //   → 回退到原始 aclnnRepeat（目前未遇到此路径触发 bug）
}
```

**原理**：对于 `repeat=[1,1,2,1]`（dim 1 重复 2 次），等价于将 src 分别拷贝到 dst 的前半段和后半段。通过 `ggml_cann_create_tensor(dst, src->ne, dst->nb, ..., offset)` 创建 dst 的偏移视图，然后用 `InplaceCopy` 完成拷贝。这完全绕过了有 bug 的 `aclnnRepeat` JIT 内核。

---

## 修改文件清单

| 文件 | 修改内容 | 新增行数 |
|------|---------|---------|
| `ggml/src/ggml-cann/aclnn_ops.h` | 添加 `ggml_cann_set()` 声明 | +12 |
| `ggml/src/ggml-cann/aclnn_ops.cpp` | 添加 `ggml_cann_set()` 实现；重写 `ggml_cann_repeat()` | +71 |
| `ggml/src/ggml-cann/ggml-cann.cpp` | `compute_forward` 添加 SET dispatch；`supports_op` 添加 SET 类型检查 | +10 |

总计：3 个文件，+88 行，-5 行。

---

## 验证结果

| 测试 | 结果 |
|------|------|
| 短 prompt "Hello"，-n 16 | ✅ 成功，Prompt 26.1 t/s，Generation 16.0 t/s |
| 长 prompt（广义相对论），-n 64 | ✅ 成功，Prompt 67.7 t/s，Generation 15.4 t/s |
| 长 prompt（广义相对论），-n 256 | ✅ 成功，完整输出 |
| 无 segfault | ✅ |
| 无 CANN 设备错误 | ✅ |

---

## 调试方法论

本次调试过程中使用的关键技巧，供后续参考：

1. **GDB backtrace**：`gdb -batch -ex run -ex bt --args ./llama-cli ...` 快速获取崩溃调用栈。
2. **逐 op 同步定位异步错误**：CANN 操作是异步的，崩溃报告的位置往往不是真正出错的 op。在 `compute_forward` 入口添加 `aclrtSynchronizeStream()` 可以精确定位哪个 op 的执行导致了设备错误。
3. **对比短/长输入**：通过不同长度的 prompt 对比，发现错误与 tensor shape 中的 token 维度（ne[2]）相关。
4. **git stash 对比法**：暂存修改后用原始代码测试，确认 bug 是否为新引入。
5. **stderr 分离**：CANN 设备错误输出到 stderr，用 `2>/tmp/err.txt` 或 `2>&1` 分别捕获以避免遗漏。
