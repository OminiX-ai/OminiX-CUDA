# CANN 后端错误排查方法论

> 基于 Qwen3.5-9B 在 Ascend 910B 上推理 segfault 的完整排查过程总结。
> 适用于 llama.cpp + CANN 后端的各类崩溃、挂起、设备错误问题。

---

## 一、排查总体思路

```
崩溃 → 定位阶段（加载/推理） → 定位层级（框架/后端/设备） → 定位具体 op → 分析根因 → 修复验证
```

核心原则：**逐层缩小范围，每次只改一个变量，用对比实验验证假设。**

---

## 二、工具箱

### 2.1 GDB 快速获取调用栈

```bash
gdb -batch -ex run -ex bt -ex quit --args ./build/bin/llama-cli \
  -m model.gguf -ngl 99 -p "Hello" -n 1 -s 42
```

- 无需交互，直接输出崩溃点的 backtrace。
- 快速区分崩溃发生在「模型加载」还是「推理计算」阶段。
- 关注调用栈中的关键函数名：
  - `supports_op` → op 支持性检查阶段
  - `compute_forward` → 实际计算阶段
  - `graph_reserve` / `sched_reserve` → 图预留阶段（模型加载时）

### 2.2 调度器 debug 模式

```bash
GGML_SCHED_DEBUG=1 ./build/bin/llama-cli ...   # 基本调度信息
GGML_SCHED_DEBUG=2 ./build/bin/llama-cli ...   # 详细调度信息（输出极大）
```

- 可以看到每个 op 被分配到哪个后端（CANN / CPU）。
- 如果某个 op 意外被分配到 CPU，说明 CANN 的 `supports_op` 未覆盖该 op。

### 2.3 在 compute_forward 中插入 op 跟踪日志

```cpp
static bool ggml_cann_compute_forward(..., struct ggml_tensor * dst) {
    fprintf(stderr, "CANN_OP: %s name=%s ne=[%ld,%ld,%ld,%ld]\n",
            ggml_op_name(dst->op), dst->name,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);
    switch (dst->op) { ... }
}
```

- 可以看到最后一个成功执行的 op 是什么。
- 崩溃点通常紧接在最后一条日志之后。
- **注意**：CANN 操作是异步的，日志中的「最后一个 op」不一定是真正出错的 op（见下节）。

### 2.4 逐 op 同步——精确定位异步错误（关键技巧）

CANN 的所有计算操作都是异步提交到 stream 的。当设备端发生错误时，报告的位置往往是后续某个 op 的同步点，而不是实际出错的 op。

**解决方法**：在每个 op 执行前插入 `aclrtSynchronizeStream`，强制同步，使错误在正确的位置暴露：

```cpp
static bool ggml_cann_compute_forward(..., struct ggml_tensor * dst) {
    fprintf(stderr, "CANN_OP: %s name=%s ne=[%ld,%ld,%ld,%ld]\n",
            ggml_op_name(dst->op), dst->name,
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3]);

    aclError sync_err = aclrtSynchronizeStream(ctx.stream());
    if (sync_err != 0) {
        fprintf(stderr, "CANN_SYNC_FAIL: at %s name=%s err=%d\n",
                ggml_op_name(dst->op), dst->name, (int)sync_err);
    }

    switch (dst->op) { ... }
}
```

**解读**：如果 `CANN_SYNC_FAIL` 出现在 op B 之前，说明前一个 op A 的异步执行出了问题。真正的出错 op 是 A，不是 B。

### 2.5 git stash 对比法

```bash
git stash                          # 暂存当前修改
cmake --build . && 运行测试         # 用原始代码测试
git stash pop                      # 恢复修改
```

- 确认 bug 是「新引入的」还是「原本就存在的」。
- 避免在错误的方向上浪费时间。

### 2.6 stderr 分离捕获

CANN 设备错误输出到 stderr，但 llama-cli 的正常输出也在 stderr。分离捕获：

```bash
# 方法 1：stderr 单独写文件
./llama-cli ... 2>/tmp/stderr.txt

# 方法 2：合并到同一文件（注意重定向顺序）
./llama-cli ... > /tmp/output.txt 2>&1    # ✅ 正确
./llama-cli ... 2>&1 > /tmp/output.txt    # ❌ 错误，stderr 仍然到终端

# 方法 3：过滤关键错误
./llama-cli ... 2>&1 | grep -E "CANN error|EZ9999|MTE|out of range"
```

### 2.7 输入长度对比法

```bash
# 短 prompt（少量 token）
echo "Hello" | ./llama-cli -m model.gguf -ngl 99 -n 8 -s 42

# 长 prompt（多 token）
echo "Explain the theory of general relativity in detail..." | ./llama-cli ...
```

- 不同输入长度会导致 tensor 的 ne[2]（token 维度）不同。
- 某些 CANN 内核仅在特定 tensor shape 下触发 bug。
- 通过对比可以缩小到「哪个维度的哪个大小」触发了问题。

---

## 三、常见问题模式与排查路径

### 模式 1：op 未实现 → CPU 回退 → 设备指针解引用崩溃

**特征**：
- segfault 发生在 CPU 端代码中
- 崩溃的 tensor 数据指针指向设备内存（非 host 可访问）
- `supports_op` 中找不到对应 op 的 case

**排查**：
```bash
# 检查 supports_op 是否覆盖了目标 op
grep "GGML_OP_XXX" ggml/src/ggml-cann/ggml-cann.cpp
# 检查 compute_forward 是否有对应 case
grep "GGML_OP_XXX" ggml/src/ggml-cann/ggml-cann.cpp
```

**修复**：在 CANN 后端实现该 op，参考 CUDA 实现和同文件中类似 op（如 ACC）的模式。

**注意**：在 `supports_op` 的 switch 中添加新 case 时，必须注意 fall-through 链。多个 case 共享一个 `return true` 是常见模式，不要在中间插入带大括号的 case 块。

### 模式 2：CANN 内核 bug → 设备端 MTE/AIC 错误

**特征**：
- 错误信息包含 `EZ9999`、`aivec error`、`MTE instruction`、`out of range`
- `CUSTOM_TILE_V3` 日志出现在崩溃附近（说明是 JIT 编译的自定义内核）
- 仅在特定 tensor shape 下触发，换 shape 就正常

**排查**：
1. 用「逐 op 同步」定位到具体的 op。
2. 在该 op 的实现函数中加 debug 日志，打印 src/dst 的 ne、nb、type、contiguity。
3. 对比成功和失败的 case，找出 tensor shape 差异。
4. 确认是 CANN SDK 的问题（非 llama.cpp 的问题）。

**修复策略**：
- **绕过有 bug 的 API**：用功能等价但实现路径不同的 API 替代。例如用 `aclnnInplaceCopy` + tensor 视图替代 `aclnnRepeat`。
- **降维处理**：将复杂操作拆解为多个简单操作的组合。
- **报告上游**：向华为 CANN 团队提交 bug report。

### 模式 3：推理挂起（无崩溃，无输出）

**特征**：
- 进程不退出，CPU 占用低
- `timeout` 后被 kill，backtrace 显示卡在条件变量等待

**排查**：
1. 检查是否是交互模式问题（`llama-cli` 默认进入对话模式等待输入）。
2. 用 `echo "prompt" | ./llama-cli ...` 管道方式提供输入。
3. 若仍挂起，用 `gdb -p <pid>` attach 查看所有线程状态。
4. 检查 CANN stream 是否有未完成的异步操作阻塞了后续计算。

---

## 四、CANN 后端开发备忘

### 关键文件

| 文件 | 用途 |
|------|------|
| `ggml/src/ggml-cann/ggml-cann.cpp` | 后端入口：`supports_op`、`compute_forward` |
| `ggml/src/ggml-cann/aclnn_ops.cpp` | 各 op 的具体实现 |
| `ggml/src/ggml-cann/aclnn_ops.h` | op 实现函数的声明 |
| `ggml/src/ggml-cann/acl_tensor.h` | tensor 创建工具（`ggml_cann_create_tensor` 等） |

### ggml ↔ ACL tensor 维度映射

ggml 使用 `[innermost → outermost]` 顺序（ne[0] 是最内层），ACL/PyTorch 使用 `[outermost → innermost]`。`ggml_cann_create_tensor` 内部会自动 `std::reverse` ne 和 stride 数组。

### 常用 ACL 操作模式

```cpp
// 1. 创建 ACL tensor（默认参数，直接从 ggml tensor 创建）
acl_tensor_ptr acl_t = ggml_cann_create_tensor(tensor);

// 2. 创建带自定义 shape/strides/offset 的 ACL tensor 视图
acl_tensor_ptr acl_view = ggml_cann_create_tensor(
    tensor, custom_ne, custom_nb, GGML_MAX_DIMS, ACL_FORMAT_ND, byte_offset);

// 3. 调用 ACL 算子（宏封装了 GetWorkspaceSize + Execute 两阶段调用）
GGML_CANN_CALL_ACLNN_OP(ctx, OpName, arg1, arg2, ...);

// 4. 设备内存拷贝
ACL_CHECK(aclrtMemcpyAsync(dst, dst_size, src, src_size,
                           ACL_MEMCPY_DEVICE_TO_DEVICE, ctx.stream()));

// 5. 同步等待（仅调试用，生产代码不要加）
ACL_CHECK(aclrtSynchronizeStream(ctx.stream()));
```

### 添加新 op 的检查清单

1. ☐ 在 `aclnn_ops.h` 中添加函数声明
2. ☐ 在 `aclnn_ops.cpp` 中实现函数（参考相似 op 的模式）
3. ☐ 在 `ggml-cann.cpp` 的 `compute_forward` 中添加 case
4. ☐ 在 `ggml-cann.cpp` 的 `supports_op` 中添加 case（**注意 fall-through**）
5. ☐ 编译测试：`cmake --build . -j$(nproc)`
6. ☐ 短 prompt 推理测试
7. ☐ 长 prompt 推理测试（不同 token 数触发不同 tensor shape）
8. ☐ 清除所有 debug 日志后再次编译验证

---

## 五、本次排查时间线

| 步骤 | 操作 | 结论 |
|------|------|------|
| 1 | GDB backtrace | 崩溃在 `supports_op`，模型加载阶段 |
| 2 | 检查代码 | SET case 插入位置破坏了 ACC→GROUP_NORM 的 fall-through |
| 3 | 修正 SET case 位置 | 模型加载通过，短 prompt 推理成功 |
| 4 | 长 prompt 测试 | 挂起后 abort，CANN 设备错误 |
| 5 | git stash 对比 | 干净代码在 supports_op 就崩了（更严重），排除新引入 |
| 6 | compute_forward 加 op 跟踪 | 最后一个 op 是 SUB，但 CANN 是异步的 |
| 7 | 逐 op 同步 | **精确定位到 REPEAT k_in-0 ne=[128,32,36,1]** |
| 8 | REPEAT 加 debug 日志 | 确认 src/dst shape、repeats、contiguity 均正确 |
| 9 | 短/长 prompt 对比 | ne[2]=2 成功，ne[2]=36 失败 → CANN 内核 shape 敏感 bug |
| 10 | 尝试 aclnnTile 替代 | 头文件不在标准路径，编译失败 |
| 11 | 用 InplaceCopy + 视图实现 REPEAT | ✅ 绕过 aclnnRepeat bug，长短 prompt 均成功 |
| 12 | 清除 debug 日志，最终验证 | ✅ 全部通过 |

**关键教训**：第 7 步的「逐 op 同步」是整个排查的转折点。没有这一步，会一直在错误的 op 附近打转。
