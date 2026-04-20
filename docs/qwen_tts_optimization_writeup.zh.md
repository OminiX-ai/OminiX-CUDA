# Ascend 910B4 上的 Qwen3-TTS:热路径原生化优化

**在固定硬件目标上,从 12.2 fps 提升至 33.8 fps(+177%),无需重写整个技术栈。**

本文档描述了 `OminiX-Ascend/tools/qwen_tts/` 中从 llama.cpp 初始基线到当前原生引擎的优化工作。本文以设计回顾的形式撰写,便于合同结束后对相关决策进行审计。

---

## 1. 第一性原理

> 定位热路径。只为目标硬件重写热路径本身。其他一切保持可移植。

这并非"为性能牺牲一切可移植性"。那种说法会导致两个糟糕的后果:(a) 维护者觉得有理由分叉一切,使 Ascend 专属的代码面积急剧膨胀;(b) 旁观者会将其解读为狂热主义。

真正的原则是外科手术式的:在自回归生成工作负载中,**80%-90% 的墙钟时间都消耗在一个狭窄的内层循环中**(transformer 前向 → codec head → 采样 → 下一步嵌入)。为固定目标原生优化这个循环。循环之外的一切——权重加载、分词、音频 I/O、API 服务器——都保留在可移植的 C++/ggml/标准库中。

回报:热循环的原生代码比通用框架(llama.cpp 的 ggml-cann 后端)快 2-3 倍,因为它可以利用目标专属的数据布局(FRACTAL_NZ)、融合注意力(FIAS)、厂商量化内核(aclnnWeightQuantBatchMatmul)以及图捕获。代价:那 10%-20% 的代码现在绑定在 Ascend 上。

在这条规则下,"我是否应该为目标硬件重写 X?"这个问题就变成了"X 是否占墙钟时间 >10%?"。如果否,保持其可移植。如果是,则在该孤立组件上权衡原生与通用方案。

---

## 2. 基线与目标

| Attribute | Value |
|---|---|
| Model | Qwen3-TTS(Talker 28 层 + Code Predictor 5 层 + decoder conv stack) |
| Hardware | 华为 Ascend 910B4,32 GB HBM,驱动 23.0.6 |
| Runtime | CANN 8.5.0(最初为 8.3.RC1;合同中期迁移) |
| Baseline path | llama.cpp + ggml-cann 后端,未修改 |
| Baseline throughput | **12.2 fps**(codec 帧每秒,长语句) |
| Contract goal | **≥ 25 fps 端到端,ASR 验证内容,人耳听感通过** |
| Final throughput | **33.8 fps**(W8 + TASK_QUEUE_ENABLE=2,CANN 8.5,cp_groups=8) |

这 2.77× 的提升是在未修改生成质量的前提下交付的——3 条基准语句的 ASR 转写结果在整个过程中保持一致。

---

## 3. 被否决的方案及原因

在决定采用原生引擎之前,我们考虑并否决了以下方案:

**(a) 停留在 ggml-cann(可移植),调优内核。** ggml-cann 后端通过通用张量抽象进行 JIT 调度。它无法消费 FRACTAL_NZ 权重,无法调度 aclnnFusedInferAttentionScoreV2,无法为 W8 使用 aclnnWeightQuantBatchMatmulV3。它在该工作负载上的上限约为 15 fps。

**(b) 等待厂商上游优化。** 华为的 llama.cpp-ggml-cann 分支按自己的节奏推进。等待不是计划。

**(c) 全面 Rust 重写。** CANN 的 C API(aclnn*、acl*)没有成熟的 Rust 绑定。Rust 移植将花费数周时间复现 C++ 中轻而易举的 FFI。换一件语言外衣的同样原生代码。

**(d) 用原生代码替换整条管线。** 会将音频 I/O、BPE 分词器、测试框架都 Ascend 化——这些都不在热路径上。维护负担巨大,性能收益为零。

**(e) 更激进的量化(Q4_K_M、Q5_K_M)。** GGUF K-quants 在 CPU/MLX 上得到支持,但没有直接的 Ascend 内核调度。通过 aclnnWeightQuantBatchMatmul 的 A16W8 才是原生量化路径;其他方案都需要加载时反量化,抵消提速收益。

---

## 4. 原生热路径策略

Qwen3-TTS 生成每个 codec 帧的热循环如下:

```
1. forward(hidden)     — 28-layer Talker transformer, dominant cost
2. codec_head(hidden)  — vocab_size projection
3. sample(logits)      — top-k/top-p with repetition penalty
4. predict_codes(h)    — CpCannEngine, 5-layer transformer, 15 codebooks
5. generation_embed    — sum 16 codec embeddings + text embed
                         → next step's hidden
```

步骤 1 和 4 占墙钟时间的 90%。我们用手写引擎替换了它们:

**`TalkerCannEngine`**(`tools/qwen_tts/talker_cann_engine.{h,cpp}`)
- 28 层 transformer,所有层都在 Ascend 上
- 直接调度 aclnn 算子:aclnnMm、aclnnFusedInferAttentionScoreV2、aclnnRotaryPositionEmbedding、aclnnRmsNorm、aclnnCast
- 拥有自己的 KV cache、逐层权重张量、rope 缓存
- 通过 dlsym 进行运行时符号加载(`cp_cann_symbols.{h,cpp}`),使 CANN 版本差异不会破坏构建

**`CpCannEngine`**(`tools/qwen_tts/cp_cann_engine.{h,cpp}`)
- 用于 15 码本预测的 5 层 code predictor
- 同样的 aclnn 调度模式
- 配备专用 CANN 后端的分块解码路径(commit `22e3e217`)

其他一切(BPE 分词器、speaker encoder、decoder conv stack、音频 I/O、stft、kissfft)都保留在可移植的 C++/ggml 上。speaker encoder 甚至被迁移到 CPU 上,因为 CANN 在小型 Conv1D/SE 算子上慢 2.7 倍——冷路径模块上通用方案反而更快。

---

## 5. 优化栈(按落地顺序)

下列每一层都在前面所有层之上**叠加**,并且**由环境变量控制**,以便在出现回归时无需重新构建即可禁用。

### Layer 0 — 原生 Talker + CpCannEngine(M1-M2.5)

| Stage | Throughput | Δ from baseline |
|---|---|---|
| Baseline ggml-cann (llama.cpp) | 12.2 fps | — |
| Native engine, iterative decode | 18.3 fps | +50% |
| Native engine, batched prefill (FIAS) | 23.2 fps | +90% |

**变更内容**:自回归循环现在端到端都停留在 aclnn 调用中。没有每步的 ggml 图重建,没有通用调度。

**关键 commit**:
- 初始原生 Talker:M1 系列
- 批量 prefill 修复:commit `5fcd1445`(在 Track D 确认 M5 已悄然修复 cos-sim=0.28 的回归后,将默认值翻转)

### Layer 1 — 从默认构建中剥离 llama.cpp(M3)

- `CMakeLists.txt` 暴露 `QWEN_TTS_LLAMA` 选项,默认 OFF
- `--llama_fallback` CLI 标志由该选项控制
- 节省约 200 MB 内存,在常规路径中移除 JIT 调度

**保留**:llama.cpp 构建仍然作为 xvec 模式的回退支持(MRoPE 4×pos 尚未进入原生 Talker)。需要 xvec 的用户使用 `-DQWEN_TTS_LLAMA=ON` 编译。

### Layer 2 — aclGraph 捕获(M4,已搁置)

尝试对 `forward_decode` 进行逐形状图捕获:

| Stage | Result |
|---|---|
| aclGraph capture on | 单语句上**慢 2.3×** |
| Reason | 一次性模式下捕获开销占主导 |

仅在**会话模式**(多语句摊薄捕获成本)下可行。不在 v1 默认路径中。保留在 `TALKER_CANN_GRAPH=1` 后,供未来会话 API 工作使用。

### Layer 3 — FRACTAL_NZ 权重布局(M5)

**它是什么**:Ascend 针对 matmul 右操作数的原生 2D 张量布局。将普通的行优先 `[K, N]` 权重转换为与 910 的 cube 单元对齐的分块 `[N/16, K/16, 16, 16]` 布局。

**我们如何使用**:权重加载时,通过 `mat2`(右操作数)描述符上的 `ACL_FORMAT_FRACTAL_NZ` 将每个 linear 层的权重转换为 NZ 格式。然后调用普通的 `aclnnMm`——它会检测 NZ 标签并调度快速内核。

| Stage | Throughput |
|---|---|
| NZ off (ND default) | 22.6 fps median |
| NZ on (mat2 as FRACTAL_NZ) | **25.9 fps** (+15%) |

**在 CANN 8.3 上发现的陷阱**:普通的 `aclnnMm` 无法正确消费带 NZ 标签的操作数。静默地生成乱码输出("哎呀!" / "嗯嗯嗯"之类)。需要迁移到 CANN 8.5,该版本修复了操作数重排序。

**由 `TALKER_NZ_WEIGHTS=1` 控制**。

### Stretch — A16W8 INT8 量化(S1)

**它是什么**:对 Q/K/V/O 和 FFN gate/up/down 权重进行逐输出通道对称 INT8 量化。

**校准**(离线,加载时):
```
for each linear layer:
    for each output channel c:
        scale_c = max(|W[c,:]|) / 127
        W_int8[c,:] = round(W[c,:] / scale_c).clip(-128, 127)
```

**调度**(解码调用点):
```
aclnnWeightQuantBatchMatmulV3(activation_f16, W_int8, scale_f16, ...)
```

| Stage | Throughput | Memory |
|---|---|---|
| NZ baseline | 29.7 fps | 6.88 GB |
| W8 on (`TALKER_W8_QUANT=1`) | **33.8 fps** (+14%) | 8.85 GB |

**内存回退**:W8 保留 F16 权重共同驻留(合同约束:不要触碰 `forward_prefill` 主体)。Prefill 仍在 F16 权重上调度普通的 aclnnMm。+28% VRAM。

**由 `TALKER_W8_QUANT=1` 控制**。

### 最后一环 — TASK_QUEUE_ENABLE=2

不是代码变更,而是 CANN 运行时环境变量。

**它是什么**:CANN 的 task queue 模式影响算子如何调度到 NPU。在 CANN 8.5 上,默认模式似乎相对 8.3 导致 27% 的回归。Track I 将其诊断为**冷缓存 / 环境漂移**,并非真正的回归。设置 `TASK_QUEUE_ENABLE=2` 使 8.5 恢复到 8.3 噪声范围内的水平。

**为何重要**:没有这一条,CANN 8.5 迁移看起来就是性能损失,NZ + W8 的延伸方案(需要 8.5)就会被否决。

---

## 6. 并行的质量轨道(与性能可分离)

性能工作期间出现了三个质量问题。由于它们与吞吐量没有交互,被作为独立轨道处理。

### (a)"Oh." 幻影前缀(commit `69c41884`)

每条语句都会产生一个前导的"Oh."音,ASR 将其标记为 2-3 个 token 的前缀垃圾。

**根因**:缺失 `tokenizer_config.json` 导致 BPE 分词器静默失败,将 `<|im_start|>` 视为原始文本并将其 BPE 成 2-3 个垃圾 token 前置在每条语句前。

**修复**:使 BPE 分词器在缺失 `tokenizer_config.json` 时**致命失败**。如果配置不存在,分词器拒绝运行,而不是静默产生垃圾。

### (b)开头喀哒声 / 前缀噪声(commit `2c2c3f6b`,本轮会话)

ICL 模式下,每条生成的语句在 ref/target 切点处都有 50-100 ms 的噪声爆破,可听为喀哒声。

**根因**:decoder 的卷积感受野跨越了 ref 到生成 codec 的接缝。生成部分的前 ~150 ms 带有稳态涟漪,fade mask 可以衰减但无法消除。

**我们最初尝试的(都是盲打)**:
1. 50 ms 线性淡入——仍有喀哒
2. 120 ms 线性淡入——RMS 降至 0.6×,仍可闻
3. 200 ms 三次方淡入——前 100 ms 降至 0.125×,160 ms 处仍有残留

**根因修复**:将切点边界前移 150 ms,使瞬态区落在被丢弃的部分。完全移除淡入。

对称应用于 ICL、xvec 和 customvoice 路径。

### (c)长克隆中的空洞 / 低频轰鸣("轰隆隆")

用户反馈 25 秒克隆输出听起来空洞,带有低频轰鸣。

**诊断**:频谱分析显示 100 Hz 以下能量比参考高 45-95 倍;2-5 kHz 存在感降低 3 倍。

**调查**:在全新 8 秒 ICL 上运行 F16(无 W8)对 W8。两者都干净。在 W8 上运行全新 24 秒 ICL:也干净。对 4 个非马云参考(doubao、luoxiang、ellen、maple)进行相同测试:全都干净。

**结论**:原始的轰鸣文件是内容特定 / 种子特定的,并非 W8 或长生成的系统性 bug。无需代码变更。

**这值得作为一次流程教训指出**:在编写任何"修复"之前,我们先跑了一个 A/B,排除了嫌疑原因。第一直觉("是 W8 量化损坏了 codec token")是错的。10 分钟的测量省下了一周盲目重写的时间。

---

## 7. 可移植性的代价

目前 Ascend 专属(针对 CANN 8.5 编译)的部分:

| Component | Lines | Reason |
|---|---|---|
| `TalkerCannEngine` | ~1,500 | 热路径;直接 aclnn 调度 |
| `CpCannEngine` | ~800 | 热路径;5 层 CP transformer |
| `cp_cann_symbols.{h,cpp}` | ~400 | 针对版本变体 aclnn 符号的 dlsym 加载器 |
| FRACTAL_NZ conversion | ~100 (inline) | Ascend 张量布局 |
| A16W8 calibration | ~200 | Ascend 量化内核调度 |
| aclGraph capture scaffolding | ~300 (parked) | Ascend 图模式 |

**原生代码面积合计:约 3,300 行**。这部分代码在没有重新实现的情况下,无法在 MLX、CUDA 或 CPU 上运行。

---

## 8. 保持可移植的部分

热循环之外的一切:

| Component | Shared with |
|---|---|
| GGUF 权重格式 | MLX、CPU、CUDA,任意 llama.cpp 下游 |
| BPE 分词器(`qwen_common`) | Qwen ASR、qwen_common 消费者 |
| 音频 I/O(`stft.cpp`、`audio_io.cpp`、kissfft) | 任意 TTS 下游 |
| Speaker encoder | 运行在 CPU(Ascend 在小算子上更慢) |
| 语音分词器 encoder/decoder | 可移植 ggml 路径 |
| 生成算法(自回归 + CP 采样 + EOS) | 在任意后端上相同算法 |
| C ABI(`qwen_tts_api.h`) | 任意宿主语言通过 FFI 接入 |
| ASR 内容门(`scripts/asr_quality_check.sh`) | 可移植 shell + mlx-whisper |
| DTW 质量测试框架(`scripts/dtw_vs_baseline.py`) | 可移植 Python |
| 合同 / 里程碑 / 测试结构 | 可移植的项目方法论 |

将此技术栈移植到新加速器只会触及 3,300 行原生引擎代码。其余约 80% 的代码库可以原样迁移。

---

## 9. 数据回顾

| Metric | Baseline | Final | Delta |
|---|---|---|---|
| Throughput (long utt) | 12.2 fps | 33.8 fps | **+177%** |
| Contract goal (§1) | ≥ 25 fps | 超额 +35% 达成 | — |
| VRAM (long utt, peak) | ~6.9 GB | ~8.8 GB | +28%(W8 门未达) |
| ASR content gate (3 utts) | PASS | PASS | — |
| Prefix noise | 50-100 ms 爆破 | 0 ms(150 ms 切点前移) | 已消除 |
| Rumble on long clones | 有反馈 | 新一轮无法复现 | 已关闭(内容特定) |
| Peak memory w/o W8 (NZ baseline) | — | 6.88 GB | — |
| Peak memory w/ W8 (current default) | — | 8.85 GB | — |

---

## 10. 为何此方案对我们有效

该策略在以下情形下有回报:
- **硬件目标固定**(单一部署平台——此处为 Ascend 910B4)。原生工程摊销到众多用户上,而非众多硬件上。
- **工作负载由热路径主导**(自回归生成 → 90% 在一个循环内)。使外科手术式优化具有高 ROI。
- **通用后端具有已知上限**(ggml-cann 约 15 fps)。原生是突破上限的唯一路径。
- **团队能够维持原生 C++ + 厂商 SDK 的专业能力**。CANN aclnn 不是一个轻松上手的 API——它需要持续投入。

不适用的场景:
- 多目标 SaaS(必须支持 CPU/CUDA/MPS/NPU 矩阵)。那 3,300 行原生代码会变成 3,300 × N。
- 延迟预算扁平的工作负载(没有可优化的热路径)。
- 摊销从未回本的短周期项目。
- 缺乏厂商 SDK 专家的小团队。

---

## 11. 经验教训

1. **通用后端存在上限**。当吞吐量重要且硬件固定时,走原生是值得的。在假设框架足够快之前,先对通用方案的上限做基准测试。

2. **环境变量控制的优化让回归可以安全上线**。每一个延伸项(W8、NZ、aclGraph)都在开关后面。如果回归落地,我们翻转开关即可回到上一个状态——不重建,不回滚。

3. **质量与性能是可分离的**。分词器致命修复和前缀裁剪修复与 Ascend 或 W8 毫无关系。将它们混为一谈会浪费数周追错根因。

4. **在"修复"前先测量**。空洞 / 轰鸣的调查差点演变成 decoder 端的重写,直到 10 分钟的频谱 A/B 证明嫌疑对象(W8)无罪。轰鸣是内容特定的,不是系统性的。

5. **带盖章的合同使并行代理工作成为可能**。`§5 里程碑 / §6 验收 / Verified-by` 结构意味着代理可以认领一项、证明其落地、盖章——无需 PM 瓶颈。

6. **根因修复胜过盲打**。淡入掩码的历史(50 ms 线性 → 120 ms 线性 → 200 ms 三次方)是一连串盲打,每一次都比上一次好,但没有一次是正确的。真正的修复是意识到切点边界位置不对。

---

## 12. 下一步(post-v1)

超出本合同的工作:
- **去重 W8 内存占用**(延伸项 S4——prefill 后释放 F16 权重)。
- **将 MRoPE 4×pos 移植到 `TalkerCannEngine`**,使 xvec 模式脱离 llama.cpp 回退路径(xvec 上 +35% fps)。
- **桥接到 OminiX-API**(见 `ASCEND_API_BRIDGE_CONTRACT.md`)——通过 C ABI 暴露原生引擎,使 HTTP 服务器将其作为库消费,而非子进程。
- **CannFusion 研究方向**——发射 AscendC 内核的 Rust DSL。长期目标;通向"每个模型只写一次,同时跑在 MLX 和 Ascend 上"的路径。

---

## Appendix A — Commit 线索(代表性)

| Commit | What |
|---|---|
| `69c41884` | BPE 分词器在缺失配置时致命失败(消除 "Oh." 前缀) |
| `5fcd1445` | 将批量 prefill 翻转为默认(16× prefill 提速) |
| `22e3e217` | 分块解码的专用 CANN 后端 |
| `f09ef578` | 文档化 TASK_QUEUE_ENABLE=2 缓解方案 |
| `f10508a4` | A16W8 逐通道 INT8 量化落地 |
| `2c2c3f6b` | 前缀喀哒声根因修复(150 ms 切点前移,移除三次方淡入) |
| `ae2832e5` | 高层 `qwen_tts_synthesize` C ABI(为 OminiX-API 桥接) |
| `d88f872d` | `libqwen_tts_api.so` 共享库目标 |

## Appendix B — 关键文件索引

| File | Purpose |
|---|---|
| `tools/qwen_tts/talker_cann_engine.{h,cpp}` | 原生 28 层 Talker |
| `tools/qwen_tts/cp_cann_engine.{h,cpp}` | 原生 5 层 Code Predictor |
| `tools/qwen_tts/cp_cann_symbols.{h,cpp}` | 面向 CANN 版本差异的 dlsym 加载器 |
| `tools/qwen_tts/qwen_tts.{cpp,h}` | 生成循环 + ICL/xvec/customvoice 调度 |
| `tools/qwen_tts/qwen_tts_api.{h,cpp}` | 供宿主语言 FFI 使用的稳定 C ABI |
| `tools/qwen_tts/qwen_tts_api.version` | 链接器版本脚本(符号导出控制) |
| `tools/qwen_common/bpe_tokenizer.{cpp,h}` | 可移植 BPE 分词器(与 ASR 共用) |
| `NATIVE_TTS_CONTRACT.md` | 交付合同,真相源 |
| `docs/gguf_quant_exploration.md` | K-quants 与 A16W8 权衡分析 |
