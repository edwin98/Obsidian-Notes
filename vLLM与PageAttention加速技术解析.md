---
title: vLLM与PagedAttention加速技术解析
date: 2026-02-24
aliases:
  - vLLM与PagedAttention加速技术解析
tags:
  - LLM
  - 推理加速
  - vLLM
  - PagedAttention
status: done
---

# vLLM与PagedAttention加速技术解析

LLM 推理的瓶颈随负载而变：prefill 常受计算吞吐约束，decode 常受显存带宽、KV Cache 容量和调度约束。伯克利团队提出的 vLLM 通过 **PagedAttention** 改善 KV Cache 管理，提高并发能力。

本文将**深入浅出**地解析这一革命性技术的底层逻辑。

---

## 1. 痛点：为什么LLM推理慢且耗显卡？

要理解 vLLM 为什么牛，首先要知道现在的 LLM 推理卡在哪里：**KV Cache**。

在自回归生成（Autoregressive Generation）过程中，每次生成一个新的 token，都需要依赖之前所有 token 的 Key (K) 和 Value (V) 向量。为了避免重复计算（即每次由于新 token 加入，还要重新算前面所有词的依赖关系），我们会把这些算好的 K 和 V 存放到显存中，这就叫 **KV Cache**。

但在 vLLM 出现之前（如早期的 HuggingFace Transformers 默认实现），**KV Cache 的内存管理非常糟糕：**
1. **预分配（Pre-allocation）**：系统在请求开始前，不知道模型最终会生成多少个字，所以通常按照模型允许的最大长度（如 2048 或 4096）直接“一刀切”地预先分配一大块连续显存。
2. **极高的显存浪费（内存碎片化）**：根据论文统计，由于预分配且生成长度不可预知，**高达 60%-80% 的 KV Cache 显存其实是被浪费掉的**（分为分配了没用到的内部碎片，以及太小凑不到一块的外部碎片）。这导致一块昂贵的显卡并发处理不了几个用户的请求，Batch Size 上不去，吞吐量极低。

---

## 2. 核心创新：PagedAttention（分页注意力机制）

vLLM 的杀手锏叫做 **PagedAttention**，它的灵感直接来源于**操作系统（OS）中的虚拟内存（Virtual Memory）和分页（Paging）机制**。

### 2.1 从“操作系统”偷师
在操作系统里，如果要分配一块很大的连续内存，但物理内存全是一块块散落的碎片怎么办？
答案是：**将逻辑内存和物理内存解耦，分成固定大小的“页（Page）”，依靠页表（Page Table）进行映射。逻辑上保持连续，但在物理上它们可以散落在任意不连续的地方。**

PagedAttention 用同样的思路彻底改造了 KV Cache：
- 它将每个序列（Sequence）的 KV Cache 划分为固定大小的 **块（Blocks / Pages）**。
- 每个 Block 可以存放固定数量的 token（连同它们的 K、V 向量），比如设定 16 个 token 为一个 Block。
- 在逻辑上，一句话生成的 token 是串联的；**但在物理显存页上，它们可以散落在任意一段空闲的物理显存块里。**每次计算 Attention 时，只需通过一个类似“页表”的结构精准查找到对应的物理块。

### 2.2 PagedAttention 带来的三大好处
1. **按需分配**：序列增长时逐块分配 KV Cache。减少预留与碎片浪费，但最后一个未填满的 block 仍有内部碎片，每个序列浪费不到一个 block。
2. **提高 KV Cache 利用效率**：接近零的是 KV Cache 管理浪费，并非整张 GPU 显存利用率接近 100%。论文中的 2～4 倍吞吐提升针对指定模型、负载及 FasterTransformer/Orca 基线；部署收益需实测。[PagedAttention 原论文](https://arxiv.org/abs/2309.06180)。
3. **强大的内存共享（Memory Sharing）**：在 Beam Search 解码、或多个用户使用相同的 System Prompt（复杂前缀）的场景下，可以做到逻辑层面独立的请求，在底层**共享相同的物理资源块**。进一步成倍压缩显存占用。

---

## 3. vLLM 的其他核心加速生态与技术

现在的 vLLM 已经进化为一个功能完备的“六边形战士”，除了 PagedAttention，它还整合了业内一众顶尖的优化方案：

### 3.1 连续批处理（Continuous Batching / In-Flight Batching）
- **传统静态批处理（Static Batching）**：系统会攒齐一波请求（比如 8 个）一起送去推理，其中如果有一个请求文思泉涌生成得特别长，其他 7 个早早生成完的请求就只能干等着，白白占用计算配额。
- **vLLM 的细粒度批处理（Continuous Batching）**：系统变成了流水线。在每一次迭代（Step）后，只要有请求完成，立刻把它踢出批次池，并把正在排队的新请求无缝塞进来。始终保持 GPU 处于“极度忙碌”和满载状态。

### 3.2 算子融合与底层优化 (如 FlashAttention / FlashInfer)
vLLM 高度集成了 **FlashAttention** 等更底层的算子：
- 当大模型的计算从“算力瓶颈”（Compute Bound）转向“内存带宽瓶颈”（Memory Bound）时，FlashAttention 家族利用块切分与合并的核心理念，将计算尽可能收拢在了 GPU 极其快速但容量极小的 SRAM 内。大量减少了往主显存 (HBM) 搬运数据的次数，让 Attention 算得更快。
- "PagedAttention 管内存组织，FlashAttention 管极致计算"，强强联手。

### 3.3 模型量化（Quantization）的全面支持
它原生且极度优化了各种轻量化方案（如 AWQ, GPTQ, FP8 甚至各类 INT4 量化）：
- 将模型庞杂的权重从 FP16/BF16 压缩。这不仅让你的小显卡能装下更大的模型，而且直接减轻了计算时往 HBM 拿取参数的显存带宽压力，使得单 Token 的推理速度（TPOT）得到跨越式提升。

### 3.4 分布式并行推理支持 (Tensor Parallelism & Pipeline Parallelism)
针对单卡根本塞不下的千亿级巨模型（如 Llama3-70B等），vLLM 使用了成熟的张量并行策略（TP）：
- 可以把庞大的计算力拆分到多张 GPU 卡上同时运算，并通过底层高速网络（如 NVLink）聚拢计算结果，完成多卡并联的性能释放。最新版vLLM也可以通过流水线并行跨多台机器甚至多个节点运行。

---

## 4. 总结升华

用一句话概括：如果说大语言模型是一个无情吞噬内存的怪兽，那么 **PagedAttention 就是一位极简风的超级收纳师**。

PagedAttention 将连续的逻辑序列映射到分散的物理 KV blocks，连续批处理则让已完成请求及时退出、新请求及时加入。两者提升了显存使用和服务调度效率，实际吞吐、时延及量化收益需要按工作负载测试。
