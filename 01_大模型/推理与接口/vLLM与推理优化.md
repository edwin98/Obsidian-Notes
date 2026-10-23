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

# vLLM 与 PagedAttention 加速技术解析

vLLM 是大模型推理与服务框架。理解其性能时，要先区分 **Prefill**（处理输入、构建状态）和 **Decode**（逐 token 生成）。长提示的 Prefill 常受计算限制，低并发 Decode 常受权重与状态读取的带宽限制；实际瓶颈还取决于批量、模型和硬件。当前 V1 重构了调度、KV 管理与执行架构。[V1 文档](https://docs.vllm.ai/en/stable/usage/v1_guide/)

## 1. KV Cache 为什么影响推理效率？

标准自回归注意力会缓存历史 token 的 Key/Value，避免每生成一个 token 都重算历史投影。缓存随序列和并发请求增长，常常限制可同时运行的请求数。

早期按最大长度预留连续缓存，会产生预留浪费与碎片。PagedAttention 论文中报告的内存浪费和吞吐倍数对应其测试基线，不能视为所有当前部署的固定比例。KV 缓存也不是模型权重；两者要分别计入显存。

## 2. PagedAttention：分块管理 KV

PagedAttention 借鉴分页思想：将序列的逻辑 KV 分成固定大小的块，通过块表映射到物理缓存。请求增长时按需分配，无须提前预留完整的最大长度；请求结束后回收块。

这带来更灵活的缓存分配，并可支持相同前缀、并行采样等场景的块共享与写时复制。末块仍可能有内部碎片，块表与管理也有开销。分页改变的是内存组织，不会消除注意力计算。

## 3. 调度、缓存复用与解码机制

| 机制 | 主要作用 | 收益条件或代价 |
| --- | --- | --- |
| Continuous Batching | 在迭代间加入新请求、移除已完成请求 | 提高调度利用率；收益依赖到达速率、请求长度和负载。 |
| Prefix Caching | 复用相同前缀已计算的 KV | 命中率决定收益，不缓存后续完整回答。 |
| Chunked Prefill | 把长输入分块，与 Decode 协调调度 | token 预算影响首 token 延迟、生成间隔与吞吐。 |
| Speculative Decoding | 候选生成器提出 token，目标模型验证 | 接受率、候选成本和负载决定是否加速；高负载不一定受益。 |
| Prefill/Decode 分离 | 分别调度或扩展提示处理和逐 token 生成 | 减少长提示对 Decode 的干扰，同时增加 KV 传输与部署复杂度。 |

这些机制可以组合，但解决的瓶颈不同。[优化文档](https://docs.vllm.ai/en/stable/configuration/optimization/)、[推测解码资料](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)

### 3.1 P/D 分离与有效吞吐

2026-09 的官方指南讨论了 Prefill/Decode 分离，以及将 tokenization、输出解析等 CPU 工作拆离 GPU 节点。P/D 交接需要搬运 KV；网络或互连较慢时，首 token 延迟可能增加。

因此评测应同时记录 TTFT（首 token 延迟）、ITL（生成 token 间隔）和满足延迟目标的有效吞吐 goodput，不能只比较总 tokens/s，也不能将一组硬件的性能倍数推广到其他部署。[分离式推理指南](https://vllm.ai/blog/2026-09-29-disaggregated-serving-guide)

## 4. 算子、精度与分布式执行

### 4.1 FlashAttention 与 FlashInfer

算子优化通过分块、融合和减少显存读写提高执行效率，与 PagedAttention 的缓存管理职责不同。FlashAttention 官方已有 FA3 和基于 CuTe DSL 的 FA4 路线，面向 Hopper/Blackwell 等硬件；版本、数据类型和 GPU 架构要匹配，H100/B200 的收益不能直接套到 A100/A10。[官方仓库](https://github.com/Dao-AILab/flash-attention)

### 4.2 量化与并行

量化可降低权重或 KV 的存储与带宽开销，实际速度还受反量化、算子支持、批量和精度损失影响，不能保证所有负载都更快。权重量化与 KV 量化要分别说明。

Tensor Parallelism 分割层内计算，Pipeline Parallelism 分割层；跨卡通信、流水线气泡和负载均衡都影响效果。例如 Llama 3 70B 是 700 亿参数模型，能否部署取决于精度、缓存预算和并行配置。

### 4.3 混合模型与结构化输出

混合注意力模型同时包含历史 KV 与循环状态，显存需按层类型拆分，详见 [[01_大模型/模型结构/MoE混合专家模型]]。

当前结构化输出接口使用 `structured_outputs`；旧 `guided_json` 等字段已在 v0.12 移除。接口名称变化与约束解码原理分别理解，详见 [[01_大模型/推理与接口/结构化输出与校验]]。[当前接口文档](https://docs.vllm.ai/en/stable/features/structured_outputs/)

## 5. 性能分析与选型

先确定权重、缓存、算子、调度或通信中的主要瓶颈，再比较候选机制。记录模型、精度、硬件、输入/输出长度、并发和延迟分位数，性能结论才有明确适用范围。历史项目的配置和数据仍按当时条件解释。
