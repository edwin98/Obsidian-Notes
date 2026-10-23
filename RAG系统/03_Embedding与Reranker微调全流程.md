---
title: "检索模型微调总览与协同部署"
updated: 2026-10-23
aliases:
  - "03_Embedding与Reranker微调全流程"
---

# 检索模型微调总览与协同部署

这是检索模型微调的总入口，维护协同关系、部署和复盘；两类模型的完整配方分别维护。所有历史配置与效果需绑定实际数据和运行日志。

## 为什么需要领域微调

通用预训练的 Embedding 和 Reranker 在通用语料上表现优秀，但在专业垂直领域存在三类系统性偏差：

1. **词汇鸿沟**：无线通信领域充斥大量缩写和专有名词（CA、HARQ、PDCP、PRACH 等），通用模型在预训练阶段见过这些 token 但缺乏语义关联训练，对 "CA 和载波聚合是同一概念" 这类等价关系无法正确建模。

2. **检索目标漂移**：预训练 Embedding 的目标是 "语义相似"，而 RAG 检索的目标是 "查询-文档相关性"。一个问句和一个答案段落的语义距离可能很大（问题是陈述句改问句，答案是解释性段落），通用模型打分偏低，导致漏召回。

3. **负样本分布失配**：通用模型见过的负样本是随机文档，而在线检索时真正难以区分的是"同话题不同子问题"的文档（Hard Negative），通用模型不能有效区分这类样本。

微调的本质是**让模型的表示空间向领域查询-文档相关性任务对齐**，而非简单地增加领域词汇量。

---

## 分主题阅读

- [[RAG系统/Embedding领域微调]]：查询与文档表示、对比学习和双维训练。
- [[RAG系统/Reranker领域微调与标注]]：相关性标注、难负样本、精排训练与校准。
- [[RAG系统/项目事实与实现状态]]：统一数据、配置、checkpoint 与实验结果的证据状态。

## 协同训练

Embedding 和 Reranker 不应孤立微调，它们在 RAG 链路中是串联关系，存在**分布依赖**：

```
Reranker 的训练数据 = Embedding 召回的结果
如果 Embedding 更新，Reranker 的训练分布也应该随之更新
```

**推荐的迭代训练方式（Bootstrap 循环）**：

```
第 0 轮：用预训练模型做 Embedding 召回，构建 Reranker 训练集 → 训练 Reranker_v1
第 1 轮：微调 Embedding_v1，用 Reranker_v1 过滤假负样本
第 2 轮：用 Embedding_v1 重新做召回，更新 Reranker 训练集 → 训练 Reranker_v2
第 3 轮：用 Reranker_v2 蒸馏知识到 Embedding_v2
...
```

每轮迭代中，两个模型互相提升对方的训练数据质量，形成正向循环。实践中 2~3 轮迭代后效果趋于饱和。

---

## 工程部署

### 5.1 Embedding 服务

不同模型的向量空间不能默认混用；同一模型的 query/document 指令、pooling、维度和归一化也影响匹配。改变这些条件后，即使数据库能接收向量，旧索引的相关性也未必成立。

软件生态上，当前 Sentence Transformers 提供 `encode_query`/`encode_document` 和 `CrossEncoderTrainer` 等入口，6.x 还调整了部分打分、量化和处理器行为。新旧分数及量化产物不要默认视为完全等价；历史训练代码保留对应版本。[官方迁移说明](https://www.sbert.net/docs/migration_guide.html)、[发布记录](https://github.com/huggingface/sentence-transformers/releases)

索引与过滤参见 [[04_工程与框架/Milvus]]、[[Agent系统/记忆与上下文管理]]。

- **批量推理**：离线入库时采用 batch_size=256 的批量编码，GPU 利用率接近 100%
- **动态批处理**：在线推理用异步批处理（等待 10ms 凑满一个 batch），将 QPS 提升 3~5 倍
- **模型量化**：INT8 量化可在损失 < 1% 性能的前提下将推理速度提升约 2 倍，内存占用减半
- **向量归一化**：推理前确认是否对输出向量做 L2 归一化（余弦相似度 = 归一化向量的点积）

### 5.2 Reranker 服务

- **候选集截断**：Reranker 输入候选数量设置为 50（而非 100+），在准确率和延迟间取平衡
- **序列长度控制**：超长文档截断时，保留前 128 token（通常是标题+导语，信息密度最高）+ 最后 128 token（通常是总结），而非简单截断前 512 token
- **缓存候选（上线状态待核实）**：可对高频 query 的重排结果做缓存；key 需包含租户/授权范围、query、候选集与文档版本、模型版本。TTL 和命中率需实测，不能把历史设想中的 1 小时、15%～20% 当作已验证结果。

### 5.3 持续迭代机制

- **在线日志收集**：记录每次问答中 Embedding 召回的 Top 10、Reranker 重排后的 Top 5，以及用户对答案的满意度信号
- **A/B 测试框架**：新版模型先在 10% 流量上灰度，监控端到端指标（答案采纳率、追问率）
- **周期性再训练**：每月将新积累的日志数据补充到训练集，定期重训，防止模型随着知识库更新而退化

---

## 历史训练框架与配置边界

**Embedding 微调**：

基于 HuggingFace `transformers` + `sentence-transformers`，可用 GradCache 扩大对比损失覆盖的逻辑 batch。分块大小 16 只控制编码时的激活峰值；单卡逻辑 batch=64、双卡表示 all-gather 后是 128，不会自动变成 1024。要获得全局 1024，需每卡逻辑 batch=512 并跨卡聚合表示；实际运行配置待训练脚本与日志核实。普通梯度累积不扩大单次损失的负样本池。

损失函数为 InfoNCE（自定义实现），第三阶段叠加 Cross-Encoder 蒸馏（KL 散度损失），联合优化。

基座模型：`gte-multilingual-base`（305M，阿里 GTE 系列，HuggingFace 上直接加载）。

训练环境：A30 × 2，CUDA 12.1，FP16，PyTorch DDP 数据并行。

**Reranker 微调**：

同样基于 HuggingFace `transformers`，Cross-Encoder 架构，输入格式为 `[CLS] query [SEP] document [SEP]`，从 `[CLS]` 位置接全连接层输出相关性分数。

损失函数为 Binary Cross-Entropy（Pointwise）或 Listwise Softmax（Listwise 格式，每个 query 配 1 正 + 4~6 负构成一个训练实例）。

基座模型：`gte-multilingual-reranker-base`（305M），选型依据是 305M 在 A30 上精排 50 个候选文档的 P99 延迟可以控制在 200ms 内。

---

## 微调复盘与待验证改进

### 走过的弯路

**弯路 1：负样本质量差**

初期负样本全用随机采样（从语料库随机取文档），太容易区分，模型很快就能把正负样本分开，但对真实检索没有提升。效果验证时 Recall@10 比预期低 15 个点才发现问题。

**弯路 2：Reranker 输入长度没有控制**

Reranker 输入格式是 `[CLS] query [SEP] document [SEP]`，早期没有对 document 做截断，超过 512 token 的文档被直接截断，关键信息（往往在文档末尾）丢失，精排效果反而不如 Bi-Encoder。

**弯路 3：Listwise 训练格式引入太晚**

Reranker 最初用 Pointwise（Binary BCE，每个 query-doc 对独立打分），效果一般。后来换 Listwise（每个 query 配 1 正 + 5 负构成一个训练实例，Softmax 联合优化），Recall 提升明显，但切换成本较高。

### 如果重做

1. **先建评测集，再做任何训练**：按文档粒度切分 train/val/test，防止泄漏；把评测集当作不可动的"真相"，所有迭代决策都以它为准。
2. **Hard Negatives 从第一轮开始**：用 BM25 召回的 Top 50 去掉真正相关文档，剩下的作为 hard negatives；再加一轮用训练中途 checkpoint 挖 false negatives（被模型误判为正样本的负例）。
3. **Reranker 直接用 Listwise**：不走 Pointwise 这条弯路，训练格式从一开始就是 1 正 + 5 负的 Listwise Softmax。
4. **输入长度从一开始就控制**：document 最多保留 256~400 token（覆盖核心内容），query 留 64~128 token，总长控制在 512 token 以内。
5. **阶段式训练**：第一阶段用大量弱监督 QA 对预热，第二阶段再加 hard negatives 精调，第三阶段叠加 Cross-Encoder 蒸馏（用 Reranker 的软标签蒸馏 Bi-Encoder）。

---
