# Embedding 与 Reranker 微调全流程

> 本文以 ICT 无线领域知识问答 RAG 系统为背景，系统性地梳理 Embedding 模型和 Reranker 模型从数据集构建到上线的完整技术链路，重点阐述每个环节背后的工程逻辑与方案选型依据。

---

## 一、为什么需要微调

通用预训练的 Embedding 和 Reranker 在通用语料上表现优秀，但在专业垂直领域存在三类系统性偏差：

1. **词汇鸿沟**：无线通信领域充斥大量缩写和专有名词（CA、HARQ、PDCP、PRACH 等），通用模型在预训练阶段见过这些 token 但缺乏语义关联训练，对 "CA 和载波聚合是同一概念" 这类等价关系无法正确建模。

2. **检索目标漂移**：预训练 Embedding 的目标是 "语义相似"，而 RAG 检索的目标是 "查询-文档相关性"。一个问句和一个答案段落的语义距离可能很大（问题是陈述句改问句，答案是解释性段落），通用模型打分偏低，导致漏召回。

3. **负样本分布失配**：通用模型见过的负样本是随机文档，而在线检索时真正难以区分的是"同话题不同子问题"的文档（Hard Negative），通用模型不能有效区分这类样本。

微调的本质是**让模型的表示空间向领域查询-文档相关性任务对齐**，而非简单地增加领域词汇量。

---

## 二、Embedding 模型微调

### 2.1 架构选型：为什么用 Bi-Encoder

Embedding 模型承担 **离线入库 + 在线查询编码** 两个职责，对延迟和吞吐量要求极高。

| 架构 | 延迟 | 适用场景 |
|------|------|----------|
| Bi-Encoder | 查询和文档独立编码，O(1) 检索 | 大规模召回（百万级） |
| Cross-Encoder | 查询+文档联合编码，O(N) 检索 | 精排（百级） |

Bi-Encoder 的核心优势是**向量可以预先离线计算并缓存**，查询时只需对 Query 编码一次，然后做向量相似度检索（ANN），这是大规模 RAG 系统唯一可行的架构。代价是表达能力弱于 Cross-Encoder（缺少 token 级别的交互注意力）。

本系统选用 `gte-multilingual-base` 作为统一的 Bi-Encoder，它原生支持 **384 维**和 **768 维**两种输出维度：
- **384 维**：负责第一层粗筛，优先保证吞吐量
- **768 维**：结合用户 SAP 画像做第二层精筛，优先保证语义覆盖

### 2.2 基座选型

基于以下维度筛选基座：

- **多语言支持**：系统同时处理中文技术文档和英文协议标准
- **参数量与延迟**：线上推理延迟约束 < 50ms（单查询 P99）
- **预训练语料**：是否包含技术文档/代码相关语料，影响迁移学习的起点

候选基座对比：

| 模型 | 参数量 | 最大序列长度 | 多语言 | 说明 |
|------|--------|-------------|--------|------|
| BGE-M3 | 568M | 8192 | 是 | FlagEmbedding 系列，支持稠密+稀疏+多向量，但参数量偏大 |
| E5-mistral-7B | 7B | 32K | 部分 | 太重，不适合高并发在线服务 |
| multilingual-e5-large | 560M | 512 | 是 | 基于 XLM-R，多语言能力强，但不支持多维度输出 |
| **gte-multilingual-base** | **305M** | **8192** | **是** | **阿里 GTE 系列，推理快，原生支持 384/768 维输出** ✓ |

**选型结论**：选用 `gte-multilingual-base`（305M），它原生支持 384 维和 768 维两种输出维度，可用同一个模型同时满足粗筛和精筛两个通道，节省部署成本。

### 2.3 数据集构建

数据集构建是微调质量的天花板，模型再强也无法超越训练数据的质量上限。

#### 2.3.1 数据格式

标准的对比学习训练样本格式：

```
(query, positive_doc, [hard_negative_1, hard_negative_2, ...])
```

- `query`：用户的自然语言问题
- `positive_doc`：与 query 相关的正样本文档段落
- `hard_negative`：与 query 表面相似但实际不相关的文档段落

#### 2.3.2 正样本来源

**来源一：历史问答日志挖掘**

系统上线后积累的真实问答日志是最宝贵的数据源：
- 用户 Query + 系统返回的 Top-1 文档（被用户点击/采纳则标记为正样本）
- 筛选条件：用户对答案满意（通过显式点赞或对话持续深入来判断）
- 问题：冷启动阶段数据量不足，且存在选择偏置（模型自身的偏好被放大）

**来源二：Qwen2.5-72B 合成问答对（主要来源）**

对每个知识 Chunk，调用 **Qwen2.5-72B** 生成对应的问题，构成 `(合成问题, Chunk)` 正样本对，共生成 **75 万条** QA 语料，经数据清洗与质量过滤后 **48 万条**参与微调训练：

```python
PROMPT = """
你是一个无线通信专家。以下是一段技术文档：

{chunk_text}

请根据这段文档内容，生成 {n} 个不同角度、不同难度的技术问题。
要求：
1. 问题必须能从文档中找到答案
2. 包含概念型、原理型、故障排查型、参数配置型等多种问题类型
3. 问题应使用真实工程师会问的口语化表达
4. 避免直接复制文档中的句子作为问题

输出格式（JSON）：
[
  {{"question": "...", "type": "概念型"}},
  ...
]
"""
```

合成策略要点：
- 每个 Chunk 生成 3~5 个问题，覆盖不同提问视角；知识库共约 19 万个 Chunk，平均每 Chunk 产出 4 个问题，共生成 75 万条原始语料
- 区分 **概念型**（X是什么）、**原理型**（X如何工作）、**配置型**（X参数如何设置）、**故障型**（X场景如何排查）
- 对长文档 Chunk，先用 Qwen2.5-72B 提取关键事实点，再对每个事实点生成问题，避免问题集中在段落前几句
- 选用 72B 而非更小模型的原因：70B+ 量级模型在无线协议等高专业度领域生成的问题更自然、歧义更少，有效降低低质样本比例

**来源三：人工标注（少量高质量种子集）**

- 邀请无线领域工程师人工编写 200~500 个高质量 QA 对
- 这部分数据主要用于：评测集构建 + 对合成数据进行质量校准
- 不应作为主要训练来源，人工标注成本高且容易引入个人偏好偏置

#### 2.3.3 难负样本挖掘（核心环节）

Hard Negative 的质量直接决定模型的区分能力。以下是从易到难的三级负样本体系：

**第一级：BM25 负样本**

对每条 query，用 BM25 检索 Top-K 文档，剔除正样本后剩余的文档即为 BM25 负样本。

```python
from rank_bm25 import BM25Okapi

# 对query做BM25检索，取Top 50，排除正样本
bm25_candidates = bm25_index.get_top_n(query_tokens, corpus, n=50)
bm25_negatives = [doc for doc in bm25_candidates if doc not in positive_set][:5]
```

BM25 负样本的特点：词汇上与 query 高度重叠但语义不匹配，是中等难度负样本。

**第二级：语义相似负样本（ANN 负样本）**

用当前版本的 Embedding 模型对 query 做向量检索，取出语义相近但不相关的文档：

```python
# 用base模型编码query
query_vec = base_model.encode(query)

# 从向量库中检索Top 100
ann_candidates = vector_index.search(query_vec, top_k=100)

# 过滤掉正样本，剩余作为ANN负样本
ann_negatives = [doc for doc in ann_candidates if doc not in positive_set][:5]
```

这类负样本是模型最难区分的，因为它们在表示空间中距离正样本很近。

**第三级：LLM 生成混淆负样本**

针对特别关键或难区分的知识点，用 LLM 专门生成"看似相关实则无关"的干扰文档：

```
对于问题："HARQ 最大重传次数如何配置？"

正样本：【介绍 HARQ 参数 maxHARQ-Tx 配置方法的段落】

LLM 生成混淆负样本：
- 【介绍 ARQ 重传机制的段落】（相关但不相同的技术）
- 【介绍 HARQ 进程数配置的段落】（同技术但不同参数）
```

**负样本质量过滤**

避免假负样本（实际相关但被标记为负样本）污染训练集：

```python
# 用交叉编码器（Reranker）对候选负样本打分
# 得分超过阈值的认为是假负样本，从训练集中移除
reranker_scores = reranker.predict([(query, neg) for neg in candidates])
filtered_negatives = [
    neg for neg, score in zip(candidates, reranker_scores)
    if score < FALSE_NEGATIVE_THRESHOLD  # 通常设 0.3~0.4
]
```

#### 2.3.4 数据增强策略

**查询侧增强**

- **同义改写**：用 LLM 将原始问题改写为不同表达方式（正式/口语、长/短、有缩写/无缩写）
- **跨语言增强**：将中文 query 翻译为英文，或将英文标准文档段落翻译为中文，构建跨语言正样本对

**文档侧增强**

- **Chunk 边界扩展**：正样本 Chunk 随机扩展 ±1 个相邻句子，增加位置鲁棒性
- **摘要替换**：用 LLM 生成 Chunk 的摘要作为正样本变体

#### 2.3.5 数据质量控制

```
Qwen2.5-72B 合成数据（75 万条）
    ↓ 规则过滤（去重、去短文本、去乱码、JSON 格式异常）
约 68 万条
    ↓ 相关性过滤（用现有 Reranker 对 (query, pos) 打分，低于阈值 0.4 过滤）
约 54 万条
    ↓ 假负样本过滤（用 Reranker 对硬负样本重新打分，高于阈值 0.3 的移除）
约 48 万条（最终训练集，共过滤掉约 36% 的低质语料）
```

训练/验证/测试集分割：
- 按**文档**级别分割（而非按样本），避免同一文档的问题同时出现在训练集和测试集
- 比例：8:1:1

### 2.4 训练方法

#### 2.4.1 损失函数选择

**InfoNCE Loss（主训练损失）**

对比学习的核心，公式如下：

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(q, d^+) / \tau)}{\exp(\text{sim}(q, d^+) / \tau) + \sum_{i=1}^{K} \exp(\text{sim}(q, d_i^-) / \tau)}$$

其中：
- $\text{sim}(q, d)$ = 余弦相似度
- $\tau$ = 温度系数（通常 0.01~0.05）
- $K$ = 负样本数量

**温度系数的作用**：温度越小，对相似度的区分越敏感（梯度集中在难分样本上），但也越容易训练不稳定。通常从 0.05 开始，随训练进行逐步降低到 0.01。

**In-batch Negatives 策略**

同一 batch 内其他样本的正文档自动作为当前样本的负样本：

```python
# batch_size = 64 时，每个样本有 63 个 in-batch 负样本 + 若干 hard 负样本
# 这使得实际负样本数量 = batch_size - 1 + num_hard_negatives

# 实现时需要注意：避免同一文档的变体同时出现在一个 batch 中
# （否则会引入假负样本污染）
```

In-batch Negatives 的优势：无需显式构造负样本即可获得大量负例，batch size 越大效果越好（因此需要大显存 GPU 或梯度累积）。

**GradCache 技术**

当 batch size 受显存限制时，用 GradCache 实现"大 batch 效果"：

```python
# 先对所有样本做前向传播并缓存表示，再统一计算梯度
# 将有效 batch size 从 64 扩大到 512+
# 代价：显存翻倍（需缓存所有表示），但可以规避一次反向传播的显存峰值
```

#### 2.4.2 多阶段训练策略

**阶段一：大规模弱监督预热（Warmup）**

- 数据：仅用 LLM 合成的 (query, positive) 对，不加 Hard Negative
- 目标：让模型快速适应领域词汇分布
- 训练参数：较大 LR（2e-4），较少 epoch（1~2 轮）
- 原因：此阶段数据质量低但量大，先建立领域基本感知，避免后期 Hard Negative 将模型带入错误方向

**阶段二：Hard Negative 精调（Fine-grained Tuning）**

- 数据：带 BM25 + ANN 硬负样本的完整训练集
- 目标：提升模型对难以区分样本的判别能力
- 训练参数：较小 LR（1e-5~5e-5），更多 epoch（3~5 轮）
- 监控：每个 epoch 后在验证集评测 Recall@10，MRR@10，若连续两轮不提升则停止

**阶段三：Reranker 知识蒸馏**

用已微调好的 `gte-multilingual-reranker-0.3B`（Cross-Encoder）对每个训练样本的候选文档列表打分，将这些连续分数作为软标签蒸馏到 Bi-Encoder。相比硬标签（0/1），软标签保留了文档之间的**相对相关性梯度**，训练信号更丰富。

**蒸馏流程**

```
Step 1：固定 Reranker 权重，遍历训练集
Step 2：对每个 (query, [doc1, doc2, ..., docN]) 组合，让 Reranker 输出分数向量 s = [s1, s2, ..., sN]
Step 3：将 s 经 softmax(s/T) 得到教师软概率分布 p_teacher（T 为蒸馏温度，通常取 1~3）
Step 4：Bi-Encoder 输出候选文档相似度向量 e = [cos(q,d1), ..., cos(q,dN)]
Step 5：将 e 经 softmax(e/τ) 得到学生概率分布 p_student
Step 6：最小化 KL 散度：L_distill = KL(p_teacher || p_student)
```

**联合损失函数**

实际训练中将蒸馏损失与 InfoNCE 损失联合优化：

```python
# alpha 控制蒸馏强度，通常取 0.3~0.5
# InfoNCE 保证正负样本区分，KL 蒸馏注入相对排序知识

loss = (1 - alpha) * InfoNCE(query, pos, hard_negs) \
     + alpha * KL(p_teacher, p_student)
```

**关键工程细节**

- **Reranker 打分缓存**：Reranker 推理成本高，提前离线对所有训练样本打分并缓存为 `.jsonl`，训练时直接读取，不需要 Reranker 在线参与
- **候选文档组成**：正样本 1 个 + BM25 硬负样本 3 个 + ANN 硬负样本 3 个 + In-batch 随机负样本，共 7~15 个文档送给 Reranker 打分
- **蒸馏温度 T**：T 越大软标签越平滑（弱化边界模糊样本的影响），T=1 时等同于直接用原始分数；本系统取 T=2
- **为什么有效**：Cross-Encoder 做 token 级别全注意力，能感知否定词、数字精确匹配、条件依赖等细粒度关系，这些信号通过软标签隐式传递给 Bi-Encoder，使其表示空间更贴近实际相关性

**实际效果**：在验证集上，单纯 Hard Negative 精调后 Recall@10 = 82.3%，叠加蒸馏后提升至 86.7%，提升约 4.4 个百分点。

#### 2.4.3 训练硬件与超参数

**训练环境**：A30 × 2（单卡 24GB VRAM，双卡数据并行，CUDA 12.1）

双卡数据并行使每步的有效样本量翻倍，结合 GradCache 进一步扩大 In-batch Negatives。

| 超参数 | 实际取值 | 选择依据 |
|--------|----------|----------|
| per_device_train_batch_size | 64 | 单卡 A30 24GB 实际上限 |
| GradCache mini_batch_size | 16 | GradCache 分块大小，显存换 batch |
| 有效 batch_size（含双卡+GradCache） | 1024 | 双卡各 512，DDP 合并后 In-batch Negatives 达 1023 |
| learning_rate | 2e-5 | 阶段一用 2e-4，阶段二/三用 2e-5 |
| warmup_steps | 总步数的 10% | 保护预训练权重，避免初期梯度震荡 |
| max_seq_length（query） | 128 | Query 平均 20~50 token，128 足够 |
| max_seq_length（doc） | 512 | 知识 Chunk 切分粒度对齐 |
| temperature τ | 0.02 | InfoNCE 温度，验证集搜索后确定 |
| 蒸馏温度 T | 2 | Reranker 软标签平滑系数 |
| num_hard_negatives | 7 | BM25×3 + ANN×4，过多引入噪声 |
| fp16 | True | A30 支持 FP16，显存节省约 40% |
| gradient_checkpointing | False | 24GB 足够，无需牺牲速度 |

**各阶段训练时间（训练集 ~48 万条样本，A30×2）**

| 阶段 | epoch | 时间 | 说明 |
|------|-------|------|------|
| 阶段一：弱监督预热 | 1 | ~3 h | 数据量大，1 epoch 足以建立领域感知 |
| 阶段二：Hard Negative 精调 | 2 | ~6 h | 大数据集下 2 epoch 即可收敛，监控验证集早停 |
| 阶段三：Reranker 蒸馏 | 2 | ~5 h | 需提前离线生成软标签（~3 h 额外开销） |
| **合计** | — | **~14 h** | 含软标签生成约 17 h，实际训练持续约两周（含多轮迭代调参） |

> **注**：概要文档记录总训练周期为两周，包含多轮超参搜索、数据配比调整与模型筛选；上表为单次完整训练流程的纯计算耗时。

#### 2.4.4 多维度输出支持

`gte-multilingual-base` 原生支持 384 维和 768 维两种输出维度，通过在推理时指定 `output_dim` 参数即可切换，无需额外训练技巧：

```python
# 384 维（粗筛通道）
embeddings = model.encode(texts, output_dim=384)

# 768 维（精筛通道）
embeddings = model.encode(texts, output_dim=768)
```

这使得同一个微调后的模型可以直接服务 384 维粗筛和 768 维精筛两个通道，节省部署和维护成本。

### 2.5 评测体系

#### 2.5.1 离线评测指标

在独立的测试集上评测以下指标：

- **Recall@K**：前 K 个召回结果中正样本的比例（K=5, 10, 20）。最重要的指标，直接对应 RAG 的上游表现
- **MRR@K**（Mean Reciprocal Rank）：正样本排名倒数的均值，衡量正样本排名的平均位置
- **NDCG@K**：考虑位置权重的归一化折现累积增益，适合多正样本场景（详见下方算法）

**NDCG@K 算法**

DCG（Discounted Cumulative Gain）对排名靠前的相关文档给予更高权重：

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{rel_i} - 1}{\log_2(i+1)}$$

其中 $rel_i$ 为第 $i$ 位文档的相关性分数（二分类时取 0 或 1；多级标注时取 0~3 等整数）。

IDCG（Ideal DCG）为将所有相关文档排在最前时的 DCG 上界：

$$\text{IDCG@K} = \text{DCG@K（按相关性降序排列的理想结果）}$$

最终 NDCG 归一化到 [0, 1]：

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

**计算示例**（K=5，二分类，正样本排在第 1、3 位）：

```
位置 i :  1      2      3      4      5
rel_i  :  1      0      1      0      0
权重   :  1/log₂(2)=1.0  1/log₂(3)≈0.63  1/log₂(4)=0.5  ...

DCG@5  = (2¹-1)/log₂(2) + 0 + (2¹-1)/log₂(4) = 1.0 + 0.5 = 1.5
IDCG@5 = (2¹-1)/log₂(2) + (2¹-1)/log₂(3) = 1.0 + 0.63 = 1.63  （理想：正样本排1、2位）
NDCG@5 = 1.5 / 1.63 ≈ 0.920
```

**为什么 RAG 用 NDCG 而不只用 Recall**：Recall@K 只关心 "正样本是否召回"，不区分排第 1 还是排第 K；而 LLM 读取检索结果时会受位置偏差影响（前几个文档贡献更大），NDCG 的位置折现正好与此对齐。

**注意**：不要用 Cosine Similarity 的均值来评测，这和下游任务相关性弱；应用排序类指标。

#### 2.5.2 端到端评测

离线指标好不等于线上效果好，还需要接入 RAG 评测流水线：

```
Query → Embedding 检索 → Reranker 精排 → LLM 生成 → 与 Ground Truth 比对
```

端到端指标：答案 EM（Exact Match）、答案 F1、RAGAS 框架下的 Faithfulness 和 Answer Relevance。

---

## 三、Reranker 模型微调

### 3.1 架构选型：为什么用 Cross-Encoder

Reranker 在召回之后工作，候选集已经缩小到 50~200 个文档，允许更重的计算。

Cross-Encoder 的工作方式：将 `[CLS] query [SEP] document [SEP]` 拼接后送入 Transformer，在 token 级别做全局注意力，最后从 `[CLS]` 位置输出相关性分数。

相比 Bi-Encoder：
- 优势：query 和 document 之间有 **完整的 token 级别交互**，能捕捉细粒度语义关系（如数字匹配、否定关系、条件依赖等）
- 劣势：无法预先离线计算 document 表示，每次检索必须重新计算，O(N) 复杂度，只适用于候选集较小（<200）的精排场景

**为什么 Reranker 更准确**：以 "HARQ 最大重传次数是多少" 为例，Bi-Encoder 对 "HARQ 重传机制" 和 "HARQ 最大重传次数" 的文档表示都很接近 query，难以区分；但 Cross-Encoder 在做注意力时，"最大" 和 "次数" 这两个关键 token 能直接 attend 到文档中对应的数字和参数名，打分差异显著。

### 3.2 基座选型

Reranker 基座的选择逻辑：

| 模型 | 参数量 | 特点 |
|------|--------|------|
| bge-reranker-v2-m3 | 568M | FlagEmbedding 系列，中英双语强 |
| gte-multilingual-reranker-base | 305M | 阿里 GTE，推理快，多语言 |
| ms-marco-MiniLM-L-12-v2 | 33M | 极轻量，但仅英文 |
| Qwen2.5-0.5B + LM-head | 500M | LLM 作 Reranker 新趋势 |

**选型结论**：选用 `gte-multilingual-reranker-0.3B`（305M）。

选择 305M 而非更大模型的原因：Reranker 在在线链路中是阻塞点（必须同步等待结果才能继续），精排 50 个候选文档时 P99 延迟不能超过 200ms，305M 在 A10G GPU 上可以满足这个约束；568M 模型在同等硬件上大约慢 40%。

### 3.3 数据集构建

> **规模说明：48 万样本无法人工打分，全量使用 LLM 评分**
>
> Reranker 需要的是细粒度相关性分数（1~5 分），而非 Embedding 训练时简单的正/负二值标签。对 48 万条 QA 对及其候选文档逐一进行人工细粒度打分，按照每人每天 500 条的标注效率，需要约 **960 人天**，完全不现实。
>
> **实际做法**：以 Qwen2.5-72B（部署于内网推理服务）作为主评分器，对所有 (query, doc) 对进行异步批量打分（1~5 分制），单卡 A100 吞吐约 **800~1000 pair/min**，48 万对约需 8~10 小时。人工只参与两个环节：
> - **校准环节**（正式评分前）：抽取 200 条样本，人工与 LLM 分数对比，调整 Prompt 直到一致率 > 85%
> - **复核环节**（评分后）：对 LLM 置信度低（3 分附近 ±0.3）的边界样本（约占 5~8%）进行人工确认

Reranker 的训练数据格式：

```
正样本对：(query, relevant_doc)  → label = 1
负样本对：(query, irrelevant_doc) → label = 0
```

实际上更多使用**列表式**格式（一个 query 对应多个文档，各有分数）以支持 Listwise 训练。

#### 3.3.1 正样本构建

**与 Embedding 共用正样本集**，来源相同（日志挖掘 + Qwen2.5-72B 合成的 48 万条 QA 语料），但有一个重要补充：

**难正样本（Hard Positive）**

对于同一个 query，可能有多个相关程度不同的正样本：
- 强相关：直接回答问题的段落（score = 1.0）
- 中等相关：包含相关上下文但需要推理的段落（score = 0.7）
- 弱相关：属于同话题但不直接回答问题的段落（score = 0.4）

使用细粒度分级标注（而非二分类）可以让 Reranker 学到更精确的打分校准，而非只学到 "相关/不相关"。

构建方式（LLM 批量打分为主，人工复核为辅）：
```python
# 第一步：LLM 主评分（覆盖全量 48 万条）
# - 调用 Qwen2.5-72B，对每个 (query, positive_doc) 对打 1~5 分
# - 并发 32 路异步请求，吞吐 ~1000 pair/min，总耗时约 8 h
# - 输出：{query, doc, llm_score: 1~5, raw_response: "..."}
#
# 第二步：边界样本人工复核（约 5% 的量）
# - 筛选 llm_score ∈ [2.7, 3.3] 的样本（LLM 最不确定的区域）
# - 人工标注者二次判断，最终以人工分数为准
# - 专家间一致性（Cohen's Kappa）> 0.7 才认为该批次合格
#
# 第三步：分数归一化
# score_normalized = (llm_score - 1) / 4  →  映射到 [0, 1]
# 强相关（5分）→ 1.0, 中等（3分）→ 0.5, 不相关（1分）→ 0.0
```

#### 3.3.2 负样本构建

Reranker 的负样本需要比 Embedding 更 "难"，因为 Reranker 收到的候选已经经过 Embedding 粗筛，随机负样本在这里几乎没有训练价值。

**第一类：Embedding 模型的 False Positive（最有价值）**

用当前 Embedding 模型检索出排名靠前但实际不相关的文档，这些正是 Reranker 最需要区分的样本：

```python
# 流程：
# 1. 用 Embedding 模型对所有 query 检索 Top 50
# 2. 对 Top 50 做人工相关性判断（或用 LLM 打分）
# 3. 将 "Embedding 排名 Top 10 但实际不相关" 的文档作为硬负样本
```

这类负样本的本质是：**Embedding 层的错误召回即是 Reranker 的训练目标**。

**第二类：BM25 召回的干扰文档**

词汇重叠高但语义无关的文档，训练 Reranker 抵抗 "关键词匹配陷阱"。

例：
- Query: "如何解决 PDCP 层 SDU 丢包问题"
- 负样本: "PDCP 层 SDU 重排序窗口配置"（含大量相同关键词，但讨论的是不同问题）

**第三类：LLM 生成的混淆文档（针对高价值 query）**

对系统中频率最高的前 100 个 query，用 GPT-4 专门生成混淆文档：

```python
PROMPT = """
以下是一个技术问题和对应的正确答案段落：

问题：{query}
正确答案：{positive_doc}

请生成 3 个干扰性文档段落，要求：
1. 与正确答案在话题上高度相关
2. 但不能直接回答上述问题
3. 包含与正确答案部分重叠的技术术语
4. 每段约 100~200 字
"""
```

#### 3.3.3 数据平衡与采样策略

正负样本比例是影响 Reranker 性能的关键因素：

- **正负比 1:1**：召回率高，精确率低（模型倾向于打高分）
- **正负比 1:4~1:7**：更接近在线检索时的真实分布（通常 Top 50 中真正相关的只有 3~5 个）
- **动态采样**：训练初期正负比 1:2，后期逐步增大到 1:5，模拟课程学习

**列表采样**：每个 query 配 1 个正样本 + 4~6 个负样本组成一个训练 instance（Listwise 训练格式），比 Pairwise 格式（一次只看一对）训练效率高 3~5 倍。

#### 3.3.4 数据质量控制

Reranker 训练数据的一个特殊问题：**标注一致性**。

同一篇文档对不同问题的相关性判断，人工标注者的主观性差异很大。解决方案：

1. **标注指南**：制定详细的相关性定义标准（如"文档中是否包含可以直接回答问题的信息"）
2. **Calibration 轮次**：正式标注前，多个标注者共同标注 50 条数据，讨论分歧，统一标准
3. **LLM 辅助标注**：用 GPT-4 作为第一轮粗筛，人工只复核 LLM 不确定的样本（置信度低于 0.7 的）

LLM 标注 Prompt：
```python
ANNOTATION_PROMPT = """
以下是一个搜索查询和一个文档段落。请判断该文档段落对于回答该查询的相关程度。

查询：{query}

文档段落：
{document}

请按以下标准打分（1-5分）：
5分：文档直接、完整地回答了查询
4分：文档包含回答查询所需的关键信息，但不够完整
3分：文档与查询话题相关，但不能直接回答
2分：文档与查询有一定关联，但主要讨论其他内容
1分：文档与查询基本无关

只输出数字分数，不需要解释。
"""
```

### 3.4 训练方法

#### 3.4.1 微调策略：全量 SFT vs LoRA

对于 Cross-Encoder 架构的 Reranker，选择**全量参数微调（Full Parameter SFT）**而非 LoRA。

**为什么 Reranker 选全量微调：**

**1. 参数量小，全量可承受**

`gte-multilingual-reranker-0.3B` 只有 305M 参数，fp16 精度下仅占约 0.6 GB 显存。单张 A10G（24 GB）加载模型后，剩余空间完全容纳 batch_size=32 时的激活值（约 8~12 GB）和 AdamW 优化器状态（约 1.2 GB）。LoRA 节省显存的核心价值在此规模下不体现。

**2. 领域偏移幅度大，需要全层适配**

预训练 Reranker 在通用语料（MSMARCO、Wikipedia）上对齐，而目标域是 ICT 无线协议文档（5G NR、LTE、PDCP/RLC 技术规范）。这类领域偏移要求**每一层同时适应**：

- 底层 token embedding：专有术语（HARQ、MCS、PDCCH 等缩写）在通用词表中表示偏差
- 中间 attention 层：key/query/value 矩阵需要学会对技术关键词做精确 attend（如数值参数、条件约束）
- 顶层分类头：相关性打分的分布需要重新校准

LoRA 仅在高层 attention 的 K/Q 矩阵插入低秩适配器，底层特征提取几乎不变，容易出现"上层学会打分但底层特征错位"的情况。

**3. Listwise Loss 需要梯度全局传播**

Listwise Softmax Loss 的梯度需要同时流过所有 (query, doc) pair 的 token 表示。若只更新 LoRA 插入的低秩矩阵而无法修改 LayerNorm 参数和 FFN 权重，模型的打分分数分布会不稳定，后续 Platt Scaling 校准效果也会打折扣。

**4. 推理无额外开销**

全量微调直接输出干净的 checkpoint，部署时无需处理 LoRA adapter 合并逻辑，减少上线风险。

**何时改用 LoRA：**

| 场景          | 说明                                              |
| ----------- | ----------------------------------------------- |
| 基座换成 1B+ 模型 | 如 Qwen2.5-1.5B 作 LLM-as-Reranker，全量微调显存不足       |
| 多领域独立适配     | 共享一个基础 Reranker，每个业务线用独立 LoRA adapter，节省存储和推理资源 |
| 快速实验验证      | 验证某个数据构建方案是否有效，用 LoRA 降低每次实验成本（训练时间减半）          |
| 资源极度受限      | 只有单张 16 GB GPU，无法放下全量微调的优化器状态                   |

**全量 SFT vs LoRA 实测对比（305M 模型，2× A10G）：**

| 维度 | 全量 SFT | LoRA (r=16, α=32) |
|------|----------|--------------------|
| 可训练参数量 | 305M（100%） | ~5M（1.6%） |
| 单次训练显存峰值 | ~18 GB（双卡合计） | ~9 GB（单卡可训） |
| 验证集 NDCG@10 | **0.847** | 0.821（低约 3 个点） |
| 收敛所需 epoch 数 | 3~5 | 5~8 |
| 训练时间（3 epoch） | ~5 h | ~2.5 h |
| 部署复杂度 | 低（单一 checkpoint） | 略高（adapter 合并） |

结论：在 300M 规模模型 + 大领域偏移场景下，全量 SFT 3 个点的 NDCG 提升值得额外的约 2.5 小时训练时间。

---

#### 3.4.2 训练环境与框架配置

**硬件配置：**

| 资源 | 规格 | 说明 |
|------|------|------|
| GPU | 2× NVIDIA A10G（24 GB VRAM × 2） | 单机双卡，数据并行（DDP） |
| CPU | 32 核 Intel Xeon | DataLoader 预处理，多进程 tokenization |
| 内存 | 128 GB RAM | 训练集全量加载至内存，消除磁盘 I/O 瓶颈 |
| 存储 | 1 TB NVMe SSD | checkpoint 存储、tokenized 数据缓存 |

**训练框架栈：**

```
sentence-transformers      # 主训练框架：原生支持 Cross-Encoder、Listwise Loss
transformers（HuggingFace）# 模型加载、tokenizer、学习率调度
accelerate                 # 多 GPU DDP 封装，代码几乎不需改动
deepspeed ZeRO-1           # 优化器状态分片（可选，305M 模型不强依赖）
torch 2.1+ (torch.compile) # 图编译加速，约提升 15% 吞吐
wandb                      # 训练全程监控（loss、NDCG、学习率曲线）
```

**为什么选 sentence-transformers 而非手写训练循环：**

- 原生的 `CrossEncoder` 类封装了 `[CLS] query [SEP] doc [SEP]` 的输入格式构造
- 内置 `CERerankingEvaluator`，每个 epoch 结束自动在验证集上计算 NDCG@10、MAP
- 内置 `RankingLoss`（Listwise Softmax）实现经社区大量验证，无需自行调试数值稳定性
- 与 HuggingFace Hub 上的 reranker checkpoint 格式完全兼容

**训练时间实测（完整链路）：**

| 阶段 | 数据量 | 耗时 | 备注 |
|------|--------|------|------|
| 数据预处理（tokenization + 本地缓存） | 48 万条 query，每条 1 正 + 5 负 | ~1.5 h | 32 核 CPU 多进程并行 |
| 离线 Reranker 打分（用于 Embedding 蒸馏阶段生成软标签） | 同上 | ~3 h | 仅推理，无梯度，单卡 A10G |
| **Reranker 全量 SFT（3 epoch）** | ~280 万 query-doc pair | **~5 h** | 2× A10G，batch=32，seq_len=512 |
| 验证集评测（每 epoch 结束） | 5000 条样本 | ~8 min/次 | NDCG@10 + Precision@3 |
| 全流程总计（数据处理 → 最终 checkpoint） | — | **~10~11 h** | 含离线打分、训练、评测 |

**训练稳定性关键配置：**

- **梯度裁剪** `max_grad_norm=1.0`：Cross-Encoder 前期训练时，Softmax 对分数尺度极为敏感，梯度容易突然放大（表现为 loss 突然跳升），裁剪可有效抑制
- **早停**：监控验证集 NDCG@10，连续 2 个 epoch 无提升则停止，防止小数据过拟合
- **fp16 混合精度**（`torch.cuda.amp.GradScaler`）：训练速度提升约 40%，显存节省约 30%，Cross-Encoder 的 Softmax 分数计算对精度不敏感，fp16 损失可忽略
- **checkpoint 策略**：每 epoch 保存一次，保留最近 3 个，以验证集最佳 NDCG@10 的 checkpoint 作为最终部署版本

---

#### 3.4.3 损失函数

**方案一：Binary Cross-Entropy（BCE）—— Pointwise**

最简单的方案，将 Reranker 视为二分类器：

```python
loss = BCE(sigmoid(score), label)
# label ∈ {0, 1}
```

优点：实现简单，与下游逻辑回归概率估计兼容
缺点：没有显式建模文档之间的相对排序关系

**方案二：Pairwise Ranking Loss**

每次比较一对 (positive, negative)，要求 positive 得分高于 negative：

$$\mathcal{L} = \max(0, \text{margin} - \text{score}(q, d^+) + \text{score}(q, d^-))$$

优点：直接优化排序目标
缺点：没有利用整个候选列表的结构信息，训练效率低

**方案三：Listwise Softmax Loss（推荐）**

将一个 query 对应的所有文档视为一个列表，用 Softmax 对整个列表建模：

$$\mathcal{L} = -\sum_{i \in \text{positives}} \log \frac{\exp(\text{score}(q, d_i))}{\sum_j \exp(\text{score}(q, d_j))}$$

优点：
- 同时建模所有文档之间的相对顺序
- 利用了负样本的分数差异（不同难度的负样本有不同的惩罚）
- 训练效率高，一个 instance 等效于多个 pairwise 比较

**本系统选用 Listwise Softmax**，配合精细分级的正样本分数（而非二值标签），进一步提升模型对分数值的校准能力。

#### 3.4.4 Score Calibration（分数校准）

Reranker 输出的分数如果没有校准，仅能用于相对排序，不能用于设置阈值过滤。校准的目标是让分数具有概率含义（P(相关 | query, doc)）：

```python
# Platt Scaling：用 Logistic Regression 对 Reranker 原始分数做校准
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve

calibrator = LogisticRegression()
calibrator.fit(raw_scores.reshape(-1, 1), binary_labels)

# 校准后：score = 0.7 表示该文档有 70% 的概率与 query 相关
calibrated_score = calibrator.predict_proba(raw_score)[:, 1]
```

校准的工程价值：允许设置动态阈值，当最高分 < 0.3 时触发 "无相关文档" 的降级策略（返回兜底回答），而非强行用低相关文档生成答案导致幻觉。

#### 3.4.5 训练超参数

| 超参数 | 推荐值 | 选择依据 |
|--------|--------|----------|
| batch_size | 32~64（每个包含 1 正 + 5 负） | 受序列长度限制（query+doc 最长 512 token） |
| learning_rate | 5e-6~2e-5 | Reranker 比 Embedding 更难训练，LR 要更小 |
| max_seq_length | 512 | query + doc 拼接后的总长度 |
| num_epochs | 3~5 | 结合早停，在验证集 NDCG@10 上评估 |
| warmup_ratio | 0.1 | 同 Embedding |
| label_smoothing | 0.1 | 避免模型对正样本过度置信，提升泛化 |

#### 3.4.6 LLM-as-Reranker 的新趋势

近期出现了用指令微调的小 LLM 作 Reranker 的方案（如 RankLLaMA、RankZephyr），本质是将打分任务转化为生成任务：

```
输入：[系统提示] 判断以下文档是否与查询相关，回答"是"或"否"
      查询：{query}
      文档：{document}

输出：提取"是"或"否" token 的 logit 比值作为相关性分数
```

优势：
- 利用 LLM 的推理能力处理复杂相关性判断
- Zero-shot 迁移能力更强

劣势：
- 推理速度显著慢于 BERT 类 Cross-Encoder（生成式 vs 判别式）
- 当前阶段不适合对延迟敏感的在线链路

**当前阶段的建议**：在精排候选集较小（< 20）且有足够计算资源时，可以尝试 LLM-as-Reranker；在本系统的生产环境中，仍然推荐 BERT 类 Cross-Encoder。

### 3.5 评测体系

Reranker 的评测需要从两个层面进行：

**层面一：精排层面（Reranker 自身指标）**

输入：Embedding 召回的 Top 50 候选文档
输出：重排后的 Top 10

- NDCG@10：综合考虑排序质量和位置权重，精排场景最关键指标
- Precision@3：前 3 个文档的精确率（LLM 实际最依赖的是前 3 个文档）
- MAP（Mean Average Precision）：全列表的平均精确率，反映整体排序质量

**层面二：端到端影响**

- Reranker 前后的 Recall@10 变化（Reranker 可能因错误排序降低召回率）
- 答案生成质量：RAGAS Faithfulness 分数（衡量答案是否有文档支持）
- 误召回率下降：以前排名靠前但最终被 Reranker 降低排名的低相关文档比例

---

## 四、Embedding 与 Reranker 的协同训练

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

## 五、工程部署注意事项

### 5.1 Embedding 服务

- **批量推理**：离线入库时采用 batch_size=256 的批量编码，GPU 利用率接近 100%
- **动态批处理**：在线推理用异步批处理（等待 10ms 凑满一个 batch），将 QPS 提升 3~5 倍
- **模型量化**：INT8 量化可在损失 < 1% 性能的前提下将推理速度提升约 2 倍，内存占用减半
- **向量归一化**：推理前确认是否对输出向量做 L2 归一化（余弦相似度 = 归一化向量的点积）

### 5.2 Reranker 服务

- **候选集截断**：Reranker 输入候选数量设置为 50（而非 100+），在准确率和延迟间取平衡
- **序列长度控制**：超长文档截断时，保留前 128 token（通常是标题+导语，信息密度最高）+ 最后 128 token（通常是总结），而非简单截断前 512 token
- **缓存策略**：对高频 query（Top 1000）的 Reranker 结果做 Redis 缓存，TTL = 1 小时，命中率约 15%~20%

### 5.3 持续迭代机制

- **在线日志收集**：记录每次问答中 Embedding 召回的 Top 10、Reranker 重排后的 Top 5，以及用户对答案的满意度信号
- **A/B 测试框架**：新版模型先在 10% 流量上灰度，监控端到端指标（答案采纳率、追问率）
- **周期性再训练**：每月将新积累的日志数据补充到训练集，定期重训，防止模型随着知识库更新而退化

---

## 六、总结

| 维度    | Embedding                    | Reranker                |
| ----- | ---------------------------- | ----------------------- |
| 架构    | Bi-Encoder，独立编码              | Cross-Encoder，联合编码      |
| 核心优势  | 速度快，可预计算                     | 精度高，token 级交互           |
| 适用阶段  | 大规模召回（百万级）                   | 精排（50~200 候选）           |
| 训练损失  | InfoNCE + In-batch Negatives | Listwise Softmax        |
| 关键数据  | LLM 合成 QA 对 + ANN 硬负样本       | Embedding 假正例 + 细粒度分级标注 |
| 最难的问题 | 假负样本污染                       | 标注一致性与分数校准              |
| 迭代依赖  | 受 Reranker 蒸馏提升              | 依赖 Embedding 召回质量       |

Embedding 和 Reranker 微调的本质，是**将两个模型分别向召回目标和排序目标对齐，再通过协同迭代消除两者之间的分布鸿沟**。数据质量 > 模型架构 > 训练技巧，这一优先级在实践中反复得到验证。
