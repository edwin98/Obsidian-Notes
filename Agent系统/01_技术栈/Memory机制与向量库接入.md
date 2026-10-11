---
tags:
  - Agent
  - 记忆系统
  - RAG
status: draft
---

# Memory 机制与向量库接入

记忆管理解决信息保存、检索和上下文组装问题。**短期与长期记忆主要按作用范围和生命周期区分，不能按是否写入数据库划分。** 向量库是检索工具，知识文档也不自动等于用户长期记忆。

## 1. 短期记忆：线程状态与上下文管理

LangChain/LangGraph 的短期记忆可以是 thread state，包含消息、任务进度及其他当前会话状态，通过 checkpointer 持久化和恢复。即使对话历史写入 Redis 或数据库，仍可属于短期记忆。[记忆概念](https://docs.langchain.com/oss/python/concepts/memory)

保存下来的历史不必全部送入模型。常见上下文管理方式包括：

- **滑动窗口**：按 token 预算保留近期内容；不能随意拆开工具调用与结果。
- **摘要**：将较早历史压缩为摘要，但可能损失细节或引入错误，重要原始信息仍需可查。
- **结构化状态**：保存用户约束、任务进度和待处理事项，避免仅靠对话文本恢复业务状态。
- **按需检索**：需要时再取回旧信息，控制上下文长度。

长上下文模型仍有输入上限、成本和有效利用范围，不能将某型号称作“无限窗口”。[短期记忆文档](https://docs.langchain.com/oss/python/langchain/short-term-memory)

## 2. 长期记忆：跨线程 Store 与知识检索

长期记忆可包括跨会话用户偏好、事实和已形成的经验。在当前 LangChain/LangGraph 体系中，Store 与 namespace 用于跨线程访问；namespace、身份和权限应与实际隔离需求一致。checkpointer 负责线程状态恢复，两者职责不同。[官方记忆概念](https://docs.langchain.com/oss/python/concepts/memory)

RAG 接入 Agent 常有两种方式：在模型调用前检索并注入上下文，或把检索暴露为工具，由 Agent 按需调用。前者便于固定流程，后者更灵活但需要评估工具选择和额外调用成本。记忆写入、更新、删除和来源可信度也需要明确策略。

## 3. 向量库选型与过滤检索

| 方案 | 定位与能力 | 选型边界 |
| --- | --- | --- |
| Chroma | 本地检索开发，以及托管 Cloud 路线 | Cloud Search API 的混合检索、RRF 和自定义排序目前仅限 Cloud，不能套用到全部本地实例。 |
| FAISS | 相似性搜索库，适合自行组装索引与服务 | 持久化、服务层和权限需应用管理；cuVS、过滤及 GPU 打包能力受索引、CUDA 和平台条件影响。 |
| Milvus | 面向向量与混合检索的数据库 | 稠密、稀疏、BM25 和过滤可组合，按规模、更新、资源与实际召回评估。 |
| Qdrant | 向量数据库，提供过滤与多租户能力 | 1.16 的 ACORN 可改善某些复杂过滤场景的召回，但增加搜索开销。 |
| pgvector | PostgreSQL 内的向量扩展 | 可复用关系数据与事务能力；ANN 与过滤的交互需要单独评测。 |
| Redis | 缓存、状态与搜索能力可组合 | Redis 8 整合 Search/JSON；向量索引与缓存各有内存、持久化和隔离要求。 |
| Elasticsearch | 全文、字段过滤、向量及混合检索 | 具体索引、量化和 GPU 建索引路线受版本、部署及许可约束。 |

来源：[Chroma Cloud](https://docs.trychroma.com/cloud/getting-started)、[Cloud Search API](https://docs.trychroma.com/cloud/search-api/overview)、[FAISS 发布记录](https://github.com/facebookresearch/faiss/releases)、[Qdrant 1.16](https://qdrant.tech/blog/qdrant-1.16.x/)、[Redis 8](https://redis.io/docs/latest/develop/whats-new/8-0/)。

### 3.1 ANN 与过滤为何可能导致结果不足？

若先近似扫描、再过滤，候选中只有少部分满足条件时，最终可能不足 Top-K。不同数据库和索引的过滤执行方式并不相同，不能只看未过滤情况下的召回率。

pgvector 0.8 起提供 iterative scan，可继续扫描以缓解结果不足；strict/relaxed order 和扫描预算体现排序、召回与成本的权衡。[pgvector 文档](https://github.com/pgvector/pgvector)

Qdrant 1.16 的 ACORN 可在过滤后探索更多图节点，改善部分多条件过滤场景，但并非无成本提高准确率。其分层多租户和磁盘能力也影响选型。[官方说明](https://qdrant.tech/blog/qdrant-1.16.x/)

数据库对照见 [[RAG系统/01_技术栈/Milvus]]、[[RAG系统/01_技术栈/Elasticsearch]]、[[RAG系统/01_技术栈/Redis]]。

## 4. 主动记忆与分层管理

MemGPT 等工作探索让 Agent 主动读写外部记忆，将当前上下文和外部存储分层管理。它不是物理上扩大模型上下文，而是通过检索、摘要和写入策略管理有限上下文。

效果应同时看任务成功、信息准确性、隔离、更新一致性和调用成本，不能用“记住得更多”替代完整评测。历史项目的数据库与记忆实现仍按当时方案解释。
