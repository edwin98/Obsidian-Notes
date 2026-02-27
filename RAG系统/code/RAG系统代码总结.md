# RAG 系统代码总结

## 项目结构

```
code/
├── config/             # 全局配置（pydantic-settings，支持环境变量覆盖）
├── models/             # Pydantic 数据模型
├── core/               # 抽象基类 + 核心算法
├── ingestion/          # 离线文档摄入流水线
├── retrieval/          # 三级混合检索
├── generation/         # 查询改写 + LLM 生成 + Token 预算管理
├── cache/              # Redis 缓存（精确 + 语义 + 会话）
├── tasks/              # Celery 异步任务
├── api/                # FastAPI 路由 + 依赖注入
├── data/               # 示例文档和测试 Query
├── tests/              # 单元测试 + E2E 测试
├── docker-compose.yml  # 基础设施一键启动
└── run_demo.py         # 系统一键演示脚本
```

---

## 核心数据模型（`models/schemas.py`）

| 模型 | 字段 | 用途 |
|---|---|---|
| `ChatRequest` | user_id, session_id, query, top_k | 问答请求，Pydantic 强校验 |
| `ChatResponse` | answer, citations, rewritten_queries, source | 问答响应 |
| `IngestRequest` | doc_id, doc_name, content, source_type | 文档摄入请求 |
| `ChunkMetadata` | chunk_id, doc_id, heading_path, node_type | chunk 元数据 |
| `DocumentChunk` | chunk_id, text, metadata, vector_384, vector_768, bm25_tokens | 核心数据单元 |
| `RetrievedChunk` | chunk, score, source | 检索结果 |

---

## 四大抽象基类（`core/abstractions.py`）

```python
class DocumentParser(ABC):
    def parse(self, raw_content: str, file_type: str) -> str: ...

class ChunkSplitter(ABC):
    def split(self, markdown_text: str, doc_id: str, doc_name: str) -> list[DocumentChunk]: ...

class PipelineRetriever(ABC):
    async def retrieve(self, query: str, rewritten_queries: list[str], top_k: int) -> list[RetrievedChunk]: ...

class LLMGenerator(ABC):
    async def generate_stream(self, query: str, context_chunks: list[DocumentChunk], history: ...) -> AsyncIterator[str]: ...
```

---

## 离线 Ingestion 流水线（写路径）

```
原始文档
  -> MarkdownDocumentParser     # 转换为统一 Markdown 格式
  -> DataCleaner                # 清洗噪声文本
  -> HierarchicalChunkSplitter  # 层次化切分（512~800 token/chunk，overlap 12%）
  -> KafkaChunkProducer         # 投递到 topic_text_slice
  -> KafkaChunkConsumer         # 消费消息（Demo 中同步，生产环境独立进程）
  -> Embedder                   # 生成 384维 + 768维双路向量
  -> jieba 分词                  # 生成 BM25 tokens
  -> BM25Engine（ES）            # 索引到 Elasticsearch
  -> MilvusVectorEngine         # 索引到 Milvus（两个 Collection）
  -> chunk_store（内存 dict）    # 保存完整文本供检索回填
```

**降级方案**：Kafka 不可用时自动调用 `ingest_document_direct()`，跳过消息队列直接处理。

---

## 在线 Query 链路（读路径）

```
用户 Query
  -> Redis 精确缓存（精确字符串匹配，命中直接返回，< 50ms）
  -> Redis 语义缓存（向量相似度 >= 0.92 视为命中）
  -> 获取会话历史（Redis，TTL 2小时）
  -> QueryRewriter（指代消解 + 问题扩展，产出多条改写 Query）
  -> ThreeLevelRetriever（三级检索）
      L1 粗筛：BM25 + Vector384 多路召回，各 top 1500，去重
      L2 精筛：RSF 动态权重融合 -> top 80
      L3 精排：CrossEncoder Rerank + 断崖截断 -> top 10
  -> TokenBudgetManager（控制传入 LLM 的 context 在 token 预算内）
  -> LLMGenerator 流式生成（AsyncIterator[str]）
  -> 写精确缓存 + 语义缓存 + 会话历史（TTL 24小时）
  -> Celery 异步触发会话摘要压缩
```

---

## RSF 核心算法（`core/algorithms.py`）

### 动态权重

```
alpha = 0.4 + 0.3 * sigmoid((token_len - 8) / 1.0)

短 query（token < 8） -> alpha -> 0.4  偏 BM25（精确关键词匹配）
长 query（token > 8） -> alpha -> 0.7  偏向量（语义匹配）
token = 8 时            alpha = 0.55  均衡
```

### 融合打分

```
综合分 = alpha * vector_norm + (1 - alpha) * bm25_norm
（两路分数均经 Min-Max 归一化到 [0, 1]）
```

### 断崖截断

相邻分差 > 0.8 且后一条绝对分 < 0.3 时截断，防止低质量结果混入。

---

## API 接口

| 方法 | 路径 | 功能 |
|---|---|---|
| GET | `/health` | 健康检查 + 已索引 chunk 数量 |
| POST | `/chat` | 非流式问答，返回完整 JSON |
| POST | `/chat/stream` | SSE 流式问答（打字机效果） |
| POST | `/ingest` | 文档摄入（解析 -> 向量化 -> 索引） |

**API 文档**：启动后访问 `http://localhost:8000/docs`

---

## 依赖注入（`api/dependencies.py`）

所有组件由 `Components` 容器统一管理，`get_components()` 返回全局单例。

```python
comp = get_components()

comp.ingestion_pipeline   # 摄入流水线
comp.retriever            # 三级检索器
comp.generator            # LLM 生成器
comp.rewriter             # 查询改写器
comp.redis_cache          # 缓存层
comp.bm25_engine          # BM25 引擎
comp.vector_engine        # 向量引擎
comp.embedder             # Embedding 工具
comp.chunk_store          # 内存 chunk 字典
```

连接失败不阻塞启动，所有外部依赖均有降级路径。

---

## 全局配置（`config/settings.py`）

所有参数可通过环境变量覆盖（前缀 `RAG_`）。

| 分类 | 参数 | 默认值 |
|---|---|---|
| 检索召回 | `level1_topk` | 1500 |
| 检索召回 | `level2_topk` | 80 |
| 检索召回 | `level3_topk` | 10 |
| RSF 算法 | `rsf_k` | 8（token 中心值）|
| RSF 算法 | `rsf_s` | 1.0（平滑系数）|
| Rerank | `rerank_diff_threshold` | 0.8 |
| Chunk | `chunk_leaf_min_tokens` | 512 |
| Chunk | `chunk_leaf_max_tokens` | 800 |
| Chunk | `chunk_overlap_ratio` | 0.12 |
| Token 预算 | `token_budget_total` | 4000 |
| 缓存 | `semantic_cache_threshold` | 0.92 |
| 缓存 | `session_ttl_seconds` | 7200（2小时）|
| 缓存 | `cache_ttl_seconds` | 86400（24小时）|

---

## 基础设施

| 服务 | 用途 | 默认地址 |
|---|---|---|
| Redis 7 | 精确缓存、语义缓存、会话历史、Celery Broker | localhost:6379 |
| Elasticsearch 8.12 | BM25 全文检索，index: `rag_chunks` | localhost:9200 |
| Milvus 2.3.7 | 向量检索（`rag_vectors_384` + `rag_vectors_768`）| localhost:19530 |
| Kafka 7.6 | chunk 消息队列（`topic_text_slice`）| localhost:9092 |
| Celery | 会话摘要异步压缩，Broker/Backend 均走 Redis | — |

---

## 启动方式

```bash
# 1. 启动基础设施
docker-compose up -d

# 2. 安装依赖
pip install -r requirements.txt

# 3. 一键演示（初始化 + 加载示例数据 + 演示查询 + 启动 API）
python run_demo.py

# 4. 运行测试
pytest tests/
```

---

## 依赖清单（`requirements.txt`）

```
fastapi, uvicorn, pydantic, pydantic-settings  # Web 框架
redis, elasticsearch, pymilvus                  # 数据存储
kafka-python                                    # 消息队列
celery[redis]                                   # 异步任务
rank-bm25                                       # BM25 检索
jieba, numpy                                    # NLP / 向量
httpx, pytest, pytest-asyncio                   # 测试
```
