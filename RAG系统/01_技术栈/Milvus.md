# Milvus 十亿级向量引擎深度利用与优化

随着知识库规模的爆发式增长（达千万级 Chunk），系统全面引入企业级专属的高维度向量数据库 Milvus，以支撑超高并发与更低延迟语义检索。

## 1. 核心作用
*   **语义特征存储**：存储经过 Embedding 模型转换后的高维向量（384维与768维）。
*   **向量近似搜索 (ANN)**：在海量数据中实现毫秒级的语义相似度检索。
*   **混合查询底座**：结合标量过滤（Scalar Filtering）实现业务维度的精准约束召回。

## 2. 核心能力
- **高性能索引 (HNSW)**：支持磁盘/内存混合索引，在保障召回率的同时大幅降低检索延迟。
- **标量/向量混合检索**：允许在单次查询中同时指定向量相似度和元数据过滤条件（如 `pdu == '基站' && date > 2024`）。
- **动态 Schema**：支持灵活的元数据字段定义，方便注入文档路径、页码等业务标签。
- **高可用分布式架构**：读写分离设计，支持通过增加 Query Node 平滑扩展检索 QPS。

## 3. 常见用法与调优策略
- **索引优化 (HNSW)**：
  - `M` (Max Connection): 设置为 32，平衡内存开销与图连通性。
  - `efConstruction`: 设置为 256，提高建索引精细度。
  - `efSearch`: 生产环境动态设为 128~200，牺牲极少性能换取 99% 以上的高精度。
- **预加载策略**：使用 `collection.load()` 将热点数据驻留内存，规避首次检索磁盘 I/O 波动。
- **分区管理 (Partitions)**：根据数据产生周期或 PDU 维度划分 Partition，进一步缩小无效搜索范围。

## 4. 技术实现示例
```python
from pymilvus import Collection, connections

# 建立连接加载 Collection (已做内存加载进驻)
connections.connect("default", host="milvus-cluster", port="19530")
collection = Collection("rag_knowledge_vectors")

# 1. 配置 HNSW 获取精度
search_params = {
    "metric_type": "COSINE", # 领域模型依赖余弦距离度量
    "params": {"ef": 128}    # 图搜索范围控制
}

# 2. 标量前置过滤: 利用表达式将检索锁定在特定基站范围并过滤软删除知识
expr = f"pdu == '基站产品' and is_deleted == False"

# 3. 高性能异步推测批量搜素
results = collection.search(
    data=[user_query_embedding_768d],
    anns_field="vector_768",
    param=search_params,
    limit=80,      # Top 80送入后续 Rerank Cross-Encoder 管线
    expr=expr,
    output_fields=["chunk_id", "text_snippet", "doc_name"]
)
```

## 5. 注意事项与坑点
- **度量指标一致性**：Embedding 模型微调时若是用的余弦相似度，Milvus 检索必须指定 `metric_type: "COSINE"`，否则分数无意义。
- **过大的输出字段限制**：如果 `output_fields` 中包含巨大的 Text Chunk，会极大拖慢 RPC 响应速度。建议只存储 ID 和 Metadata，正文通过 Redis 或文件系统二次获取。
- **内存水位线**：HNSW 极其吃内存，需严密监控处理节点的 RAM 占用，防止 OOM 导致服务断流。
- **删除延迟**：Milvus 的删除是软删除，且 Compaction 动作较重。高频更新场景下应配合过滤标志位（如 `is_deleted`）使用。
