# Elasticsearch 在 RAG 项目中的应用

在向量检索盛行的 RAG 时代，Elasticsearch (ES) 依然是不可或缺的组件，主要承担基于关键词的**精确匹配（BM25）**任务，与向量检索形成双路互补。

## 1. 核心作用
*   **精确术语召回**：针对专业术语、型号（项如 `WS-C2960X-48FPS-L`）、特定错误码，解决向量模型在专有名词上“语义弥散”的问题。
*   **长文本全文检索**：在初筛阶段快速扫描全量知识库，提供高分候选集。
*   **高性能排序底座**：利用 ES 成熟的分布式能力实现 BM25 算法。

## 2. 核心能力
- **BM25 算法**：比传统的 TF-IDF 更稳健，能有效处理词频饱和问题。
- **倒排索引**：支持极高速的关键字查询。
- **自定义分词 (Analyzer)**：允许集成 Jieba 等分词器并挂载项目专属的**无线词典**，确保领域词汇不被切碎。
- **多字段联邦查询**：支持同时在标题、正文、作者等多个维度进行加权查询。

## 3. 常见用法
- **混合召回策略**：作为“粗筛”层的一员，与 384 维向量召回并行运行，合并结果送入下一层。
- **领域词典分词**：通过自定义 `analysis-ik` 或 `jieba` 插件，强制保留公司内部术语分词。
- **过滤召回 (Pre-Filtering)**：利用 ES 的 `filter` 子句进行密级或 PDU 的权限前置过滤，不参与打分，性能极佳。

## 4. 技术实现示例
```python
from elasticsearch import AsyncElasticsearch

# 初始化异步 ES 客户端
# 采用集群模式，配置重试机制
es_client = AsyncElasticsearch(
    ["http://es-node1:9200", "http://es-node2:9200"],
    retry_on_timeout=True,
    max_retries=3
)

async def retrieve_bm25(query_text: str, top_k: int = 1500):
    """
    基于非阻塞 Python 客户端执行高并发检索
    """
    # 构建轻量化 match 查询
    search_body = {
        "size": top_k,
        "query": {
            "match": {
                "content": {
                    "query": query_text,
                    "operator": "or",      # 平衡召回率
                    "minimum_should_match": "30%" # 过滤低噪声匹配
                }
            }
        },
        "_source": ["chunk_id", "text_snippet", "doc_name"] # 仅取关键字段
    }
    
    response = await es_client.search(index="rag_knowledge_bm25", body=search_body)
    return [hit["_source"] for hit in response["hits"]["hits"]]
```

## 5. 注意事项与坑点
- **深度分页问题**：ES 的 `size + from` 超过 10,000 会抛异常。在 RAG 粗筛中建议 `size` 限制在 2000 以内。
- **分词器一致性**：建索引时的分词器（Index Analyzer）必须与搜索时（Search Analyzer）保持一致，否则搜索结果会大幅偏离预期。
- **JVM 内存管理**：作为 Java 程序，ES 容易发生 Full GC。建议将堆内存设为物理内存的 50% 且不超过 32GB。
- **刷新频率 (Refresh Interval)**：默认 1s 刷新一次会导致入库性能下降。在大批量离线灌库时建议设置为 `-1` 或 `60s` 提高吞吐。
