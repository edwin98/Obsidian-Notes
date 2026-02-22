from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """RAG 系统全局配置，所有参数均可通过环境变量覆盖。"""

    # ---- 基础设施连接 ----
    redis_url: str = "redis://localhost:6379/0"
    elasticsearch_url: str = "http://localhost:9200"
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    kafka_bootstrap_servers: str = "localhost:9092"

    # ---- RSF 算法参数 ----
    rsf_alpha_base: float = 0.4
    rsf_alpha_range: float = 0.3
    rsf_k: int = 8  # Token 中心值
    rsf_s: float = 1.0  # 平滑系数

    # ---- 三级召回参数 ----
    level1_topk: int = 1500
    level2_topk: int = 80
    level3_topk: int = 10

    # ---- Rerank 截断参数 ----
    rerank_diff_threshold: float = 0.8

    # ---- Chunk 参数 ----
    chunk_leaf_min_tokens: int = 512
    chunk_leaf_max_tokens: int = 800
    chunk_overlap_ratio: float = 0.12
    chunk_nonleaf_threshold: int = 2000

    # ---- Token 预算 ----
    token_budget_total: int = 4000
    system_prompt_token_reserve: int = 500

    # ---- 缓存 ----
    session_ttl_seconds: int = 7200
    cache_ttl_seconds: int = 86400
    semantic_cache_threshold: float = 0.92

    # ---- Embedding ----
    embedding_dim_light: int = 384
    embedding_dim_dense: int = 768

    # ---- Milvus Collection ----
    milvus_collection_384: str = "rag_vectors_384"
    milvus_collection_768: str = "rag_vectors_768"

    # ---- Elasticsearch Index ----
    es_index_name: str = "rag_chunks"

    # ---- Kafka Topics ----
    kafka_topic_text: str = "topic_text_slice"
    kafka_topic_gpu: str = "topic_gpu_task"

    # ---- Celery ----
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"

    model_config = {"env_prefix": "RAG_"}
