"""依赖注入：初始化并管理所有组件实例。"""

from __future__ import annotations

import logging

from cache.redis_cache import RedisCache
from config.settings import Settings
from generation.llm_generator import GeminiLLMGenerator, MockLLMGenerator
from generation.query_rewriter import QueryRewriter
from generation.token_budget import TokenBudgetManager
from ingestion.chunk_splitter import HierarchicalChunkSplitter
from ingestion.data_cleaner import DataCleaner
from ingestion.document_parser import MarkdownDocumentParser
from ingestion.embedder import Embedder
from ingestion.kafka_consumer import KafkaChunkConsumer
from ingestion.kafka_producer import KafkaChunkProducer
from ingestion.pipeline import IngestionPipeline
from models.schemas import DocumentChunk
from retrieval.bm25_engine import BM25Engine
from retrieval.pipeline_retriever import ThreeLevelRetriever
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_engine import MilvusVectorEngine

logger = logging.getLogger(__name__)


class Components:
    """所有组件的容器。"""

    def __init__(self):
        self.settings = Settings()

        # 存储层
        self.redis_cache = RedisCache(self.settings)
        self.bm25_engine = BM25Engine(self.settings)
        self.vector_engine = MilvusVectorEngine(self.settings)

        # 消息队列
        self.kafka_producer = KafkaChunkProducer(self.settings)
        self.kafka_consumer = KafkaChunkConsumer(self.settings)

        # 核心组件
        self.embedder = Embedder(self.settings)
        self.parser = MarkdownDocumentParser()
        self.cleaner = DataCleaner()
        self.splitter = HierarchicalChunkSplitter(self.settings)
        self.rewriter = QueryRewriter(self.settings)
        self.token_budget = TokenBudgetManager(
            total_budget=self.settings.token_budget_total,
            system_reserve=self.settings.system_prompt_token_reserve,
        )
        if self.settings.llm_provider == "gemini":
            self.generator = GeminiLLMGenerator(self.token_budget, self.settings)
            logger.info("[Init] LLM Provider: Gemini (%s)", self.settings.gemini_model)
        else:
            self.generator = MockLLMGenerator(self.token_budget)
            logger.info("[Init] LLM Provider: Mock")

        # chunk 全局存储
        self.chunk_store: dict[str, DocumentChunk] = {}

        # Reranker
        self.reranker = CrossEncoderReranker(self.chunk_store)

        # 三级检索器
        self.retriever = ThreeLevelRetriever(
            settings=self.settings,
            bm25_engine=self.bm25_engine,
            vector_engine=self.vector_engine,
            reranker=self.reranker,
            embedder=self.embedder,
            chunk_store=self.chunk_store,
        )

        # Ingestion 流水线
        self.ingestion_pipeline = IngestionPipeline(
            settings=self.settings,
            parser=self.parser,
            cleaner=self.cleaner,
            splitter=self.splitter,
            embedder=self.embedder,
            kafka_producer=self.kafka_producer,
            kafka_consumer=self.kafka_consumer,
            bm25_engine=self.bm25_engine,
            vector_engine=self.vector_engine,
        )
        # 共享 chunk_store 引用
        self.ingestion_pipeline.chunk_store = self.chunk_store


_components: Components | None = None


def init_components() -> Components:
    """初始化所有组件并建立连接。"""
    global _components
    _components = Components()

    # 尝试连接各基础设施（连接失败不阻塞启动，走降级路径）
    _try_connect("Redis", _components.redis_cache.connect)
    _try_connect("Elasticsearch", _components.bm25_engine.connect)
    _try_connect("Milvus", _components.vector_engine.connect)
    _try_connect("Kafka Producer", _components.kafka_producer.connect)
    _try_connect("Kafka Consumer", _components.kafka_consumer.connect)

    logger.info("[Init] 所有组件初始化完毕")
    return _components


def _try_connect(name: str, connect_fn) -> None:
    """尝试连接，失败只记录警告不阻塞。"""
    try:
        connect_fn()
        logger.info("[Init] %s 连接成功", name)
    except Exception as e:
        logger.warning("[Init] %s 连接失败（将使用降级模式）: %s", name, e)


def get_components() -> Components:
    """获取全局组件实例。"""
    global _components
    if _components is None:
        _components = init_components()
    return _components


def shutdown_components() -> None:
    """关闭所有连接。"""
    global _components
    if _components:
        _try_close("Kafka Producer", _components.kafka_producer.close)
        _try_close("Kafka Consumer", _components.kafka_consumer.close)
        _components = None
        logger.info("[Shutdown] 所有连接已关闭")


def _try_close(name: str, close_fn) -> None:
    try:
        close_fn()
    except Exception as e:
        logger.warning("[Shutdown] %s 关闭失败: %s", name, e)
