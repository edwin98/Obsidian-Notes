"""离线知识摄入流水线：完整的文档解析 -> 清洗 -> 切分 -> Kafka -> Embedding -> 索引。"""

from __future__ import annotations

import logging

from config.settings import Settings
from ingestion.chunk_splitter import HierarchicalChunkSplitter
from ingestion.data_cleaner import DataCleaner
from ingestion.document_parser import MarkdownDocumentParser
from ingestion.embedder import Embedder
from ingestion.kafka_producer import KafkaChunkProducer
from ingestion.kafka_consumer import KafkaChunkConsumer
from models.schemas import DocumentChunk
from retrieval.bm25_engine import BM25Engine
from retrieval.vector_engine import MilvusVectorEngine

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """离线摄入流水线编排。

    流程：原始文档 -> Parser -> Cleaner -> ChunkSplitter
          -> Kafka Producer -> Kafka Consumer
          -> Embedder -> BM25 索引 + Milvus 向量索引
    """

    def __init__(
        self,
        settings: Settings,
        parser: MarkdownDocumentParser,
        cleaner: DataCleaner,
        splitter: HierarchicalChunkSplitter,
        embedder: Embedder,
        kafka_producer: KafkaChunkProducer,
        kafka_consumer: KafkaChunkConsumer,
        bm25_engine: BM25Engine,
        vector_engine: MilvusVectorEngine,
    ):
        self.settings = settings
        self.parser = parser
        self.cleaner = cleaner
        self.splitter = splitter
        self.embedder = embedder
        self.kafka_producer = kafka_producer
        self.kafka_consumer = kafka_consumer
        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine

        # chunk_id -> DocumentChunk 的全局存储（供检索后取回完整文本）
        self.chunk_store: dict[str, DocumentChunk] = {}

    def ingest_document(
        self,
        doc_id: str,
        doc_name: str,
        raw_content: str,
        file_type: str = "markdown",
    ) -> list[DocumentChunk]:
        """同步执行完整的 ingestion 流程。"""
        logger.info("[Ingestion] 开始处理文档: %s (%s)", doc_name, doc_id)

        # 1. 解析为 Markdown
        markdown = self.parser.parse(raw_content, file_type)
        logger.info("[Ingestion] 解析完成: %s", doc_name)

        # 2. 清洗
        cleaned = self.cleaner.clean(markdown)
        logger.info("[Ingestion] 清洗完成")

        # 3. 层次化切分
        chunks = self.splitter.split(cleaned, doc_id, doc_name)
        logger.info("[Ingestion] 切分为 %d 个 chunk", len(chunks))

        # 4. 投递到 Kafka
        for chunk in chunks:
            self.kafka_producer.send_chunk(chunk.model_dump())
        self.kafka_producer.flush()
        logger.info("[Kafka] 投递 %d 条消息", len(chunks))

        # 5. 消费并索引（Demo 中同步执行，生产环境由独立 Consumer 进程处理）
        messages = self.kafka_consumer.consume_batch(max_records=len(chunks))
        indexed_chunks = self._process_and_index(messages, chunks)

        logger.info(
            "[Ingestion] 文档 %s 处理完毕, 索引 %d 个 chunk",
            doc_name,
            len(indexed_chunks),
        )
        return indexed_chunks

    def ingest_document_direct(
        self,
        doc_id: str,
        doc_name: str,
        raw_content: str,
        file_type: str = "markdown",
    ) -> list[DocumentChunk]:
        """跳过 Kafka 直接处理（当 Kafka 不可用时的降级方案）。"""
        logger.info("[Ingestion-Direct] 开始处理文档: %s (%s)", doc_name, doc_id)

        markdown = self.parser.parse(raw_content, file_type)
        cleaned = self.cleaner.clean(markdown)
        chunks = self.splitter.split(cleaned, doc_id, doc_name)
        logger.info("[Ingestion-Direct] 切分为 %d 个 chunk", len(chunks))

        indexed_chunks = self._embed_and_index(chunks)
        logger.info("[Ingestion-Direct] 文档 %s 索引完毕", doc_name)
        return indexed_chunks

    def _process_and_index(
        self, messages: list[dict], original_chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """从 Kafka 消息恢复 chunk 并执行 embedding + 索引。"""
        # 如果 Kafka 消费到的消息为空（如 Kafka 不可用），回退到原始 chunks
        if not messages:
            logger.warning("[Kafka] 未消费到消息，使用原始 chunks 直接索引")
            return self._embed_and_index(original_chunks)

        chunks = []
        for msg in messages:
            chunk = DocumentChunk(**msg)
            chunks.append(chunk)

        return self._embed_and_index(chunks)

    def _embed_and_index(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """为 chunks 生成 embedding 并索引到 ES + Milvus。"""
        import jieba

        for chunk in chunks:
            # 生成双路向量
            chunk.vector_384 = self.embedder.embed_384(chunk.text)
            chunk.vector_768 = self.embedder.embed_768(chunk.text)

            # 中文分词（用于 BM25）
            chunk.bm25_tokens = list(jieba.cut(chunk.text))

            # 索引到 Elasticsearch (BM25)
            self.bm25_engine.index_chunk(chunk)

            # 索引到 Milvus (向量)
            self.vector_engine.insert_chunk(chunk)

            # 存入全局 chunk store
            self.chunk_store[chunk.chunk_id] = chunk

        logger.info("[Index] BM25 + Vector 索引完成: %d chunks", len(chunks))
        return chunks
