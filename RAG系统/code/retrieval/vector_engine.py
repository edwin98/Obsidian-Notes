"""向量检索引擎：基于 Milvus 的密集向量检索。

管理两个 Collection：
- rag_vectors_384：384 维轻量级向量（第一层粗筛）
- rag_vectors_768：768 维密集语义向量（第二层精筛）

索引类型：HNSW（M=32, efConstruction=256）
"""

from __future__ import annotations

import logging

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from config.settings import Settings
from models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class MilvusVectorEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._collection_384: Collection | None = None
        self._collection_768: Collection | None = None

    def connect(self) -> None:
        connections.connect(
            alias="default",
            host=self.settings.milvus_host,
            port=self.settings.milvus_port,
        )
        self._collection_384 = self._ensure_collection(
            self.settings.milvus_collection_384,
            self.settings.embedding_dim_light,
        )
        self._collection_768 = self._ensure_collection(
            self.settings.milvus_collection_768,
            self.settings.embedding_dim_dense,
        )
        logger.info(
            "[Milvus] 已连接: %s:%d",
            self.settings.milvus_host,
            self.settings.milvus_port,
        )

    def _ensure_collection(self, name: str, dim: int) -> Collection:
        """创建 Collection（如不存在），配置 HNSW 索引。"""
        if utility.has_collection(name):
            col = Collection(name)
            col.load()
            return col

        fields = [
            FieldSchema(
                name="chunk_id", dtype=DataType.VARCHAR, is_primary=True, max_length=128
            ),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        ]
        schema = CollectionSchema(fields=fields, description=f"RAG vectors {dim}d")
        col = Collection(name=name, schema=schema)

        # 创建 HNSW 索引
        index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 32, "efConstruction": 256},
        }
        col.create_index(field_name="embedding", index_params=index_params)
        col.load()

        logger.info("[Milvus] 创建 Collection: %s (dim=%d, HNSW)", name, dim)
        return col

    def insert_chunk(self, chunk: DocumentChunk) -> None:
        """将 chunk 的双路向量分别插入两个 Collection。"""
        if self._collection_384 is None:
            self.connect()

        if chunk.vector_384:
            self._collection_384.insert(
                [
                    [chunk.chunk_id],
                    [chunk.metadata.doc_id],
                    [chunk.vector_384],
                ]
            )

        if chunk.vector_768:
            self._collection_768.insert(
                [
                    [chunk.chunk_id],
                    [chunk.metadata.doc_id],
                    [chunk.vector_768],
                ]
            )

    def flush(self) -> None:
        if self._collection_384:
            self._collection_384.flush()
        if self._collection_768:
            self._collection_768.flush()

    def search_384(
        self, query_vector: list[float], top_k: int = 1500
    ) -> list[tuple[str, float]]:
        """384 维向量检索（第一层粗筛）。"""
        return self._search(self._collection_384, query_vector, top_k)

    def search_768(
        self, query_vector: list[float], top_k: int = 80
    ) -> list[tuple[str, float]]:
        """768 维向量检索（第二层精筛）。"""
        return self._search(self._collection_768, query_vector, top_k)

    def _search(
        self, collection: Collection | None, query_vector: list[float], top_k: int
    ) -> list[tuple[str, float]]:
        if collection is None:
            self.connect()
            collection = self._collection_384  # fallback

        search_params = {"metric_type": "COSINE", "params": {"ef": 128}}

        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["chunk_id"],
        )

        output = []
        for hits in results:
            for hit in hits:
                output.append((hit.entity.get("chunk_id"), hit.score))

        logger.debug("[Milvus] 向量搜索: %d 结果 (top_k=%d)", len(output), top_k)
        return output

    def drop_collections(self) -> None:
        """删除所有 Collection（测试清理用）。"""
        for name in [
            self.settings.milvus_collection_384,
            self.settings.milvus_collection_768,
        ]:
            if utility.has_collection(name):
                Collection(name).drop()
