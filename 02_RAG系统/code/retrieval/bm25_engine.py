"""BM25 检索引擎：基于 Elasticsearch 的全文检索。

使用 Elasticsearch 倒排索引 + BM25 评分，挂载领域专有词表分词器。
"""

from __future__ import annotations

import logging

from elasticsearch import Elasticsearch

from config.settings import Settings
from models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class BM25Engine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.index_name = settings.es_index_name
        self._es: Elasticsearch | None = None

    def connect(self) -> None:
        self._es = Elasticsearch(self.settings.elasticsearch_url)
        self._ensure_index()
        logger.info("[ES] 已连接: %s", self.settings.elasticsearch_url)

    def _ensure_index(self) -> None:
        """创建索引（如果不存在），配置 IK 分词器映射。"""
        if self._es.indices.exists(index=self.index_name):
            return

        mappings = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "analysis": {
                    "analyzer": {
                        "ik_smart_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",  # 生产环境替换为 ik_smart
                        }
                    }
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "doc_name": {"type": "text"},
                    "text": {
                        "type": "text",
                        "analyzer": "ik_smart_analyzer",
                    },
                    "heading_path": {"type": "text"},
                    "node_type": {"type": "keyword"},
                }
            },
        }
        self._es.indices.create(index=self.index_name, **mappings)
        logger.info("[ES] 创建索引: %s", self.index_name)

    def index_chunk(self, chunk: DocumentChunk) -> None:
        """将 chunk 索引到 Elasticsearch。"""
        if self._es is None:
            self.connect()

        doc = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.metadata.doc_id,
            "doc_name": chunk.metadata.doc_name,
            "text": chunk.text,
            "heading_path": chunk.metadata.heading_path,
            "node_type": chunk.metadata.node_type,
        }
        self._es.index(index=self.index_name, id=chunk.chunk_id, document=doc)

    def refresh(self) -> None:
        """刷新索引使文档可搜索。"""
        if self._es:
            self._es.indices.refresh(index=self.index_name)

    def search(self, query: str, top_k: int = 1500) -> list[tuple[str, float]]:
        """BM25 检索，返回 [(chunk_id, bm25_score), ...]。"""
        if self._es is None:
            self.connect()

        body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^3", "heading_path^2", "doc_name"],
                    "type": "best_fields",
                }
            },
        }

        resp = self._es.search(index=self.index_name, **body)
        results = []
        for hit in resp["hits"]["hits"]:
            results.append((hit["_id"], hit["_score"]))

        logger.debug("[ES] BM25 搜索 '%s': %d 结果", query[:30], len(results))
        return results

    def delete_index(self) -> None:
        """删除索引（测试清理用）。"""
        if self._es and self._es.indices.exists(index=self.index_name):
            self._es.indices.delete(index=self.index_name)
