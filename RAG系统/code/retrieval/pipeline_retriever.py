"""三级渐进式混合检索器。

L1（粗筛）: BM25 + 384 维向量，各召回 top_k，合并
L2（精筛）: RSF 归一化融合打分 -> 80 docs
L3（精排）: Cross-encoder rerank + 断崖截断 -> 10 docs
"""

from __future__ import annotations

import logging

from config.settings import Settings
from core.abstractions import PipelineRetriever
from core.algorithms import compute_rsf_alpha, rerank_with_threshold_cutoff, rsf_fusion
from ingestion.embedder import Embedder
from models.schemas import DocumentChunk, RetrievedChunk
from retrieval.bm25_engine import BM25Engine
from retrieval.reranker import CrossEncoderReranker
from retrieval.vector_engine import MilvusVectorEngine

logger = logging.getLogger(__name__)


class ThreeLevelRetriever(PipelineRetriever):
    def __init__(
        self,
        settings: Settings,
        bm25_engine: BM25Engine,
        vector_engine: MilvusVectorEngine,
        reranker: CrossEncoderReranker,
        embedder: Embedder,
        chunk_store: dict[str, DocumentChunk],
    ):
        self.settings = settings
        self.bm25_engine = bm25_engine
        self.vector_engine = vector_engine
        self.reranker = reranker
        self.embedder = embedder
        self.chunk_store = chunk_store

    async def retrieve(
        self, query: str, rewritten_queries: list[str], top_k: int = 10
    ) -> list[RetrievedChunk]:
        # ---- L1 粗筛：多路召回 ----
        all_bm25: list[tuple[str, float]] = []
        all_vector: list[tuple[str, float]] = []

        for q in rewritten_queries:
            # BM25 检索
            bm25_hits = self.bm25_engine.search(q, top_k=self.settings.level1_topk)
            all_bm25.extend(bm25_hits)

            # 384 维向量检索
            vec_384 = self.embedder.embed_384(q)
            vec_hits = self.vector_engine.search_384(
                vec_384, top_k=self.settings.level1_topk
            )
            all_vector.extend(vec_hits)

        # 去重，保留最高分
        bm25_deduped = self._deduplicate(all_bm25)
        vec_deduped = self._deduplicate(all_vector)

        logger.info(
            "[L1 粗筛] BM25: %d 条, Vector-384: %d 条",
            len(bm25_deduped),
            len(vec_deduped),
        )

        # ---- L2 精筛：RSF 融合 ----
        token_len = self._count_tokens(query)
        alpha = compute_rsf_alpha(
            token_len, k=self.settings.rsf_k, s=self.settings.rsf_s
        )
        fused = rsf_fusion(
            bm25_deduped, vec_deduped, alpha, top_k=self.settings.level2_topk
        )

        logger.info(
            "[L2 RSF] alpha=%.3f (token_len=%d), 输出: %d docs",
            alpha,
            token_len,
            len(fused),
        )

        # ---- L3 精排：Rerank + 断崖截断 ----
        reranked = self.reranker.rerank(query, fused)
        final = rerank_with_threshold_cutoff(
            reranked,
            diff_threshold=self.settings.rerank_diff_threshold,
            max_output=top_k,
        )

        logger.info("[L3 Rerank] 精排后: %d docs", len(final))

        # 构建结果
        results: list[RetrievedChunk] = []
        for chunk_id, score in final:
            chunk = self.chunk_store.get(chunk_id)
            if chunk:
                results.append(
                    RetrievedChunk(chunk=chunk, score=score, source="rerank")
                )

        return results

    def _deduplicate(self, results: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """去重，保留每个 chunk_id 的最高分。"""
        best: dict[str, float] = {}
        for cid, score in results:
            best[cid] = max(best.get(cid, 0.0), score)
        return list(best.items())

    def _count_tokens(self, text: str) -> int:
        """粗略统计 token 数。"""
        import jieba

        return len(list(jieba.cut(text)))
