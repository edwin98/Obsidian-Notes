"""Cross-Encoder Reranker：交叉互动打分精排。

生产环境使用 gte-multilingual-reranker-0.3B，
Demo 中使用基于词汇重叠 + 位置加权的模拟打分。
"""

from __future__ import annotations

import logging
import math

from models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """模拟 Cross-Encoder 交叉编码器。

    生产环境替换为 GPU 批量推理 gte-multilingual-reranker-0.3B。
    Demo 中使用 Jaccard + TF-IDF 风格的模拟打分，
    能产生合理的排序梯度以演示断崖截断逻辑。
    """

    def __init__(self, chunk_store: dict[str, DocumentChunk]):
        self._chunk_store = chunk_store

    def rerank(
        self, query: str, candidates: list[tuple[str, float]]
    ) -> list[tuple[str, float]]:
        """对候选列表进行 cross-encoder 精排。

        输入: [(chunk_id, fusion_score), ...]
        输出: [(chunk_id, rerank_score), ...]
        """
        results = []
        for chunk_id, _prev_score in candidates:
            chunk = self._chunk_store.get(chunk_id)
            if chunk is None:
                continue
            score = self._compute_relevance(query, chunk.text)
            results.append((chunk_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        logger.debug("[Reranker] 精排 %d 个候选", len(results))
        return results

    def _compute_relevance(self, query: str, text: str) -> float:
        """模拟 cross-encoder 打分。

        综合考虑：
        1. 词汇重叠度 (Jaccard)
        2. 查询词在文档中的覆盖率
        3. 位置加权（前面出现的匹配更重要）
        """
        q_tokens = set(self._tokenize(query))
        t_tokens = set(self._tokenize(text))

        if not q_tokens or not t_tokens:
            return 0.0

        # Jaccard 相似度
        intersection = q_tokens & t_tokens
        jaccard = len(intersection) / len(q_tokens | t_tokens)

        # 查询词覆盖率
        coverage = len(intersection) / len(q_tokens) if q_tokens else 0.0

        # 位置加权：匹配词在文档前部出现，得分更高
        position_score = 0.0
        text_lower = text.lower()
        for token in intersection:
            pos = text_lower.find(token.lower())
            if pos >= 0:
                # 越靠前得分越高（指数衰减）
                position_score += math.exp(-pos / max(len(text), 1) * 3)
        position_score = position_score / max(len(intersection), 1)

        # 综合打分
        score = 0.4 * jaccard + 0.35 * coverage + 0.25 * position_score
        return min(score, 1.0)

    def _tokenize(self, text: str) -> list[str]:
        """简易分词：中文按字，英文按空格。"""
        import jieba

        return [w for w in jieba.cut(text) if w.strip()]
