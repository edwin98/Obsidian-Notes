"""核心算法单元测试：RSF 动态权重、归一化、Rerank 截断。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.algorithms import (
    compute_rsf_alpha,
    normalize_scores,
    rerank_with_threshold_cutoff,
    rsf_fusion,
)


class TestRSFAlpha:
    """RSF 动态权重公式测试。"""

    def test_short_query_favors_bm25(self):
        """短 query (L=3) 应偏向 BM25，alpha 接近 0.4。"""
        alpha = compute_rsf_alpha(token_length=3)
        assert 0.4 <= alpha <= 0.45

    def test_long_query_favors_vector(self):
        """长 query (L=20) 应偏向向量，alpha 接近 0.7。"""
        alpha = compute_rsf_alpha(token_length=20)
        assert 0.65 <= alpha <= 0.71

    def test_midpoint(self):
        """L=k=8 时 alpha 应约为 0.55（中点）。"""
        alpha = compute_rsf_alpha(token_length=8, k=8, s=1.0)
        assert abs(alpha - 0.55) < 0.01

    def test_monotonic_increasing(self):
        """alpha 应随 token_length 单调递增。"""
        prev = 0.0
        for length in range(1, 30):
            alpha = compute_rsf_alpha(length)
            assert alpha >= prev
            prev = alpha

    def test_bounded(self):
        """alpha 应始终在 [0.4, 0.7] 范围内。"""
        for length in range(0, 100):
            alpha = compute_rsf_alpha(length)
            assert 0.4 <= alpha <= 0.71


class TestNormalize:
    """归一化测试。"""

    def test_basic(self):
        result = normalize_scores([1.0, 2.0, 3.0])
        assert result == [0.0, 0.5, 1.0]

    def test_equal_scores(self):
        result = normalize_scores([5.0, 5.0, 5.0])
        assert result == [1.0, 1.0, 1.0]

    def test_empty(self):
        result = normalize_scores([])
        assert result == []

    def test_single(self):
        result = normalize_scores([42.0])
        assert result == [1.0]


class TestRSFFusion:
    """RSF 融合打分测试。"""

    def test_basic_fusion(self):
        bm25 = [("c1", 5.0), ("c2", 3.0), ("c3", 1.0)]
        vector = [("c1", 0.9), ("c3", 0.8), ("c4", 0.7)]
        result = rsf_fusion(bm25, vector, alpha=0.5, top_k=10)

        # c1 在两路中都有高分，应排在前面
        ids = [cid for cid, _ in result]
        assert "c1" in ids
        assert ids[0] == "c1"

    def test_top_k_limit(self):
        bm25 = [(f"c{i}", float(i)) for i in range(100)]
        vector = [(f"c{i}", float(100 - i)) for i in range(100)]
        result = rsf_fusion(bm25, vector, alpha=0.5, top_k=10)
        assert len(result) == 10

    def test_alpha_effect(self):
        """alpha 偏向 BM25 时，BM25 高分文档应排更前。"""
        bm25 = [("bm_doc", 10.0), ("vec_doc", 1.0)]
        vector = [("bm_doc", 0.1), ("vec_doc", 0.9)]

        result_bm25 = rsf_fusion(bm25, vector, alpha=0.1, top_k=2)  # 偏 BM25
        result_vec = rsf_fusion(bm25, vector, alpha=0.9, top_k=2)  # 偏向量

        bm25_rank = {cid: i for i, (cid, _) in enumerate(result_bm25)}
        vec_rank = {cid: i for i, (cid, _) in enumerate(result_vec)}

        # 偏 BM25 时，bm_doc 应排更前
        assert bm25_rank["bm_doc"] < bm25_rank["vec_doc"]
        # 偏向量时，vec_doc 应排更前
        assert vec_rank["vec_doc"] < vec_rank["bm_doc"]


class TestRerankCutoff:
    """Rerank 断崖截断测试。"""

    def test_cutoff_at_cliff(self):
        """分差 > 0.8 且绝对分低时应截断。"""
        scores = [("c1", 0.95), ("c2", 0.92), ("c3", 0.90), ("c4", 0.05)]
        result = rerank_with_threshold_cutoff(scores, diff_threshold=0.8)
        ids = [cid for cid, _ in result]
        assert "c4" not in ids
        assert len(result) == 3

    def test_no_cutoff_gradual(self):
        """分数均匀递减时不应截断。"""
        scores = [("c1", 0.9), ("c2", 0.8), ("c3", 0.7), ("c4", 0.6)]
        result = rerank_with_threshold_cutoff(scores, diff_threshold=0.8)
        assert len(result) == 4

    def test_max_output_limit(self):
        """不超过 max_output。"""
        scores = [(f"c{i}", 0.9 - i * 0.01) for i in range(20)]
        result = rerank_with_threshold_cutoff(scores, max_output=5)
        assert len(result) <= 5

    def test_empty_input(self):
        result = rerank_with_threshold_cutoff([])
        assert result == []
