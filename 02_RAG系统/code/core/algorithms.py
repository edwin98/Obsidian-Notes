"""核心算法实现：RSF 动态权重、归一化融合、Rerank 断崖截断。"""

from __future__ import annotations

import math


def compute_rsf_alpha(token_length: int, k: int = 8, s: float = 1.0) -> float:
    """RSF 动态权重公式。

    alpha = 0.4 + 0.3 / (1 + e^(-(L-k)/s))

    短 query -> alpha 趋近 0.4 -> 偏 BM25
    长 query -> alpha 趋近 0.7 -> 偏向量
    L=k 时 alpha = 0.55（中点）
    """
    sigmoid = 1.0 / (1.0 + math.exp(-(token_length - k) / s))
    return 0.4 + 0.3 * sigmoid


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-Max 归一化到 [0, 1]。"""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if max_s == min_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def rsf_fusion(
    bm25_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    alpha: float,
    top_k: int = 80,
) -> list[tuple[str, float]]:
    """RSF 归一化融合打分。

    综合分 = alpha * vector_norm + (1 - alpha) * bm25_norm
    """
    bm25_dict: dict[str, float] = {}
    for cid, s in bm25_results:
        bm25_dict[cid] = max(bm25_dict.get(cid, 0.0), s)

    vec_dict: dict[str, float] = {}
    for cid, s in vector_results:
        vec_dict[cid] = max(vec_dict.get(cid, 0.0), s)

    all_ids = list(set(bm25_dict.keys()) | set(vec_dict.keys()))

    bm25_raw = [bm25_dict.get(cid, 0.0) for cid in all_ids]
    vec_raw = [vec_dict.get(cid, 0.0) for cid in all_ids]

    bm25_norm = normalize_scores(bm25_raw)
    vec_norm = normalize_scores(vec_raw)

    fused = []
    for i, cid in enumerate(all_ids):
        combined = alpha * vec_norm[i] + (1 - alpha) * bm25_norm[i]
        fused.append((cid, combined))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def rerank_with_threshold_cutoff(
    scored_results: list[tuple[str, float]],
    diff_threshold: float = 0.8,
    max_output: int = 10,
) -> list[tuple[str, float]]:
    """Rerank 后断崖截断。

    对降序排列的分数序列，若相邻分差 > diff_threshold 且后者绝对分低，则截断。
    """
    if not scored_results:
        return []

    sorted_results = sorted(scored_results, key=lambda x: x[1], reverse=True)
    output = [sorted_results[0]]

    for i in range(1, len(sorted_results)):
        diff = sorted_results[i - 1][1] - sorted_results[i][1]
        if diff > diff_threshold and sorted_results[i][1] < 0.3:
            break
        output.append(sorted_results[i])
        if len(output) >= max_output:
            break

    return output
