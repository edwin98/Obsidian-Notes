"""Embedding 生成器：产出 384 维与 768 维向量表征。

Demo 中使用确定性伪随机向量（基于文本 hash），保证相同文本生成相同向量。
生产环境替换为领域微调后的 gte-multilingual-base 模型。
"""

from __future__ import annotations

import numpy as np

from config.settings import Settings


class Embedder:
    def __init__(self, settings: Settings):
        self.dim_light = settings.embedding_dim_light
        self.dim_dense = settings.embedding_dim_dense

    def embed_384(self, text: str) -> list[float]:
        """生成 384 维轻量级向量（第一层粗筛用）。"""
        return self._deterministic_embed(text, self.dim_light)

    def embed_768(self, text: str) -> list[float]:
        """生成 768 维密集语义向量（第二层精筛用）。"""
        return self._deterministic_embed(text, self.dim_dense)

    def _deterministic_embed(self, text: str, dim: int) -> list[float]:
        """确定性伪随机向量：相同文本 -> 相同向量，且语义相近的文本向量有一定相似度。"""
        # 基于 n-gram 特征生成稳定向量
        vec = np.zeros(dim, dtype=np.float32)

        # 字符级 n-gram (n=3) 叠加
        for i in range(len(text) - 2):
            trigram = text[i : i + 3]
            seed = hash(trigram) % (2**31)
            rng = np.random.RandomState(seed)
            vec += rng.randn(dim).astype(np.float32)

        # 词级特征叠加（更强的语义信号）
        words = text.split()
        for word in words:
            seed = hash(word) % (2**31)
            rng = np.random.RandomState(seed)
            vec += rng.randn(dim).astype(np.float32) * 2.0

        # L2 归一化
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        return vec.tolist()
