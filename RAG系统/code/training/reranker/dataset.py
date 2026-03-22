"""
dataset.py
===========
Reranker 训练用的 PyTorch Dataset 类。

与 Embedding 的 dataset.py 的核心区别：
  1. 输入格式：[CLS] query [SEP] doc [SEP]（Cross-Encoder 拼接格式）
     Embedding 是 query 和 doc 分别编码（Bi-Encoder）
  2. 标签类型：Listwise 连续分数（来自 LLM 评分归一化），不是二值标签
  3. 每个 instance 包含多个 doc（listwise），不是单个 (query, doc) pair
  4. 序列长度限制：query + doc 拼接后最长 512 token（Embedding 只有 doc 部分）

提供两个 Dataset：
  RerankerListwiseDataset  —— Listwise 训练（推荐）：一次处理整个候选列表
  RerankerPairwiseDataset  —— Pairwise 回退选项：每次比较一对 (pos, neg)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase


# ---------------------------------------------------------------------------
# Listwise Dataset（主要使用）
# ---------------------------------------------------------------------------

class RerankerListwiseDataset(Dataset):
    """
    每个 instance 是一个 query + 多个候选文档（含连续分数标签）。

    __getitem__ 返回原始文本（未 tokenize），
    tokenization 在 collate_fn 中批量完成，充分利用 DataLoader 多进程预取。

    输入 JSONL 格式（每行）：
      {
        "query": "HARQ最大重传次数是多少",
        "docs": [
          {"text": "HARQ最大重传次数为4次...", "score": 1.0, "is_positive": true},
          {"text": "HARQ重传机制概述...",     "score": 0.25, "is_positive": false},
          ...
        ]
      }
    """

    def __init__(self, data_file: Path | str, max_docs_per_query: int = 7) -> None:
        self.data_file = Path(data_file)
        self.max_docs_per_query = max_docs_per_query
        self.samples: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        with open(self.data_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                # 截断超出 max_docs_per_query 的文档（保留所有正样本 + 随机采样负样本）
                docs = item["docs"]
                positives = [d for d in docs if d.get("is_positive", False)]
                negatives = [d for d in docs if not d.get("is_positive", False)]
                if len(positives) + len(negatives) > self.max_docs_per_query:
                    import random
                    neg_budget = self.max_docs_per_query - len(positives)
                    negatives = random.sample(negatives, min(neg_budget, len(negatives)))
                item["docs"] = positives + negatives
                self.samples.append(item)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def listwise_collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> dict[str, Any]:
    """
    将一个 batch 的 Listwise instances 转为模型输入 tensor。

    Cross-Encoder 输入格式：
      [CLS] query [SEP] doc [SEP]
    每个 query 对应多个 doc，所有 (query, doc) pair 在 batch 维度展平。

    Returns:
        {
          "input_ids":      [total_pairs, max_length],  # total_pairs = sum(len(docs) for b in batch)
          "attention_mask": [total_pairs, max_length],
          "labels":         [total_pairs],               # 归一化分数 [0, 1]
          "query_ids":      [total_pairs],               # 标记每个 pair 属于哪个 query（用于 Listwise Loss）
          "doc_counts":     [batch_size],                # 每个 query 的 doc 数量
        }
    """
    all_queries: list[str] = []
    all_docs: list[str] = []
    all_labels: list[float] = []
    doc_counts: list[int] = []
    query_ids: list[int] = []

    for query_idx, instance in enumerate(batch):
        query = instance["query"]
        docs = instance["docs"]
        for doc_info in docs:
            all_queries.append(query)
            all_docs.append(doc_info["text"])
            all_labels.append(float(doc_info["score"]))
            query_ids.append(query_idx)
        doc_counts.append(len(docs))

    # Cross-Encoder tokenization：拼接 query + doc
    encodings = tokenizer(
        all_queries,
        all_docs,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return {
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "token_type_ids": encodings.get("token_type_ids"),
        "labels": torch.tensor(all_labels, dtype=torch.float32),
        "query_ids": torch.tensor(query_ids, dtype=torch.long),
        "doc_counts": torch.tensor(doc_counts, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Pairwise Dataset（回退选项）
# ---------------------------------------------------------------------------

class RerankerPairwiseDataset(Dataset):
    """
    每个 instance 是一对 (query, positive_doc, negative_doc)，
    用于 Pairwise Ranking Loss 训练。

    适用场景：快速实验，或负样本数量少（< 3）无法组 Listwise 时的降级方案。
    """

    def __init__(self, data_file: Path | str) -> None:
        self.samples: list[dict[str, str]] = []
        self._load(Path(data_file))

    def _load(self, data_file: Path) -> None:
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                query = item["query"]
                positives = [d for d in item["docs"] if d.get("is_positive", False)]
                negatives = [d for d in item["docs"] if not d.get("is_positive", False)]
                if not positives or not negatives:
                    continue
                # 每个正样本 × 每个负样本 展开为独立 pair
                for pos in positives:
                    for neg in negatives:
                        self.samples.append({
                            "query": query,
                            "positive": pos["text"],
                            "negative": neg["text"],
                            "pos_score": pos["score"],
                            "neg_score": neg["score"],
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]


def pairwise_collate_fn(
    batch: list[dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 512,
) -> dict[str, Any]:
    """Pairwise collate：每次输出 (positive pair, negative pair) 各自的 token。"""
    pos_queries = [b["query"] for b in batch]
    pos_docs = [b["positive"] for b in batch]
    neg_queries = [b["query"] for b in batch]
    neg_docs = [b["negative"] for b in batch]

    pos_enc = tokenizer(pos_queries, pos_docs, max_length=max_length,
                        truncation=True, padding="max_length", return_tensors="pt")
    neg_enc = tokenizer(neg_queries, neg_docs, max_length=max_length,
                        truncation=True, padding="max_length", return_tensors="pt")

    return {
        "pos_input_ids":      pos_enc["input_ids"],
        "pos_attention_mask": pos_enc["attention_mask"],
        "neg_input_ids":      neg_enc["input_ids"],
        "neg_attention_mask": neg_enc["attention_mask"],
    }
