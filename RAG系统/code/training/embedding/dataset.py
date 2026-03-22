"""
dataset.py
===========
三个训练阶段对应的 PyTorch Dataset 类。

  Stage1Dataset  —— 弱监督预热：仅 (query, positive) 正样本对
  Stage2Dataset  —— Hard Negative 精调：(query, positive, hard_negs) 三元组
  Stage3Dataset  —— Reranker 知识蒸馏：(query, docs, soft_labels)

每个 Dataset 的 __getitem__ 直接返回原始文本，tokenization 在 collate_fn 中完成，
这样可以利用 DataLoader 的多进程预取，tokenizer 调用不阻塞 GPU。
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Stage 1：弱监督预热 Dataset
# ---------------------------------------------------------------------------

class Stage1Dataset(Dataset):
    """
    仅使用 (query, positive) 正样本对，不含 Hard Negative。

    目标：让模型快速适应领域词汇分布。
    数据来源：triplets.jsonl（只取 query 和 positive 字段）。
    """

    def __init__(self, data_file: str) -> None:
        self.records: list[dict] = []
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    self.records.append({
                        "query":    r["query"],
                        "positive": r["positive"],
                    })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# ---------------------------------------------------------------------------
# Stage 2：Hard Negative 精调 Dataset
# ---------------------------------------------------------------------------

class Stage2Dataset(Dataset):
    """
    (query, positive, hard_negatives) 三元组格式。

    每条样本包含：
      - query：用户问题
      - positive：正样本文档
      - negatives：经 Reranker 过滤后的 2~7 个硬负样本

    collate_fn 会将所有文档（positive + negatives）打包成一个 batch，
    InfoNCE 损失在 batch 内计算相似度矩阵。
    """

    def __init__(self, data_file: str) -> None:
        self.records: list[dict] = []
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    self.records.append({
                        "query":     r["query"],
                        "positive":  r["positive"],
                        "negatives": r["negatives"],  # list[str]
                    })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# ---------------------------------------------------------------------------
# Stage 3：Reranker 知识蒸馏 Dataset
# ---------------------------------------------------------------------------

class Stage3Dataset(Dataset):
    """
    (query, docs, soft_labels) 格式，docs[0] 为正样本，docs[1:] 为负样本。

    soft_labels 是 Reranker 对 [positive, neg1, neg2, ...] 的 softmax(score/T) 概率，
    用于 KL 散度蒸馏损失。

    注意：不同样本的 docs 数量可能不同（因为硬负样本数量不固定），
    collate_fn 中需要做 padding 或逐条处理。
    """

    def __init__(self, data_file: str) -> None:
        self.records: list[dict] = []
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    docs = [r["positive"]] + r["negatives"]
                    self.records.append({
                        "query":       r["query"],
                        "docs":        docs,           # list[str]，长度可变
                        "soft_labels": r["soft_labels"],  # list[float]，与 docs 等长
                    })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        return self.records[idx]


# ---------------------------------------------------------------------------
# Collate 函数
# ---------------------------------------------------------------------------

def make_collate_fn(
    tokenizer: AutoTokenizer,
    query_max_len: int = 128,
    doc_max_len: int = 512,
):
    """
    工厂函数：返回适用于 Stage1/Stage2/Stage3 的 collate_fn。

    tokenizer 在 collate_fn 中调用（而非 __getitem__），
    利用 DataLoader num_workers 实现 CPU 并行 tokenization。

    返回的 batch 格式：
      {
        "query_input_ids":      Tensor[B, Lq]
        "query_attention_mask": Tensor[B, Lq]
        "doc_input_ids":        Tensor[B * (1 + num_neg), Ld]
        "doc_attention_mask":   Tensor[B * (1 + num_neg), Ld]
        "soft_labels":          Tensor[B, 1 + num_neg]  （Stage3 专属）
        "num_docs_per_sample":  list[int]               （每条样本的文档数）
      }
    """

    def collate_fn(batch: list[dict]) -> dict:
        queries = [item["query"] for item in batch]

        # 判断当前是哪个阶段
        has_negatives = "negatives" in batch[0]
        has_soft_labels = "soft_labels" in batch[0]
        has_docs = "docs" in batch[0]

        # 收集所有文档文本
        if has_docs:
            # Stage 3：每条样本有不同数量的 docs
            all_docs = []
            num_docs_per_sample = []
            for item in batch:
                all_docs.extend(item["docs"])
                num_docs_per_sample.append(len(item["docs"]))
        elif has_negatives:
            # Stage 2：positive + negatives
            all_docs = []
            num_docs_per_sample = []
            for item in batch:
                docs = [item["positive"]] + item["negatives"]
                all_docs.extend(docs)
                num_docs_per_sample.append(len(docs))
        else:
            # Stage 1：只有 positive
            all_docs = [item["positive"] for item in batch]
            num_docs_per_sample = [1] * len(batch)

        # Tokenize queries
        q_encoding = tokenizer(
            queries,
            padding=True,
            truncation=True,
            max_length=query_max_len,
            return_tensors="pt",
        )

        # Tokenize 所有文档（flatten 后一次性编码，效率更高）
        d_encoding = tokenizer(
            all_docs,
            padding=True,
            truncation=True,
            max_length=doc_max_len,
            return_tensors="pt",
        )

        result = {
            "query_input_ids":      q_encoding["input_ids"],
            "query_attention_mask": q_encoding["attention_mask"],
            "doc_input_ids":        d_encoding["input_ids"],
            "doc_attention_mask":   d_encoding["attention_mask"],
            "num_docs_per_sample":  num_docs_per_sample,
        }

        # Stage 3 专属：软标签（需要 padding 到相同长度以便堆叠成 Tensor）
        if has_soft_labels:
            max_n = max(len(item["soft_labels"]) for item in batch)
            padded_labels = []
            for item in batch:
                labels = item["soft_labels"]
                # 用 0 填充至 max_n，计算 KL 时 mask 掉 padding 位置
                padded = labels + [0.0] * (max_n - len(labels))
                padded_labels.append(padded)
            result["soft_labels"] = torch.tensor(padded_labels, dtype=torch.float32)

        return result

    return collate_fn
