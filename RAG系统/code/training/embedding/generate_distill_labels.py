"""
generate_distill_labels.py
===========================
离线蒸馏标签生成：用已微调好的 Reranker 对训练集的候选文档列表打分，
将分数作为软标签缓存至磁盘，供阶段三蒸馏训练使用。

为什么要离线预生成？
  - Reranker（305M CrossEncoder）推理成本是 Bi-Encoder 的 N 倍（N = 候选文档数）
  - 如果在训练循环内实时调用 Reranker，每步要额外做 7 次 CrossEncoder 前向，
    会把训练速度拖慢 5~8 倍
  - 提前缓存后，训练时只需从磁盘读软标签，开销可忽略

软标签格式（每行一条 JSONL）：
  {
    "query": "...",
    "positive": "...",
    "negatives": ["...", "..."],
    "soft_labels": [0.92, 0.05, 0.03, ...],   # 经 softmax(score/T) 归一化后的概率
    "raw_scores":  [2.31, -1.02, -2.45, ...]   # Reranker 原始 logit 分数
  }

运行示例：
  python generate_distill_labels.py \
      --triplets_file  data/triplets.jsonl \
      --output_file    data/distill_labels.jsonl \
      --reranker_model Alibaba-NLP/gte-multilingual-reranker-base \
      --temperature    2.0 \
      --batch_size     64
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def softmax_with_temperature(scores: list[float], temperature: float) -> list[float]:
    """
    对 Reranker 原始打分做带温度的 softmax，得到软标签概率分布。

    temperature > 1：分布更平滑，弱化边界模糊样本的影响
    temperature = 1：等同于标准 softmax
    temperature < 1：分布更尖锐（接近 one-hot）

    本系统取 temperature=2，让模型专注于学习相对排序而非绝对分值。
    """
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exp_scores = [math.exp(s - max_s) for s in scaled]  # 减 max 防数值溢出
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载三元组数据
    triplets = []
    with open(args.triplets_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                triplets.append(json.loads(line))
    logger.info("加载三元组 %d 条", len(triplets))

    # 加载 Reranker（gte-multilingual-reranker-base, 305M）
    logger.info("加载 Reranker：%s", args.reranker_model)
    reranker = CrossEncoder(args.reranker_model, device=device, max_length=512)
    reranker.model.eval()

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_file, "w", encoding="utf-8")

    processed = 0
    for record in tqdm(triplets, desc="生成软标签"):
        query = record["query"]
        positive = record["positive"]
        negatives = record["negatives"]

        # 构建候选文档列表：正样本排首位，负样本跟后
        # 注意：打分时不告诉 Reranker 哪个是正样本，它只做相关性判断
        docs = [positive] + negatives

        # Reranker 批量打分：输入 [(query, doc1), (query, doc2), ...]
        pairs = [(query, doc) for doc in docs]
        with torch.no_grad():
            raw_scores = reranker.predict(pairs, convert_to_numpy=True).tolist()

        # 用带温度的 softmax 转换为软标签概率
        soft_labels = softmax_with_temperature(raw_scores, args.temperature)

        result = {
            "query": query,
            "positive": positive,
            "negatives": negatives,
            "raw_scores": raw_scores,
            "soft_labels": soft_labels,  # 长度 = 1 + len(negatives)
            "chunk_id": record.get("chunk_id", ""),
        }
        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")

        processed += 1
        if processed % 10000 == 0:
            logger.info("已处理 %d / %d", processed, len(triplets))

    out_f.close()
    logger.info("软标签生成完毕，已写出至 %s", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="离线生成 Reranker 软标签")
    parser.add_argument("--triplets_file",  default="data/triplets.jsonl")
    parser.add_argument("--output_file",    default="data/distill_labels.jsonl")
    parser.add_argument("--reranker_model", default="Alibaba-NLP/gte-multilingual-reranker-base")
    parser.add_argument("--temperature",    type=float, default=2.0,
                        help="蒸馏温度 T，越大软标签越平滑")
    parser.add_argument("--batch_size",     type=int, default=64)
    args = parser.parse_args()
    main(args)
