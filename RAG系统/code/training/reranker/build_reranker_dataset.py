"""
build_reranker_dataset.py
==========================
从 LLM 评分结果出发，构建 Reranker 的 Listwise 格式训练数据。

与 Embedding 数据准备的核心区别：
  1. 分数来自 LLM（1~5 细粒度），不是二值标签
  2. 负样本必须是"硬负样本"（Embedding 的 False Positive 或 BM25 混淆文档）
     ——因为 Reranker 的输入已经是 Embedding 粗筛后的候选，随机负样本没有训练价值
  3. 输出格式是 Listwise（1 query + 1 正 + 4~6 负），不是三元组
  4. 需要做"假负样本过滤"（LLM 评分发现某些"负样本"实际相关度 >= 3 分的要移除）

数据流：
  scored_pairs.jsonl          ← score_with_llm.py 的输出
        ↓ 正样本筛选（score >= 4）
  positives.jsonl
        ↓ Embedding False Positive 挖掘
  embedding_fps.jsonl         ← Embedding 模型排名靠前但 LLM 评分 <= 2 的文档
        ↓ BM25 召回负样本
  bm25_negatives.jsonl
        ↓ 组装 Listwise 格式 + 假负样本过滤
  reranker_train.jsonl        ← 最终训练集

输出格式（每行一条 Listwise instance）：
  {
    "query": "HARQ最大重传次数是多少",
    "docs": [
      {"text": "HARQ最大重传次数为4次...", "score": 1.0, "is_positive": true},
      {"text": "HARQ重传机制概述...",     "score": 0.25, "is_positive": false},
      ...
    ]
  }

启动命令：
  python build_reranker_dataset.py \
      --scored_file      data/scored_pairs.jsonl \
      --output_file      data/reranker_train.jsonl \
      --embedding_model  checkpoints/gte-finetuned \
      --corpus_file      data/corpus.jsonl \
      --top_k            50 \
      --neg_per_query    5
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Step 1：从 LLM 评分结果中提取正样本
# ---------------------------------------------------------------------------

def load_positives(scored_file: Path, pos_score_threshold: int = 4) -> dict[str, list[dict]]:
    """
    从评分结果中提取正样本（LLM 评分 >= pos_score_threshold）。

    Returns:
        query → [{"text": doc_text, "score": normalized_score}, ...]
    """
    query2positives: dict[str, list[dict]] = defaultdict(list)
    total, kept = 0, 0

    with open(scored_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            total += 1
            if item["llm_score"] < 0:  # 评分失败，跳过
                continue
            if item["llm_score"] >= pos_score_threshold:
                query2positives[item["query"]].append({
                    "text": item["doc"],
                    "score": item["normalized_score"],
                    "is_positive": True,
                })
                kept += 1

    logger.info("Positives: %d / %d (threshold=%d)", kept, total, pos_score_threshold)
    return dict(query2positives)


# ---------------------------------------------------------------------------
# Step 2：挖掘 Embedding False Positive 作为硬负样本
# ---------------------------------------------------------------------------

def mine_embedding_false_positives(
    queries: list[str],
    corpus: list[str],
    embedding_model_path: str,
    top_k: int,
    scored_map: dict[tuple[str, str], int],  # (query, doc) → llm_score
    fp_score_threshold: int = 2,
) -> dict[str, list[str]]:
    """
    用已微调的 Embedding 模型对每个 query 检索 Top-K，
    将"Embedding 排名靠前但 LLM 评分 <= fp_score_threshold"的文档作为硬负样本。

    这类负样本的含义：Embedding 认为相关，但 LLM/人工确认不相关。
    正是 Reranker 需要纠正的错误。
    """
    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        raise ImportError("pip install sentence-transformers torch")

    logger.info("Loading embedding model from %s", embedding_model_path)
    model = SentenceTransformer(embedding_model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    logger.info("Encoding %d corpus docs...", len(corpus))
    corpus_embeddings = model.encode(
        corpus, batch_size=256, show_progress_bar=True,
        normalize_embeddings=True, convert_to_tensor=True,
    )

    query2hard_negs: dict[str, list[str]] = defaultdict(list)

    logger.info("Mining False Positives for %d queries...", len(queries))
    for query in tqdm(queries, desc="FP Mining"):
        q_emb = model.encode([query], normalize_embeddings=True, convert_to_tensor=True)
        scores = (q_emb @ corpus_embeddings.T).squeeze(0)
        top_indices = scores.topk(top_k).indices.tolist()

        for idx in top_indices:
            doc = corpus[idx]
            llm_score = scored_map.get((query, doc), -1)
            # Embedding 认为相关（Top-K 内），但 LLM 评分低 → 硬负样本
            if llm_score != -1 and llm_score <= fp_score_threshold:
                query2hard_negs[query].append(doc)

    total_fps = sum(len(v) for v in query2hard_negs.values())
    logger.info("Mined %d Embedding False Positives across %d queries", total_fps, len(query2hard_negs))
    return dict(query2hard_negs)


# ---------------------------------------------------------------------------
# Step 3：BM25 召回负样本
# ---------------------------------------------------------------------------

def mine_bm25_negatives(
    queries: list[str],
    corpus: list[str],
    scored_map: dict[tuple[str, str], int],
    top_k: int = 20,
    neg_score_threshold: int = 2,
) -> dict[str, list[str]]:
    """
    用 BM25 召回词汇重叠高但语义不相关的文档作为负样本（"关键词匹配陷阱"类型）。
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        raise ImportError("pip install rank-bm25")

    logger.info("Building BM25 index over %d docs...", len(corpus))
    tokenized_corpus = [doc.split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    query2bm25_negs: dict[str, list[str]] = defaultdict(list)

    for query in tqdm(queries, desc="BM25 Neg Mining"):
        tokenized_q = query.split()
        scores = bm25.get_scores(tokenized_q)
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()

        for idx in top_indices:
            doc = corpus[idx]
            llm_score = scored_map.get((query, doc), -1)
            # BM25 排名靠前但 LLM 评分低 → "关键词相似但语义无关"的负样本
            if llm_score != -1 and llm_score <= neg_score_threshold:
                query2bm25_negs[query].append(doc)
            elif llm_score == -1:
                # LLM 未评分的文档（不在 scored_pairs 中），保守策略：不使用
                pass

    total = sum(len(v) for v in query2bm25_negs.values())
    logger.info("Mined %d BM25 negatives across %d queries", total, len(query2bm25_negs))
    return dict(query2bm25_negs)


# ---------------------------------------------------------------------------
# Step 4：假负样本过滤
# ---------------------------------------------------------------------------

def filter_false_negatives(
    negatives: list[str],
    query: str,
    scored_map: dict[tuple[str, str], int],
    fake_neg_threshold: int = 3,
) -> list[str]:
    """
    移除被 LLM 打高分的负样本（即假负样本）。
    LLM 评分 >= fake_neg_threshold 的文档不应作为负样本。
    """
    return [
        doc for doc in negatives
        if scored_map.get((query, doc), 0) < fake_neg_threshold
    ]


# ---------------------------------------------------------------------------
# Step 5：组装 Listwise 格式训练数据
# ---------------------------------------------------------------------------

def assemble_listwise_dataset(
    query2positives: dict[str, list[dict]],
    query2fp_negs: dict[str, list[str]],
    query2bm25_negs: dict[str, list[str]],
    scored_map: dict[tuple[str, str], int],
    neg_per_query: int = 5,
    min_negs: int = 3,
) -> list[dict]:
    """
    为每个 query 组装一个 Listwise instance：1 个正样本 + neg_per_query 个负样本。

    负样本优先级：Embedding FP（最有价值）> BM25 负样本 > 随机抽取其他文档

    Returns:
        list of {"query": ..., "docs": [{"text": ..., "score": ..., "is_positive": ...}]}
    """
    dataset = []
    skipped = 0

    all_queries = list(query2positives.keys())

    for query in tqdm(all_queries, desc="Assembling listwise instances"):
        positives = query2positives.get(query, [])
        if not positives:
            skipped += 1
            continue

        # 挑选一个正样本（优先使用分数最高的）
        positives.sort(key=lambda x: x["score"], reverse=True)
        positive = positives[0]

        # 收集候选负样本并过滤假负样本
        fp_negs = query2fp_negs.get(query, [])
        bm25_negs = query2bm25_negs.get(query, [])

        fp_negs = filter_false_negatives(fp_negs, query, scored_map)
        bm25_negs = filter_false_negatives(bm25_negs, query, scored_map)

        # 去重（同一文档可能出现在两个来源）
        seen = {positive["text"]}
        all_negs: list[str] = []
        for doc in fp_negs + bm25_negs:
            if doc not in seen:
                all_negs.append(doc)
                seen.add(doc)

        if len(all_negs) < min_negs:
            skipped += 1
            continue

        # 采样：FP 负样本优先，不足时用 BM25 补充
        sampled_negs = all_negs[:neg_per_query]
        if len(sampled_negs) < neg_per_query:
            # 从 BM25 池中补充（已去重）
            sampled_negs = sampled_negs  # 不足时就用现有的

        # 组装 docs 列表
        docs = [positive]  # 正样本放第一位（训练时会 shuffle）
        for neg_text in sampled_negs:
            llm_score = scored_map.get((query, neg_text), 1)
            docs.append({
                "text": neg_text,
                "score": round((llm_score - 1) / 4, 4) if llm_score > 0 else 0.0,
                "is_positive": False,
            })

        # Shuffle（避免模型学到位置偏差）
        random.shuffle(docs)

        dataset.append({"query": query, "docs": docs})

    logger.info(
        "Assembled %d listwise instances (skipped=%d due to insufficient negs)",
        len(dataset), skipped,
    )
    return dataset


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def build_scored_map(scored_file: Path) -> dict[tuple[str, str], int]:
    """构建 (query, doc) → llm_score 的快速查找字典。"""
    scored_map: dict[tuple[str, str], int] = {}
    with open(scored_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item["llm_score"] > 0:
                scored_map[(item["query"], item["doc"])] = item["llm_score"]
    return scored_map


def load_corpus(corpus_file: Path) -> list[str]:
    corpus = []
    with open(corpus_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            corpus.append(item.get("text", item.get("content", "")))
    return corpus


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Reranker listwise training dataset")
    p.add_argument("--scored_file",      required=True, type=Path, help="score_with_llm.py 的输出")
    p.add_argument("--corpus_file",      required=True, type=Path, help="文档库 JSONL，每行 {text: ...}")
    p.add_argument("--output_file",      required=True, type=Path, help="输出训练集 JSONL")
    p.add_argument("--embedding_model",  default="checkpoints/gte-finetuned", help="已微调 Embedding 模型路径")
    p.add_argument("--top_k",            type=int, default=50, help="Embedding 检索 TopK，用于挖掘 FP")
    p.add_argument("--neg_per_query",    type=int, default=5,  help="每个 query 的负样本数")
    p.add_argument("--pos_threshold",    type=int, default=4,  help="正样本 LLM 分数下限")
    p.add_argument("--no_embedding_fp",  action="store_true",  help="跳过 Embedding FP 挖掘（仅用 BM25 负样本）")
    p.add_argument("--val_ratio",        type=float, default=0.02, help="验证集比例")
    p.add_argument("--val_output",       type=Path, default=None, help="验证集输出路径（默认不拆分）")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # 1. 构建评分查找表
    logger.info("Building scored map...")
    scored_map = build_scored_map(args.scored_file)
    logger.info("Scored map size: %d", len(scored_map))

    # 2. 加载正样本
    query2positives = load_positives(args.scored_file, args.pos_threshold)
    queries = list(query2positives.keys())
    logger.info("Queries with positives: %d", len(queries))

    # 3. 加载文档库
    corpus = load_corpus(args.corpus_file)
    logger.info("Corpus size: %d", len(corpus))

    # 4. 挖掘负样本
    query2fp_negs: dict[str, list[str]] = {}
    if not args.no_embedding_fp:
        query2fp_negs = mine_embedding_false_positives(
            queries=queries,
            corpus=corpus,
            embedding_model_path=args.embedding_model,
            top_k=args.top_k,
            scored_map=scored_map,
        )

    query2bm25_negs = mine_bm25_negatives(
        queries=queries,
        corpus=corpus,
        scored_map=scored_map,
        top_k=20,
    )

    # 5. 组装 Listwise 数据集
    dataset = assemble_listwise_dataset(
        query2positives=query2positives,
        query2fp_negs=query2fp_negs,
        query2bm25_negs=query2bm25_negs,
        scored_map=scored_map,
        neg_per_query=args.neg_per_query,
    )

    # 6. 拆分训练/验证集
    random.shuffle(dataset)
    if args.val_output and args.val_ratio > 0:
        val_size = int(len(dataset) * args.val_ratio)
        val_set = dataset[:val_size]
        train_set = dataset[val_size:]

        args.val_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.val_output, "w", encoding="utf-8") as f:
            for item in val_set:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        logger.info("Validation set: %d → %s", len(val_set), args.val_output)
    else:
        train_set = dataset

    # 7. 写入训练集
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info("Training set: %d → %s", len(train_set), args.output_file)


if __name__ == "__main__":
    main()
