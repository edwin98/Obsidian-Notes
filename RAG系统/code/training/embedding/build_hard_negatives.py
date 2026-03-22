"""
build_hard_negatives.py
========================
负样本挖掘：为每条 QA 样本构建 BM25×3 + ANN×4 共 7 个硬负样本，
并用 Reranker 过滤假负样本（实际相关但被标记为负）。

流程：
  1. 加载 raw_qa.jsonl（规则过滤后的 QA 语料）
  2. BM25 检索：用 query 从全量 Chunk 检索 Top50，去掉正样本，取前 3 条
  3. ANN 检索：用当前 Embedding 模型编码 query，从向量库检索 Top100，取前 4 条
  4. Reranker 假负样本过滤：对候选负样本打分，分数 > 0.3 的认为是假负，移除
  5. Reranker 相关性过滤：对 (query, pos) 打分，分数 < 0.4 的去掉（正样本本身不相关）
  6. 写出 triplets.jsonl 供三阶段训练使用

运行示例：
  python build_hard_negatives.py \
      --qa_file          data/raw_qa.jsonl \
      --chunks_file      data/chunks.jsonl \
      --output_file      data/triplets.jsonl \
      --embedding_model  Alibaba-NLP/gte-multilingual-base \
      --reranker_model   Alibaba-NLP/gte-multilingual-reranker-base \
      --batch_size       256
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 假负样本判定阈值：Reranker 认为"相关"的负样本要从训练集中移除
FALSE_NEG_THRESHOLD = 0.3
# 正样本质量阈值：Reranker 对 (query, pos) 打分低于此值的样本整条丢弃
POS_QUALITY_THRESHOLD = 0.4


# ---------------------------------------------------------------------------
# BM25 索引
# ---------------------------------------------------------------------------

def build_bm25_index(chunks: list[dict]) -> tuple[BM25Okapi, list[dict]]:
    """
    对全量 Chunk 构建 BM25 倒排索引。

    分词策略：直接按字切分（char-level tokenization），对中英混合文本更友好。
    生产环境可替换为结巴分词。
    """
    tokenized_corpus = [list(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 索引构建完毕，共 %d 条文档", len(chunks))
    return bm25, chunks


def bm25_retrieve(
    bm25: BM25Okapi,
    chunks: list[dict],
    query: str,
    positive_chunk_id: str,
    top_k: int = 3,
    candidate_size: int = 50,
) -> list[str]:
    """
    BM25 检索：返回 top_k 个 BM25 硬负样本文本。

    - 先检索 candidate_size 个候选，再排除正样本，取前 top_k 条
    - BM25 负样本特点：词汇重叠高但语义不匹配，是中等难度负样本
    """
    tokenized_query = list(query)
    scores = bm25.get_scores(tokenized_query)

    # 按分数降序，取前 candidate_size 个索引
    top_indices = np.argsort(scores)[::-1][:candidate_size]

    negatives = []
    for idx in top_indices:
        if chunks[idx]["chunk_id"] == positive_chunk_id:
            continue  # 排除正样本
        negatives.append(chunks[idx]["text"])
        if len(negatives) >= top_k:
            break

    return negatives


# ---------------------------------------------------------------------------
# ANN 向量检索
# ---------------------------------------------------------------------------

def build_ann_index(
    model: SentenceTransformer,
    chunks: list[dict],
    batch_size: int = 256,
) -> tuple[np.ndarray, list[dict]]:
    """
    用 Embedding 模型对全量 Chunk 编码，得到向量矩阵（用于暴力内积检索）。

    生产环境应换成 Milvus/Faiss，这里为了流程清晰做 numpy 暴力检索。
    384 维用于加速，精度可接受。
    """
    logger.info("开始编码全量 Chunk（%d 条）…", len(chunks))
    texts = [c["text"] for c in chunks]
    # output_dim=384：gte-multilingual-base 原生支持，推理更快
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # 余弦相似度 = 内积（归一化后）
        output_dim=384,
    )
    return embeddings.astype(np.float32), chunks


def ann_retrieve(
    query_vec: np.ndarray,
    corpus_vecs: np.ndarray,
    chunks: list[dict],
    positive_chunk_id: str,
    top_k: int = 4,
    candidate_size: int = 100,
) -> list[str]:
    """
    ANN（近似近邻）检索：返回 top_k 个语义相似负样本文本。

    - 这类负样本在向量空间中距离正样本很近，是最难区分的 Hard Negative
    - 也是微调中最有价值的训练信号
    """
    # 暴力内积检索（向量已 L2 归一化，内积 = 余弦相似度）
    scores = corpus_vecs @ query_vec  # (N,)
    top_indices = np.argsort(scores)[::-1][:candidate_size]

    negatives = []
    for idx in top_indices:
        if chunks[idx]["chunk_id"] == positive_chunk_id:
            continue
        negatives.append(chunks[idx]["text"])
        if len(negatives) >= top_k:
            break

    return negatives


# ---------------------------------------------------------------------------
# Reranker 过滤
# ---------------------------------------------------------------------------

def filter_false_negatives(
    reranker: CrossEncoder,
    query: str,
    candidates: list[str],
    threshold: float = FALSE_NEG_THRESHOLD,
) -> list[str]:
    """
    用 Reranker 对候选负样本打分，过滤假负样本。

    原理：负样本中可能混入实际相关的文档（因为 BM25/ANN 不完美），
    如果 Reranker 认为某文档与 query 相关（分数 > threshold），
    则不能作为负样本，否则会给模型注入错误的训练信号。
    """
    if not candidates:
        return []

    pairs = [(query, doc) for doc in candidates]
    scores = reranker.predict(pairs, convert_to_numpy=True)

    # 只保留 Reranker 认为"不相关"的候选
    filtered = [doc for doc, score in zip(candidates, scores) if score < threshold]
    return filtered


def filter_low_quality_positives(
    reranker: CrossEncoder,
    query: str,
    positive_doc: str,
    threshold: float = POS_QUALITY_THRESHOLD,
) -> bool:
    """
    检查正样本质量：如果 Reranker 认为 (query, positive_doc) 不相关，
    说明这条 QA 对本身有问题（LLM 生成偏题），整条丢弃。

    返回 True 表示质量合格，False 表示需要过滤。
    """
    score = reranker.predict([(query, positive_doc)], convert_to_numpy=True)[0]
    return float(score) >= threshold


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载 QA 语料
    qa_records = []
    with open(args.qa_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_records.append(json.loads(line))
    logger.info("加载 QA 语料 %d 条", len(qa_records))

    # 加载全量 Chunk
    chunks = []
    with open(args.chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("加载 Chunk %d 条", len(chunks))

    # 构建 BM25 索引
    bm25, _ = build_bm25_index(chunks)

    # 加载 Embedding 模型（用于 ANN 检索）
    logger.info("加载 Embedding 模型：%s", args.embedding_model)
    embed_model = SentenceTransformer(args.embedding_model, device=device)

    # 构建向量索引（全量 Chunk 编码，384 维）
    corpus_vecs, _ = build_ann_index(embed_model, chunks, batch_size=args.batch_size)

    # 加载 Reranker（用于假负样本过滤）
    logger.info("加载 Reranker：%s", args.reranker_model)
    reranker = CrossEncoder(args.reranker_model, device=device, max_length=512)

    # 逐条处理
    triplets = []
    skipped_low_quality = 0
    skipped_no_neg = 0

    for record in tqdm(qa_records, desc="挖掘硬负样本"):
        query = record["question"]
        positive_doc = record["positive_doc"]
        positive_chunk_id = record["chunk_id"]

        # Step 1：过滤低质量正样本（LLM 生成偏题）
        if not filter_low_quality_positives(reranker, query, positive_doc):
            skipped_low_quality += 1
            continue

        # Step 2：BM25 检索 3 个硬负样本
        bm25_negs = bm25_retrieve(bm25, chunks, query, positive_chunk_id, top_k=3)

        # Step 3：ANN 检索 4 个语义相似负样本
        query_vec = embed_model.encode(
            [query], normalize_embeddings=True, output_dim=384
        )[0]
        ann_negs = ann_retrieve(query_vec, corpus_vecs, chunks, positive_chunk_id, top_k=4)

        # Step 4：合并所有候选负样本，用 Reranker 过滤假负样本
        all_neg_candidates = list(dict.fromkeys(bm25_negs + ann_negs))  # 去重保序
        filtered_negs = filter_false_negatives(reranker, query, all_neg_candidates)

        if len(filtered_negs) < 2:
            # 负样本太少（可能真的是一个非常独特的问题），跳过
            skipped_no_neg += 1
            continue

        triplets.append({
            "query": query,
            "positive": positive_doc,
            "negatives": filtered_negs,      # 最终 2~7 个硬负样本
            "chunk_id": positive_chunk_id,
            "doc_id": record.get("doc_id", ""),
            "question_type": record.get("type", ""),
        })

    logger.info(
        "处理完毕：有效三元组 %d 条，"
        "因正样本低质过滤 %d 条，因负样本不足过滤 %d 条",
        len(triplets), skipped_low_quality, skipped_no_neg,
    )

    # 写出
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for t in triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    logger.info("已写出至 %s", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 BM25+ANN 硬负样本三元组")
    parser.add_argument("--qa_file",          default="data/raw_qa.jsonl")
    parser.add_argument("--chunks_file",      default="data/chunks.jsonl")
    parser.add_argument("--output_file",      default="data/triplets.jsonl")
    parser.add_argument("--embedding_model",  default="Alibaba-NLP/gte-multilingual-base")
    parser.add_argument("--reranker_model",   default="Alibaba-NLP/gte-multilingual-reranker-base")
    parser.add_argument("--batch_size",       type=int, default=256)
    args = parser.parse_args()
    main(args)
