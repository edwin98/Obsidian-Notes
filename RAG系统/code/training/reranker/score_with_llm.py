"""
score_with_llm.py
==================
用 Qwen2.5-72B（或任意 OpenAI 兼容 API）对 (query, doc) pair 进行 1~5 分相关性打分。

背景：
  Reranker 需要细粒度分数（1~5），而非二值标签。
  48 万条样本人工打分需要约 960 人天，完全不可行。
  本脚本用 LLM 作为主评分器，以 32 路异步并发处理，吞吐约 800~1000 pair/min，
  48 万条约需 8~10 小时（单 A100 部署 Qwen2.5-72B）。

  人工只参与两个环节：
    1. 校准（正式评分前）：200 条样本与 LLM 对比，一致率 > 85% 才上线
    2. 复核（评分后）：对 score ∈ [2.7, 3.3] 的边界样本（约 5~8%）做人工确认

输入 JSONL（每行一条）：
  {"query": "HARQ最大重传次数是多少", "doc": "HARQ重传..."}

输出 JSONL（每行一条）：
  {"query": "...", "doc": "...", "llm_score": 4, "normalized_score": 0.75, "raw_response": "4"}

启动命令：
  python score_with_llm.py \
      --input_file  data/raw_pairs.jsonl \
      --output_file data/scored_pairs.jsonl \
      --model       Qwen/Qwen2.5-72B-Instruct \
      --api_base    http://localhost:8000/v1 \
      --concurrency 32 \
      --batch_size  50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

import aiohttp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

SCORING_PROMPT = """\
你是一个专业的信息检索相关性评估专家，熟悉 ICT 无线通信技术领域（5G NR、LTE、协议栈等）。

请判断以下文档段落对于回答给定查询的相关程度，按以下标准打分（只输出数字，不要解释）：

5分：文档直接、完整地回答了查询，包含所有关键信息
4分：文档包含回答查询所需的核心信息，但可能不够完整或需少量推理
3分：文档与查询话题相关，但不能直接回答，或只回答了部分内容
2分：文档与查询有一定关联，但主要讨论其他内容，帮助有限
1分：文档与查询基本无关

查询：{query}

文档段落：
{doc}

只输出数字（1/2/3/4/5）："""


# ---------------------------------------------------------------------------
# 单条请求
# ---------------------------------------------------------------------------

async def score_one(
    session: aiohttp.ClientSession,
    api_base: str,
    model: str,
    query: str,
    doc: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> dict:
    """对单个 (query, doc) 调用 LLM 打分，返回原始 dict（包含分数和原始回复）。"""
    prompt = SCORING_PROMPT.format(query=query, doc=doc[:1500])  # 截断过长文档

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 5,
        "temperature": 0.0,  # 评分任务需要确定性输出
    }

    for attempt in range(max_retries):
        try:
            async with semaphore:
                async with session.post(
                    f"{api_base}/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            raw = data["choices"][0]["message"]["content"].strip()
            score = int(raw[0])  # 取第一个字符，兼容 "4\n" 等格式
            if score not in (1, 2, 3, 4, 5):
                raise ValueError(f"invalid score: {raw!r}")

            return {
                "query": query,
                "doc": doc,
                "llm_score": score,
                "normalized_score": round((score - 1) / 4, 4),  # 映射到 [0, 1]
                "raw_response": raw,
            }

        except (ValueError, KeyError, aiohttp.ClientError) as e:
            if attempt == max_retries - 1:
                logger.warning("Failed after %d retries: %s | query=%s", max_retries, e, query[:50])
                return {
                    "query": query,
                    "doc": doc,
                    "llm_score": -1,       # -1 表示评分失败，后处理时需过滤
                    "normalized_score": -1.0,
                    "raw_response": str(e),
                }
            await asyncio.sleep(2 ** attempt)  # 指数退避

    # 不会到达，只是为了类型检查
    return {}


# ---------------------------------------------------------------------------
# 批量异步评分主函数
# ---------------------------------------------------------------------------

async def score_batch(
    pairs: list[dict],
    api_base: str,
    model: str,
    concurrency: int,
    output_file: Path,
    resume_offset: int = 0,
) -> None:
    """
    异步批量对 pairs 列表评分，实时写入 output_file（支持断点续传）。

    Args:
        pairs:         [{"query": ..., "doc": ...}, ...]
        api_base:      OpenAI 兼容 API 地址，如 http://localhost:8000/v1
        model:         模型名称
        concurrency:   最大并发请求数（建议 16~64，取决于 API 服务吞吐）
        output_file:   输出 JSONL 文件路径
        resume_offset: 已处理条数，用于断点续传
    """
    semaphore = asyncio.Semaphore(concurrency)
    pairs_to_process = pairs[resume_offset:]

    logger.info(
        "Total pairs: %d | Already done: %d | To process: %d",
        len(pairs), resume_offset, len(pairs_to_process),
    )

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            score_one(session, api_base, model, p["query"], p["doc"], semaphore)
            for p in pairs_to_process
        ]

        start = time.time()
        with open(output_file, "a", encoding="utf-8") as fout:
            for i, coro in enumerate(asyncio.as_completed(tasks)):
                result = await coro
                fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                fout.flush()

                if (i + 1) % 1000 == 0:
                    elapsed = time.time() - start
                    speed = (i + 1) / elapsed * 60  # pair/min
                    eta_min = (len(pairs_to_process) - i - 1) / (speed + 1e-9)
                    logger.info(
                        "Progress: %d/%d | Speed: %.0f pair/min | ETA: %.1f min",
                        i + 1 + resume_offset, len(pairs), speed, eta_min,
                    )

    logger.info("Scoring finished. Total: %d", len(pairs))


# ---------------------------------------------------------------------------
# 统计 & 校准检查
# ---------------------------------------------------------------------------

def print_score_distribution(output_file: Path) -> None:
    """打印分数分布，用于与人工标注对比校准。"""
    from collections import Counter
    scores = []
    failed = 0
    with open(output_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item["llm_score"] == -1:
                failed += 1
            else:
                scores.append(item["llm_score"])

    dist = Counter(scores)
    total = len(scores)
    print(f"\n=== Score Distribution (total={total}, failed={failed}) ===")
    for s in [1, 2, 3, 4, 5]:
        bar = "#" * (dist[s] * 50 // max(dist.values(), default=1))
        print(f"  {s}分: {dist[s]:6d} ({dist[s]/total*100:5.1f}%)  {bar}")

    # 边界样本（需要人工复核）
    boundary = sum(1 for sc in scores if abs(sc - 3) <= 0.3)
    # LLM 输出整数，3分即为边界
    boundary_count = dist[3]
    print(f"\n  需要人工复核（3分边界）: {boundary_count} 条 ({boundary_count/total*100:.1f}%)")


# ---------------------------------------------------------------------------
# 断点续传：检查已处理条数
# ---------------------------------------------------------------------------

def count_done(output_file: Path) -> int:
    if not output_file.exists():
        return 0
    with open(output_file, encoding="utf-8") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM-based relevance scorer for Reranker training data")
    p.add_argument("--input_file",  required=True, type=Path, help="输入 JSONL，每行 {query, doc}")
    p.add_argument("--output_file", required=True, type=Path, help="输出 JSONL，每行追加评分结果")
    p.add_argument("--model",       default="Qwen/Qwen2.5-72B-Instruct")
    p.add_argument("--api_base",    default="http://localhost:8000/v1", help="OpenAI 兼容 API 地址")
    p.add_argument("--concurrency", type=int, default=32, help="最大并发请求数")
    p.add_argument("--stats_only",  action="store_true", help="只打印已有结果的分布统计，不评分")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.stats_only:
        print_score_distribution(args.output_file)
        return

    # 加载输入
    pairs = []
    with open(args.input_file, encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line.strip()))
    logger.info("Loaded %d pairs from %s", len(pairs), args.input_file)

    # 断点续传
    resume_offset = count_done(args.output_file)
    if resume_offset > 0:
        logger.info("Resuming from offset %d", resume_offset)

    # 开始评分
    asyncio.run(
        score_batch(
            pairs=pairs,
            api_base=args.api_base,
            model=args.model,
            concurrency=args.concurrency,
            output_file=args.output_file,
            resume_offset=resume_offset,
        )
    )

    print_score_distribution(args.output_file)


if __name__ == "__main__":
    main()
