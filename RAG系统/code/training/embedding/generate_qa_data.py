"""
generate_qa_data.py
====================
阶段零：用 Qwen2.5-72B 从知识库 Chunk 批量合成 QA 训练语料。

流程：
  1. 从 Milvus / JSONL 加载全量 Chunk（~19 万条）
  2. 每个 Chunk 调用 Qwen2.5-72B 生成 3~5 个问题（共 ~75 万条）
  3. 规则过滤（去重、去乱码、JSON 格式异常）
  4. 写出 raw_qa.jsonl 供后续负样本挖掘与质量过滤使用

运行示例：
  python generate_qa_data.py \
      --chunks_file  data/chunks.jsonl \
      --output_file  data/raw_qa.jsonl \
      --model        Qwen2.5-72B-Instruct \
      --n_questions  4 \
      --concurrency  32
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

import aiofiles
from openai import AsyncOpenAI  # Qwen2.5-72B 兼容 OpenAI 接口

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

# 按问题类型生成，覆盖四类工程师常见提问视角：
#   概念型 —— "X 是什么"
#   原理型 —— "X 为什么 / 如何工作"
#   配置型 —— "X 参数如何设置"
#   故障型 —— "X 场景如何排查"
SYSTEM_PROMPT = "你是一名资深无线通信领域工程师，擅长撰写技术培训题目。"

USER_PROMPT_TEMPLATE = """\
以下是一段无线通信技术文档：

【文档标题路径】{heading_path}
【文档内容】
{chunk_text}

请根据上述内容，生成 {n} 个不同角度、不同难度的技术问题。

要求：
1. 每个问题必须能从文档中找到明确答案，不得引入文档未提及的信息
2. 覆盖以下类型（尽量均衡）：概念型、原理型、配置型、故障型
3. 使用真实工程师的口语化表达（允许缩写，如 "CA 怎么配"）
4. 不得直接复制文档原句作为问题
5. 输出严格为 JSON 数组，不添加任何额外说明

输出格式（JSON 数组）：
[
  {{"question": "...", "type": "概念型"}},
  {{"question": "...", "type": "原理型"}},
  ...
]"""


# ---------------------------------------------------------------------------
# 核心生成函数
# ---------------------------------------------------------------------------

async def generate_questions_for_chunk(
    client: AsyncOpenAI,
    chunk: dict,
    model: str,
    n_questions: int,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    对单个 Chunk 调用 Qwen2.5-72B 生成 n_questions 个问题。

    返回：[{"chunk_id": ..., "question": ..., "type": ..., "positive_doc": ...}, ...]
    """
    async with semaphore:  # 限制并发，防止把 API 打爆
        chunk_id = chunk["chunk_id"]
        chunk_text = chunk["text"]
        heading_path = chunk.get("heading_path", "")

        # 截断过长 Chunk，避免超出模型上下文；512 token ≈ 800 汉字
        if len(chunk_text) > 1500:
            chunk_text = chunk_text[:1500] + "…（截断）"

        prompt = USER_PROMPT_TEMPLATE.format(
            heading_path=heading_path,
            chunk_text=chunk_text,
            n=n_questions,
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,       # 适当温度增加问题多样性
                max_tokens=1024,
                response_format={"type": "json_object"},  # 强制 JSON 输出
            )
            raw_text = response.choices[0].message.content.strip()

            # 从响应中提取 JSON 数组（有时模型会在外层包一层 key）
            questions = _parse_json_array(raw_text)
        except Exception as e:
            logger.warning("chunk_id=%s 生成失败: %s", chunk_id, e)
            return []

        results = []
        for item in questions:
            q = item.get("question", "").strip()
            q_type = item.get("type", "未知")
            if not q:
                continue
            results.append({
                "chunk_id": chunk_id,
                "doc_id": chunk.get("doc_id", ""),
                "question": q,
                "type": q_type,
                "positive_doc": chunk_text,      # 正样本文档
                "heading_path": heading_path,
            })

        return results


def _parse_json_array(text: str) -> list[dict]:
    """
    鲁棒地从 LLM 输出中提取 JSON 数组。
    LLM 有时会输出 {"questions": [...]} 或直接 [...]，统一兼容处理。
    """
    # 尝试直接解析
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
        # 模型把数组包在某个 key 里（如 {"questions": [...]}）
        for v in obj.values():
            if isinstance(v, list):
                return v
    except json.JSONDecodeError:
        pass

    # 用正则兜底：找到第一个完整的 JSON 数组
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


# ---------------------------------------------------------------------------
# 规则过滤
# ---------------------------------------------------------------------------

def _is_valid_question(q: str, chunk_text: str) -> bool:
    """
    简单规则过滤，去除明显低质样本：
      - 长度不合理（太短或太长）
      - 直接复制文档原句
      - 包含乱码控制字符
    """
    if not (10 <= len(q) <= 200):
        return False
    # 问题与文档前 100 字的重叠率超过 80% 视为直接抄写
    overlap = len(set(q) & set(chunk_text[:100])) / max(len(set(q)), 1)
    if overlap > 0.85:
        return False
    # 含有控制字符或乱码
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", q):
        return False
    return True


def deduplicate(records: list[dict]) -> list[dict]:
    """
    按问题文本的 MD5 去重。
    同一问题可能被多个相似 Chunk 重复生成。
    """
    seen: set[str] = set()
    deduped = []
    for r in records:
        key = hashlib.md5(r["question"].encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return deduped


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

async def main(args: argparse.Namespace) -> None:
    client = AsyncOpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
    )
    semaphore = asyncio.Semaphore(args.concurrency)

    # 读取 Chunk 列表
    chunks = []
    with open(args.chunks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    logger.info("共加载 %d 个 Chunk", len(chunks))

    # 并发生成所有 Chunk 的问题
    tasks = [
        generate_questions_for_chunk(
            client, chunk, args.model, args.n_questions, semaphore
        )
        for chunk in chunks
    ]

    all_records: list[dict] = []
    for i, coro in enumerate(asyncio.as_completed(tasks), 1):
        results = await coro
        all_records.extend(results)
        if i % 1000 == 0:
            logger.info("已处理 %d / %d 个 Chunk，当前累计 %d 条 QA", i, len(chunks), len(all_records))

    logger.info("生成完毕，原始总量 %d 条", len(all_records))

    # 规则过滤
    filtered = [
        r for r in all_records
        if _is_valid_question(r["question"], r["positive_doc"])
    ]
    logger.info("规则过滤后剩余 %d 条（过滤掉 %d 条）", len(filtered), len(all_records) - len(filtered))

    # 去重
    deduped = deduplicate(filtered)
    logger.info("去重后剩余 %d 条", len(deduped))

    # 写出
    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(args.output_file, "w", encoding="utf-8") as f:
        for r in deduped:
            await f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("已写出至 %s", args.output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 Qwen2.5-72B 批量生成 QA 训练语料")
    parser.add_argument("--chunks_file",  default="data/chunks.jsonl",   help="输入 Chunk JSONL 路径")
    parser.add_argument("--output_file",  default="data/raw_qa.jsonl",   help="输出 QA JSONL 路径")
    parser.add_argument("--model",        default="Qwen2.5-72B-Instruct",help="模型名称")
    parser.add_argument("--api_base",     default="http://localhost:8000/v1", help="vLLM API 地址")
    parser.add_argument("--api_key",      default="EMPTY")
    parser.add_argument("--n_questions",  type=int, default=4,           help="每个 Chunk 生成的问题数")
    parser.add_argument("--concurrency",  type=int, default=32,          help="最大并发请求数")
    args = parser.parse_args()

    asyncio.run(main(args))
