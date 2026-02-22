"""LLM 生成器：防幻觉 System Prompt + 内联引用 + SSE 流式输出。

生产环境对接大语言模型 API（携带防幻觉强约束 System Prompt）。
Demo 中使用上下文片段拼装模拟，但展示完整的 Prompt 结构。
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator

from core.abstractions import LLMGenerator
from generation.token_budget import TokenBudgetManager
from models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


# ---- 防幻觉 System Prompt ----

ANTI_HALLUCINATION_SYSTEM_PROMPT = """\
你是一个严谨的华为无线通信技术专家。请仅基于以下<context>和</context>标签内部的参考资料回答问题。
如果参考资料中不包含相关答案，请输出标准回复："根据当前已知知识库，暂时无法回答该问题"。
严禁捏造不存在的术语、协议编号与事实。
在表述关键论点后必须添加引用标记，格式为 [chunk_id]。
回答要求：
1. 结构清晰，使用标题和列表组织内容
2. 关键技术点必须附带引用来源
3. 如果多个参考资料有互补信息，需综合整理
4. 输出语言与用户提问语言保持一致"""


class MockLLMGenerator(LLMGenerator):
    """Mock LLM 生成器，展示完整的 Prompt 构建与流式输出逻辑。"""

    def __init__(self, token_budget: TokenBudgetManager):
        self.token_budget = token_budget

    async def generate_stream(
        self,
        query: str,
        context_chunks: list[DocumentChunk],
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """SSE 流式生成。"""
        # 1. 构建上下文
        context_text = self._build_context(context_chunks)

        # 2. Token 预算修剪历史
        trimmed_history = []
        if history:
            trimmed_history = self.token_budget.trim_history(
                ANTI_HALLUCINATION_SYSTEM_PROMPT,
                history,
                query,
            )

        # 3. 构建完整消息列表（生产环境将其发送给 LLM API）
        _messages = self._build_messages(query, context_text, trimmed_history)

        logger.info(
            "[LLM] Prompt 构建完成: system=%d chars, context=%d chars, history=%d 轮, messages=%d",
            len(ANTI_HALLUCINATION_SYSTEM_PROMPT),
            len(context_text),
            len(trimmed_history),
            len(_messages),
        )

        # 4. Mock 生成答案
        answer = self._mock_generate(query, context_chunks)

        # 5. 模拟逐字流式输出（30-50 token/s）
        for char in answer:
            yield char
            await asyncio.sleep(0.02)

    def _build_context(self, chunks: list[DocumentChunk]) -> str:
        """构建 <context> 标签包裹的参考资料。"""
        parts = []
        for chunk in chunks:
            parts.append(f"[{chunk.chunk_id}] {chunk.text}")
        return "<context>\n" + "\n---\n".join(parts) + "\n</context>"

    def _build_messages(
        self, query: str, context: str, history: list[dict]
    ) -> list[dict]:
        """构建发送给 LLM 的完整消息列表。"""
        messages = [{"role": "system", "content": ANTI_HALLUCINATION_SYSTEM_PROMPT}]

        if history:
            messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": f"参考资料：\n{context}\n\n问题：{query}",
            }
        )

        return messages

    def _mock_generate(self, query: str, chunks: list[DocumentChunk]) -> str:
        """从上下文 chunk 中提取相关片段，拼装为带引用的回答。"""
        if not chunks:
            return "根据当前已知知识库，暂时无法回答该问题。"

        answer_parts = [f"关于「{query}」，根据检索到的资料回答如下：\n\n"]

        for i, chunk in enumerate(chunks[:5]):
            # 提取每个 chunk 的关键内容
            text = chunk.text.strip()
            # 取前 200 字作为摘要
            snippet = text[:200]
            if len(text) > 200:
                snippet += "..."

            heading = chunk.metadata.heading_path or chunk.metadata.doc_name
            answer_parts.append(
                f"**{i + 1}. {heading}**\n{snippet} [{chunk.chunk_id}]\n"
            )

        answer_parts.append(
            "\n以上信息均来源于检索到的参考资料，如需更详细信息请进一步查阅原文档。"
        )

        return "\n".join(answer_parts)
