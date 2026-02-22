"""查询改写：指代消解 + 问题扩展。

使用 Prompt + LLM 方案，Demo 中使用规则模拟。
生产环境使用 Qwen3-4B 执行改写（单次 < 1s）。
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ---- Prompt 模板（展示完整结构，生产环境发送给 Qwen3-4B）----

SYSTEM_PROMPT = """你是一个专业的查询改写助手，服务于无线通信领域的 RAG 检索系统。
你的任务是：
1. 指代消解：将多轮对话中的代词（它、这个、那个、该技术）还原为具体概念
2. 问题扩展：将用户问题扩展为 1~3 个语义等价或相关的检索问题，提高检索命中率

输出格式（JSON）:
{
  "resolved_query": "消解指代后的完整问题",
  "expanded_queries": ["扩展问题1", "扩展问题2"]
}"""

USER_PROMPT_TEMPLATE = """对话历史：
{history}

当前问题：{query}

请执行指代消解和问题扩展。注意：
- 术语和关键词保留原文
- 缩写需扩展为全称（如 CA -> 载波聚合）
- 排序逻辑：原始问题优先，扩展问题按相关性降序"""


# ---- 常见缩写词表 ----

ABBREVIATION_MAP = {
    "CA": "载波聚合",
    "MIMO": "多输入多输出",
    "PRACH": "物理随机接入信道",
    "HARQ": "混合自动重传请求",
    "RRC": "无线资源控制",
    "NR": "New Radio",
    "gNB": "gNodeB 基站",
    "SSB": "同步信号块",
    "BWP": "带宽部分",
    "UE": "用户设备",
    "DCI": "下行控制信息",
    "RAR": "随机接入响应",
    "RACH": "随机接入信道",
    "PDCCH": "物理下行控制信道",
    "PDSCH": "物理下行共享信道",
}


class QueryRewriter:
    """指代消解 + 问题扩展。

    优先调用与 LLM 生成器相同的 Gemini 模型；
    如未配置或调用失败，自动降级到规则模式。
    """

    def __init__(self, settings=None):
        self.settings = settings
        self._client = None

    def _get_client(self):
        """懒加载 Gemini 同步客户端（与 GeminiLLMGenerator 共用同一 API Key）。"""
        if self._client is None:
            try:
                from google import genai
            except ImportError as e:
                raise ImportError(
                    "缺少 google-genai 依赖，请执行: uv add google-genai"
                ) from e
            self._client = genai.Client(api_key=self.settings.gemini_api_key)
        return self._client

    def rewrite(self, query: str, history: list[dict] | None = None) -> list[str]:
        """返回 1~3 个改写后的查询（首位为最优查询）。"""
        if self.settings and self.settings.llm_provider == "gemini":
            try:
                result = self._rewrite_with_gemini(query, history)
                logger.info("[Rewrite][Gemini] '%s' -> %s", query, result)
                return result
            except Exception as e:
                logger.warning("[Rewrite] Gemini 改写失败，降级到规则模式: %s", e)

        result = self._rewrite_rule_based(query, history)
        logger.info("[Rewrite][Rule] '%s' -> %s", query, result)
        return result

    def _rewrite_with_gemini(
        self, query: str, history: list[dict] | None = None
    ) -> list[str]:
        """调用 Gemini 执行指代消解和问题扩展，返回改写查询列表。"""
        import json

        from google.genai import types

        client = self._get_client()

        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in (history or [])[-4:]
        )
        prompt = USER_PROMPT_TEMPLATE.format(
            history=history_text or "（无历史）",
            query=query,
        )

        response = client.models.generate_content(
            model=self.settings.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
                max_output_tokens=512,
                response_mime_type="application/json",  # 强制输出裸 JSON，无 fence 无前缀
            ),
        )

        text = (response.text or "").strip()
        if not text:
            raise ValueError("Gemini 返回空响应")

        data = json.loads(text)

        queries: list[str] = []

        # resolved_query 作为首选（已消解指代的完整问题）
        resolved = (data.get("resolved_query") or "").strip()
        queries.append(resolved if resolved else query)

        # 追加扩展问题
        for eq in data.get("expanded_queries") or []:
            eq = eq.strip()
            if eq and eq not in queries:
                queries.append(eq)

        return queries[:3]

    # ── 规则降级实现 ────────────────────────────────────────────────────────

    def _rewrite_rule_based(
        self, query: str, history: list[dict] | None = None
    ) -> list[str]:
        queries = [query]

        if history:
            resolved = self._resolve_references(query, history)
            if resolved and resolved != query:
                queries.append(resolved)

        expanded = self._expand_abbreviations(query)
        if expanded and expanded != query and expanded not in queries:
            queries.append(expanded)

        paraphrased = self._paraphrase(query)
        if paraphrased and paraphrased not in queries:
            queries.append(paraphrased)

        return queries[:3]

    def _resolve_references(self, query: str, history: list[dict]) -> str | None:
        """指代消解：将代词还原为上文提到的具体概念。"""
        pronouns = ["它", "这个", "那个", "该技术", "该方案", "这种", "那种", "上述"]

        has_pronoun = any(p in query for p in pronouns)
        if not has_pronoun:
            return None

        last_topic = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                content = msg["content"]
                match = re.search(r"(.+?)(?:是什么|有什么|怎么|如何)", content)
                if match:
                    last_topic = match.group(1).strip()
                else:
                    last_topic = content.strip()[:20]
                break

        if not last_topic:
            return None

        resolved = query
        for pronoun in pronouns:
            resolved = resolved.replace(pronoun, last_topic)
        return resolved

    def _expand_abbreviations(self, query: str) -> str | None:
        """将缩写扩展为全称。"""
        expanded = query
        for abbr, full in ABBREVIATION_MAP.items():
            if abbr in expanded:
                expanded = expanded.replace(abbr, f"{abbr}({full})")
        return expanded if expanded != query else None

    def _paraphrase(self, query: str) -> str | None:
        """同义改写（规则版）。"""
        replacements = [
            ("是什么", "的定义和概念"),
            ("怎么工作", "的工作原理"),
            ("有什么优势", "的优点和好处"),
            ("有什么区别", "之间的差异对比"),
            ("如何配置", "的配置方法和步骤"),
        ]
        for old, new in replacements:
            if old in query:
                return query.replace(old, new)
        return None
