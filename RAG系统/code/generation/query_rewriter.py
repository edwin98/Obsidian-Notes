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
    """指代消解 + 问题扩展。"""

    def rewrite(self, query: str, history: list[dict] | None = None) -> list[str]:
        """返回 1~3 个改写后的查询（原始查询始终在首位）。"""
        queries = [query]

        # 1. 指代消解
        if history:
            resolved = self._resolve_references(query, history)
            if resolved and resolved != query:
                queries.append(resolved)

        # 2. 缩写扩展
        expanded = self._expand_abbreviations(query)
        if expanded and expanded != query and expanded not in queries:
            queries.append(expanded)

        # 3. 同义改写
        paraphrased = self._paraphrase(query)
        if paraphrased and paraphrased not in queries:
            queries.append(paraphrased)

        queries = queries[:3]
        logger.info("[Rewrite] '%s' -> %s", query, queries)
        return queries

    def _resolve_references(self, query: str, history: list[dict]) -> str | None:
        """指代消解：将代词还原为上文提到的具体概念。"""
        pronouns = ["它", "这个", "那个", "该技术", "该方案", "这种", "那种", "上述"]

        has_pronoun = any(p in query for p in pronouns)
        if not has_pronoun:
            return None

        # 从历史中提取最近的主题
        last_topic = None
        for msg in reversed(history):
            if msg.get("role") == "user":
                # 提取名词短语作为主题
                content = msg["content"]
                # 简单提取：取「是什么」前面的部分，或取整个问题
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
        """同义改写（Demo 规则版，生产环境由 LLM 生成）。"""
        # 问句转换
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
