"""Token 预算管理与动态滑动窗口。

规则：
1. System Prompt 永驻（固定 Token 占用）
2. 最新一轮 Q&A 高优保留
3. 中间轮次从旧到新依次剔除
"""

from __future__ import annotations


class TokenBudgetManager:
    def __init__(self, total_budget: int = 4000, system_reserve: int = 500):
        self.total_budget = total_budget
        self.system_reserve = system_reserve

    def estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数。

        中文约 1.5 字/token，英文约 0.75 词/token。
        生产环境使用 tiktoken 或 Qwen 原生分词器精确计算。
        """
        cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        en_words = len(text.split())
        return int(cn_chars * 1.5 + en_words * 0.75) + 1

    def trim_history(
        self,
        system_prompt: str,
        history: list[dict],
        current_query: str,
    ) -> list[dict]:
        """修剪历史记录使其在 token 预算内。

        保底逻辑：
        1. system_prompt 永驻
        2. 当前 query + 上一轮 Q&A 绝对保留
        3. 更早的轮次从新到旧保留，超预算则剔除
        """
        sys_tokens = self.estimate_tokens(system_prompt)
        query_tokens = self.estimate_tokens(current_query)
        remaining = self.total_budget - sys_tokens - query_tokens

        if remaining <= 0:
            return []

        # 保留最新一轮 Q&A（最后两条消息）
        if len(history) >= 2:
            last_qa = history[-2:]
            last_qa_tokens = sum(
                self.estimate_tokens(m.get("content", "")) for m in last_qa
            )
            remaining -= last_qa_tokens
            older = history[:-2]
        else:
            last_qa = list(history)
            older = []
            remaining -= sum(
                self.estimate_tokens(m.get("content", "")) for m in last_qa
            )

        # 从新到旧尽可能多地保留更早的历史
        kept: list[dict] = []
        for msg in reversed(older):
            t = self.estimate_tokens(msg.get("content", ""))
            if remaining - t >= 0:
                kept.insert(0, msg)
                remaining -= t
            else:
                break

        return kept + last_qa
