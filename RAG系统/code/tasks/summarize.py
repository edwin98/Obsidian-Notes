"""Celery 异步摘要任务：对话历史超长时触发压缩。

当用户在一个 Session 中深度探讨，历史 Token 超过预算时：
1. Celery 调度异步任务
2. 调用轻量级模型（Qwen3-4B）生成历史摘要
3. 将摘要作为 system 消息替换旧历史
4. 覆盖 Redis 缓存
"""

from __future__ import annotations

import json
import logging

import redis

from config.settings import Settings
from generation.token_budget import TokenBudgetManager
from tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=10,
    name="tasks.summarize_history",
)
def summarize_history(
    self,
    user_id: str,
    session_id: str,
    budget_threshold: int = 4000,
) -> dict:
    """检查并压缩对话历史。

    返回 {"summarized": True/False, "token_before": int, "token_after": int}
    """
    settings = Settings()
    redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    tbm = TokenBudgetManager(
        total_budget=settings.token_budget_total,
        system_reserve=settings.system_prompt_token_reserve,
    )

    session_key = f"session:{user_id}:{session_id}:messages"
    raw_messages = redis_client.lrange(session_key, 0, -1)
    messages = [json.loads(m) for m in raw_messages]

    # 计算当前总 token
    total_tokens = sum(tbm.estimate_tokens(m.get("content", "")) for m in messages)
    logger.info(
        "[Celery] 检查 Session %s: %d 条消息, %d tokens",
        session_key,
        len(messages),
        total_tokens,
    )

    if total_tokens <= budget_threshold:
        return {
            "summarized": False,
            "token_before": total_tokens,
            "token_after": total_tokens,
        }

    # 执行摘要压缩（Mock LLM，生产环境调用 Qwen3-4B）
    summary = _mock_summarize(messages)

    # 替换旧历史：摘要 + 保留最近 4 条
    summary_msg = {"role": "system", "content": f"前情提要: {summary}"}
    recent = messages[-4:] if len(messages) >= 4 else messages
    new_messages = [summary_msg] + recent

    # 写回 Redis
    pipe = redis_client.pipeline()
    pipe.delete(session_key)
    for msg in new_messages:
        pipe.rpush(session_key, json.dumps(msg, ensure_ascii=False))
    pipe.expire(session_key, settings.session_ttl_seconds)
    pipe.execute()

    token_after = sum(tbm.estimate_tokens(m.get("content", "")) for m in new_messages)
    logger.info(
        "[Celery] 摘要压缩完成: %d tokens -> %d tokens",
        total_tokens,
        token_after,
    )
    return {
        "summarized": True,
        "token_before": total_tokens,
        "token_after": token_after,
    }


def _mock_summarize(messages: list[dict]) -> str:
    """Mock 摘要生成。生产环境替换为 Qwen3-4B 调用。

    Prompt: "请总结上述用户与 AI 的交互核心技术点与已确认的客观事实，需以精简的要点呈现。"
    """
    topics = []
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            topics.append(content[:50])

    summary = "用户先后探讨了以下技术主题：" + "；".join(topics[:8])
    if len(topics) > 8:
        summary += f"等共 {len(topics)} 个问题"
    return summary
