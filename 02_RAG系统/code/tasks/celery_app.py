"""Celery 异步任务配置与定义。

使用 Redis 作为 Broker 和 Result Backend。
主要任务：对话历史超长时触发摘要压缩。
"""

from __future__ import annotations

from celery import Celery

from config.settings import Settings

settings = Settings()

celery_app = Celery(
    "rag_tasks",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    # 指数退避重试
    task_acks_late=True,
    task_reject_on_worker_lost=True,
)
