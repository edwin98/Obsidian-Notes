"""Kafka Producer：文档切片消息投递。

将切片任务按类型分发到不同 Topic：
- topic_text_slice: 通用文本处理（CPU 节点消费）
- topic_gpu_task: 图像/PPT 密集计算（GPU 节点消费）
"""

from __future__ import annotations

import json
import logging

from kafka import KafkaProducer as _KafkaProducer

from config.settings import Settings

logger = logging.getLogger(__name__)


class KafkaChunkProducer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._producer: _KafkaProducer | None = None

    def connect(self) -> None:
        self._producer = _KafkaProducer(
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode(
                "utf-8"
            ),
            acks="all",
            retries=3,
        )
        logger.info(
            "[Kafka] Producer 已连接: %s", self.settings.kafka_bootstrap_servers
        )

    def send_chunk(self, chunk_data: dict, topic: str | None = None) -> None:
        """将 chunk 数据投递到 Kafka Topic。"""
        if self._producer is None:
            self.connect()

        target_topic = topic or self.settings.kafka_topic_text
        self._producer.send(target_topic, value=chunk_data)
        logger.debug(
            "[Kafka] 投递消息到 %s: chunk_id=%s",
            target_topic,
            chunk_data.get("chunk_id"),
        )

    def flush(self) -> None:
        if self._producer:
            self._producer.flush()

    def close(self) -> None:
        if self._producer:
            self._producer.close()
            self._producer = None
