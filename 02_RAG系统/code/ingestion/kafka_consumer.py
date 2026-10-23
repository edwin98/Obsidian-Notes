"""Kafka Consumer：消费 chunk 消息并执行 Embedding + 索引。

Consumer Group 分区隔离：
- CPU 组消费 topic_text_slice
- GPU 组消费 topic_gpu_task（Demo 中合并处理）
"""

from __future__ import annotations

import json
import logging

from kafka import KafkaConsumer as _KafkaConsumer

from config.settings import Settings

logger = logging.getLogger(__name__)


class KafkaChunkConsumer:
    def __init__(
        self,
        settings: Settings,
        topic: str | None = None,
        group_id: str = "rag_indexer",
    ):
        self.settings = settings
        self.topic = topic or settings.kafka_topic_text
        self.group_id = group_id
        self._consumer: _KafkaConsumer | None = None

    def connect(self) -> None:
        self._consumer = _KafkaConsumer(
            self.topic,
            bootstrap_servers=self.settings.kafka_bootstrap_servers,
            group_id=self.group_id,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            consumer_timeout_ms=5000,
        )
        logger.info(
            "[Kafka] Consumer 已连接: topic=%s, group=%s", self.topic, self.group_id
        )

    def consume_batch(self, max_records: int = 100) -> list[dict]:
        """批量消费消息。"""
        if self._consumer is None:
            self.connect()

        messages = []
        for message in self._consumer:
            messages.append(message.value)
            if len(messages) >= max_records:
                break

        logger.info("[Kafka] 消费 %d 条消息 from %s", len(messages), self.topic)
        return messages

    def close(self) -> None:
        if self._consumer:
            self._consumer.close()
            self._consumer = None
