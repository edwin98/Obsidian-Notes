"""Redis 多级缓存与状态管理。

三级缓存机制：
1. Session State：List 结构持久化会话，Hash 存元信息，TTL 2小时
2. 精确匹配缓存：MD5(query) -> answer，TTL 24小时
3. 语义相似度缓存：向量余弦相似度 >= 0.92 时命中

缓存穿透防范：分布式锁保障只允许一个请求穿透到后端。
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import redis

from config.settings import Settings

logger = logging.getLogger(__name__)


class RedisCache:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.session_ttl = settings.session_ttl_seconds
        self.cache_ttl = settings.cache_ttl_seconds
        self.semantic_threshold = settings.semantic_cache_threshold
        self._redis: redis.Redis | None = None

    def connect(self) -> None:
        self._redis = redis.from_url(self.settings.redis_url, decode_responses=True)
        logger.info("[Redis] 已连接: %s", self.settings.redis_url)

    @property
    def client(self) -> redis.Redis:
        if self._redis is None:
            self.connect()
        return self._redis

    # ---- Session State (List + Hash) ----

    def _session_key(self, user_id: str, session_id: str) -> str:
        return f"session:{user_id}:{session_id}:messages"

    def push_message(self, user_id: str, session_id: str, message: dict) -> None:
        """追加消息到会话历史。"""
        key = self._session_key(user_id, session_id)
        self.client.rpush(key, json.dumps(message, ensure_ascii=False))
        self.client.expire(key, self.session_ttl)

    def get_session_messages(self, user_id: str, session_id: str) -> list[dict]:
        """获取完整会话历史。"""
        key = self._session_key(user_id, session_id)
        raw_messages = self.client.lrange(key, 0, -1)
        return [json.loads(m) for m in raw_messages]

    def trim_session(self, user_id: str, session_id: str, keep_last: int = 20) -> None:
        """修剪会话历史，保留最近 N 条。"""
        key = self._session_key(user_id, session_id)
        self.client.ltrim(key, -keep_last, -1)

    def replace_session(
        self, user_id: str, session_id: str, messages: list[dict]
    ) -> None:
        """替换整个会话历史（摘要压缩后使用）。"""
        key = self._session_key(user_id, session_id)
        pipe = self.client.pipeline()
        pipe.delete(key)
        for msg in messages:
            pipe.rpush(key, json.dumps(msg, ensure_ascii=False))
        pipe.expire(key, self.session_ttl)
        pipe.execute()

    # ---- 精确匹配缓存 (MD5) ----

    def _exact_cache_key(self, query: str) -> str:
        md5 = hashlib.md5(query.strip().lower().encode()).hexdigest()
        return f"cache:exact:{md5}"

    def get_exact_cache(self, query: str) -> str | None:
        """精确匹配缓存查询。"""
        key = self._exact_cache_key(query)
        result = self.client.get(key)
        if result:
            logger.info("[Cache] 精确缓存命中: %s", query[:30])
        return result

    def set_exact_cache(self, query: str, answer: str) -> None:
        """写入精确匹配缓存。"""
        key = self._exact_cache_key(query)
        self.client.setex(key, self.cache_ttl, answer)

    # ---- 语义相似度缓存 ----

    def get_semantic_cache(self, query_vector: list[float]) -> str | None:
        """语义缓存查询：遍历已缓存的向量，余弦相似度 >= 阈值则命中。"""
        keys = self.client.keys("cache:semantic:*")
        if not keys:
            return None

        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return None
        q = q / q_norm

        for key in keys:
            data = self.client.hgetall(key)
            if not data:
                continue
            cached_vec = np.array(json.loads(data["vector"]), dtype=np.float32)
            cached_norm = np.linalg.norm(cached_vec)
            if cached_norm == 0:
                continue
            cached_vec = cached_vec / cached_norm
            sim = float(np.dot(q, cached_vec))
            if sim >= self.semantic_threshold:
                logger.info("[Cache] 语义缓存命中: sim=%.4f", sim)
                return data["answer"]

        return None

    def set_semantic_cache(
        self, query: str, query_vector: list[float], answer: str
    ) -> None:
        """写入语义缓存。"""
        md5 = hashlib.md5(query.strip().lower().encode()).hexdigest()
        key = f"cache:semantic:{md5}"
        self.client.hset(
            key,
            mapping={
                "query": query,
                "vector": json.dumps(query_vector),
                "answer": answer,
            },
        )
        self.client.expire(key, self.cache_ttl)

    # ---- 分布式锁（缓存穿透防范）----

    def acquire_lock(self, lock_name: str, timeout: int = 10) -> bool:
        """获取分布式锁。"""
        return bool(self.client.set(f"lock:{lock_name}", "1", nx=True, ex=timeout))

    def release_lock(self, lock_name: str) -> None:
        """释放分布式锁。"""
        self.client.delete(f"lock:{lock_name}")
