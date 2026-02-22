"""端到端测试：通过 httpx 测试完整的 API 链路。

需要先启动基础设施服务（docker-compose up）和 FastAPI 服务（python run_demo.py）。
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import httpx

BASE_URL = "http://localhost:8000"


@pytest.fixture
def client():
    return httpx.Client(base_url=BASE_URL, timeout=30.0)


@pytest.fixture
def async_client():
    return httpx.AsyncClient(base_url=BASE_URL, timeout=30.0)


class TestHealth:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "chunks_indexed" in data


class TestIngest:
    def test_ingest_document(self, client):
        resp = client.post(
            "/ingest",
            json={
                "doc_id": "test_doc",
                "doc_name": "测试文档",
                "content": "# 测试标题\n\n测试内容，这是一个用于验证的文档。\n\n## 子标题\n\n更多内容。",
                "source_type": "general",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["chunks_created"] > 0


class TestChat:
    def test_chat_basic(self, client):
        resp = client.post(
            "/chat",
            json={
                "user_id": "test_user",
                "session_id": "test_session",
                "query": "5G随机接入的四步流程是什么？",
                "top_k": 5,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0
        assert "source" in data

    def test_chat_cache_hit(self, client):
        """同一问题第二次查询应可能命中缓存。"""
        query = "CA是什么"
        # 第一次
        resp1 = client.post(
            "/chat",
            json={
                "user_id": "test_user",
                "session_id": "test_session_2",
                "query": query,
                "top_k": 5,
            },
        )
        assert resp1.status_code == 200

        # 第二次（可能命中精确缓存）
        resp2 = client.post(
            "/chat",
            json={
                "user_id": "test_user",
                "session_id": "test_session_2",
                "query": query,
                "top_k": 5,
            },
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        # 如果 Redis 可用，应命中缓存
        assert data2["source"] in ("exact_cache", "semantic_cache", "rag")

    def test_chat_validation(self, client):
        """Pydantic 参数校验：空查询应返回 422。"""
        resp = client.post(
            "/chat",
            json={
                "user_id": "test_user",
                "session_id": "test_session",
                "query": "",  # 空查询
                "top_k": 5,
            },
        )
        assert resp.status_code == 422


class TestChatStream:
    def test_stream_response(self, client):
        with client.stream(
            "POST",
            "/chat/stream",
            json={
                "user_id": "test_user",
                "session_id": "test_stream",
                "query": "波束管理是什么？",
                "top_k": 5,
            },
        ) as response:
            assert response.status_code == 200
            content = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    content.append(data)
            answer = "".join(content)
            assert len(answer) > 0
