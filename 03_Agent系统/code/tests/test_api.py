"""接口集成：用 HTTP 请求提交/审批，同一后台图确实从断点恢复。"""

import asyncio
import json
import threading

import httpx
import pytest
from langchain_core.messages import AIMessage

import api
from demo import OfflineModel, create_demo_graph
from graph import create_graph_in_memory


@pytest.fixture(autouse=True)
def clean_tasks():
    api._task_store.clear()
    yield
    api._task_store.clear()


@pytest.mark.parametrize(
    "approved,verdict,count", [(True, "PASS", 4), (False, "REJECTED", 0)]
)
def test_http_resume_uses_original_graph(monkeypatch, approved, verdict, count):
    monkeypatch.setattr(
        api, "create_graph_in_memory", lambda: create_demo_graph("danger")
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            response = await client.post("/run", json={"query": "验证切换"})
            task_id = response.json()["task_id"]
            store = api._task_store[task_id]
            await store["runner"]
            assert store["status"] == "hitl_pending" and store["result"] is None
            response = await client.post(
                f"/resume/{task_id}",
                json={"approved": approved, "feedback": "人工决定"},
            )
            assert response.status_code == 200
            duplicate = await client.post(
                f"/resume/{task_id}", json={"approved": approved}
            )
            assert duplicate.status_code == 409
            await store["runner"]
            result = (await client.get(f"/result/{task_id}")).json()
            assert result["result"]["verdict"] == verdict
            state = store["graph"].get_state(store["config"]).values
            assert len(state["tool_history"]) == count
            stream = (await client.get(f"/stream/{task_id}")).text
            assert '"event": "complete"' in stream and '"event": "stream_end"' in stream
            # 独立订阅可以完整回放，不会互相取走事件。
            assert (await client.get(f"/stream/{task_id}")).text == stream

    asyncio.run(run())


def test_stream_emits_before_graph_finishes(monkeypatch):
    release = threading.Event()

    class SlowModel(OfflineModel):
        def invoke(self, messages):
            if self.with_tools and len(messages) > 3 and not release.wait(10):
                raise RuntimeError("测试等待超时")
            return super().invoke(messages)

    monkeypatch.setattr(
        api,
        "create_graph_in_memory",
        lambda: create_graph_in_memory(
            llm_factory=lambda tools: SlowModel(with_tools=tools)
        ),
    )

    async def run():
        first_event = asyncio.Event()
        original = api._push_event

        async def capture(task_id, event):
            await original(task_id, event)
            if event.get("node") == "tools":
                first_event.set()

        monkeypatch.setattr(api, "_push_event", capture)
        response = await api.submit_task(api.RunRequest(query="验证切换"))
        store = api._task_store[response["task_id"]]
        try:
            await asyncio.wait_for(first_event.wait(), 10)
            assert store["status"] == "running"
            assert not store["runner"].done()
            assert any(event.get("node") == "tools" for event in store["events"])
        finally:
            release.set()
            await store["runner"]
        assert store["status"] == "complete"

    asyncio.run(run())


def test_http_clarification_and_validation(monkeypatch):
    class Model(OfflineModel):
        def invoke(self, messages):
            if not self.with_tools and "判断对话" in str(messages[0].content):
                return AIMessage(
                    content=json.dumps(
                        {
                            "clear": "Xn" in messages[-1].content,
                            "question": "请补充具体功能",
                        }
                    )
                )
            return super().invoke(messages)

    monkeypatch.setattr(
        api,
        "create_graph_in_memory",
        lambda: create_graph_in_memory(
            llm_factory=lambda tools: Model(with_tools=tools)
        ),
    )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            assert (await client.post("/run", json={"query": "  "})).status_code == 422
            assert (await client.get("/result/missing")).status_code == 404
            response = await client.post(
                "/run", json={"query": "测试", "thread_id": "same"}
            )
            task_id = response.json()["task_id"]
            store = api._task_store[task_id]
            await store["runner"]
            assert store["status"] == "awaiting_clarification"
            assert (
                await client.post("/run", json={"query": "测试", "thread_id": "same"})
            ).status_code == 409
            assert (
                await client.post(
                    f"/resume/{task_id}",
                    json={"approved": True, "patched_state": {"final_result": "PASS"}},
                )
            ).status_code == 422
            assert (
                await client.post(
                    f"/clarify/{task_id}", json={"answer": "验证 Xn 切换回归"}
                )
            ).status_code == 200
            await store["runner"]
            assert store["status"] == "complete"

    asyncio.run(run())
