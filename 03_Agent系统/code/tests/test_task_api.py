import asyncio

import httpx
import pytest

import api
import task_api
from task_service import TaskService


@pytest.fixture(autouse=True)
def fresh_service(monkeypatch):
    instance = TaskService()
    monkeypatch.setattr(task_api, "_service", instance)
    monkeypatch.setattr(task_api, "_workers", {})
    monkeypatch.setattr(task_api, "_automatic", {})
    yield
    instance.store.close()


def test_unified_api_query_steps_report_and_message_conflict():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            response = await client.post(
                "/tasks/messages",
                json={
                    "message_id": "create",
                    "text": "复现升级后移动视频卡顿",
                    "auto_run": False,
                },
            )
            assert response.status_code == 200
            task_id = response.json()["task_id"]
            query = await client.post(
                "/tasks/messages",
                json={
                    "message_id": "query",
                    "text": "现在怎么样了",
                    "target_task_id": task_id,
                },
            )
            assert not query.json()["run_requested"]
            assert not task_api._workers
            for _ in range(9):
                result = await client.post(f"/tasks/{task_id}/step")
                assert result.status_code == 200
                if result.json()["status"] != "READY":
                    break
            assert result.json()["status"] == "COMPLETED"
            report = (await client.get(f"/tasks/{task_id}/report")).json()
            assert report["test_verdict"] == "FAIL"
            assert len((await client.get(f"/tasks/{task_id}/experiments")).json()) == 7
            conflict = await client.post(
                "/tasks/messages",
                json={
                    "message_id": "create",
                    "text": "取消",
                    "auto_run": False,
                },
            )
            assert conflict.status_code == 409
            assert (await client.get("/tasks/missing")).status_code == 404
            assert (
                await client.post(
                    "/tasks/messages", json={"message_id": "blank", "text": " "}
                )
            ).status_code == 422

    asyncio.run(run())


def test_ui_and_health_use_only_task_service():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            assert "测试与排查助手" in (await client.get("/")).text
            assert (await client.get("/health")).json() == {
                "status": "ok",
                "active_tasks": 0,
            }
            for name in ("tasks.js", "tasks.css"):
                assert (await client.get("/assets/" + name)).status_code == 200
            for path in (
                "/legacy",
                "/demo/config",
                "/assets/app.js",
                "/assets/index.html",
            ):
                assert (await client.get(path)).status_code == 404
            for path in ("/run", "/demo/run"):
                assert (
                    await client.post(path, json={"query": "测试"})
                ).status_code == 404

    asyncio.run(run())


def test_default_background_execution_and_duplicate_delivery():
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=api.app), base_url="http://test"
        ) as client:
            payload = {"message_id": "automatic", "text": "复现移动视频卡顿"}
            response = await client.post("/tasks/messages", json=payload)
            assert response.status_code == 200
            task_id = response.json()["task_id"]
            await task_api._workers[task_id]
            task = (await client.get(f"/tasks/{task_id}")).json()
            assert task["status"] == "COMPLETED"
            assert len(task["experiments"]) == 7
            duplicate = (await client.post("/tasks/messages", json=payload)).json()
            assert duplicate["duplicate"]
            assert not duplicate["run_requested"]
            assert len(task_api.service().get(task_id).experiments) == 7

    asyncio.run(run())
