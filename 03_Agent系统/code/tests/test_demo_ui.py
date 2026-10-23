"""教学界面：真实单步、状态快照、离线场景与人工审批边界。"""

import asyncio
import json

import httpx
import pytest

import api


@pytest.fixture(autouse=True)
def clean_demo_tasks():
    api._task_store.clear()
    yield
    api._task_store.clear()


def client():
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    )


async def settled(task_id):
    store = api._task_store[task_id]
    await store["runner"]
    assert store["status"] != "error", store["events"]
    return store


def test_ui_and_assets_are_served_without_api_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    async def run():
        async with client() as http:
            page = await http.get("/")
            assert page.status_code == 200
            assert "Agent 运行实验室" in page.text
            for name in ("app.js", "style.css"):
                assert (await http.get(f"/assets/{name}")).status_code == 200
            guides = (await http.get("/demo/config")).json()["nodes"]
            assert set(guides) == {
                "clarify",
                "agent",
                "guardrail",
                "tools",
                "hitl",
                "result_judge",
            }

    asyncio.run(run())


def test_step_executes_one_node_and_snapshots_use_real_reducer():
    async def run():
        async with client() as http:
            response = await http.post("/demo/run", json={"query": "验证 Xn 切换"})
            task_id = response.json()["task_id"]
            store = await settled(task_id)
            assert store["status"] == "step_paused"
            for expected in ("clarify", "agent", "guardrail", "tools"):
                events = [e for e in store["events"] if e["event"] == "node_update"]
                event = events[-1]
                assert event["node"] == expected
                assert event["input"]["messages"][0]["content"] == "验证 Xn 切换"
                assert event["state"]["current_step"] == expected
                if expected == "agent":
                    assert event["input"] == events[-2]["state"]
                    assert len(event["update"]["messages"]) == 1
                    assert len(event["state"]["messages"]) == 2
                    assert not event["state"]["tool_history"]
                    assert event["next_node"] == "guardrail"
                if expected == "guardrail":
                    assert not event["state"]["tool_history"]
                if expected == "tools":
                    assert len(event["state"]["tool_history"]) == 1
                    call = event["input"]["messages"][-1]["tool_calls"][0]
                    observation = event["update"]["messages"][0]
                    assert observation["tool_call_id"] == call["id"]
                    assert observation["type"] == "tool"
                    break
                count = len(events)
                assert (await http.post(f"/step/{task_id}")).status_code == 200
                # _start 同步切换 running，连续点击不会重复执行。
                assert (await http.post(f"/step/{task_id}")).status_code == 409
                store = await settled(task_id)
                assert (
                    len([e for e in store["events"] if e["event"] == "node_update"])
                    == count + 1
                )

    asyncio.run(run())


@pytest.mark.parametrize(
    "scenario,approved,verdict,tool_count",
    [
        ("pass", None, "PASS", 4),
        ("regression", None, "FAIL", 5),
        ("danger", True, "PASS", 4),
        ("danger", False, "REJECTED", 0),
        ("review", True, "PASS", 4),
        ("review", False, "REJECTED", 4),
    ],
)
@pytest.mark.parametrize("step_mode", [True, False])
def test_demo_scenarios_preserve_real_approval(
    scenario, approved, verdict, tool_count, step_mode
):
    async def run():
        async with client() as http:
            response = await http.post(
                "/demo/run",
                json={
                    "query": "验证 Xn 切换",
                    "scenario": scenario,
                    "step_mode": step_mode,
                },
            )
            task_id = response.json()["task_id"]
            reviews = 0
            for _ in range(35):
                store = await settled(task_id)
                if store["status"] in {"complete", "rejected"}:
                    break
                if store["status"] == "hitl_pending":
                    reviews += 1
                    assert (await http.post(f"/step/{task_id}")).status_code == 409
                    assert store["events"][-1]["review"]["reason"]
                    if scenario == "danger":
                        assert (
                            not store["graph"]
                            .get_state(store["config"])
                            .values["tool_history"]
                        )
                    response = await http.post(
                        f"/resume/{task_id}",
                        json={"approved": approved, "feedback": "界面审核"},
                    )
                else:
                    assert step_mode and store["status"] == "step_paused"
                    response = await http.post(f"/step/{task_id}")
                assert response.status_code == 200
            else:
                pytest.fail("单步执行未在预期步数内完成")
            assert store["result"]["verdict"] == verdict
            state = store["graph"].get_state(store["config"]).values
            assert len(state["tool_history"]) == tool_count
            assert reviews == (approved is not None)
            if scenario == "review" and approved:
                assert store["result"]["confidence_score"] == 0.4
            events = store["events"]
            assert [event["sequence"] for event in events] == list(range(len(events)))
            serialized = (await http.get(f"/stream/{task_id}")).text
            replayed = [
                json.loads(line[6:])
                for line in serialized.splitlines()
                if line.startswith("data: ")
            ]
            assert replayed[1:] == json.loads(json.dumps(events))
            assert (await http.post(f"/step/{task_id}")).status_code == 409

    asyncio.run(run())


def test_demo_rejects_invalid_scenario_and_blank_query():
    async def run():
        async with client() as http:
            assert (
                await http.post(
                    "/demo/run", json={"query": "测试", "scenario": "unknown"}
                )
            ).status_code == 422
            assert (
                await http.post("/demo/run", json={"query": "  "})
            ).status_code == 422
            assert (await http.post("/step/missing")).status_code == 404
            assert not api._task_store

    asyncio.run(run())
