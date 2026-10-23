"""FastAPI 教学接口：后台执行真实状态图，逐节点推送 SSE，支持断点审批。

任务、事件和 Checkpoint 均保存在单进程内存中。生产部署还需要身份认证、
审批权限、共享任务存储、持久化 Checkpoint 和任务队列。
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, Field

from config import RECURSION_LIMIT
from graph import create_graph_in_memory
from state import initial_state
from teaching import NODE_GUIDE, explain_step, next_node


class RunRequest(BaseModel):
    query: str = Field(min_length=1)
    thread_id: str = ""
    site_id: str = "SITE-DEFAULT"


class ResumeRequest(BaseModel):
    # 审批不是任意写 State 的后门。修改工具参数应重新提交明确的测试请求。
    model_config = ConfigDict(extra="forbid")
    approved: bool = Field(strict=True)
    feedback: str = ""


class ClarificationRequest(BaseModel):
    answer: str = Field(min_length=1)


class DemoRequest(RunRequest):
    scenario: Literal["pass", "regression", "danger", "review"] = "pass"
    step_mode: bool = True


app = FastAPI(title="5G Agent 教学 API")
_task_store: dict[str, dict] = {}
logger = logging.getLogger(__name__)
WEB_DIR = Path(__file__).resolve().parent / "web"
app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
async def learning_ui():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/demo/config")
async def demo_config():
    return {"nodes": NODE_GUIDE}


def _now():
    return datetime.now(UTC).isoformat()


def _store(task_id):
    if task_id not in _task_store:
        raise HTTPException(404, "任务不存在")
    return _task_store[task_id]


async def _push_event(task_id, event):
    store = _task_store[task_id]
    store["events"].append(
        {"sequence": len(store["events"]), "timestamp": _now(), **event}
    )
    store["changed"].set()


async def _run_agent_async(task_id, graph_input):
    """在线程中迭代同步图，每个节点完成后立即向事件循环投递。

    不能使用 list(graph.stream(...))：那会等整张图结束才发送所有事件。
    run_coroutine_threadsafe 保证 asyncio 状态只在所属事件循环中更新。
    """
    store = _store(task_id)
    graph, config = store["graph"], store["config"]
    loop = asyncio.get_running_loop()
    pending_review = None

    async def on_chunk(chunk, before, after):
        nonlocal pending_review
        for name, update in chunk.items():
            if name == "__interrupt__":
                # interrupt_after 的静态暂停不携带人工审批 payload。
                if not update:
                    continue
                # 等流结束、Checkpoint 提交后才开放审批，避免抢先恢复。
                pending_review = update[0].value
                continue
            await _push_event(
                task_id,
                {
                    "event": "node_update",
                    "node": name,
                    "confidence_score": update.get("confidence_score"),
                    "input": jsonable_encoder(before),
                    "update": jsonable_encoder(update),
                    "state": jsonable_encoder(after),
                    "next_node": next_node(name, after),
                    "explanation": explain_step(name, after),
                },
            )

    def run():
        before, pending = {}, {}
        # updates 是节点增量，values 是 reducer 合并后的真实状态。
        # 两者配对，避免从尚未提交的 Checkpoint 猜测本步的输出。
        for mode, chunk in graph.stream(
            graph_input, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                if "__interrupt__" in chunk:
                    asyncio.run_coroutine_threadsafe(
                        on_chunk(chunk, before, before), loop
                    ).result()
                else:
                    pending = chunk
            else:
                after = {
                    key: value for key, value in chunk.items() if key != "__interrupt__"
                }
                if pending:
                    asyncio.run_coroutine_threadsafe(
                        on_chunk(pending, before, after), loop
                    ).result()
                    pending = {}
                before = after
        return graph.get_state(config)

    try:
        snapshot = await asyncio.to_thread(run)
        if snapshot.next:
            # interrupt 不是完成；保留图实例、thread_id 和完整 Checkpoint。
            if any(task.interrupts for task in snapshot.tasks):
                store["status"] = "hitl_pending"
                await _push_event(
                    task_id, {"event": "hitl_required", "review": pending_review}
                )
            else:
                store["status"] = "step_paused"
                await _push_event(
                    task_id, {"event": "step_paused", "next_nodes": snapshot.next}
                )
        elif snapshot.values.get("needs_clarification"):
            store["status"] = "awaiting_clarification"
            await _push_event(
                task_id,
                {
                    "event": "clarification_required",
                    "question": snapshot.values["clarification_question"],
                },
            )
        elif snapshot.values.get("final_result"):
            store["result"] = json.loads(snapshot.values["final_result"])
            store["status"] = (
                "rejected" if store["result"]["verdict"] == "REJECTED" else "complete"
            )
            await _push_event(task_id, {"event": "complete", **store["result"]})
            await _push_event(task_id, {"event": "stream_end"})
        else:
            raise RuntimeError("图结束但未生成结果或澄清问题")
    except Exception as exc:
        logger.exception("任务 %s 执行失败", task_id)
        store["status"] = "error"
        await _push_event(task_id, {"event": "error", "message": str(exc)})
        await _push_event(task_id, {"event": "stream_end"})


def _start(task_id, graph_input):
    store = _store(task_id)
    # 创建后台任务前同步设置状态，阻止并发重复审批同一个断点。
    store["status"] = "running"
    store["runner"] = asyncio.create_task(_run_agent_async(task_id, graph_input))


def _create_task_graph():
    if os.getenv("AGENT_MODE", "online") == "offline":
        from demo import create_demo_graph

        return create_demo_graph(os.getenv("AGENT_SCENARIO", "pass"))
    return create_graph_in_memory()


@app.post("/run")
async def submit_task(req: RunRequest):
    return _submit_task(req, _create_task_graph())


@app.post("/demo/run")
async def submit_demo(req: DemoRequest):
    from demo import create_demo_graph

    return _submit_task(req, create_demo_graph(req.scenario, step_mode=req.step_mode))


def _submit_task(req, graph):
    if not req.query.strip():
        raise HTTPException(422, "query 不能为空白")
    task_id = uuid.uuid4().hex
    thread_id = req.thread_id or task_id
    if any(s["thread_id"] == thread_id for s in _task_store.values()):
        raise HTTPException(409, "thread_id 已存在，请通过原任务的审批/澄清接口继续")
    _task_store[task_id] = {
        "thread_id": thread_id,
        "site_id": req.site_id,
        "graph": graph,
        "config": {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": RECURSION_LIMIT,
        },
        "status": "pending",
        "result": None,
        "events": [],
        "changed": asyncio.Event(),
    }
    _start(task_id, initial_state(req.query))
    return {"task_id": task_id, "thread_id": thread_id}


@app.post("/step/{task_id}")
async def step_task(task_id: str):
    if _store(task_id)["status"] != "step_paused":
        raise HTTPException(
            409, "仅教学单步暂停的任务可以执行下一步；人工审批请使用审批接口"
        )
    _start(task_id, None)
    return {"task_id": task_id, "message": "正在执行下一节点"}


@app.post("/resume/{task_id}")
async def resume_task(task_id: str, req: ResumeRequest):
    store = _store(task_id)
    if store["status"] != "hitl_pending":
        raise HTTPException(409, "仅等待审批的任务可以恢复")
    _start(task_id, Command(resume=req.model_dump()))
    return {"task_id": task_id, "message": "已提交审批决定，正在从断点继续"}


@app.post("/clarify/{task_id}")
async def clarify_task(task_id: str, req: ClarificationRequest):
    if _store(task_id)["status"] != "awaiting_clarification":
        raise HTTPException(409, "任务未在等待澄清")
    if not req.answer.strip():
        raise HTTPException(422, "补充信息不能为空白")
    _start(task_id, {"messages": [HumanMessage(content=req.answer)]})
    return {"task_id": task_id, "message": "已追加补充信息"}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    store = _store(task_id)
    return {"task_id": task_id, "status": store["status"], "result": store["result"]}


@app.get("/stream/{task_id}")
async def stream_task(task_id: str):
    store = _store(task_id)

    async def events():
        # 每个订阅者有独立游标，重连会回放历史；不会互相消费掉消息。
        index = 0
        yield 'data: {"event": "connected"}\n\n'
        while True:
            store["changed"].clear()
            while index < len(store["events"]):
                event = store["events"][index]
                index += 1
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["event"] == "stream_end":
                    return
            try:
                await asyncio.wait_for(store["changed"].wait(), 15)
            except TimeoutError:
                yield ": heartbeat\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_tasks": sum(s["status"] == "running" for s in _task_store.values()),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
