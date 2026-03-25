"""
api.py — FastAPI + SSE 接口层

对应全景报告第七章工程化部署：FastAPI 接口层 + SSE 实时状态推送

架构：
  用户 / CI 系统
      │ POST /run  (测试需求，自然语言/YAML)
      ▼
  FastAPI 接口层
      │ 长时任务 → Celery 异步队列（生产；Demo 模式在线程池执行）
      │ 状态推送 → SSE 长连接（GET /stream/{task_id}）
      ▼
  LangGraph 状态机引擎
      │ 节点状态变化 → 推送至 SSE 队列
      ▼
  前端实时状态展示

主要端点：
  POST /run                  提交测试任务，返回 task_id
  GET  /stream/{task_id}     SSE 长连接，实时推送节点状态
  POST /resume/{task_id}     HITL 审批后恢复 Agent 执行
  GET  /result/{task_id}     查询最终结果（非流式）
  GET  /health               健康检查

SSE 消息格式：
  data: {"event": "node_start", "node": "guardrail", "timestamp": "..."}
  data: {"event": "tool_call", "tool": "simulation_runner", "args": {...}}
  data: {"event": "hitl_required", "reason": "...", "task_id": "..."}
  data: {"event": "complete", "verdict": "PASS", "confidence": 0.92}
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from config import RECURSION_LIMIT


# ═══════════════════════════════════════════════════════════════════
# 第一部分：请求/响应 Schema
# ═══════════════════════════════════════════════════════════════════


class RunRequest(BaseModel):
    query: str = Field(
        ..., description="测试需求（自然语言或 YAML 格式）", min_length=1
    )
    thread_id: str = Field(default="", description="会话 ID，为空时自动生成")
    site_id: str = Field(default="SITE-DEFAULT", description="局点标识符")


class RunResponse(BaseModel):
    task_id: str
    thread_id: str
    message: str


class ResumeRequest(BaseModel):
    approved: bool = True
    feedback: str = ""
    patched_state: dict = Field(
        default_factory=dict, description="人工修正后的 State 字段"
    )


class TaskResult(BaseModel):
    task_id: str
    status: str  # "running" | "complete" | "hitl_pending" | "error"
    verdict: str = ""
    confidence: float = 0.0
    root_cause: str = ""
    issues: list[str] = Field(default_factory=list)
    hitl_reason: str = ""


# ═══════════════════════════════════════════════════════════════════
# 第二部分：任务状态存储（生产用 Redis；Demo 用内存）
# ═══════════════════════════════════════════════════════════════════

# task_id → {"status": ..., "thread_id": ..., "result": ..., "sse_queue": asyncio.Queue}
_task_store: dict[str, dict] = {}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _push_event(task_id: str, event: dict) -> None:
    """向指定 task 的 SSE 队列推送事件"""
    store = _task_store.get(task_id)
    if store and "sse_queue" in store:
        await store["sse_queue"].put(event)


# ═══════════════════════════════════════════════════════════════════
# 第三部分：Agent 执行逻辑（异步包装）
# ═══════════════════════════════════════════════════════════════════


async def _run_agent_async(task_id: str, query: str, thread_id: str) -> None:
    """
    在后台线程中运行 LangGraph Agent，将节点状态变化推送至 SSE 队列。

    生产环境：此函数打包为 Celery task，通过 Kafka 接收回调事件。
    Demo 环境：直接在 asyncio 线程池中运行，通过 Queue 通信。
    """
    from graph import create_graph_in_memory

    graph = create_graph_in_memory()
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "current_step": "start",
        "tool_outputs": {},
        "error_count": 0,
        "confidence_score": 1.0,
        "hitl_required": False,
        "hitl_feedback": "",
        "final_result": None,
    }

    try:
        _task_store[task_id]["status"] = "running"

        # stream_mode="updates" 每个节点执行后推送一次 State 增量
        loop = asyncio.get_event_loop()

        def _stream():
            return list(
                graph.stream(initial_state, config=config, stream_mode="updates")
            )

        chunks = await loop.run_in_executor(None, _stream)

        for chunk in chunks:
            for node_name, update in chunk.items():
                event: dict = {
                    "event": "node_update",
                    "node": node_name,
                    "timestamp": _now(),
                }

                # 工具调用事件
                msgs = update.get("messages", [])
                tool_calls_list = []
                for msg in msgs:
                    tcs = getattr(msg, "tool_calls", []) or []
                    for tc in tcs:
                        tool_calls_list.append(
                            {"tool": tc["name"], "args": tc.get("args", {})}
                        )

                if tool_calls_list:
                    event["event"] = "tool_call"
                    event["tool_calls"] = tool_calls_list

                # HITL 触发事件
                if update.get("hitl_required"):
                    event["event"] = "hitl_required"
                    event["reason"] = "guardrail 或置信度触发"
                    _task_store[task_id]["status"] = "hitl_pending"

                # 置信度更新
                if "confidence_score" in update:
                    event["confidence_score"] = update["confidence_score"]

                # 最终结果
                if update.get("final_result"):
                    try:
                        result_data = json.loads(update["final_result"])
                        event["event"] = "complete"
                        event["verdict"] = result_data.get("verdict", "UNKNOWN")
                        event["confidence"] = result_data.get("confidence_score", 0.0)
                        event["root_cause"] = result_data.get("root_cause", "")
                        event["issues"] = result_data.get("issues", [])
                        _task_store[task_id]["status"] = "complete"
                        _task_store[task_id]["result"] = result_data
                    except json.JSONDecodeError:
                        pass

                await _push_event(task_id, event)

        # 确保 complete 事件发出后 SSE 流关闭
        if _task_store[task_id]["status"] == "running":
            _task_store[task_id]["status"] = "complete"
        await _push_event(task_id, {"event": "stream_end", "timestamp": _now()})

    except Exception as e:
        _task_store[task_id]["status"] = "error"
        await _push_event(
            task_id, {"event": "error", "message": str(e), "timestamp": _now()}
        )
        await _push_event(task_id, {"event": "stream_end", "timestamp": _now()})


# ═══════════════════════════════════════════════════════════════════
# 第四部分：FastAPI 应用
# ═══════════════════════════════════════════════════════════════════


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] 5G Test Agent API 启动")
    yield
    print("[API] 5G Test Agent API 关闭")


app = FastAPI(
    title="5G Test Verification Agent API",
    description="LangGraph 驱动的 5G 智能测试验证 Agent，支持 SSE 实时状态推送与 HITL 审批",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── POST /run ────────────────────────────────────────────────────


@app.post("/run", response_model=RunResponse, summary="提交测试任务")
async def submit_task(req: RunRequest):
    """
    提交测试需求，异步启动 Agent 执行。

    - `query`: 自然语言测试需求，例如 "验证基站A→B Xn切换，近期有配置变更"
    - `thread_id`: 可选，用于断点恢复；为空时自动生成

    返回 `task_id`，用于 SSE 订阅和结果查询。
    """
    task_id = str(uuid.uuid4())
    thread_id = req.thread_id or f"thread-{task_id[:8]}"

    _task_store[task_id] = {
        "status": "pending",
        "thread_id": thread_id,
        "query": req.query,
        "site_id": req.site_id,
        "result": None,
        "sse_queue": asyncio.Queue(),
    }

    # 异步启动 Agent（生产：Celery delay；Demo：asyncio 任务）
    asyncio.create_task(_run_agent_async(task_id, req.query, thread_id))

    return RunResponse(
        task_id=task_id,
        thread_id=thread_id,
        message=f"任务已提交，通过 GET /stream/{task_id} 订阅实时状态",
    )


# ── GET /stream/{task_id} ────────────────────────────────────────


@app.get("/stream/{task_id}", summary="SSE 实时状态流")
async def stream_task(task_id: str):
    """
    SSE 长连接，实时推送 Agent 节点状态。

    消息格式：`data: <json>\n\n`

    事件类型：
    - `node_update`   : 节点执行更新
    - `tool_call`     : 工具调用（含参数）
    - `hitl_required` : 需要人工介入
    - `complete`      : 任务完成（含 verdict/confidence/root_cause）
    - `error`         : 执行错误
    - `stream_end`    : 流结束信号
    """
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' 不存在")

    queue: asyncio.Queue = _task_store[task_id]["sse_queue"]

    async def event_generator() -> AsyncGenerator[str, None]:
        # 发送初始连接确认
        yield f"data: {json.dumps({'event': 'connected', 'task_id': task_id, 'timestamp': _now()})}\n\n"

        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event.get("event") == "stream_end":
                    break
            except asyncio.TimeoutError:
                # 心跳，防止连接超时
                yield f": heartbeat {_now()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Nginx 反代时禁用缓冲
        },
    )


# ── POST /resume/{task_id} ───────────────────────────────────────


@app.post("/resume/{task_id}", summary="HITL 审批后恢复 Agent")
async def resume_task(task_id: str, req: ResumeRequest):
    """
    HITL 审批接口。专家在 Review 平台点击 Approve/Reject 后调用。

    生产实现：
      1. 从 Postgres 拉取 thread_id 对应的 Checkpoint State
      2. 若有修正参数，调用 graph.update_state(config, patch) 注入
      3. 调用 graph.invoke(None, config) 从断点恢复执行
      4. None 表示不修改 State，直接继续

    Demo 实现：
      模拟状态更新，推送 resume 事件。
    """
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' 不存在")

    store = _task_store[task_id]
    if store["status"] != "hitl_pending":
        raise HTTPException(
            status_code=400,
            detail=f"任务状态为 '{store['status']}'，仅 hitl_pending 状态可 Resume",
        )

    if not req.approved:
        store["status"] = "rejected"
        await _push_event(
            task_id,
            {
                "event": "hitl_rejected",
                "feedback": req.feedback,
                "timestamp": _now(),
            },
        )
        await _push_event(task_id, {"event": "stream_end", "timestamp": _now()})
        return {"message": "任务已拒绝", "task_id": task_id}

    # 模拟 Resume（生产：从 Postgres 恢复 Checkpoint）
    store["status"] = "running"
    await _push_event(
        task_id,
        {
            "event": "hitl_approved",
            "feedback": req.feedback,
            "patched_fields": list(req.patched_state.keys()),
            "timestamp": _now(),
        },
    )

    # Demo：直接标记完成（生产：重新触发 graph.invoke）
    mock_result = {
        "verdict": "PASS",
        "confidence_score": 0.82,
        "root_cause": "人工审批后确认参数合规，测试通过",
        "issues": [],
    }
    store["status"] = "complete"
    store["result"] = mock_result
    await _push_event(
        task_id,
        {
            "event": "complete",
            "verdict": mock_result["verdict"],
            "confidence": mock_result["confidence_score"],
            "root_cause": mock_result["root_cause"],
            "timestamp": _now(),
        },
    )
    await _push_event(task_id, {"event": "stream_end", "timestamp": _now()})

    return {
        "message": "任务已恢复执行",
        "task_id": task_id,
        "thread_id": store["thread_id"],
    }


# ── GET /result/{task_id} ────────────────────────────────────────


@app.get(
    "/result/{task_id}", response_model=TaskResult, summary="查询任务结果（非流式）"
)
async def get_result(task_id: str):
    """查询任务最终结果，适合 CI 系统轮询使用。"""
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail=f"task_id '{task_id}' 不存在")

    store = _task_store[task_id]
    result = store.get("result") or {}

    return TaskResult(
        task_id=task_id,
        status=store["status"],
        verdict=result.get("verdict", ""),
        confidence=result.get("confidence_score", 0.0),
        root_cause=result.get("root_cause", ""),
        issues=result.get("issues", []),
    )


# ── GET /health ──────────────────────────────────────────────────


@app.get("/health", summary="健康检查")
async def health():
    return {
        "status": "ok",
        "active_tasks": sum(
            1 for s in _task_store.values() if s["status"] == "running"
        ),
        "total_tasks": len(_task_store),
        "timestamp": _now(),
    }


# ═══════════════════════════════════════════════════════════════════
# 本地启动入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    # 开发模式：uvicorn api:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
