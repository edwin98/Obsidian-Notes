"""
graph.py — LangGraph 状态机装配

LangGraph 核心概念
──────────────────
StateGraph：
    以 TypedDict（AgentState）为"黑板"，图中每个节点读写同一份 State，
    节点函数签名为 (state) -> dict，返回值会被 merge 进全局 State。
    messages 字段使用 Annotated[list, add_messages]，LangGraph 自动追加而非覆盖。

add_node / add_edge / add_conditional_edges：
    add_node(name, fn)          注册节点
    add_edge(src, dst)          无条件跳转
    add_conditional_edges(      按路由函数返回值决定下一节点
        src, router_fn, mapping
    )

compile(checkpointer)：
    checkpointer     每个节点执行后自动将 State 序列化至存储后端（内存/Postgres）。
                     配合 thread_id 可跨调用恢复同一对话上下文。
    hitl 节点调用 interrupt(payload) 暂停，外部用 Command(resume=decision) 恢复。

recursion_limit（传入 invoke config）：
    硬熔断。超过此轮次时 LangGraph 抛出 GraphRecursionError，防止死循环。

LangSmith 与 LangGraph 的集成：
    只要 LANGCHAIN_TRACING_V2=true，LangGraph 会自动为每个节点创建子 Span，
    在 LangSmith UI 可以看到完整的节点调用链、State 变化和耗时瀑布图。
"""

from functools import partial

from langgraph.graph import END, START, StateGraph

from config import CONFIDENCE_LOW, MAX_ERRORS, POSTGRES_URI
from nodes import (
    agent_node,
    clarify_node,
    guardrail_node,
    hitl_node,
    result_judge_node,
    tool_node,
)
from state import AgentState

# ── Routing functions ────────────────────────────────────────────────────────


def after_clarify(state: AgentState) -> str:
    """Route from clarify_node: ambiguous → end（等待用户追问）, clear → agent."""
    return "end" if state.get("needs_clarification", False) else "agent"


def should_continue(state: AgentState) -> str:
    """
    Route from agent_node:
      - tool calls pending  → guardrail (safety check before execution)
      - consecutive errors  → hitl      (soft circuit breaker)
      - low confidence      → hitl      (soft circuit breaker)
      - no tool calls       → result_judge
    """
    # Soft circuit breaker: too many consecutive tool errors
    if state.get("error_count", 0) >= MAX_ERRORS:
        return "hitl"

    last_msg = state["messages"][-1]
    has_tool_calls = bool(getattr(last_msg, "tool_calls", None))

    if has_tool_calls:
        return "guardrail"

    # Soft circuit breaker: final answer but confidence too low
    if state.get("confidence_score", 1.0) < CONFIDENCE_LOW:
        return "hitl"

    return "result_judge"


def after_hitl(state: AgentState) -> str:
    """拒绝/结果复核结束；批准待执行请求后直接执行，避免再次规划同一请求。"""
    if state.get("hitl_rejected") or state.get("final_result"):
        return "end"
    calls = getattr(state["messages"][-1], "tool_calls", [])
    return "tools" if calls else "result_judge"


def after_guardrail(state: AgentState) -> str:
    """Route from guardrail_node: dangerous op → hitl, safe → tools."""
    return "hitl" if state.get("hitl_required", False) else "tools"


def after_result_judge(state: AgentState) -> str:
    """Route from result_judge_node: low-confidence result → hitl, otherwise done."""
    return "hitl" if state.get("confidence_score", 1.0) < CONFIDENCE_LOW else "end"


# ── Graph builder ────────────────────────────────────────────────────────────


def _build_workflow(*, llm_factory=None, tool_map=None) -> StateGraph:
    # StateGraph(AgentState)：以 AgentState 作为整张图共享的状态结构
    workflow = StateGraph(AgentState)

    # 注册节点：每个节点是一个普通 Python 函数 (state: AgentState) -> dict
    options = {"llm_factory": llm_factory} if llm_factory is not None else {}
    workflow.add_node("clarify", partial(clarify_node, **options))
    workflow.add_node("agent", partial(agent_node, **options))
    workflow.add_node("guardrail", guardrail_node)
    workflow.add_node("tools", partial(tool_node, tool_map=tool_map))
    workflow.add_node("hitl", hitl_node)
    workflow.add_node("result_judge", partial(result_judge_node, **options))

    # 固定入口：先经过意图澄清节点
    workflow.add_edge(START, "clarify")
    workflow.add_conditional_edges(
        "clarify",
        after_clarify,
        {"agent": "agent", "end": END},
    )

    # 条件边：路由函数返回字符串 key，mapping 将 key 映射到目标节点名
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {"guardrail": "guardrail", "result_judge": "result_judge", "hitl": "hitl"},
    )
    workflow.add_conditional_edges(
        "guardrail",
        after_guardrail,
        {"tools": "tools", "hitl": "hitl"},
    )
    # 固定边：工具执行完毕后无条件返回 agent 继续推理（ReAct 循环）
    workflow.add_edge("tools", "agent")
    # 审批后直接执行已审核工具；结果复核或拒绝则结束。
    workflow.add_conditional_edges(
        "hitl",
        after_hitl,
        {"tools": "tools", "result_judge": "result_judge", "end": END},
    )
    workflow.add_conditional_edges(
        "result_judge",
        after_result_judge,
        {"end": END, "hitl": "hitl"},
    )

    return workflow


def create_graph_with_postgres():
    """
    生产模式：PostgreSQL Checkpointer。

    PostgresSaver.setup() 会在首次运行时自动建表（checkpoint / checkpoint_blobs 等）。
    hitl_node 内部的动态 interrupt 使图暂停：
      - LangGraph 将当前 State 序列化写入 Postgres
      - 进程可以安全退出（释放 GPU/内存）
      - 人工审批后通过 graph.invoke(Command(resume={"approved": True}), config) 恢复
      - 调用方负责管理图和数据库连接的生命周期；此工厂适合常驻 CLI 演示
    """
    import psycopg
    from langgraph.checkpoint.postgres import PostgresSaver

    conn = psycopg.connect(POSTGRES_URI, autocommit=True, prepare_threshold=0)
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()  # 幂等：表已存在时不报错

    return _build_workflow().compile(checkpointer=checkpointer)


def create_graph_in_memory(*, llm_factory=None, tool_map=None, interrupt_after=None):
    """
    Demo/测试模式：MemorySaver Checkpointer。

    State 存储在进程内存中，进程退出即丢失。
    hitl_node 使用动态 interrupt；内存模式同样必须收到真实审批才能继续。
    thread_id 仍然有效：同一 thread_id 多次 invoke 可延续对话上下文。
    """
    from langgraph.checkpoint.memory import MemorySaver

    return _build_workflow(llm_factory=llm_factory, tool_map=tool_map).compile(
        checkpointer=MemorySaver(), interrupt_after=interrupt_after
    )
