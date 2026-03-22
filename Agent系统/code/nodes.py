"""
nodes.py — 图节点实现

LangGraph 节点说明
──────────────────
每个节点函数签名为 (state: AgentState) -> dict：
  - 入参是当前完整 State（只读）
  - 返回值是要 merge 进 State 的字段增量（只需返回需要更新的 key）
  - messages 字段因为使用了 add_messages reducer，返回的列表会被追加而非覆盖

LangSmith 自动追踪
──────────────────
只要 LANGCHAIN_TRACING_V2=true，以下内容会被自动追踪，无需在节点内手动添加代码：
  - 每次 LLM 调用：prompt、completion、token 用量、延迟
  - 每个节点的输入 State 和输出 dict（作为子 Span）
  - 工具调用的名称、入参、返回值
  在 LangSmith UI 选择一条 Run，点击 Trace 可看到完整节点调用树。

bind_tools 与 DeepSeek
──────────────────────
llm.bind_tools(TOOLS) 将工具的 JSON Schema 注入到每次请求的 tools 参数中。
DeepSeek 兼容 OpenAI Function Calling 格式，模型会在需要时返回带 tool_calls 字段的
AIMessage，LangGraph 的路由函数通过检查该字段决定是否进入工具执行分支。

ToolMessage 与 tool_call_id
────────────────────────────
每个 ToolMessage 必须携带与 AIMessage.tool_calls[i].id 匹配的 tool_call_id，
LLM 才能将工具结果与对应的调用请求关联起来。这是 OpenAI/DeepSeek 协议要求。
"""

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from config import (
    CONFIDENCE_LOW,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    MAX_ERRORS,
)
from state import AgentState
from tools import TOOL_MAP, TOOLS

# ── Danger keywords for guardrail ───────────────────────────────────────────
DANGER_KEYWORDS = {"format_c", "reset_all", "delete_all", "shutdown", "kill_process", "wipe"}

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a 5G test verification agent. Follow this workflow:
1. Parse the test request and identify which 5G feature to test.
2. Call test_case_query to retrieve relevant test cases.
3. Call simulation_runner with the retrieved test case IDs.
4. Call metrics_collector with the simulation session ID.
5. Call baseline_comparator with the collected KPI metrics.
6. Call log_analyzer if anomalies need further diagnosis.
7. When analysis is complete, output ONLY this JSON block:

```json
{"confidence_score": <0.0-1.0>, "verdict": "<PASS|FAIL|INCONCLUSIVE>", "summary": "<one-line summary>"}
```

Rules:
- Always start with test_case_query, then simulation_runner.
- Set confidence_score < 0.65 if you are unsure; this triggers human review.
- Never invent tool parameters — use values returned by previous tool calls.
"""


# ── LLM factory ─────────────────────────────────────────────────────────────
def _get_llm(with_tools: bool = True) -> ChatOpenAI:
    llm = ChatOpenAI(
        model=DEEPSEEK_MODEL,
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL,
        temperature=0.1,
    )
    return llm.bind_tools(TOOLS) if with_tools else llm


# ── Node: agent ──────────────────────────────────────────────────────────────
def agent_node(state: AgentState) -> dict:
    """Main ReAct reasoning node — LLM decides the next action or outputs final verdict."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = _get_llm(with_tools=True).invoke(messages)

    # Extract confidence_score when the LLM produces its final JSON block
    confidence = state.get("confidence_score", 1.0)
    if not response.tool_calls and isinstance(response.content, str):
        match = re.search(r'"confidence_score"\s*:\s*([\d.]+)', response.content)
        if match:
            confidence = float(match.group(1))

    return {
        "messages": [response],
        "current_step": "agent",
        "confidence_score": confidence,
    }


# ── Node: guardrail ──────────────────────────────────────────────────────────
def guardrail_node(state: AgentState) -> dict:
    """
    Safety guardrail — inspect pending tool calls for dangerous parameters
    before allowing execution.
    Triggers HITL if any danger keyword is found in tool arguments.
    """
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    for tc in tool_calls:
        args_str = json.dumps(tc.get("args", {})).lower()
        if any(kw in args_str for kw in DANGER_KEYWORDS):
            return {"hitl_required": True, "current_step": "hitl"}

    return {"hitl_required": False}


# ── Node: tool executor ──────────────────────────────────────────────────────
def tool_node(state: AgentState) -> dict:
    """Execute pending tool calls, append ToolMessages, and cache results."""
    last_msg = state["messages"][-1]
    tool_calls = getattr(last_msg, "tool_calls", []) or []

    tool_messages = []
    tool_outputs = dict(state.get("tool_outputs", {}))
    error_count = state.get("error_count", 0)

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        tc_id = tc["id"]

        if name not in TOOL_MAP:
            content = f"Unknown tool: {name}"
            error_count += 1
        else:
            try:
                result = TOOL_MAP[name].invoke(args)
                content = json.dumps(result)
                tool_outputs[name] = result  # keyed by tool name; latest call wins
            except Exception as e:
                content = f"Tool execution error: {e}"
                error_count += 1

        tool_messages.append(ToolMessage(content=content, tool_call_id=tc_id))

    return {
        "messages": tool_messages,
        "tool_outputs": tool_outputs,
        "error_count": error_count,
        "current_step": "tools",
    }


# ── Node: HITL ───────────────────────────────────────────────────────────────
def hitl_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop node.

    Production behaviour:
      1. Serialize AgentState to Postgres checkpoint (already handled by LangGraph).
      2. Send alert via webhook (DingTalk / Slack).
      3. Release the compute thread — wait for /resume API call with human decision.

    Demo behaviour:
      Simulate auto-approval and reset the error counter so the agent can proceed.
    """
    reason_parts = []
    if state.get("hitl_required"):
        reason_parts.append("dangerous operation detected by guardrail")
    if state.get("confidence_score", 1.0) < CONFIDENCE_LOW:
        reason_parts.append(f"low confidence ({state.get('confidence_score', 0):.2f})")
    if state.get("error_count", 0) >= MAX_ERRORS:
        reason_parts.append(f"consecutive tool errors ({state.get('error_count', 0)})")

    print(f"\n[HITL] Human review triggered: {'; '.join(reason_parts) or 'unknown reason'}")
    print("[HITL] Simulating human approval (demo mode)...")

    return {
        "hitl_required": False,
        "hitl_feedback": "Approved by human reviewer (simulated)",
        "confidence_score": 0.75,
        "error_count": 0,
        "current_step": "agent",
    }


# ── Node: result_judge ───────────────────────────────────────────────────────
def result_judge_node(state: AgentState) -> dict:
    """
    Dual-track result judgment:
      Statistical track — KPI threshold checks on cached tool_outputs.
      Semantic track    — LLM synthesises all evidence into a final verdict.
    """
    tool_outputs = state.get("tool_outputs", {})

    # ── Statistical track ────────────────────────────────────────────────────
    stat_issues = []

    if "baseline_comparator" in tool_outputs:
        stat_issues.extend(tool_outputs["baseline_comparator"].get("degradations", []))

    if "log_analyzer" in tool_outputs:
        anomalies = tool_outputs["log_analyzer"].get("anomalies", [])
        stat_issues.extend(anomalies)

    if "simulation_runner" in tool_outputs:
        pass_rate = tool_outputs["simulation_runner"].get("pass_rate", 1.0)
        if pass_rate < 0.8:
            stat_issues.append(f"low simulation pass rate: {pass_rate:.0%}")

    stat_verdict = "FAIL" if stat_issues else "PASS"

    # ── Semantic track ───────────────────────────────────────────────────────
    prompt = (
        f"You are a 5G test result judge. Given the evidence below, return a final verdict.\n\n"
        f"Tool outputs:\n{json.dumps(tool_outputs, indent=2)[:3000]}\n\n"
        f"Statistical analysis: verdict={stat_verdict}, issues={stat_issues}\n\n"
        f"Return ONLY valid JSON with no markdown fences:\n"
        f'{{"verdict": "PASS|FAIL|INCONCLUSIVE", "confidence_score": 0.0-1.0, "root_cause": "brief"}}'
    )
    response = _get_llm(with_tools=False).invoke([HumanMessage(content=prompt)])

    try:
        data = json.loads(response.content)
        confidence = float(data.get("confidence_score", 0.8))
        verdict = data.get("verdict", stat_verdict)
        root_cause = data.get("root_cause", "")
    except (json.JSONDecodeError, ValueError):
        confidence = 0.8
        verdict = stat_verdict
        root_cause = "; ".join(stat_issues) if stat_issues else "no issues detected"

    final = {
        "verdict": verdict,
        "confidence_score": confidence,
        "stat_verdict": stat_verdict,
        "issues": stat_issues,
        "root_cause": root_cause,
    }

    return {
        "final_result": json.dumps(final, ensure_ascii=False),
        "confidence_score": confidence,
        "current_step": "complete",
    }
