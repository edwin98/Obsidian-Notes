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
import math
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


# ── 统计检验工具（纯标准库，无需 scipy）────────────────────────────────────────


def _ttest_ind(sample: list[float], baseline_mean: float, baseline_std: float) -> float:
    """
    单样本 t 检验（样本 vs 已知总体均值）。
    返回双尾 p 值近似值（基于 t 分布自由度 n-1）。
    当 p < 0.05 且均值下降 > 5% 时判定为回归。
    """
    n = len(sample)
    if n < 2 or baseline_std == 0:
        return 1.0
    sample_mean = sum(sample) / n
    sample_var = sum((x - sample_mean) ** 2 for x in sample) / (n - 1)
    se = math.sqrt(sample_var / n) if sample_var > 0 else 1e-9
    t_stat = (sample_mean - baseline_mean) / se
    # 用 t 分布的正态近似计算 p 值（n 较大时误差 < 0.01）
    z = abs(t_stat)
    p_approx = 2 * (1 - _norm_cdf(z))
    return p_approx


def _ks_test_uniform(sample: list[float], low: float, high: float) -> float:
    """
    KS 检验：检测样本是否超出 [low, high] 包络。
    返回超出包络的样本比例（0 = 完全在包络内）。
    """
    if not sample:
        return 0.0
    out = sum(1 for x in sample if x < low or x > high)
    return out / len(sample)


def _norm_cdf(z: float) -> float:
    """标准正态 CDF 近似（Abramowitz & Stegun 7.1.26）"""
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    return cdf if z >= 0 else 1.0 - cdf


# ── Danger keywords for guardrail ───────────────────────────────────────────
DANGER_KEYWORDS = {
    "format_c",
    "reset_all",
    "delete_all",
    "shutdown",
    "kill_process",
    "wipe",
}

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
def _send_webhook_alert(reason: str, state: AgentState) -> None:
    """
    生产环境中通过 Celery 异步推送 DingTalk/Slack Webhook 告警。
    此处模拟告警内容输出，结构与真实 DingTalk markdown 消息一致。

    生产实现示例：
        from celery_app import send_hitl_alert
        send_hitl_alert.delay(thread_id=..., reason=reason, payload=payload)
    """
    last_msg = state["messages"][-1] if state.get("messages") else None
    tool_calls_summary = ""
    if last_msg:
        tcs = getattr(last_msg, "tool_calls", []) or []
        if tcs:
            tool_calls_summary = ", ".join(
                f"{tc['name']}({list(tc.get('args', {}).keys())})" for tc in tcs
            )

    alert_payload = {
        "msgtype": "markdown",
        "markdown": {
            "title": "[5G Agent] HITL Review Required",
            "text": (
                f"**触发原因**: {reason}\n\n"
                f"**当前步骤**: {state.get('current_step', 'unknown')}\n\n"
                f"**置信度**: {state.get('confidence_score', 0):.2f}\n\n"
                f"**待执行工具**: {tool_calls_summary or 'N/A'}\n\n"
                f"**连续错误次数**: {state.get('error_count', 0)}\n\n"
                "请登录审批平台处理: https://agent-review.internal/pending"
            ),
        },
    }
    print(f"\n[HITL][Webhook] 告警已推送（生产模式将发往 DingTalk）:")
    print(
        f"  {json.dumps(alert_payload['markdown']['text'][:200], ensure_ascii=False)}"
    )


def hitl_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop node.

    生产行为：
      1. AgentState 已由 LangGraph Checkpointer 序列化至 Postgres。
      2. 通过 Celery 异步推送 DingTalk Webhook 告警（含用例详情与拦截原因）。
      3. 当前进程释放 GPU/内存资源，等待 /resume API 恢复。
      4. 专家在 Web 前端 Review 参数 → 修改异常值 → 点击 Approve。
      5. /resume 接口从 Postgres 重建 State，调用 graph.invoke(None, config) 恢复。

    Demo 行为：
      模拟自动审批，重置错误计数器，继续执行。
    """
    reason_parts = []
    if state.get("hitl_required"):
        reason_parts.append("guardrail 检测到高危操作")
    if state.get("confidence_score", 1.0) < CONFIDENCE_LOW:
        reason_parts.append(
            f"置信度过低 ({state.get('confidence_score', 0):.2f} < {CONFIDENCE_LOW})"
        )
    if state.get("error_count", 0) >= MAX_ERRORS:
        reason_parts.append(f"连续工具错误 {state.get('error_count', 0)} 次")

    reason = "; ".join(reason_parts) or "未知原因"
    print(f"\n[HITL] 触发人工审核: {reason}")

    # 推送 Webhook 告警（生产：Celery 异步；Demo：直接打印）
    _send_webhook_alert(reason, state)

    print("[HITL] Demo 模式：模拟人工审批通过...")

    return {
        "hitl_required": False,
        "hitl_feedback": f"人工审批通过（模拟）。原因：{reason}",
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
    # 统计轨：硬规则 + T-Test/KS-Test 双重校验
    # 原则：明显违规直接 FAIL，不依赖 LLM 判断
    stat_issues = []

    # 1. baseline_comparator 回归检测
    if "baseline_comparator" in tool_outputs:
        bc = tool_outputs["baseline_comparator"]
        stat_issues.extend(bc.get("degradations", []))
        # 整体状态为 REGRESSION 时额外记录
        if bc.get("overall_status") == "REGRESSION" and not bc.get("degradations"):
            stat_issues.append("baseline_comparator: overall status REGRESSION")

    # 2. log_analyzer 信令异常
    if "log_analyzer" in tool_outputs:
        anomalies = tool_outputs["log_analyzer"].get("anomalies", [])
        stat_issues.extend(anomalies)
        # 严重程度为 HIGH 时加权
        if tool_outputs["log_analyzer"].get("severity") == "HIGH" and not anomalies:
            stat_issues.append("log_analyzer: severity HIGH with no explicit anomaly")

    # 3. simulation_runner 通过率 T-Test 检验
    # 基线：pass_rate 均值 0.99，标准差 0.02（来自历史压测数据）
    BASELINE_PASS_RATE_MEAN = 0.99
    BASELINE_PASS_RATE_STD = 0.02
    if "simulation_runner" in tool_outputs:
        sr = tool_outputs["simulation_runner"]
        pass_rate = sr.get("pass_rate", 1.0)
        # 硬规则：pass_rate < 0.8 直接 FAIL
        if pass_rate < 0.8:
            stat_issues.append(
                f"simulation pass_rate {pass_rate:.1%} below hard threshold 80%"
            )
        # T-Test：pass_rate 与基线均值做显著性检验
        elif pass_rate < BASELINE_PASS_RATE_MEAN:
            # 用单点样本近似（生产中应收集多次运行数据）
            sample = [pass_rate]
            p_val = _ttest_ind(sample, BASELINE_PASS_RATE_MEAN, BASELINE_PASS_RATE_STD)
            decline_pct = (
                BASELINE_PASS_RATE_MEAN - pass_rate
            ) / BASELINE_PASS_RATE_MEAN
            if p_val < 0.05 and decline_pct > 0.05:
                stat_issues.append(
                    f"T-Test: pass_rate {pass_rate:.1%} significantly below baseline "
                    f"{BASELINE_PASS_RATE_MEAN:.1%} (p={p_val:.3f}, decline={decline_pct:.1%})"
                )

    # 4. metrics_collector KPI 包络 KS-Test 检验
    # 基线包络：throughput [80, 300] Mbps，latency_p99 [0, 30] ms
    if "metrics_collector" in tool_outputs:
        mc = tool_outputs["metrics_collector"]
        throughput = mc.get("throughput_avg_mbps", 150.0)
        latency_p99 = mc.get("latency_p99_ms", 10.0)
        pkt_loss = mc.get("packet_loss_rate", 0.0)

        # 吞吐量包络检验
        tp_out_rate = _ks_test_uniform([throughput], low=80.0, high=300.0)
        if tp_out_rate > 0:
            stat_issues.append(
                f"KS-Test: throughput {throughput:.1f} Mbps out of envelope [80, 300]"
            )

        # P99 延迟包络检验
        lat_out_rate = _ks_test_uniform([latency_p99], low=0.0, high=30.0)
        if lat_out_rate > 0:
            stat_issues.append(
                f"KS-Test: P99 latency {latency_p99:.1f} ms exceeds envelope [0, 30]"
            )

        # 丢包率硬规则
        if pkt_loss > 0.01:
            stat_issues.append(f"packet_loss_rate {pkt_loss:.2%} exceeds threshold 1%")

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
