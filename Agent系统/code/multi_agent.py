"""
multi_agent.py — Multi-Agent Map-Reduce 全网验收架构

对应全景报告第六章场景二：新局点 5G SA 全网验收跑批（Multi-Agent 自动驾驶模式）

架构：
  ┌─────────────────────────────────────────────────────┐
  │              主控 Agent（Orchestrator）               │
  │  任务规划 → 任务拆分 → 聚合研判 → 签发验收报告         │
  └─────────────────────────────────────────────────────┘
          │               │               │
          ▼               ▼               ▼
  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
  │ Worker-01   │ │ Worker-02   │ │ Worker-N    │
  │ VoNR 验收   │ │ 切换验收    │ │ 干扰验收    │
  └─────────────┘ └─────────────┘ └─────────────┘
          │               │               │
          └───────────────┴───────────────┘
                          │
                    Reduce: 汇总研判

LangGraph 实现方案：
  - 主控 Agent 是一个 StateGraph，Worker 也是 StateGraph
  - 主控通过 Send API 将子任务并发分发给 Worker
  - Worker 完成后将结果写入 Reduce 节点的 State
  - 所有 Worker 完成后，主控汇总生成最终验收报告

本文件完整实现 Map-Reduce 多 Agent 协调逻辑。
Worker 使用 graph.py 中已有的单 Agent 图，主控图新建。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


# ═══════════════════════════════════════════════════════════════════
# 第一部分：数据结构
# ═══════════════════════════════════════════════════════════════════


@dataclass
class AcceptanceTask:
    """单项验收任务"""

    task_id: str
    feature: str  # 被测特性：vonr / handover / interference / capacity
    site_id: str  # 局点 ID
    priority: str  # P0 / P1 / P2
    description: str  # 自然语言描述


@dataclass
class WorkerResult:
    """Worker Agent 的执行结果"""

    task_id: str
    feature: str
    verdict: str  # PASS / FAIL / INCONCLUSIVE
    confidence: float
    issues: list[str] = field(default_factory=list)
    root_cause: str = ""


class OrchestratorState(TypedDict):
    """主控 Agent 的全局状态"""

    site_id: str
    site_type: str  # "室分" | "高铁宏站" | "普通宏站"
    tasks: list[dict]  # AcceptanceTask 的 dict 表示
    worker_results: Annotated[list, lambda a, b: a + b]  # Reduce：并发追加
    final_report: str
    all_pass: bool


class WorkerState(TypedDict):
    """Worker Agent 的局部状态"""

    messages: Annotated[list, add_messages]
    task_id: str
    feature: str
    site_id: str
    verdict: str
    confidence: float
    issues: list[str]
    root_cause: str


# ═══════════════════════════════════════════════════════════════════
# 第二部分：任务规划（Planning Node）
# ═══════════════════════════════════════════════════════════════════

# 不同局点类型的标准验收项
_SITE_TYPE_TASKS: dict[str, list[dict]] = {
    "室分": [
        {
            "feature": "vonr",
            "priority": "P0",
            "description": "VoNR 室内覆盖呼叫成功率验收",
        },
        {
            "feature": "handover",
            "priority": "P0",
            "description": "室内外切换 A3 事件验收",
        },
        {
            "feature": "capacity",
            "priority": "P1",
            "description": "室分容量上限并发用户验收",
        },
    ],
    "高铁宏站": [
        {
            "feature": "handover",
            "priority": "P0",
            "description": "高速移动 Xn 切换（速度 > 200km/h）验收",
        },
        {"feature": "vonr", "priority": "P0", "description": "VoNR 高速移动连续性验收"},
        {
            "feature": "interference",
            "priority": "P1",
            "description": "高铁沿线干扰边界隔离验收",
        },
        {
            "feature": "capacity",
            "priority": "P2",
            "description": "高铁宏站峰值容量验收",
        },
    ],
    "普通宏站": [
        {"feature": "vonr", "priority": "P0", "description": "VoNR 基础呼叫 QoS 验收"},
        {
            "feature": "handover",
            "priority": "P0",
            "description": "Xn 接口切换成功率验收",
        },
        {
            "feature": "interference",
            "priority": "P1",
            "description": "PCI 干扰和 PRACH 碰撞验收",
        },
        {"feature": "capacity", "priority": "P1", "description": "满配 PRB 利用率验收"},
        {
            "feature": "nas_auth",
            "priority": "P2",
            "description": "NAS 鉴权端到端流程验收",
        },
    ],
}


def planning_node(state: OrchestratorState) -> dict:
    """
    任务规划节点：根据局点类型生成验收任务编排图。
    生产中通过网管北向接口获取基站配置后动态生成。
    """
    site_id = state["site_id"]
    site_type = state.get("site_type", "普通宏站")
    task_templates = _SITE_TYPE_TASKS.get(site_type, _SITE_TYPE_TASKS["普通宏站"])

    tasks = [
        {
            "task_id": f"{site_id}-{tmpl['feature']}-{i + 1:02d}",
            "feature": tmpl["feature"],
            "site_id": site_id,
            "priority": tmpl["priority"],
            "description": tmpl["description"],
        }
        for i, tmpl in enumerate(task_templates)
    ]

    print(
        f"\n[Orchestrator] 局点 {site_id}（{site_type}）任务规划完成，共 {len(tasks)} 项"
    )
    for t in tasks:
        print(f"  [{t['priority']}] {t['task_id']}: {t['description']}")

    return {"tasks": tasks}


# ═══════════════════════════════════════════════════════════════════
# 第三部分：Map 节点（分发任务给 Worker）
# ═══════════════════════════════════════════════════════════════════


def map_node(state: OrchestratorState) -> list[Send]:
    """
    Map 节点：使用 LangGraph Send API 将每个任务并发分发给独立 Worker。

    Send(node_name, state) 创建一个新的并发子流，Worker 之间完全独立运行。
    所有 Worker 完成后，结果通过 Reduce Reducer（list + list）自动汇聚。
    """
    sends = []
    for task in state["tasks"]:
        worker_state: WorkerState = {
            "messages": [HumanMessage(content=task["description"])],
            "task_id": task["task_id"],
            "feature": task["feature"],
            "site_id": task["site_id"],
            "verdict": "PENDING",
            "confidence": 0.0,
            "issues": [],
            "root_cause": "",
        }
        sends.append(Send("worker", worker_state))
    print(f"\n[Orchestrator] Map: 并发分发 {len(sends)} 个 Worker 任务")
    return sends


# ═══════════════════════════════════════════════════════════════════
# 第四部分：Worker Agent（单特性验收执行器）
# ═══════════════════════════════════════════════════════════════════


def worker_node(state: WorkerState) -> dict:
    """
    Worker Agent 节点：执行单项特性验收。

    生产模式：调用 graph.py 中完整的 LangGraph Agent 图（含 Guardrail/HITL）。
    Demo 模式：基于特性规则模拟验收结果，完整展示 Map-Reduce 数据流。
    """
    import random

    task_id = state["task_id"]
    feature = state["feature"]
    site_id = state["site_id"]

    print(f"\n  [Worker] 执行: {task_id} ({feature} @ {site_id})")

    # Demo：模拟各特性的验收逻辑
    # 生产中替换为: from graph import create_graph_in_memory; graph.invoke(worker_state)
    feature_results: dict[str, dict] = {
        "vonr": {
            "verdict": "PASS",
            "confidence": round(random.uniform(0.85, 0.97), 2),
            "issues": [],
            "root_cause": "",
        },
        "handover": {
            "verdict": random.choice(["PASS", "PASS", "FAIL"]),
            "confidence": round(random.uniform(0.75, 0.95), 2),
            "issues": ["handover_success_rate 87% below threshold 99%"]
            if random.random() < 0.3
            else [],
            "root_cause": "Xn backhaul RTT > 20ms" if random.random() < 0.3 else "",
        },
        "interference": {
            "verdict": "PASS",
            "confidence": round(random.uniform(0.80, 0.92), 2),
            "issues": [],
            "root_cause": "",
        },
        "capacity": {
            "verdict": "PASS",
            "confidence": round(random.uniform(0.88, 0.96), 2),
            "issues": [],
            "root_cause": "",
        },
        "nas_auth": {
            "verdict": "PASS",
            "confidence": round(random.uniform(0.90, 0.98), 2),
            "issues": [],
            "root_cause": "",
        },
    }

    result = feature_results.get(
        feature,
        {
            "verdict": "INCONCLUSIVE",
            "confidence": 0.6,
            "issues": [f"unknown feature: {feature}"],
            "root_cause": "",
        },
    )

    if result["issues"]:
        result["verdict"] = "FAIL"

    print(
        f"  [Worker] 完成: {task_id} → {result['verdict']} (conf={result['confidence']:.2f})"
    )

    # Worker 将结果写入 worker_results（主控 Reduce 字段）
    worker_result = {
        "task_id": task_id,
        "feature": feature,
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "issues": result["issues"],
        "root_cause": result["root_cause"],
    }

    return {
        "verdict": result["verdict"],
        "confidence": result["confidence"],
        "issues": result["issues"],
        "root_cause": result["root_cause"],
        # worker_results 是 Orchestrator State 的 Reduce 字段
        # Worker 通过写入同名字段实现向主控汇报
        "worker_results": [worker_result],
    }


# ═══════════════════════════════════════════════════════════════════
# 第五部分：Reduce 节点（汇总研判）
# ═══════════════════════════════════════════════════════════════════


def reduce_node(state: OrchestratorState) -> dict:
    """
    Reduce 节点：汇总所有 Worker 结果，生成最终验收报告。

    逻辑：
      - P0 任何一项 FAIL → 整体 FAIL（必须立即处理）
      - P1 超过 50% FAIL → 整体 FAIL
      - 其余 → PASS（附 Warning 列表）
    """
    results = state.get("worker_results", [])
    print(f"\n[Orchestrator] Reduce: 汇总 {len(results)} 个 Worker 结果")

    # 按优先级分组
    p0_fails = []
    p1_fails = []
    p1_total = 0
    warnings = []

    # 从 tasks 中获取优先级映射
    priority_map = {t["task_id"]: t["priority"] for t in state.get("tasks", [])}

    for r in results:
        priority = priority_map.get(r["task_id"], "P2")
        if r["verdict"] == "FAIL":
            if priority == "P0":
                p0_fails.append(r)
            elif priority == "P1":
                p1_fails.append(r)
        if priority == "P1":
            p1_total += 1
        if r["verdict"] == "INCONCLUSIVE":
            warnings.append(r)

    all_pass = len(p0_fails) == 0 and (p1_total == 0 or len(p1_fails) / p1_total < 0.5)

    # 生成验收报告
    site_id = state["site_id"]
    site_type = state.get("site_type", "普通宏站")
    total = len(results)
    pass_count = sum(1 for r in results if r["verdict"] == "PASS")

    report_data = {
        "site_id": site_id,
        "site_type": site_type,
        "overall_verdict": "PASS" if all_pass else "FAIL",
        "total_items": total,
        "pass_count": pass_count,
        "fail_count": total - pass_count - len(warnings),
        "warning_count": len(warnings),
        "p0_fails": [
            {"task_id": r["task_id"], "root_cause": r["root_cause"]} for r in p0_fails
        ],
        "p1_fails": [
            {"task_id": r["task_id"], "root_cause": r["root_cause"]} for r in p1_fails
        ],
        "action_required": "立即处理 P0 故障后重新验收"
        if p0_fails
        else ("处理 P1 故障" if p1_fails else "无"),
        "sign_off_ready": all_pass and len(warnings) == 0,
    }

    report_json = json.dumps(report_data, ensure_ascii=False, indent=2)

    print(f"\n[Orchestrator] 验收结论: {report_data['overall_verdict']}")
    print(
        f"  通过 {pass_count}/{total}，P0故障 {len(p0_fails)} 项，P1故障 {len(p1_fails)} 项"
    )
    if all_pass:
        print("  [OK] 可签发验收报告，调用 Network_Config_Write 完成割接")
    else:
        print(f"  [FAIL] {report_data['action_required']}")

    return {
        "final_report": report_json,
        "all_pass": all_pass,
    }


# ═══════════════════════════════════════════════════════════════════
# 第六部分：图装配
# ═══════════════════════════════════════════════════════════════════


def _build_fleet_graph() -> StateGraph:
    """
    构建 Multi-Agent Map-Reduce 验收图。

    图结构：
      START → planning → map_fanout → [worker×N（并发）] → reduce → END
    """
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("planning", planning_node)
    workflow.add_node("map_fanout", map_node)
    workflow.add_node("worker", worker_node)
    workflow.add_node("reduce", reduce_node)

    workflow.add_edge(START, "planning")
    workflow.add_edge("planning", "map_fanout")
    # map_fanout 返回 list[Send]，LangGraph 自动并发分发到 "worker" 节点
    workflow.add_conditional_edges("map_fanout", lambda _: "worker", ["worker"])
    # 所有 Worker 完成后汇入 reduce
    workflow.add_edge("worker", "reduce")
    workflow.add_edge("reduce", END)

    return workflow


def create_fleet_graph():
    """创建多局点验收 Graph（内存模式）"""
    return _build_fleet_graph().compile(checkpointer=MemorySaver())


# ═══════════════════════════════════════════════════════════════════
# 演示入口
# ═══════════════════════════════════════════════════════════════════


def run_fleet_acceptance(
    site_id: str = "SITE-GZ-001", site_type: str = "普通宏站"
) -> dict:
    """
    运行全网验收跑批演示。

    Args:
        site_id:   局点标识符
        site_type: 局点类型（"室分" | "高铁宏站" | "普通宏站"）

    Returns:
        最终验收报告 dict
    """
    graph = create_fleet_graph()
    thread_id = str(uuid.uuid4())

    initial_state: OrchestratorState = {
        "site_id": site_id,
        "site_type": site_type,
        "tasks": [],
        "worker_results": [],
        "final_report": "",
        "all_pass": False,
    }

    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n{'=' * 60}")
    print(f"5G SA 全网验收跑批: {site_id} ({site_type})")
    print(f"{'=' * 60}")

    final_state = graph.invoke(initial_state, config=config)

    report = json.loads(final_state.get("final_report", "{}"))

    print(f"\n{'=' * 60}")
    print("验收报告摘要:")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    return report


if __name__ == "__main__":
    # 演示三种局点类型
    for site_type in ["普通宏站", "室分", "高铁宏站"]:
        run_fleet_acceptance(site_id=f"SITE-{site_type[:2]}-001", site_type=site_type)
        print()
