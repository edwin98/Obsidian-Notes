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
from functools import partial
from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Send
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


def worker_node(state: WorkerState, *, worker_graph_factory=None) -> dict:
    """每个 Worker 运行完整单 Agent 图，使用独立 Checkpoint 和 thread_id。

    默认使用离线模型；调用者可注入在线图。遇到 HITL 时只能标记未决，
    不能把子图的暂停误认为 PASS。跨进程审批需要持久化子图，当前未提供。
    """
    from demo import create_demo_graph
    from state import initial_state

    graph = (worker_graph_factory or create_demo_graph)()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 40}
    result = graph.invoke(initial_state(state["messages"][0].content), config)
    raw = result.get("final_result")
    data = (
        json.loads(raw)
        if raw
        else {
            "verdict": "INCONCLUSIVE",
            "confidence_score": 0.0,
            "issues": ["子任务等待澄清或审批"],
            "root_cause": "",
        }
    )
    return {
        "worker_results": [
            {
                "task_id": state["task_id"],
                "feature": state["feature"],
                "verdict": data["verdict"],
                "confidence": data["confidence_score"],
                "issues": data.get("issues", []),
                "root_cause": data.get("root_cause", ""),
            }
        ]
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
      - 缺失、重复或未决结果 → INCONCLUSIVE，禁止签发
      - 其余按优先级策略判定；仅全部子项 PASS 才可签发
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

    # 检查缺失/重复/未知任务，空报告也不能签发。
    expected = set(priority_map)
    actual = [r["task_id"] for r in results]
    complete = (
        bool(expected) and set(actual) == expected and len(actual) == len(expected)
    )
    policy_failed = bool(p0_fails) or (p1_total > 0 and len(p1_fails) / p1_total > 0.5)
    uncertain = (
        not complete
        or bool(warnings)
        or any(r["verdict"] not in {"PASS", "FAIL", "INCONCLUSIVE"} for r in results)
    )
    overall = "FAIL" if policy_failed else ("INCONCLUSIVE" if uncertain else "PASS")
    all_pass = overall == "PASS"

    # 生成验收报告
    site_id = state["site_id"]
    site_type = state.get("site_type", "普通宏站")
    total = len(results)
    pass_count = sum(1 for r in results if r["verdict"] == "PASS")

    report_data = {
        "site_id": site_id,
        "site_type": site_type,
        "overall_verdict": overall,
        "results_complete": complete,
        "total_items": total,
        "pass_count": pass_count,
        "fail_count": sum(r["verdict"] == "FAIL" for r in results),
        "warning_count": len(warnings),
        "p0_fails": [
            {"task_id": r["task_id"], "root_cause": r["root_cause"]} for r in p0_fails
        ],
        "p1_fails": [
            {"task_id": r["task_id"], "root_cause": r["root_cause"]} for r in p1_fails
        ],
        "action_required": "立即处理 P0 故障后重新验收"
        if p0_fails
        else (
            "补齐或复核子任务结果"
            if uncertain
            else ("处理失败子项" if pass_count != total else "无")
        ),
        "sign_off_ready": all_pass and pass_count == total,
    }

    report_json = json.dumps(report_data, ensure_ascii=False, indent=2)

    print(f"\n[Orchestrator] 验收结论: {report_data['overall_verdict']}")
    print(
        f"  通过 {pass_count}/{total}，P0故障 {len(p0_fails)} 项，P1故障 {len(p1_fails)} 项"
    )
    if report_data["sign_off_ready"]:
        print("  全部子项通过，可以进入人工签发流程")
    else:
        print(f"  待处理: {report_data['action_required']}")

    return {
        "final_report": report_json,
        "all_pass": all_pass,
    }


# ═══════════════════════════════════════════════════════════════════
# 第六部分：图装配
# ═══════════════════════════════════════════════════════════════════


def _build_fleet_graph(*, worker_graph_factory=None) -> StateGraph:
    """
    构建 Multi-Agent Map-Reduce 验收图。

    图结构：
      START → planning → Send 路由 → [worker×N（并发）] → reduce → END
    """
    workflow = StateGraph(OrchestratorState)

    workflow.add_node("planning", planning_node)
    workflow.add_node(
        "worker", partial(worker_node, worker_graph_factory=worker_graph_factory)
    )
    workflow.add_node("reduce", reduce_node)

    workflow.add_edge(START, "planning")
    # Send 列表必须作为条件路由返回，不能作为普通节点的 State 增量。
    workflow.add_conditional_edges("planning", map_node, ["worker"])
    # 所有 Worker 完成后汇入 reduce
    workflow.add_edge("worker", "reduce")
    workflow.add_edge("reduce", END)

    return workflow


def create_fleet_graph(*, worker_graph_factory=None):
    """创建多局点验收 Graph（内存模式）"""
    return _build_fleet_graph(worker_graph_factory=worker_graph_factory).compile(
        checkpointer=MemorySaver()
    )


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
