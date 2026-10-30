"""【学习定位：旧单次图回归】
执行实际离线图而不是只匹配输出文字；不能代替新版完整调查评测。
对应07；与task_evaluation的不同在于业务范围。

在实际状态图上运行固定故障场景，不上传云端，也不声称测出真实 LLM 能力。
"""

import json
import uuid

from demo import create_demo_graph
from state import initial_state


def run_offline_evaluation():
    cases = [
        ("pass", "PASS", False),
        ("regression", "FAIL", False),
        ("danger", None, True),
        ("review", None, True),
    ]
    rows = []
    for scenario, expected, pending in cases:
        graph = create_demo_graph(scenario)
        config = {
            "configurable": {"thread_id": uuid.uuid4().hex},
            "recursion_limit": 40,
        }
        state = graph.invoke(initial_state("验证 Xn 切换回归"), config)
        paused = bool(graph.get_state(config).next)
        # 暂停后的候选报告还不是对用户发布的最终判定。
        verdict = (
            json.loads(state["final_result"])["verdict"]
            if state.get("final_result") and not paused
            else None
        )
        names = [call["name"] for call in state.get("tool_history", [])]
        expected_prefix = [
            "test_case_query",
            "simulation_runner",
            "metrics_collector",
            "baseline_comparator",
        ]
        rows.append(
            {
                "scenario": scenario,
                "verdict": verdict,
                "pending_review": paused,
                "passed": verdict == expected and paused == pending,
                "tool_order_correct": names[:4] == expected_prefix
                if scenario != "danger"
                else not names,
                "tool_calls": len(names),
                "safety_intercepted": paused and not names
                if scenario == "danger"
                else None,
            }
        )
    return {
        "scope": "离线替身的工程行为回归，不是在线模型能力评测",
        "cases": rows,
        "all_passed": all(row["passed"] and row["tool_order_correct"] for row in rows),
    }


if __name__ == "__main__":
    print(json.dumps(run_offline_evaluation(), ensure_ascii=False, indent=2))
