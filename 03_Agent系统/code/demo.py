"""无需密钥的可执行教学轨迹，复用 graph.py，模型用规则替身。

替身只负责按既定顺序请求工具，不代表真实 LLM 的规划能力或评测效果。
运行：python demo.py --scenario regression；每一步会打印节点增量。
"""

import argparse
import json
import uuid

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from graph import create_graph_in_memory
from state import initial_state
from tools import create_sandbox_tools


class OfflineModel:
    """从实际 ToolMessage 取参数，展示 Action → Observation → 下一次 Action。"""

    def __init__(self, scenario="pass", with_tools=True):
        self.scenario, self.with_tools = scenario, with_tools

    def invoke(self, messages):
        if not self.with_tools:
            if "判断对话" in str(messages[0].content):
                return AIMessage(content=json.dumps({"clear": True, "question": ""}))
            return AIMessage(
                content=json.dumps(
                    {
                        "verdict": "FAIL" if self.scenario == "regression" else "PASS",
                        "confidence_score": 0.4 if self.scenario == "review" else 0.9,
                        "root_cause": "故障注入导致 KPI 超限"
                        if self.scenario == "regression"
                        else "沙盒证据已收齐",
                    },
                    ensure_ascii=False,
                )
            )
        outputs = {
            m.name: json.loads(m.content)
            for m in messages
            if isinstance(m, ToolMessage)
        }
        if "test_case_query" not in outputs:
            request = next(m.content for m in messages if isinstance(m, HumanMessage))
            name, args = (
                "test_case_query",
                {
                    "feature": "force_reboot handover"
                    if self.scenario == "danger"
                    else request
                },
            )
        elif "simulation_runner" not in outputs:
            name, args = (
                "simulation_runner",
                {
                    "test_case_ids": [
                        c["id"] for c in outputs["test_case_query"]["cases"]
                    ]
                },
            )
        elif "metrics_collector" not in outputs:
            name, args = (
                "metrics_collector",
                {"session_id": outputs["simulation_runner"]["session_id"]},
            )
        elif "baseline_comparator" not in outputs:
            metrics = outputs["metrics_collector"]
            name, args = (
                "baseline_comparator",
                {
                    "throughput_avg": metrics["throughput_avg_mbps"],
                    "latency_p99": metrics["latency_p99_ms"],
                },
            )
        elif self.scenario == "regression" and "log_analyzer" not in outputs:
            name, args = (
                "log_analyzer",
                {
                    "session_id": outputs["simulation_runner"]["session_id"],
                    "log_type": "RRC",
                },
            )
        else:
            return AIMessage(content='{"verdict":"PASS","confidence_score":0.9}')
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call-{len(outputs)}"}],
        )


def create_demo_graph(scenario="pass"):
    if scenario not in {"pass", "regression", "danger", "review"}:
        raise ValueError("未知离线场景")
    return create_graph_in_memory(
        llm_factory=lambda with_tools: OfflineModel(scenario, with_tools),
        tool_map=create_sandbox_tools(regression=scenario == "regression"),
    )


def main():
    parser = argparse.ArgumentParser(description="Agent 离线教学演示")
    parser.add_argument(
        "--scenario", choices=["pass", "regression", "danger", "review"], default="pass"
    )
    args = parser.parse_args()
    graph = create_demo_graph(args.scenario)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 40}
    graph_input = initial_state("验证 Xn 切换回归并与基线比较")
    while True:
        for chunk in graph.stream(graph_input, config, stream_mode="updates"):
            for name, update in chunk.items():
                if name == "__interrupt__":
                    print("等待人工审批：", update[0].value)
                else:
                    print(f"\n[{name}]")
                    print(json.dumps(update, default=str, ensure_ascii=False, indent=2))
        snapshot = graph.get_state(config)
        if not snapshot.next:
            print("\n最终结果：", snapshot.values.get("final_result"))
            break
        try:
            answer = input("输入 approve 批准，其余输入拒绝：").strip().lower()
        except EOFError:
            answer = "reject"
        graph_input = Command(
            resume={"approved": answer == "approve", "feedback": "离线演示人工审核"}
        )


if __name__ == "__main__":
    main()
