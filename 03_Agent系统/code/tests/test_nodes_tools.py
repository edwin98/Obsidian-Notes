"""节点边界与证据规则；检查行为而非照抄实现。"""

import json
from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage

from nodes import agent_node, clarify_node, guardrail_node, result_judge_node, tool_node
from state import initial_state
from tools import create_sandbox_tools


class FixedModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, _):
        return AIMessage(content=self.content)


def factory(content):
    return lambda _: FixedModel(content)


def evidence():
    return {
        "simulation_runner": {"pass_rate": 1.0},
        "metrics_collector": {
            "status": "completed",
            "throughput_avg_mbps": 150.0,
            "latency_p99_ms": 10.0,
            "packet_loss_rate": 0.001,
        },
        "baseline_comparator": {"degradations": [], "overall_status": "STABLE"},
    }


@pytest.mark.parametrize(
    "word", ["force_reboot", "FLASH_FW", "reset_all", "delete_all"]
)
def test_guardrail_checks_nested_arguments(word):
    state = initial_state("测试")
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "1", "name": "simulation_runner", "args": {"nested": [word]}}
            ],
        )
    ]
    assert guardrail_node(state)["hitl_required"]


@pytest.mark.parametrize("content", ["oops", "[]", '{"clear":"false"}', "null"])
def test_invalid_clarification_response_does_not_start_tools(content):
    assert clarify_node(initial_state("测试"), llm_factory=factory(content))[
        "needs_clarification"
    ]


@pytest.mark.parametrize(
    "content",
    [
        "bad",
        '{"confidence_score":NaN}',
        '{"confidence_score":1.1}',
        '{"confidence_score":true}',
        '{"confidence_score":-0.2}',
    ],
)
def test_invalid_final_confidence_requires_review(content):
    assert (
        agent_node(initial_state("测试"), llm_factory=factory(content))[
            "confidence_score"
        ]
        == 0
    )


@pytest.mark.parametrize(
    "content", ['{"verdict":"PASS","confidence_score":0.99}', "bad", "[]"]
)
def test_hard_failure_cannot_be_overruled_by_model(content):
    state = initial_state("测试")
    state["tool_outputs"] = evidence()
    state["tool_outputs"]["metrics_collector"]["packet_loss_rate"] = 0.5
    result = result_judge_node(state, llm_factory=factory(content))
    assert json.loads(result["final_result"])["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "missing", ["simulation_runner", "metrics_collector", "baseline_comparator"]
)
def test_missing_evidence_never_passes(missing):
    state = initial_state("测试")
    state["tool_outputs"] = evidence()
    del state["tool_outputs"][missing]
    result = result_judge_node(
        state, llm_factory=factory('{"verdict":"PASS","confidence_score":0.99}')
    )
    assert json.loads(result["final_result"])["verdict"] == "INCONCLUSIVE"
    assert result["confidence_score"] < 0.65


def test_tool_success_resets_consecutive_errors_and_preserves_input():
    state = initial_state("测试")
    state["error_count"] = 1
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {"id": "a", "name": "missing", "args": {}},
                {"id": "b", "name": "test_case_query", "args": {"feature": "handover"}},
            ],
        )
    ]
    before = deepcopy(state)
    patch = tool_node(state, tool_map=create_sandbox_tools())
    assert patch["error_count"] == 0
    assert [m.status for m in patch["messages"]] == ["error", "success"]
    assert state == before


def test_sessions_are_real_dependencies_and_factories_are_isolated():
    a, b = create_sandbox_tools(), create_sandbox_tools()
    session = a["simulation_runner"].invoke({"test_case_ids": ["TC-0001"]})
    assert (
        a["metrics_collector"].invoke({"session_id": session["session_id"]})["status"]
        == "completed"
    )
    with pytest.raises(ValueError):
        b["metrics_collector"].invoke({"session_id": session["session_id"]})
    with pytest.raises(ValueError):
        a["simulation_runner"].invoke({"test_case_ids": []})
    with pytest.raises(ValueError):
        a["baseline_comparator"].invoke(
            {"throughput_avg": float("nan"), "latency_p99": 10}
        )


def test_failed_repeated_call_discards_stale_cache():
    state = initial_state("测试")
    state["tool_outputs"] = {"metrics_collector": {"session_id": "old"}}
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "new",
                    "name": "metrics_collector",
                    "args": {"session_id": "unknown"},
                }
            ],
        )
    ]
    patch = tool_node(state, tool_map=create_sandbox_tools())
    assert "metrics_collector" not in patch["tool_outputs"]
    assert patch["tool_history"][0]["status"] == "error"


def test_model_cannot_replace_collected_metrics():
    state = initial_state("测试")
    state["tool_outputs"] = evidence()
    state["messages"] = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "fake",
                    "name": "baseline_comparator",
                    "args": {"throughput_avg": 999, "latency_p99": 1},
                }
            ],
        )
    ]
    patch = tool_node(state, tool_map=create_sandbox_tools())
    assert patch["messages"][0].status == "error"
    assert "baseline_comparator" not in patch["tool_outputs"]


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_invalid_numeric_evidence_never_passes(value):
    state = initial_state("测试")
    state["tool_outputs"] = evidence()
    state["tool_outputs"]["metrics_collector"]["latency_p99_ms"] = value
    result = result_judge_node(
        state, llm_factory=factory('{"verdict":"PASS","confidence_score":0.99}')
    )
    assert json.loads(result["final_result"])["verdict"] == "INCONCLUSIVE"
