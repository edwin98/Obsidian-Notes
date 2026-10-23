"""真实 LangGraph 集成测试：只有模型与业务平台用本地替身。"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from demo import create_demo_graph
from graph import create_graph_in_memory
from state import initial_state


@pytest.mark.parametrize(
    "scenario,verdict,count", [("pass", "PASS", 4), ("regression", "FAIL", 5)]
)
def test_full_react_uses_real_tool_observations(config, scenario, verdict, count):
    graph = create_demo_graph(scenario)
    state = graph.invoke(initial_state("验证 Xn 切换回归"), config)
    assert json.loads(state["final_result"])["verdict"] == verdict
    assert len(state["tool_history"]) == count
    calls = [
        tc
        for msg in state["messages"]
        if isinstance(msg, AIMessage)
        for tc in msg.tool_calls
    ]
    observations = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert [tc["id"] for tc in calls] == [m.tool_call_id for m in observations]
    sid = state["tool_outputs"]["simulation_runner"]["session_id"]
    assert state["tool_outputs"]["metrics_collector"]["session_id"] == sid
    assert not graph.get_state(config).next


@pytest.mark.parametrize("approved,verdict", [(True, "PASS"), (False, "REJECTED")])
def test_danger_pauses_before_side_effect_and_resumes_once(config, approved, verdict):
    graph = create_demo_graph("danger")
    state = graph.invoke(initial_state("验证切换"), config)
    assert "__interrupt__" in state
    assert not state.get("tool_history")  # 拦截发生在工具执行前。
    assert graph.get_state(config).next == ("hitl",)
    final = graph.invoke(
        Command(resume={"approved": approved, "feedback": "专家已复核"}), config
    )
    assert json.loads(final["final_result"])["verdict"] == verdict
    assert len(final["review_history"]) == 1
    assert len(final["tool_history"]) == (4 if approved else 0)
    if approved:
        assert sum(c["name"] == "test_case_query" for c in final["tool_history"]) == 1


def test_low_confidence_review_does_not_raise_confidence(config):
    graph = create_demo_graph("review")
    state = graph.invoke(initial_state("验证切换"), config)
    assert graph.get_state(config).next
    assert state["confidence_score"] == 0.4
    state = graph.invoke(Command(resume={"approved": True}), config)
    assert not graph.get_state(config).next
    assert state["confidence_score"] == 0.4
    assert json.loads(state["final_result"])["confidence_score"] == 0.4


def test_clarification_keeps_user_history(config):
    from demo import OfflineModel

    class Model(OfflineModel):
        def invoke(self, messages):
            if not self.with_tools and "判断对话" in str(messages[0].content):
                return AIMessage(
                    content=json.dumps(
                        {
                            "clear": "Xn" in str(messages[-1].content),
                            "question": "测试什么功能？",
                        }
                    )
                )
            return super().invoke(messages)

    graph = create_graph_in_memory(llm_factory=lambda tools: Model(with_tools=tools))
    first = graph.invoke(initial_state("跑一下测试"), config)
    assert first["needs_clarification"] and first["final_result"] is None
    result = graph.invoke(
        {"messages": [HumanMessage(content="验证 Xn 切换回归")]}, config
    )
    assert result["final_result"]
    assert len([m for m in result["messages"] if isinstance(m, HumanMessage)]) == 2


def test_hard_recursion_limit_is_not_reported_as_pass(config):
    graph = create_demo_graph()
    with pytest.raises(GraphRecursionError):
        graph.invoke(initial_state("验证切换"), {**config, "recursion_limit": 2})


def test_checkpoint_threads_are_isolated(config):
    graph = create_demo_graph()
    a = graph.invoke(initial_state("验证切换 A"), config)
    b_config = {**config, "configurable": {"thread_id": "thread-b"}}
    b = graph.invoke(initial_state("验证切换 B"), b_config)
    assert len(a["tool_history"]) == len(b["tool_history"]) == 4
    assert (
        a["tool_outputs"]["simulation_runner"]["session_id"]
        != b["tool_outputs"]["simulation_runner"]["session_id"]
    )


def test_error_circuit_breaker_pauses(config):
    class FailingModel:
        def invoke(self, messages):
            if "判断对话" in str(messages[0].content):
                return AIMessage(content='{"clear":true}')
            return AIMessage(
                content="",
                tool_calls=[{"name": "unknown", "args": {}, "id": "bad-call"}],
            )

    graph = create_graph_in_memory(llm_factory=lambda _: FailingModel())
    state = graph.invoke(initial_state("验证切换"), config)
    assert state["error_count"] == 2
    assert graph.get_state(config).next == ("hitl",)
