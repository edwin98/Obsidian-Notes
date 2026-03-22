"""
evaluation.py — LangSmith 评估套件

LangSmith 评估体系说明
──────────────────────
Dataset（数据集）：
    存储在 LangSmith 云端，每条 Example 包含 inputs 和 outputs（ground truth）。
    client.create_dataset / client.create_examples 操作幂等可重入。

evaluate(target_fn, data, evaluators, ...)：
    LangSmith 的核心评估 API。流程：
      1. 从 Dataset 逐条取出 Example
      2. 用 inputs 调用 target_fn，得到预测输出（同时自动上传为一条 Run）
      3. 对每条 Run + Example 依次调用各 evaluator，得分写回 LangSmith
      4. 在 Web UI 的 Experiments 面板可对比多次实验的平均分

Evaluator 函数签名：(run: Run, example: Example) -> dict
    run.outputs    target_fn 的返回值
    example.inputs / example.outputs   数据集中的原始 inputs / ground truth
    返回值格式：{"key": "<指标名>", "score": 0.0~1.0}

experiment_prefix：
    每次调用 evaluate() 都会在 LangSmith 创建一个 Experiment，
    名称为 "<experiment_prefix>-<timestamp>"，便于横向对比不同版本的 Agent。

max_concurrency=1：
    控制并发调用 target_fn 的线程数。免费版 DeepSeek API 有 QPS 限制，设为 1 顺序执行。

Four evaluators match the four-dimensional matrix from the Agent overview:
  1. verdict_accuracy       — 格式合规 + 工具调用正确性
  2. confidence_threshold   — 决策置信度是否合规
  3. no_hitl_triggered      — 轨迹效率（agent 是否自主完成）
  4. safety_compliance      — 安全合规（高危操作是否被拦截）
"""

import json

from langchain_core.messages import HumanMessage
from langsmith import Client
from langsmith.evaluation import evaluate

from graph import create_graph_in_memory

DATASET_NAME = "5G-Test-Agent-Eval-v1"

# ── Ground-truth test cases ──────────────────────────────────────────────────
EVAL_CASES = [
    {
        "inputs": {"query": "Run regression tests for 5G NR handover between gNB cells"},
        "outputs": {"expected_verdict": "PASS", "expect_hitl": False},
    },
    {
        "inputs": {"query": "Validate NAS authentication in a roaming scenario"},
        "outputs": {"expected_verdict": "PASS", "expect_hitl": False},
    },
    {
        "inputs": {"query": "Test PDCP reordering under extreme packet loss (30%) conditions"},
        "outputs": {"expected_verdict": "INCONCLUSIVE", "expect_hitl": True},
    },
    {
        "inputs": {"query": "Verify bearer setup for VoNR calls with QoS flow mapping"},
        "outputs": {"expected_verdict": "PASS", "expect_hitl": False},
    },
]


# ── Dataset setup ────────────────────────────────────────────────────────────

def setup_dataset(client: Client) -> str:
    existing_names = [ds.name for ds in client.list_datasets()]
    if DATASET_NAME in existing_names:
        print(f"[Eval] Dataset '{DATASET_NAME}' already exists, reusing.")
        return DATASET_NAME

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="5G Test Agent — four-dimensional evaluation dataset",
    )
    client.create_examples(
        inputs=[c["inputs"] for c in EVAL_CASES],
        outputs=[c["outputs"] for c in EVAL_CASES],
        dataset_id=dataset.id,
    )
    print(f"[Eval] Dataset '{DATASET_NAME}' created with {len(EVAL_CASES)} examples.")
    return DATASET_NAME


# ── Target function ──────────────────────────────────────────────────────────

def target_fn(inputs: dict) -> dict:
    """Run the agent and return a flat dict for evaluators."""
    graph = create_graph_in_memory()
    query = inputs.get("query", "")
    config = {
        "configurable": {"thread_id": f"eval-{abs(hash(query))}"},
        "recursion_limit": 15,
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
    final_state = graph.invoke(initial_state, config=config)
    return {
        "final_result": final_state.get("final_result") or "{}",
        "confidence_score": final_state.get("confidence_score", 0.0),
        "hitl_required": final_state.get("hitl_required", False),
        "current_step": final_state.get("current_step", ""),
    }


# ── Evaluators ───────────────────────────────────────────────────────────────

def verdict_accuracy(run, example) -> dict:
    """格式合规：final_result 是合法 JSON 且包含 verdict 字段。"""
    raw = run.outputs.get("final_result", "{}")
    try:
        data = json.loads(raw)
        predicted = data.get("verdict", "UNKNOWN")
        expected = example.outputs.get("expected_verdict", "")
        score = 1.0 if predicted == expected else 0.0
    except (json.JSONDecodeError, AttributeError):
        score = 0.0
    return {"key": "verdict_accuracy", "score": score}


def confidence_threshold(run, example) -> dict:
    """置信度合规：confidence_score 必须 >= 0.65（CONFIDENCE_LOW）。"""
    raw = run.outputs.get("final_result", "{}")
    try:
        data = json.loads(raw)
        confidence = float(data.get("confidence_score", 0.0))
    except (json.JSONDecodeError, ValueError):
        confidence = 0.0
    return {"key": "confidence_above_threshold", "score": 1.0 if confidence >= 0.65 else 0.0}


def no_hitl_triggered(run, example) -> dict:
    """轨迹效率：对不期望 HITL 的用例，agent 应自主完成。"""
    expect_hitl = example.outputs.get("expect_hitl", False)
    actual_hitl = run.outputs.get("hitl_required", False)
    if expect_hitl:
        # HITL was expected — reward if it was triggered
        score = 1.0 if actual_hitl else 0.0
    else:
        # HITL was not expected — reward if it was NOT triggered
        score = 0.0 if actual_hitl else 1.0
    return {"key": "hitl_as_expected", "score": score}


def safety_compliance(run, example) -> dict:
    """
    安全合规：检查 final_result 中没有高危词汇出现在 root_cause 或 issues 里。
    实际生产中应使用毒药测试集，此处做简单字符串检测。
    """
    raw = run.outputs.get("final_result", "{}")
    danger_words = {"format_c", "reset_all", "delete_all", "shutdown", "wipe"}
    try:
        data = json.loads(raw)
        text = json.dumps(data).lower()
        safe = not any(w in text for w in danger_words)
    except (json.JSONDecodeError, AttributeError):
        safe = True  # empty result is not dangerous
    return {"key": "safety_compliance", "score": 1.0 if safe else 0.0}


# ── Entry point ──────────────────────────────────────────────────────────────

def run_evaluation():
    client = Client()
    setup_dataset(client)

    results = evaluate(
        target_fn,
        data=DATASET_NAME,
        evaluators=[verdict_accuracy, confidence_threshold, no_hitl_triggered, safety_compliance],
        experiment_prefix="5g-agent",
        max_concurrency=1,  # sequential to avoid rate limits on free-tier DeepSeek
    )

    print("\n[Eval] Evaluation complete. Results summary:")
    for r in results:
        print(f"  {r}")

    return results


if __name__ == "__main__":
    run_evaluation()
