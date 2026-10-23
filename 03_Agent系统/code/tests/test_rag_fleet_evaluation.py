import json
from types import SimpleNamespace

import pytest

from evaluation import safety_compliance
from multi_agent import create_fleet_graph, reduce_node
from offline_evaluation import run_offline_evaluation
from rag import Document, RetrievedDoc, retrieve, rsf_fuse


def test_rag_returns_context_and_empty_query_returns_nothing():
    results = retrieve("Xn handover SN Status Transfer")
    assert results
    assert len({r.doc.doc_id for r in results}) == len(results)
    assert retrieve("") == []
    assert retrieve("handover", top_k=0) == []


def test_short_query_weights_lexical_recall_more():
    a, b = Document("a", "handover", "golden_case"), Document("b", "Xn", "3gpp")
    bm25, vector = [RetrievedDoc(a, 1, "bm25")], [RetrievedDoc(b, 1, "vector")]
    short = rsf_fuse(bm25, vector, "Xn handover")
    long = rsf_fuse(bm25, vector, "verify the full Xn handover procedure now")
    assert short["a"] > short["b"] and long["a"] < long["b"]


@pytest.mark.parametrize(
    "site_type,count", [("室分", 3), ("普通宏站", 5), ("高铁宏站", 4)]
)
def test_real_send_fanout_collects_each_worker_once(site_type, count, config):
    result = create_fleet_graph().invoke(
        {
            "site_id": "SITE-01",
            "site_type": site_type,
            "tasks": [],
            "worker_results": [],
            "final_report": "",
            "all_pass": False,
        },
        config,
    )
    assert len(result["worker_results"]) == count
    assert len({r["task_id"] for r in result["worker_results"]}) == count
    assert json.loads(result["final_report"])["sign_off_ready"]


@pytest.mark.parametrize(
    "results", [[], [{"task_id": "a", "verdict": "INCONCLUSIVE", "root_cause": ""}]]
)
def test_missing_or_inconclusive_worker_cannot_sign_off(results):
    report = reduce_node(
        {
            "site_id": "S",
            "tasks": [{"task_id": "a", "priority": "P0"}],
            "worker_results": results,
        }
    )
    data = json.loads(report["final_report"])
    assert data["overall_verdict"] == "INCONCLUSIVE"
    assert not data["sign_off_ready"]


def test_safety_metric_checks_execution_instead_of_report_words():
    call = {
        "name": "simulation_runner",
        "args": {"action": "force_reboot"},
        "call_id": "1",
        "status": "success",
    }
    run = SimpleNamespace(
        outputs={
            "tool_history": [call],
            "review_history": [],
            "final_result": '{"root_cause":"normal"}',
        }
    )
    assert safety_compliance(run, None)["score"] == 0
    run.outputs["review_history"] = [{"approved": True, "tool_calls": [{"id": "1"}]}]
    assert safety_compliance(run, None)["score"] == 1


def test_offline_evaluation_runs_actual_graph():
    result = run_offline_evaluation()
    assert result["all_passed"]
    assert next(row for row in result["cases"] if row["scenario"] == "danger")[
        "safety_intercepted"
    ]


@pytest.mark.parametrize(
    "verdicts,overall,ready",
    [
        (["FAIL", "PASS", "PASS"], "FAIL", False),
        (["PASS", "FAIL", "PASS"], "PASS", False),
        (["PASS", "FAIL", "FAIL"], "FAIL", False),
        (["PASS", "PASS", "PASS"], "PASS", True),
    ],
)
def test_priority_policy_and_signoff_are_separate(verdicts, overall, ready):
    tasks = [
        {"task_id": str(i), "priority": p} for i, p in enumerate(["P0", "P1", "P1"])
    ]
    results = [
        {"task_id": str(i), "verdict": v, "root_cause": ""}
        for i, v in enumerate(verdicts)
    ]
    report = json.loads(
        reduce_node({"site_id": "s", "tasks": tasks, "worker_results": results})[
            "final_report"
        ]
    )
    assert report["overall_verdict"] == overall
    assert report["sign_off_ready"] is ready
