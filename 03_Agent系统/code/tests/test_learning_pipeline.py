"""对照文档的行为测试：竞态、图路由、证据核验与训练目标，不只检查打印文字。"""

import json
import math

import pytest

from experiments import SandboxExecutor
from learning import DecisionReview, export_decision_reviews
from task_grading import grade_task
from task_models import MessageRequest, PlanDecision
from task_service import TaskService
from task_workflow import run_task_workflow


def create(service, text="复现升级后视频卡顿"):
    return service.message(MessageRequest(message_id="message-1", text=text))["task_id"]


def test_old_prepared_plan_cannot_commit_after_user_update():
    service = TaskService()
    task_id = create(service)
    prepared = service.prepare_step(task_id)
    service.message(
        MessageRequest(
            message_id="message-2",
            target_task_id=task_id,
            text="带宽改为40 MHz，其他不变",
        )
    )
    assert service.commit_step(prepared) is None
    assert service.get(task_id).experiments == []
    assert service.executor.sessions == {}


def test_graph_runs_same_service_phases_and_preserves_product_failure():
    service = TaskService()
    visited = []
    task = run_task_workflow(
        service, create(service), on_node=lambda name, _: visited.append(name)
    )
    assert visited[:4] == ["plan", "commit", "execute", "read"]
    assert task.status == "COMPLETED" and task.report["test_verdict"] == "FAIL"
    assert grade_task(task)["strict_success"]


def test_wait_is_persistent_and_does_not_spin_planner():
    class WaitingPlanner:
        calls = 0

        def choose(self, task):
            self.calls += 1
            return PlanDecision(action="WAIT", explanation="等待外部日志")

    planner = WaitingPlanner()
    service = TaskService(planner=planner)
    task = run_task_workflow(service, create(service))
    assert task.status == "WAITING_EVIDENCE" and task.experiments == []
    assert planner.calls == 1
    response = service.message(
        MessageRequest(
            message_id="resume-1",
            text="继续",
            target_task_id=task.task_id,
        )
    )
    assert response["status"] == "READY"


@pytest.mark.parametrize(
    "change", ["verdict", "reference", "cleanup", "cause", "metrics"]
)
def test_grading_rejects_report_or_evidence_tampering(change):
    service = TaskService()
    task = service.run(create(service))
    assert grade_task(task)["strict_success"]
    if change == "verdict":
        task.report["test_verdict"] = "PASS"
    elif change == "reference":
        task.report["evidence_ids"].append("other-task-evidence")
    elif change == "cleanup":
        task.experiments[0].runs[0].cleanup_status = "unknown"
    elif change == "cause":
        task.report["cause_status"] = "CONFIRMED"
    else:
        task.experiments[0].runs[0].metrics["outage_seconds"] = float("nan")
    assert not grade_task(task)["strict_success"]


def test_missing_monitor_does_not_become_successful_reproduction():
    service = TaskService(executor=SandboxExecutor(missing_monitor=True))
    grading = grade_task(service.run(create(service)))
    assert not grading["goal_met"] and grading["invalid_run_rate"] == 1


def test_training_target_matches_deployment_wrapper_and_preserves_roles(tmp_path):
    service = TaskService()
    task = service.run(create(service))
    event = next(e for e in task.events if e["event"] == "experiment_started")
    corrected = {**event["plan"], "reason": "专家确认本轮应先保持条件进行比较"}
    export_decision_reviews(
        task,
        [
            DecisionReview(
                experiment_id=event["experiment_id"],
                reviewer="test-expert",
                corrected_plan=corrected,
                confirmed=True,
            )
        ],
        tmp_path,
    )
    sft = json.loads((tmp_path / "sft.jsonl").read_text(encoding="utf-8"))
    dpo = json.loads((tmp_path / "dpo.jsonl").read_text(encoding="utf-8"))
    decision = PlanDecision.model_validate_json(sft["messages"][-1]["content"])
    assert decision.action == "EXPERIMENT"
    assert [m["role"] for m in dpo["prompt"]] == ["system", "user"]
    context = json.loads(sft["messages"][1]["content"])
    assert context["previous_experiments"] == []  # 没有把未来结果补回第一次决策。
    assert context["active_skill"]["content_hash"]


def test_tensor_losses_match_document_and_reference_is_frozen():
    torch = pytest.importorskip(
        "torch"
    )  # 基础Agent可不装训练依赖，此测试明确标记可选。
    from training.objectives import (
        build_labels,
        causal_token_loss,
        dpo_pair_loss,
        sequence_log_probs,
    )

    labels = torch.tensor([build_labels([0, 0, 1], [0, 1, 1], [1, 1, 1])])
    logits = torch.tensor(
        [[[math.log(0.8), math.log(0.2)], [math.log(0.5), math.log(0.5)], [0.0, 0.0]]],
        requires_grad=True,
    )
    loss, count = causal_token_loss(logits, labels)
    assert count.item() == 2
    assert loss.item() == pytest.approx(0.45814536)
    assert sequence_log_probs(logits, labels).item() == pytest.approx(-0.91629073)
    reference = torch.tensor([-82.0], requires_grad=True)
    dpo = dpo_pair_loss(
        torch.tensor([-80.0], requires_grad=True),
        torch.tensor([-100.0]),
        reference,
        torch.tensor([-95.0]),
    )
    assert dpo.item() == pytest.approx(0.40318605)
    dpo.backward()
    assert reference.grad is None
    with pytest.raises(ValueError, match="有效目标"):
        causal_token_loss(logits, torch.full_like(labels, -100))


def test_skill_path_cannot_escape_library():
    from skill_library import SkillLibrary

    with pytest.raises(ValueError, match="超出"):
        SkillLibrary().load("../../other")


def test_training_only_accepts_the_deployed_decision_schema():
    from training.prepare_investigation_data import normalize_target

    target = PlanDecision(action="WAIT", explanation="等待日志").model_dump_json()
    assert normalize_target(target, "PlanDecision") == target
    for schema in (None, "ExperimentPlan", "report"):
        with pytest.raises(ValueError, match="PlanDecision"):
            normalize_target(target, schema)


def test_bad_training_pair_does_not_write_partial_dataset(tmp_path):
    from training.prepare_investigation_data import prepare

    target = PlanDecision(action="WAIT", explanation="等待日志").model_dump_json()
    metadata = {
        "sample_type": "experiment_decision",
        "source_group": "same-issue",
        "reviewer": "expert",
        "target_schema": "PlanDecision",
    }
    # SFT 合法，DPO 两个候选相同；不能留下看似已准备好的半套训练文件。
    (tmp_path / "sft.jsonl").write_text(
        json.dumps(
            {
                "messages": [{"role": "assistant", "content": target}],
                "metadata": metadata,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "dpo.jsonl").write_text(
        json.dumps(
            {
                "prompt": "日志不足",
                "chosen": target,
                "rejected": target,
                "metadata": metadata,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="相同候选"):
        prepare(tmp_path, output)
    assert not output.exists()
