"""新版入口与调查的业务回归：实际执行沙盒，不预写最终结论。"""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from experiments import SandboxExecutor
from task_models import MessageRequest, identifier
from task_service import TaskService
from task_store import ConflictError, TaskStore


def send(service, text, target=None, **kwargs):
    return service.message(
        MessageRequest(
            message_id=identifier("msg"), text=text, target_task_id=target, **kwargs
        )
    )


def investigate(service):
    response = send(service, "升级后移动视频卡顿，请查到能复现")
    return service.run(response["task_id"])


def test_investigation_changes_conditions_and_independently_reproduces():
    service = TaskService()
    task = investigate(service)
    assert task.status == "COMPLETED"
    assert task.report["test_verdict"] == "FAIL"
    assert len(task.experiments) == 7
    assert task.reproduction.handovers == 2
    assert not task.reproduction.background_traffic
    assert task.report["cause_status"] == "UNCONFIRMED"
    runs = [r for e in task.experiments for r in e.runs]
    assert len({r.run_id for r in runs}) == len(runs) == 21
    assert all(r.cleanup_status == "completed" for r in runs)
    assert task.experiments[-1].plan.kind == "confirm"


def test_plain_test_ends_without_investigation():
    service = TaskService()
    response = send(service, "验证新版本 A 到 B 的切换")
    task = service.run(response["task_id"])
    assert task.task_type == "TEST"
    assert len(task.experiments) == 1
    assert task.report["test_verdict"] == "PASS"
    assert task.issue_id is None


def test_failed_test_enters_investigation_only_when_requested():
    service = TaskService()
    task = service.run(
        send(service, "测试新版本高负载连续3次切换，失败再继续查")["task_id"]
    )
    assert task.task_type == "INVESTIGATE"
    assert task.experiments[0].plan_version == 1
    assert task.plan_version == 2
    assert task.reproduction is not None


def test_query_never_starts_ready_task_and_duplicate_message_is_not_executed():
    service = TaskService()
    task_id = send(service, "验证切换")["task_id"]
    before = service.get(task_id).model_dump()
    request = MessageRequest(
        message_id="query-1", target_task_id=task_id, text="现在怎么样了"
    )
    response = service.message(request)
    assert not response["run_requested"]
    assert service.get(task_id).model_dump() == before
    assert service.message(request)["duplicate"]
    with pytest.raises(ConflictError):
        service.message(request.model_copy(update={"text": "取消这次任务"}))


def test_duplicate_create_does_not_create_second_task():
    service = TaskService()
    request = MessageRequest(message_id="create-1", text="验证切换")
    first, duplicate = service.message(request), service.message(request)
    assert first["task_id"] == duplicate["task_id"]
    assert not duplicate["run_requested"]
    assert len(service.store.list()) == 1


def test_update_preserves_other_conditions_and_refuses_stale_version():
    service = TaskService()
    task_id = send(service, "复现 A 到 B 卡顿，带宽100 MHz，n78")["task_id"]
    first = service.get(task_id)
    service.step(task_id)
    send(
        service,
        "改成 C 到 D，带宽改为40 MHz，其他不变",
        task_id,
        expected_task_version=first.task_version,
    )
    task = service.get(task_id)
    assert task.conditions.source_cell == "C"
    assert task.conditions.target_cell == "D"
    assert task.conditions.bandwidth_mhz == 40
    assert task.conditions.band == "n78"
    assert task.plan_version == 2
    assert task.current_experiments() == []
    assert len(task.experiments) == 1
    with pytest.raises(ConflictError):
        send(
            service, "带宽改为20 MHz", task_id, expected_task_version=first.task_version
        )


@pytest.mark.parametrize("text", ["先别跑，只给已有结果", "取消这次任务"])
def test_stop_before_execution_creates_no_run(text):
    service = TaskService()
    task_id = send(service, "复现移动卡顿")["task_id"]
    response = send(service, text, task_id)
    assert not response["run_requested"]
    service.run(task_id)
    assert not service.get(task_id).experiments


class BlockingExecutor(SandboxExecutor):
    def __init__(self):
        super().__init__()
        self.started, self.release = Event(), Event()

    def execute(self, *args):
        self.started.set()
        assert self.release.wait(5)
        return super().execute(*args)


@pytest.mark.parametrize(
    "text,status", [("暂停", "PAUSED"), ("取消这次任务", "CANCELLED")]
)
def test_stop_inflight_finishes_cleanup_and_does_not_start_next_run(text, status):
    executor = BlockingExecutor()
    service = TaskService(executor=executor)
    task_id = send(service, "复现移动视频卡顿")["task_id"]
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.step, task_id)
        assert executor.started.wait(5)
        response = send(service, text, task_id)
        assert response["status"] in {"PAUSING", "CANCELLING"}
        executor.release.set()
        task = future.result(timeout=5)
    assert task.status == status
    assert len(task.experiments[0].runs) == 1
    assert task.experiments[0].runs[0].cleanup_status == "completed"
    assert task.reserved_runs == 1
    if status == "PAUSED":
        assert send(service, "继续", task_id)["run_requested"]
        finished = service.run(task_id)
        assert finished.status == "COMPLETED"
        assert len(finished.experiments) == 7
        assert len(finished.experiments[0].runs) == 3
        assert finished.reserved_runs == 21


def test_late_result_stays_with_old_plan_after_update():
    executor = BlockingExecutor()
    service = TaskService(executor=executor)
    task_id = send(service, "复现 A 到 B 移动卡顿")["task_id"]
    with ThreadPoolExecutor() as pool:
        future = pool.submit(service.step, task_id)
        assert executor.started.wait(5)
        send(service, "改成 C 到 D，其他不变", task_id)
        executor.release.set()
        task = future.result(timeout=5)
    assert task.conditions.source_cell == "C"
    assert task.experiments[0].status == "SUPERSEDED"
    assert task.experiments[0].runs[0].conditions.source_cell == "A"
    assert task.current_experiments() == []


def test_budget_handoff_does_not_overrun_or_claim_success():
    service = TaskService()
    task = service.run(send(service, "复现视频卡顿，最多5次")["task_id"])
    assert task.status == "HANDED_OFF"
    assert task.reserved_runs <= 5
    assert task.reproduction is None


@pytest.mark.parametrize("options", [{"missing_monitor": True}, {"fail_cleanup": True}])
def test_missing_monitor_and_failed_cleanup_stop_investigation(options):
    service = TaskService(executor=SandboxExecutor(**options))
    task = service.run(send(service, "复现视频卡顿")["task_id"])
    assert task.status == "ERROR"
    assert len(task.experiments[0].runs) == 1
    assert task.report["cleanup_status"] != "completed" or options.get(
        "missing_monitor"
    )


@pytest.mark.parametrize(
    "version,verdict", [("修复版", "PASS"), ("修复变慢版", "FAIL")]
)
def test_fix_keeps_original_reproduction_and_detects_side_effect(version, verdict):
    service = TaskService()
    original = investigate(service)
    task = service.run(
        send(service, f"这是{version}，按上次验证", original.task_id)["task_id"]
    )
    assert task.task_id != original.task_id
    assert task.related_task_id == original.task_id
    assert task.issue_id == original.issue_id
    assert task.report["test_verdict"] == verdict
    assert task.experiments[0].plan.conditions == original.reproduction
    assert service.get(original.task_id).report == original.report


def test_fix_without_reproduction_waits_and_readonly_never_runs():
    service = TaskService()
    fix = send(service, "请验证修复版")
    assert fix["status"] == "WAITING_INFO"
    readonly = send(service, "只看日志排查异常，不跑测试", conversation_id="readonly")
    assert not readonly["run_requested"]
    assert not service.get(readonly["task_id"]).experiments

    from task_models import IntentDecision

    class WrongIntent:
        def interpret(self, text, task):
            return IntentDecision(task_type="TEST", goal="TEST", read_only=False)

    guarded = TaskService(interpreter=WrongIntent())
    denied = send(guarded, "只看日志，不跑测试")
    assert not denied["run_requested"]
    assert not guarded.get(denied["task_id"]).experiments


def test_database_survives_restart_and_conversation_cannot_access_other_task(tmp_path):
    path = str(tmp_path / "tasks.sqlite")
    first = TaskService(store=TaskStore(path))
    original = investigate(first)
    first.store.close()
    second = TaskService(store=TaskStore(path))
    assert second.get(original.task_id).report == original.report
    with pytest.raises(KeyError):
        send(second, "现在怎么样了", original.task_id, conversation_id="other")
    second.store.close()


def test_recovery_does_not_replay_unknown_execution(tmp_path):
    path = str(tmp_path / "tasks.sqlite")
    first = TaskService(store=TaskStore(path))
    task_id = send(first, "复现卡顿")["task_id"]
    with first.store.lock, first.store.connection:
        task = first.get(task_id)
        task.status = "RUNNING"
        first.store.save(task)
    first.store.close()
    second = TaskService(store=TaskStore(path))
    second.recover()
    assert second.get(task_id).status == "HANDED_OFF"
    assert not second.executor.sessions


def test_cause_is_supported_not_code_confirmed():
    service = TaskService()
    task = service.run(send(service, "升级后视频卡顿，请查原因")["task_id"])
    assert task.report["cause_status"] == "SUPPORTED"
    assert task.hypotheses[0]["evidence_ids"]


def test_model_cannot_change_fixed_condition():
    from task_models import Conditions, ExperimentPlan, PlanDecision

    class BadPlanner:
        def choose(self, task):
            return PlanDecision(
                action="EXPERIMENT",
                plan=ExperimentPlan(
                    question="试着改参数",
                    reason="错误示范",
                    kind="screen",
                    conditions=Conditions(bandwidth_mhz=100),
                ),
            )

    service = TaskService(planner=BadPlanner())
    task = service.run(send(service, "复现卡顿，带宽40 MHz")["task_id"])
    assert task.status == "HANDED_OFF"
    assert not task.experiments


def test_decision_export_uses_pre_execution_context_and_requires_expert(tmp_path):
    import json

    from learning import DecisionReview, export_decision_reviews

    service = TaskService()
    task = investigate(service)
    event = next(e for e in task.events if e["event"] == "experiment_started")
    result = export_decision_reviews(
        task,
        [
            DecisionReview(
                experiment_id=event["experiment_id"],
                reviewer="expert",
                corrected_plan=event["plan"],
                confirmed=True,
            )
        ],
        tmp_path / "reviewed",
    )
    sample = json.loads(result["paths"][0].read_text(encoding="utf-8").strip())
    context = json.loads(sample["messages"][1]["content"])
    assert context["previous_experiments"] == []
    assert "reproduction" not in context
    assert result["sft_count"] == 1
    assert result["dpo_count"] == 0


def test_new_offline_evaluation_runs_real_service():
    from task_evaluation import run_task_evaluation

    assert run_task_evaluation()["all_passed"]


def test_decision_preparation_keeps_sft_and_dpo_in_same_source_group(tmp_path):
    import json

    from learning import DecisionReview, export_decision_reviews
    from training.prepare_investigation_data import prepare

    task = investigate(TaskService())
    event = next(e for e in task.events if e["event"] == "experiment_started")
    corrected = {**event["plan"], "reason": "专家确认：先做普通负载对照"}
    export_decision_reviews(
        task,
        [
            DecisionReview(
                experiment_id=event["experiment_id"],
                reviewer="expert",
                corrected_plan=corrected,
                confirmed=True,
            )
        ],
        tmp_path / "source",
    )
    counts = prepare(tmp_path / "source", tmp_path / "prepared")
    sft_split = next(
        k.removeprefix("sft_") for k, n in counts.items() if k.startswith("sft_") and n
    )
    dpo_split = next(
        k.removeprefix("dpo_") for k, n in counts.items() if k.startswith("dpo_") and n
    )
    assert sft_split == dpo_split
    assert json.loads(
        (tmp_path / "prepared" / "groups.json").read_text(encoding="utf-8")
    ) == {task.issue_id: sft_split}


def test_failed_cleanup_cannot_be_bypassed_by_pause_or_parameter_update():
    service = TaskService(executor=SandboxExecutor(fail_cleanup=True))
    task = service.run(send(service, "复现视频卡顿")["task_id"])
    assert send(service, "暂停", task.task_id)["status"] == "ERROR"
    with pytest.raises(ConflictError):
        send(service, "带宽改为40 MHz", task.task_id)


def test_followup_cause_reuses_existing_evidence_without_rerunning():
    service = TaskService()
    original = investigate(service)
    task = service.run(send(service, "继续查原因", original.task_id)["task_id"])
    assert task.status == "HANDED_OFF"
    assert task.related_task_id == original.task_id
    assert task.experiments == []
    assert task.report["referenced_experiments"]
    assert task.report["cause_status"] == "SUPPORTED"
    assert task.reserved_runs == 0


def test_unclear_parameter_update_waits_and_knowledge_query_is_readonly():
    service = TaskService()
    original = send(service, "验证切换")
    response = send(service, "带宽改小", original["task_id"])
    assert response["status"] == "WAITING_INFO"
    assert not response["run_requested"]
    task = service.get(response["task_id"])
    assert task.conditions.bandwidth_mhz == 100
    knowledge = send(service, "Xn 切换是什么意思", conversation_id="knowledge")
    assert knowledge["status"] == "READ_ONLY"
    assert not knowledge["run_requested"]
    assert knowledge["task_id"] is None
    unsupported = send(service, "测试 v9.9 的切换", conversation_id="unsupported")
    assert unsupported["status"] == "WAITING_INFO"
    assert not unsupported["run_requested"]


def test_report_before_fix_is_ordered_and_training_splits_reject_overlap(tmp_path):
    import json

    from training.prepare_investigation_data import read_training_splits

    service = TaskService()
    original = investigate(service)
    response = send(service, "先给我上次报告，再用修复版验证", original.task_id)
    assert response["previous_report"] == original.report
    assert response["task"]["task_type"] == "VERIFY_FIX"
    assert not response["task"]["experiments"]
    sample = {"metadata": {"source_group": "same-issue"}}
    train_path, validation_path = (
        tmp_path / "train.jsonl",
        tmp_path / "validation.jsonl",
    )
    train_path.write_text(json.dumps(sample), encoding="utf-8")
    validation_path.write_text(json.dumps(sample), encoding="utf-8")
    with pytest.raises(ValueError, match="同一原始问题"):
        read_training_splits(train_path, validation_path)
