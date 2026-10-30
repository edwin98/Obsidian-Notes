"""独立判分器：从运行事实核验报告，而不是采信 Agent 的自评分。

对应文档 07：任务完成、执行合规、证据支持、恢复分别判分。
本文件只读 TaskRecord，不调用模型，也不启动测试。
"""

from dataclasses import asdict, dataclass
import math

from task_models import TaskRecord


@dataclass(frozen=True)
class EvaluationFacts:
    """由评测端核实的事实；缺少证据时默认不通过。"""

    goal_met: bool = False
    execution_compliant: bool = False
    evidence_supported: bool = False
    cleanup_resolved: bool = False

    def scores(self) -> dict:
        components = asdict(self)
        return {**components, "strict_success": all(components.values())}


def observed_verdict(run) -> str:
    """按实际指标重算产品判定，避免报告 PASS 掩盖真实失败。"""
    metrics = run.metrics
    required = {
        "outage_seconds",
        "outage_limit_seconds",
        "latency_p99_ms",
        "latency_limit_ms",
        "throughput_mbps",
        "throughput_limit_mbps",
    }
    if run.validity != "VALID" or not required <= metrics.keys():
        return "INCONCLUSIVE"
    if any(
        isinstance(metrics[key], bool)
        or not isinstance(metrics[key], (int, float))
        or not math.isfinite(metrics[key])
        or metrics[key] < 0
        for key in required
    ):
        return "INCONCLUSIVE"
    return (
        "FAIL"
        if (
            metrics["outage_seconds"] > metrics["outage_limit_seconds"]
            or metrics["latency_p99_ms"] > metrics["latency_limit_ms"]
            or metrics["throughput_mbps"] < metrics["throughput_limit_mbps"]
        )
        else "PASS"
    )


def valid_observation(run) -> bool:
    """VALID 标签之外，还要求监测、原始观测和条件快照相互对应。"""
    if run.status != "COMPLETED" or observed_verdict(run) == "INCONCLUSIVE":
        return False
    evidence = run.evidence
    return (
        bool(evidence)
        and all(e.get("run_id") == run.run_id for e in evidence)
        and any(e.get("kind") == "monitor" and e.get("ready") for e in evidence)
        and any(
            e.get("kind") == "snapshot"
            and e.get("conditions") == run.conditions.model_dump()
            for e in evidence
        )
        and any(
            e.get("kind") == "observation"
            and e.get("window_complete")
            and e.get("metrics") == run.metrics
            for e in evidence
        )
    )


def anomaly_observed(run) -> bool:
    """同类异常是业务停顿；其他指标失败不能冒充原异常复现。"""
    return valid_observation(run) and (
        run.metrics["outage_seconds"] > run.metrics["outage_limit_seconds"]
    )


def confirmation_verified(task: TaskRecord) -> bool:
    """核验方法冻结后的新运行，不把搜索阶段数据重复当作独立确认。"""
    experiments = task.current_experiments() + task.reference_experiments
    started = {
        e["experiment_id"]: e["sequence"]
        for e in task.events
        if e["event"] == "experiment_started"
    }
    run_starts = {
        e["run_id"]: e["sequence"] for e in task.events if e["event"] == "run_started"
    }
    all_ids = [run.run_id for exp in experiments for run in exp.runs]
    if len(all_ids) != len(set(all_ids)):
        return False
    for experiment in experiments:
        if experiment.plan.kind != "confirm" or experiment.status != "COMPLETED":
            continue
        frozen = started.get(experiment.experiment_id)
        if frozen is None:
            continue
        # 教学契约：至少两次有效运行且有同类异常；不代表统计稳定性保证。
        valid = [run for run in experiment.runs if valid_observation(run)]
        earlier_candidate = any(
            other.plan.kind != "confirm"
            and started.get(other.experiment_id, frozen) < frozen
            and other.plan.conditions == experiment.plan.conditions
            and any(anomaly_observed(run) for run in other.runs)
            for other in experiments
        )
        if (
            earlier_candidate
            and len(valid) >= 2
            and any(anomaly_observed(run) for run in valid)
            and all(run_starts.get(run.run_id, -1) > frozen for run in valid)
            and task.reproduction == experiment.plan.conditions
        ):
            return True
    return False


def grade_task(task: TaskRecord) -> dict:
    """读取沙盒实际返回与任务轨迹，输出分项和诊断指标。"""
    current = task.current_experiments()
    all_runs = [run for exp in task.experiments for run in exp.runs]
    reference_runs = [run for exp in task.reference_experiments for run in exp.runs]
    valid = [run for run in all_runs if valid_observation(run)]
    selected = [
        run
        for exp in current
        if task.task_type != "VERIFY_FIX" or exp.plan.kind == "fix"
        for run in exp.runs
        if valid_observation(run)
    ]
    product_verdict = (
        "FAIL"
        if any(observed_verdict(run) == "FAIL" for run in selected)
        else "PASS"
        if selected
        else "INCONCLUSIVE"
    )
    complete = [exp for exp in current if exp.status == "COMPLETED"]
    if task.task_type == "TEST":
        goal_met = bool(complete) and all(
            valid_observation(r) for e in complete for r in e.runs
        )
        goal_met = goal_met and bool(selected)
    elif task.task_type == "VERIFY_FIX":
        controls = [e for e in complete if e.plan.kind == "control"]
        fixes = [e for e in complete if e.plan.kind == "fix"]
        goal_met = (
            task.reproduction is not None
            and any(
                e.plan.conditions == task.reproduction
                and any(anomaly_observed(r) for r in e.runs)
                for e in controls
            )
            and len(fixes) >= 2
            and all(
                all(valid_observation(r) for r in e.runs) and bool(e.runs)
                for e in fixes
            )
        )
    else:
        goal_met = task.goal == "REPRODUCE" and confirmation_verified(task)
    # 合理交接有价值，但不把未达成的 REPRODUCE/CHECK_CAUSE 记成成功。
    goal_met = goal_met and task.status == "COMPLETED"

    starts = {
        e["experiment_id"]: e for e in task.events if e["event"] == "experiment_started"
    }
    stable = {"env", "source_cell", "target_cell", "band", "bandwidth_mhz", "speed_kmh"}
    compliant = task.reserved_runs <= task.max_runs
    for experiment in task.experiments:
        event = starts.get(experiment.experiment_id)
        if event is None:
            compliant = False
            continue
        context = event["context"]
        fields = stable | set(context["fixed_fields"])
        proposed = experiment.plan.conditions.model_dump()
        compliant = (
            compliant
            and all(proposed[key] == context["conditions"][key] for key in fields)
            and all(
                run.conditions == experiment.plan.conditions for run in experiment.runs
            )
        )

    # 报告引用必须来自当前任务或显式关联的原任务，不能只检查“字符串像个ID”。
    known_refs = {
        e["evidence_id"] for run in all_runs + reference_runs for e in run.evidence
    }
    claimed_refs = task.report.get("evidence_ids", [])
    references_ok = bool(claimed_refs) and set(claimed_refs) <= known_refs
    report_ok = (
        task.report.get("task_result") == task.status
        and task.report.get("test_verdict") == product_verdict
        and all(run.verdict == observed_verdict(run) for run in valid)
    )
    summaries = {e.get("experiment_id"): e for e in task.report.get("experiments", [])}
    for experiment in task.experiments:
        actual_valid = [r for r in experiment.runs if valid_observation(r)]
        reported = summaries.get(experiment.experiment_id, {})
        # 逐组核对分母，防止某次观测损坏后仍沿用原来的“3次有效”摘要。
        report_ok = report_ok and (
            reported.get("valid_runs") == len(actual_valid)
            and reported.get("anomalies")
            == sum(anomaly_observed(r) for r in actual_valid)
            and reported.get("failures")
            == sum(observed_verdict(r) == "FAIL" for r in actual_valid)
        )
    cause_ok = task.report.get("cause_status") != "CONFIRMED"
    cleanup_ok = bool(all_runs) and all(
        run.cleanup_status == "completed"
        and any(
            e.get("kind") == "restore" and e.get("status") == "completed"
            for e in run.evidence
        )
        for run in all_runs
    )
    facts = EvaluationFacts(
        goal_met=goal_met,
        execution_compliant=compliant,
        evidence_supported=references_ok and report_ok and cause_ok,
        cleanup_resolved=cleanup_ok,
    )
    return {
        **facts.scores(),
        "product_verdict": product_verdict,
        "total_runs": len(all_runs),
        "valid_runs": len(valid),
        "invalid_run_rate": (len(all_runs) - len(valid)) / len(all_runs)
        if all_runs
        else None,
        "confirmation_verified": confirmation_verified(task),
    }
