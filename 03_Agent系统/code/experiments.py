"""【学习定位：规划与环境分离】
OfflinePlanner/ModelPlanner提出下一组；SandboxExecutor拥有隐藏故障规则；summary只计算已发生的数据。
对应04、12、13；从choose到execute，再看build_planning_context。

新版实验的沙盒执行与计划选择。隐藏故障规则只存在执行端。
"""

from threading import RLock

from task_models import (
    Conditions,
    ExperimentPlan,
    PlanDecision,
    RunRecord,
    TaskRecord,
    identifier,
    timestamp,
)


class SandboxExecutor:
    """模拟已登记的准备、执行、采集和恢复，不连接现场设备。"""

    def __init__(self, *, missing_monitor=False, fail_cleanup=False):
        self.lock = RLock()
        self.sessions = {}
        self.missing_monitor = missing_monitor
        self.fail_cleanup = fail_cleanup

    def execute(self, conditions: Conditions, run_id: str, ordinal: int) -> RunRecord:
        # 运行是一整段动作及其观测，不是每次切换；相同run_id重传与新一次重复运行的语义不同。
        """同一运行键幂等；另一次重复测试必须使用新的 run_id。"""
        with self.lock:
            if run_id in self.sessions:
                return self.sessions[run_id].model_copy(deep=True)
            record = RunRecord(run_id=run_id, conditions=conditions)
            record.session_id = identifier("sim")
            evidence = [
                {
                    "evidence_id": identifier("ev"),
                    "kind": "snapshot",
                    "run_id": run_id,
                    "conditions": conditions.model_dump(),
                    "source": "sandbox",
                },
                {
                    "evidence_id": identifier("ev"),
                    "kind": "monitor",
                    "run_id": run_id,
                    "ready": not self.missing_monitor,
                    "source": "sandbox",
                },
            ]
            try:
                if self.missing_monitor:
                    raise ValueError("业务监测未就绪，未执行测试动作")
                # 故障规则只在环境端，计划器只能读已发生的观测。
                # ordinal 是组内重复编号，不是切换动作次数，也不是随机概率。
                trigger = (
                    conditions.version == "new"
                    and conditions.load == "high"
                    and conditions.handovers >= 2
                    and ordinal % 3 == 0
                )
                metrics = {
                    "outage_seconds": 2.0 if trigger else 0.05,
                    "latency_p99_ms": 50.0
                    if conditions.version == "slow_fix"
                    else 10.0,
                    "throughput_mbps": 60.0 if trigger else 150.0,
                    "throughput_limit_mbps": 90.0,
                    "attempts": 1,
                    "outage_limit_seconds": 0.5,
                    "latency_limit_ms": 25.0,
                    "rule_id": "SANDBOX-BUSINESS-1",
                }
                events = ["旧连接资源回收延迟", "业务传输停顿"] if trigger else []
                evidence.append(
                    {
                        "evidence_id": identifier("ev"),
                        "kind": "observation",
                        "run_id": run_id,
                        "metrics": metrics,
                        "events": events,
                        "source": "sandbox",
                        "window_complete": True,
                    }
                )
                record.metrics = metrics
                record.validity = "VALID"
                record.verdict = (
                    "FAIL"
                    if (
                        metrics["outage_seconds"] > metrics["outage_limit_seconds"]
                        or metrics["latency_p99_ms"] > metrics["latency_limit_ms"]
                        or metrics["throughput_mbps"] < metrics["throughput_limit_mbps"]
                    )
                    else "PASS"
                )
                record.status = "COMPLETED"
            except ValueError as exc:
                record.status = "FAILED"
                record.error = str(exc)
            finally:
                record.cleanup_status = "failed" if self.fail_cleanup else "completed"
                evidence.append(
                    {
                        "evidence_id": identifier("ev"),
                        "kind": "restore",
                        "run_id": run_id,
                        "status": record.cleanup_status,
                        "source": "sandbox",
                    }
                )
                record.evidence = evidence
                record.finished_at = timestamp()
                self.sessions[run_id] = record.model_copy(deep=True)
            return record


def summary(experiment) -> dict:
    """从有效运行计算摘要；业务停顿次数和全部产品失败次数分别统计。"""
    valid = [r for r in experiment.runs if r.validity == "VALID"]
    violations = set()
    for run in valid:
        metrics = run.metrics
        if metrics.get("outage_seconds", 0) > metrics.get("outage_limit_seconds", 0.5):
            violations.add("业务停顿超过本次沙盒门限")
        if metrics.get("latency_p99_ms", 0) > metrics.get("latency_limit_ms", 25):
            violations.add("P99 时延超过本次沙盒门限")
        if metrics.get("throughput_mbps", 150) < metrics.get(
            "throughput_limit_mbps", 90
        ):
            violations.add("吞吐量低于本次沙盒门限")
    return {
        "experiment_id": experiment.experiment_id,
        "kind": experiment.plan.kind,
        "conditions": experiment.plan.conditions.model_dump(),
        "valid_runs": len(valid),
        "anomalies": sum(r.metrics.get("outage_seconds", 0) > 0.5 for r in valid),
        "failures": sum(r.verdict == "FAIL" for r in valid),
        "violations": sorted(violations),
        "status": experiment.status,
    }


class OfflinePlanner:
    """教学搜索路线：规则选择实验，但每次必须查看实际结果。"""

    def choose(self, task: TaskRecord) -> PlanDecision:
        # 每次根据已完成实验重新选择路线：初筛、版本对照、简化、确认、交付；不预先写最终报告。
        experiments = task.current_experiments()
        complete = [e for e in experiments if e.status == "COMPLETED"]
        base = task.conditions.model_dump()
        if (
            task.goal == "CHECK_CAUSE"
            and task.reproduction
            and task.reference_experiments
        ):
            return PlanDecision(
                action="HANDOFF",
                explanation="已复用原复现和日志；现有工具缺少内部状态与区分原因的干预能力，需要开发继续核查",
            )

        def plan(question, kind, **changes):
            condition = {**base, **changes}
            for key in task.fixed_fields:
                condition[key] = base[key]
            return PlanDecision(
                action="EXPERIMENT",
                plan=ExperimentPlan(
                    question=question,
                    reason=question,
                    conditions=Conditions(**condition),
                    kind=kind,
                    repeats=min(3, task.max_runs),
                    comparison_id=complete[-1].experiment_id if complete else None,
                ),
            )

        if task.task_type == "TEST":
            if complete:
                return PlanDecision(action="REPORT", explanation="已完成指定条件测试")
            return plan("指定条件下是否满足沙盒业务规则？", "test")

        if task.task_type == "VERIFY_FIX":
            if task.reproduction is None:
                return PlanDecision(
                    action="HANDOFF", explanation="缺少经过独立复核的复现方法"
                )
            original = task.reproduction.model_dump()
            if not complete:
                return plan("原异常版本在原条件下是否仍能复现？", "control", **original)
            if summary(complete[0])["anomalies"] == 0:
                return PlanDecision(
                    action="HANDOFF", explanation="原异常版本也未复现，先核查环境"
                )
            if len(complete) == 1:
                return plan(
                    "修复版是否改善原问题，有没有时延副作用？",
                    "fix",
                    **{**original, "version": task.conditions.version},
                )
            if len(complete) == 2:
                return plan(
                    "修复版在更多连续切换时是否仍正常？",
                    "fix",
                    **{
                        **original,
                        "version": task.conditions.version,
                        "handovers": min(original["handovers"] + 1, 10),
                    },
                )
            return PlanDecision(
                action="REPORT", explanation="原问题对照与相关补测已完成"
            )

        failures = [e for e in complete if summary(e)["anomalies"] > 0]
        if not failures:
            candidates = [
                ("普通负载连续切换能否复现？", {"load": "normal", "handovers": 3}),
                ("高负载单次切换能否复现？", {"load": "high", "handovers": 1}),
                ("高负载连续切换能否复现？", {"load": "high", "handovers": 3}),
            ]
            seen = [e.plan.conditions for e in complete]
            for question, changes in candidates:
                candidate = plan(question, "screen", **changes)
                if candidate.plan.conditions not in seen:
                    return candidate
            return PlanDecision(
                action="HANDOFF", explanation="已测组合未复现，需要补充现场条件"
            )
        first = failures[0].plan.conditions.model_dump()
        controls = [e for e in complete if e.plan.kind == "control"]
        if not controls and "version" not in task.fixed_fields:
            return plan(
                "相同条件下，旧版本是否也出现异常？",
                "control",
                **{**first, "version": "old"},
            )
        if controls and summary(controls[-1])["anomalies"]:
            return PlanDecision(
                action="HANDOFF", explanation="旧版本也异常，现有路线不能只归因于升级"
            )
        minimized = [e for e in complete if e.plan.kind == "minimize"]
        if not minimized:
            return plan(
                "减少连续切换次数，异常还在吗？",
                "minimize",
                **{**first, "handovers": max(first["handovers"] - 1, 1)},
            )
        if len(minimized) == 1:
            previous = (
                minimized[0] if summary(minimized[0])["anomalies"] else failures[0]
            )
            return plan(
                "去掉背景业务，是否仍复现同类异常？",
                "minimize",
                **{
                    **previous.plan.conditions.model_dump(),
                    "background_traffic": False,
                },
            )
        confirmed = [e for e in complete if e.plan.kind == "confirm"]
        if not confirmed:
            candidate = next(
                (e for e in reversed(minimized) if summary(e)["anomalies"]), failures[0]
            )
            return plan(
                "用新的运行独立复核简化方法",
                "confirm",
                **candidate.plan.conditions.model_dump(),
            )
        return PlanDecision(
            action="REPORT", explanation="已完成复现和简化方法的独立复核"
        )


class ModelPlanner:
    """真实模型只提出结构化候选，TaskService 仍校验版本、规则和预算。"""

    def __init__(self, model):
        self.model = model.with_structured_output(
            PlanDecision, method="function_calling"
        )

    def choose(self, task: TaskRecord) -> PlanDecision:
        visible = build_planning_context(task)
        return self.choose_with_context(visible)

    def choose_with_context(self, visible: dict) -> PlanDecision:
        """接收已冻结投影；执行记录与训练样本保留同一份输入。"""
        return PlanDecision.model_validate(
            self.model.invoke(
                "你负责在沙盒中安排测试和调查。每次只选择一组实验或报告/交接/等待。"
                "等待外部证据时用 WAIT 并说明原因，不反复安排无价值实验。"
                "使用实际结果，考虑反证。条件遵守用户固定字段；不猜真实设备。"
                "版本仅 old/new/fixed/slow_fix，load 为 normal/high。"
                "repeats 通常为3，有限次数不能保证任何概率。"
                "复现需 confirm 类型的独立复核；简化前后使用同类异常。"
                "修复验证先在 reproduction 条件复测原异常版本，再测指定修复版本，"
                "补查相关时延与连续动作。无法区分解释时交接，不声称确认代码原因。"
                f"\n当前记录：{visible}"
            )
        )


def build_planning_context(task: TaskRecord) -> dict:
    """在线规划和训练导出共用输入投影，防止训练时塞入未来答案。"""
    from rag import retrieve
    from skill_library import SkillLibrary

    records = task.current_experiments() + task.reference_experiments
    skills = SkillLibrary()
    return {
        "query": task.query,
        "task_type": task.task_type,
        "goal": task.goal,
        "conditions": task.conditions.model_dump(),
        "fixed_fields": task.fixed_fields,
        "remaining_runs": task.max_runs - task.reserved_runs,
        **(
            {"reproduction": task.reproduction.model_dump()}
            if task.reproduction
            else {}
        ),
        "previous_experiments": [summary(e) for e in records],
        "log_events": [
            {"evidence_id": evidence["evidence_id"], "events": evidence["events"]}
            for experiment in records
            for run in experiment.runs
            for evidence in run.evidence
            if evidence.get("events")
        ],
        "hypotheses": task.hypotheses,
        # 本地检索内容是教学资料，不把其中示例门限升级为正式判定规则。
        "knowledge": [
            {
                "document_id": hit.doc.doc_id,
                "excerpt": hit.doc.content,
                "source_type": "teaching_fixture",
            }
            for hit in retrieve(task.query)[:3]
        ],
        "skill_catalog": skills.catalog(),
        # 首版按业务范围选择一个方法，后续可改为模型目录选择。
        "active_skill": skills.load("mobility-investigation"),
    }
