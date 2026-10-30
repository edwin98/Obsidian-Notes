"""【学习定位：新版业务协调核心】
message处理用户操作；prepare_step锁外规划；commit_step事务校验与登记；execute_step固定执行。
对应14；跟踪task_version、plan_version、run_id即可理解并发与迟到结果。

统一任务入口与可中断实验循环；模型不能直接写入执行事实。
"""

import logging

from experiments import OfflinePlanner, SandboxExecutor, build_planning_context, summary
from intent import OfflineIntent
from task_models import (
    Conditions,
    ExperimentRecord,
    ExecutionBatch,
    PreparedStep,
    MessageRequest,
    PlanDecision,
    RunRecord,
    TaskRecord,
    identifier,
    timestamp,
)
from task_store import ConflictError, TaskStore

logger = logging.getLogger(__name__)


class TaskService:
    """把意图、存储、规划和执行串起来；入口层不直接操作这四种依赖。

    可注入规则替身或在线模型，仍共用同一套业务校验。锁只保护本进程的
    读写临界区；跨进程执行与真实设备幂等需要另接平台能力。
    """

    def __init__(self, *, store=None, interpreter=None, planner=None, executor=None):
        """默认依赖全为本地教学实现，构造服务本身不会调用模型或启动实验。"""
        self.store = store or TaskStore()
        self.interpreter = interpreter or OfflineIntent()
        self.planner = planner or OfflinePlanner()
        self.executor = executor or SandboxExecutor()

    def get(self, task_id, conversation_id=None):
        """读取独立副本；可选对话编号用于校验归属，不等于生产身份认证。"""
        with self.store.lock:
            task = self.store.get(task_id)
            if conversation_id is not None and task.conversation_id != conversation_id:
                raise KeyError(task_id)
            return task

    def _target(self, request):
        """优先使用明确编号；多个活跃任务时拒绝猜测用户要控制哪一个。"""
        if request.target_task_id:
            return self.get(request.target_task_id, request.conversation_id)
        tasks = self.store.list(request.conversation_id)
        active = [
            t for t in tasks if t.status not in {"COMPLETED", "CANCELLED", "HANDED_OFF"}
        ]
        if len(active) > 1:
            return None
        return active[0] if active else tasks[-1] if tasks else None

    def message(self, request: MessageRequest) -> dict:
        """解释一条消息并原子保存业务变化与去重回应，不在此执行实验。

        返回的 run_requested 只是调度信号。CLI 或 HTTP 后台驱动器随后
        调用 step；查询、澄清和重复投递都不能顺带产生工具副作用。
        """
        # 先读再解释，解释期间不持锁；写入前再次检查版本和去重，处理用户中途修改的竞态。
        if not request.text.strip():
            raise ValueError("消息不能为空白")
        raw = request.model_dump()
        with self.store.lock:
            duplicate = self.store.replay(raw)
            if duplicate:
                return duplicate
            target = self._target(request)
        decision = self.interpreter.interpret(request.text, target)
        with self.store.lock, self.store.connection:
            duplicate = self.store.replay(raw)
            if duplicate:
                return duplicate
            latest = self._target(request)
            if target and (not latest or latest.task_version != target.task_version):
                raise ConflictError("解释消息期间任务已变化，请按最新状态重新提交")
            if (
                latest
                and request.expected_task_version is not None
                and latest.task_version != request.expected_task_version
            ):
                raise ConflictError("任务版本已变化")
            controls = {"QUERY", "PAUSE", "RESUME", "CANCEL"}
            if decision.actions in (["QUERY", "CREATE"], ["QUERY", "UPDATE"]):
                if latest is None:
                    response = self._clarify(
                        "请指定原任务，才能先读取报告再安排后续工作"
                    )
                else:
                    old_view = latest.model_copy(deep=True)
                    if not old_view.report:
                        self._report(old_view)
                    followup = decision.model_copy(
                        update={"actions": decision.actions[1:]}
                    )
                    response = self._apply(request, followup, latest)
                    response["previous_report"] = old_view.report
            elif decision.actions[0] in controls:
                response = self._control(latest, decision.actions, request.text)
            else:
                if decision.actions not in (["CREATE"], ["UPDATE"]):
                    raise ValueError("新建或修改请求不能混入其他执行动作")
                if decision.actions == ["UPDATE"] and latest is None:
                    response = self._clarify("请指定要修改的任务")
                else:
                    response = self._apply(request, decision, latest)
            self.store.remember(raw, response)
            return response

    @staticmethod
    def _clarify(question):
        """信息不足时只返回问题；此时没有创建可执行的任务。"""
        return {
            "task_id": None,
            "status": "WAITING_INFO",
            "message": question,
            "run_requested": False,
        }

    def _control(self, task, actions, text=""):
        """查询只读，控制修改版本；暂停和取消在下一运行边界落实。

        已启动的动作仍需保存真实结果并收尾，不能把 CANCELLING 当成
        远端动作已经停止。RESUME 还要检查此前每次运行的恢复状态。
        """
        if actions == ["QUERY"] and any(
            w in text for w in ("是什么意思", "怎么理解", "解释一下")
        ):
            from rag import retrieve

            docs = retrieve(text)
            snippets = "\n".join(
                f"{doc.doc.doc_id}: {doc.doc.content[:500]}" for doc in docs
            )
            return {
                "task_id": task.task_id if task else None,
                "status": "READ_ONLY",
                "message": "本地资料节选（未执行测试）：\n" + snippets
                if docs
                else "本地资料没有找到适用说明，未执行测试。",
                "references": [doc.doc.doc_id for doc in docs],
                "run_requested": False,
            }
        if task is None:
            return self._clarify("没有唯一的目标任务，请提供任务编号")
        changed = False
        for action in actions:
            if action == "QUERY":
                continue
            if action == "PAUSE":
                if task.status in {"READY", "RUNNING", "PAUSING", "PAUSED"}:
                    task.status = "PAUSING" if task.status == "RUNNING" else "PAUSED"
                    changed = True
            elif action == "CANCEL":
                task.status = (
                    "CANCELLING"
                    if task.status in {"RUNNING", "PAUSING"}
                    else "CANCELLED"
                )
                changed = True
            elif action == "RESUME":
                if task.status not in {"PAUSED", "WAITING_EVIDENCE"}:
                    raise ConflictError(
                        "只有暂停或等待证据的任务可以继续；缺信息或恢复失败需先解决"
                    )
                if any(
                    r.cleanup_status != "completed"
                    for e in task.experiments
                    for r in e.runs
                ):
                    raise ConflictError("环境恢复尚未确认，不能继续")
                task.status = "READY"
                task.stop_reason = ""
                changed = True
            else:
                raise ValueError("控制消息包含不允许的动作")
        if changed:
            task.task_version += 1
            task.event("control", actions=actions, status=task.status)
            self.store.save(task)
        response = self._response(
            task, "已读取当前记录" if not changed else "已处理任务要求"
        )
        response["run_requested"] = "RESUME" in actions and task.status == "READY"
        return response

    def _apply(self, request, decision, previous):
        """将候选意图落实为完整条件与任务状态，而不是直接采用模型文本。

        活跃任务按字段补丁更新；结束后的修复等请求建立关联任务。
        条件版本变化会清空当前推断，保留旧实验作为可追溯历史。
        """
        # 未结束任务使用字段补丁；已完成任务创建关联后续记录，原报告不能被静默重开或覆盖。
        updating = decision.actions == ["UPDATE"]
        if updating and previous.status in {"COMPLETED", "CANCELLED", "HANDED_OFF"}:
            updating = False  # 已结束记录不重新打开；创建可追溯的后续任务。
        if updating:
            if previous.status == "ERROR" and any(
                r.cleanup_status != "completed"
                for e in previous.experiments
                for r in e.runs
            ):
                raise ConflictError("原环境尚未确认恢复，修改条件不能绕过恢复检查")
            task = previous.model_copy(deep=True)
            task.task_version += 1
            task.plan_version += 1
            task.reproduction = None
            task.reference_experiments = []
            task.report = {}
            task.hypotheses = []
        else:
            kind = decision.task_type or "TEST"
            task = TaskRecord(
                conversation_id=request.conversation_id,
                task_type=kind,
                goal=decision.goal
                or {
                    "TEST": "TEST",
                    "INVESTIGATE": "REPRODUCE",
                    "VERIFY_FIX": "VERIFY_FIX",
                }[kind],
                query=request.text,
            )
            if previous and (kind == "VERIFY_FIX" or decision.actions == ["UPDATE"]):
                task.related_task_id = previous.task_id
                task.issue_id = previous.issue_id
                task.conditions = previous.conditions.model_copy(deep=True)
                if kind == "VERIFY_FIX":
                    task.reproduction = previous.reproduction
        if decision.task_type:
            task.task_type = decision.task_type
        if task.task_type == "VERIFY_FIX" and previous:
            task.reproduction = previous.reproduction
        if (
            previous
            and task.goal == "CHECK_CAUSE"
            and previous.reproduction
            and not decision.patch.model_dump(exclude_none=True)
        ):
            task.reproduction = previous.reproduction
            task.reference_experiments = [
                e.model_copy(deep=True) for e in previous.current_experiments()
            ]
        if decision.goal:
            task.goal = decision.goal
        allowed_goals = {
            "TEST": {"TEST"},
            "INVESTIGATE": {"REPRODUCE", "NARROW", "CHECK_CAUSE"},
            "VERIFY_FIX": {"VERIFY_FIX"},
        }
        if task.goal not in allowed_goals[task.task_type]:
            raise ValueError("任务类型与目标不一致")
        patch = decision.patch.model_dump(exclude_none=True)
        task.conditions = Conditions.model_validate(
            {**task.conditions.model_dump(), **patch}
        )
        task.fixed_fields = sorted(set(task.fixed_fields) | (set(patch) - {"version"}))
        if decision.read_only is not None:
            task.read_only = decision.read_only
        if any(
            word in request.text
            for word in ("只读", "只看日志", "不要运行", "不跑测试")
        ):
            task.read_only = True
        if decision.investigate_on_failure is not None:
            task.investigate_on_failure = decision.investigate_on_failure
        if decision.max_runs is not None:
            if decision.max_runs < task.reserved_runs:
                raise ValueError("新次数上限不能小于已使用次数")
            task.max_runs = decision.max_runs
        if task.task_type != "TEST" and task.issue_id is None:
            task.issue_id = identifier("issue")
        task.question = decision.question
        task.status = "READY"
        if task.task_type == "VERIFY_FIX" and task.reproduction is None:
            task.question = "请先提供已完成复现的原任务；当前不能证明修复有效"
        if task.task_type == "VERIFY_FIX" and task.reproduction:
            original = task.reproduction.model_dump()
            if any(
                key != "version" and value != original[key]
                for key, value in patch.items()
            ):
                task.question = (
                    "本次条件与原复现不同，请先在原条件验证，或建立新的复现任务"
                )
        if task.question:
            task.status = "WAITING_INFO"
        if decision.unsupported_environment:
            task.status = "WAITING_APPROVAL"
            task.stop_reason = (
                "当前仅支持 sandbox；现场和生产环境尚未接入，不能批准后假装执行"
            )
        elif task.read_only and not task.question:
            task.status = "COMPLETED"
            task.report = {
                "summary": "只读分析，未执行新测试。需要实际运行才能确认复现和原因。",
                "cause_status": "UNCONFIRMED",
            }
        task.messages.append(
            {
                "text": request.text,
                "decision": decision.model_dump(),
                "time": timestamp(),
            }
        )
        task.event(
            "intent", actions=decision.actions, task_type=task.task_type, goal=task.goal
        )
        self._update_evidence(task)
        self.store.save(task)
        return self._response(
            task, task.question or task.stop_reason or "已理解要求，可开始下一组实验"
        )

    @staticmethod
    def _response(task, message):
        """统一交互回应；READY 表示允许调度，尚不代表一次运行已发生。"""
        return {
            "task_id": task.task_id,
            "status": task.status,
            "message": message,
            "task": task.model_dump(),
            "run_requested": task.status == "READY",
            "duplicate": False,
        }

    def _validate_plan(self, task, plan, continuation=None):
        """在真实状态上校验候选计划；失败时不登记实验、不调用执行器。

        新组预留全部次数；续跑只预留未完成次数。结构 schema 之外，
        还要核对固定条件、对照归属、修复范围、重复目的和任务预算。
        """
        # 结构合法不代表可执行；逐项验证对象、固定条件、对照来源、规则与预算。
        if plan is None:
            raise ValueError("实验计划缺失")
        if plan.rule_id != "SANDBOX-BUSINESS-1":
            raise ValueError("沙盒未登记该判定规则")
        base, proposed = task.conditions.model_dump(), plan.conditions.model_dump()
        stable = {
            "env",
            "source_cell",
            "target_cell",
            "band",
            "bandwidth_mhz",
            "speed_kmh",
        }
        if any(base[key] != proposed[key] for key in stable):
            raise ValueError("本轮不允许擅自更换设备、环境或固定无线配置")
        if any(base[key] != proposed[key] for key in task.fixed_fields):
            raise ValueError("实验改变了用户固定的条件")
        if task.task_type == "TEST" and proposed != base:
            raise ValueError("按需测试不能擅自改变测试条件")
        if task.task_type == "VERIFY_FIX" and task.reproduction:
            original = task.reproduction.model_dump()
            if plan.kind == "control" and proposed != original:
                raise ValueError("修复对照必须使用原复现条件和原异常版本")
            if plan.kind == "fix":
                expected = {**original, "version": base["version"]}
                fixed_keys = set(expected) - {"handovers"}
                if any(expected[key] != proposed[key] for key in fixed_keys):
                    raise ValueError("修复版必须保持原条件；邻近补测仅允许调整切换次数")
            if plan.kind not in {"control", "fix"}:
                raise ValueError("修复验证只接受原问题对照和修复补测")
        if plan.comparison_id and not any(
            e.experiment_id == plan.comparison_id for e in task.current_experiments()
        ):
            raise ValueError("对照实验不属于当前计划")
        remaining = (
            plan.repeats - len(continuation.runs) if continuation else plan.repeats
        )
        if task.reserved_runs + remaining > task.max_runs:
            raise ValueError("已到运行次数上限")
        if continuation is None and len(task.experiments) >= task.max_experiments:
            raise ValueError("已到实验组数上限")
        if any(
            e.plan.conditions == plan.conditions
            and e.plan.kind == plan.kind
            and (continuation is None or e.experiment_id != continuation.experiment_id)
            for e in task.current_experiments()
        ):
            raise ValueError("没有新的验证目的，不能无限重复同一组实验")

    def step(self, task_id):
        """推进一组实验。直接循环和 LangGraph 都复用同样三个阶段。"""
        prepared = self.prepare_step(task_id)
        if prepared is None:
            return self.get(task_id)
        batch = self.commit_step(prepared)
        return self.execute_step(batch) if batch else self.get(task_id)

    def prepare_step(self, task_id):
        """读取快照，在锁外调用计划器；期间用户仍可以查询或修改任务。"""
        with self.store.lock:
            task = self.store.get(task_id)
            if task.status != "READY":
                return None
            if any(e.status == "RUNNING" for e in task.experiments):
                return None
            version, plan_version = task.task_version, task.plan_version
            continuation = next(
                (
                    e
                    for e in task.current_experiments()
                    if e.status == "PARTIAL" and len(e.runs) < e.plan.repeats
                ),
                None,
            )
        try:
            context = build_planning_context(task)
            if continuation:
                decision = PlanDecision(action="EXPERIMENT", plan=continuation.plan)
            elif hasattr(self.planner, "choose_with_context"):
                decision = self.planner.choose_with_context(context)
            else:
                # 离线规则替身与已有测试计划器保持简单的choose(task)协议。
                decision = self.planner.choose(task)
            decision = PlanDecision.model_validate(decision)
        except Exception as exc:
            logger.exception("任务 %s 的实验计划生成失败", task_id)
            self._handoff(task_id, f"计划生成失败：{exc}", version)
            return None
        return PreparedStep(
            task_id=task_id,
            task_version=version,
            plan_version=plan_version,
            continuation_id=continuation.experiment_id if continuation else None,
            decision=PlanDecision.model_validate(decision),
            context_snapshot_id=identifier("ctx"),
            context=context,
        )

    def commit_step(self, prepared: PreparedStep):
        """短事务：复查版本、校验计划、预留预算并登记实验，然后才允许执行。"""
        task_id, decision = prepared.task_id, prepared.decision
        plan_version = prepared.plan_version
        with self.store.lock, self.store.connection:
            task = self.store.get(task_id)
            if (
                task.status != "READY"
                or task.task_version != prepared.task_version
                or task.plan_version != plan_version
                or any(e.status == "RUNNING" for e in task.experiments)
            ):
                return None
            continuation = next(
                (
                    e
                    for e in task.current_experiments()
                    if e.experiment_id == prepared.continuation_id
                ),
                None,
            )
            if prepared.continuation_id and continuation is None:
                return None
            if decision.action == "WAIT":
                # 把等待变成持久业务状态，run()/后台调度随即返回，不空转请求模型。
                task.status = "WAITING_EVIDENCE"
                task.stop_reason = decision.explanation or "等待补充信息或外部事件"
                task.event("waiting", reason=task.stop_reason)
                self.store.save(task)
                return None
            if decision.action != "EXPERIMENT":
                self._finish(task, decision.action == "REPORT", decision.explanation)
                self.store.save(task)
                return None
            try:
                self._validate_plan(task, decision.plan, continuation)
            except ValueError as exc:
                self._finish(task, False, str(exc))
                self.store.save(task)
                return None
            # 在登记本轮实验和预留额度之前冻结输入，不能混入本轮结果。
            planning_context = prepared.context
            if continuation:
                experiment = next(
                    e
                    for e in task.experiments
                    if e.experiment_id == continuation.experiment_id
                )
                experiment.status = "RUNNING"
            else:
                experiment = ExperimentRecord(
                    plan_version=plan_version, plan=decision.plan
                )
                task.experiments.append(experiment)
            finished_before = len(experiment.runs)
            task.reserved_runs += experiment.plan.repeats - finished_before
            task.status = "RUNNING"
            task.event(
                "experiment_resumed" if continuation else "experiment_started",
                experiment_id=experiment.experiment_id,
                context_snapshot_id=prepared.context_snapshot_id,
                plan=experiment.plan.model_dump(),
                context=planning_context,
            )
            self.store.save(task)
            return ExecutionBatch(
                task_id=task_id,
                experiment_id=experiment.experiment_id,
                plan_version=plan_version,
            )

    def execute_step(self, batch: ExecutionBatch):
        # 先落库UNKNOWN运行，再调用外部动作；进程若在两者之间退出，也有记录可供核查。
        """固定执行流程。每次动作前读最新状态，返回仍归属原计划的真实结果。"""
        task_id, plan_version = batch.task_id, batch.plan_version
        task = self.get(task_id)
        experiment = next(
            e for e in task.experiments if e.experiment_id == batch.experiment_id
        )
        finished_before = len(experiment.runs)
        completed = finished_before
        for ordinal in range(finished_before, experiment.plan.repeats):
            with self.store.lock, self.store.connection:
                current = self.store.get(task_id)
                if current.status != "RUNNING" or current.plan_version != plan_version:
                    break
                run_id = identifier("run")
                stored = next(
                    e
                    for e in current.experiments
                    if e.experiment_id == experiment.experiment_id
                )
                stored.runs.append(
                    RunRecord(
                        run_id=run_id,
                        conditions=experiment.plan.conditions,
                        status="UNKNOWN",
                        cleanup_status="unknown",
                    )
                )
                current.event(
                    "run_started", run_id=run_id, experiment_id=experiment.experiment_id
                )
                self.store.save(current)
            try:
                result = self.executor.execute(
                    experiment.plan.conditions, run_id, ordinal
                )
                if (
                    result.run_id != run_id
                    or result.conditions != experiment.plan.conditions
                ):
                    raise ValueError("工具返回的实际对象或条件与本轮计划不一致")
            except Exception as exc:
                logger.exception("任务 %s 的执行结果未知", task_id)
                return self._handoff(task_id, f"执行状态未知，需核查和恢复：{exc}")
            with self.store.lock, self.store.connection:
                current = self.store.get(task_id)
                stored = next(
                    e
                    for e in current.experiments
                    if e.experiment_id == experiment.experiment_id
                )
                stored.runs = [result if r.run_id == run_id else r for r in stored.runs]
                completed += 1
                current.event(
                    "run_completed",
                    run_id=result.run_id,
                    verdict=result.verdict,
                    cleanup_status=result.cleanup_status,
                )
                if result.cleanup_status != "completed" or result.validity != "VALID":
                    current.status = "ERROR"
                    current.stop_reason = (
                        result.error or "环境恢复失败，不能继续使用该环境"
                    )
                self.store.save(current)
        with self.store.lock, self.store.connection:
            task = self.store.get(task_id)
            stored = next(
                e
                for e in task.experiments
                if e.experiment_id == experiment.experiment_id
            )
            task.reserved_runs -= experiment.plan.repeats - completed
            stored.status = (
                "COMPLETED" if completed == stored.plan.repeats else "PARTIAL"
            )
            if task.plan_version != plan_version:
                stored.status = "SUPERSEDED"
            stored.interpretation = f"实际完成 {completed} 次；{summary(stored)['anomalies']} 次业务停顿；{summary(stored)['failures']} 次未满足要求"
            if task.status == "PAUSING":
                task.status = "PAUSED"
            elif task.status == "CANCELLING":
                task.status = "CANCELLED"
            elif task.status == "RUNNING":
                task.status = "READY"
            task.event("experiment_finished", experiment=summary(stored))
            self._update_evidence(task)
            if task.status == "ERROR":
                self._report(task)
            if (
                task.status == "READY"
                and task.task_type == "TEST"
                and (
                    any(r.verdict == "FAIL" for r in stored.runs)
                    and task.investigate_on_failure
                )
            ):
                task.task_type, task.goal = "INVESTIGATE", "REPRODUCE"
                task.issue_id = task.issue_id or identifier("issue")
                task.plan_version += 1
                task.event("investigation_started", reason="测试失败，按原要求继续复现")
            self.store.save(task)
            return task

    def _handoff(self, task_id, reason, expected_version=None):
        """保存无法继续的原因；旧版本规划失败不能覆盖用户后来修改的任务。"""
        with self.store.lock, self.store.connection:
            task = self.store.get(task_id)
            if expected_version is not None and task.task_version != expected_version:
                return task
            task.status = "HANDED_OFF"
            task.stop_reason = reason
            self._report(task)
            task.event("handoff", reason=reason)
            self.store.save(task)
            return task

    @staticmethod
    def _update_evidence(task):
        """由当前实验登记复现条件与日志线索，不凭模型文本确认原因。

        这里生成运行时摘要；评测端 task_grading 还会重新核验原始观测、
        冻结时序和独立运行，避免摘要自身成为验收真值。
        """
        # 复现由confirm运行登记，原因线索必须引用实际日志；模型主观判断不能升级为代码确认。
        current = task.current_experiments() + task.reference_experiments
        confirmed = [
            e
            for e in current
            if e.plan.kind == "confirm"
            and e.status == "COMPLETED"
            and summary(e)["valid_runs"] >= 2
            and summary(e)["anomalies"]
        ]
        if confirmed:
            task.reproduction = confirmed[-1].plan.conditions.model_copy(deep=True)
        if task.goal == "CHECK_CAUSE" and task.reproduction:
            records = [
                v
                for e in current
                for r in e.runs
                for v in r.evidence
                if v.get("events")
            ]
            task.hypotheses = [
                {
                    "description": "连续切换后的资源处理异常",
                    "status": "SUPPORTED" if records else "PROPOSED",
                    "evidence_ids": [v["evidence_id"] for v in records],
                    "limitation": "相关日志与条件对照支持线索，未检查代码；不确认具体缺陷",
                }
            ]

    def _finish(self, task, success, reason):
        """按目标核对结束条件；产品 FAIL 可以是成功完成的复现任务。

        计划器说 REPORT 不能替代目标检查。缺少独立复现或原因干预
        能力时交接，并在报告中保留尚未完成的部分。
        """
        current = [e for e in task.current_experiments() if e.status == "COMPLETED"]
        if task.task_type == "INVESTIGATE" and not task.reproduction:
            success = False
            reason = "尚未得到经过独立复核的复现方法；" + reason
        if task.goal == "CHECK_CAUSE":
            success = False
            reason = (
                "复现与日志线索已整理；缺少内部状态或干预能力，尚不能确认具体原因。"
                + reason
            )
        if task.task_type == "TEST" and not current:
            success = False
        if task.task_type == "VERIFY_FIX":
            success = (
                success
                and any(
                    e.plan.kind == "control" and summary(e)["anomalies"]
                    for e in current
                )
                and any(e.plan.kind == "fix" for e in current)
            )
        task.status = "COMPLETED" if success else "HANDED_OFF"
        task.stop_reason = reason
        self._report(task)
        task.event("finished", status=task.status, reason=reason)

    @staticmethod
    def _report(task):
        """把已有记录投影为报告，不调用模型、不补测、不创建新证据。

        修复任务的产品判定只汇总修复组；原异常组用于对照。全部运行
        仍参与恢复状态汇总，避免旧组的未完成收尾被当前 PASS 掩盖。
        """
        # 这里是事实到视图的投影：任务完成、产品失败、原因未确认与恢复完成可以同时出现。
        current = task.current_experiments()
        selected = [
            e for e in current if task.task_type != "VERIFY_FIX" or e.plan.kind == "fix"
        ]
        runs = [r for e in selected for r in e.runs if r.validity == "VALID"]
        verdict = (
            "FAIL"
            if any(r.verdict == "FAIL" for r in runs)
            else "PASS"
            if runs
            else "INCONCLUSIVE"
        )
        all_runs = [r for e in task.experiments for r in e.runs]
        reference_runs = [r for e in task.reference_experiments for r in e.runs]
        task.report = {
            "task_result": task.status,
            "test_verdict": verdict,
            "investigation_progress": "已独立复现"
            if task.reproduction
            else "尚无独立复现",
            "cause_status": task.hypotheses[0]["status"]
            if task.hypotheses
            else "UNCONFIRMED",
            "cleanup_status": "unknown"
            if any(r.cleanup_status == "unknown" for r in all_runs)
            else "failed"
            if any(r.cleanup_status != "completed" for r in all_runs)
            else "completed"
            if all_runs
            else "not_needed",
            "summary": task.stop_reason,
            "reproduction": task.reproduction.model_dump()
            if task.reproduction
            else None,
            "experiments": [summary(e) for e in task.experiments],
            "reference_task_id": task.related_task_id,
            "referenced_experiments": [summary(e) for e in task.reference_experiments],
            "evidence_ids": [
                v["evidence_id"] for r in all_runs + reference_runs for v in r.evidence
            ],
            "limitations": [
                "确定性沙盒演示，不代表真实设备或真实模型效果",
                "未进行代码级原因确认",
                "未观察到异常不代表其他条件不会发生",
            ],
        }

    def run(self, task_id):
        """连续推进到本轮需要返回；它不是常驻轮询器，也不等待外部输入。"""
        # READY只表示可以安排下一轮；WAITING/PAUSED等状态应立即返回，不能等待时持续调用模型。
        while True:
            task = self.get(task_id)
            if task.status != "READY":
                return task
            result = self.step(task_id)
            if result.status == "READY" and any(
                e.status == "RUNNING" for e in result.experiments
            ):
                return result

    def recover(self):
        """新执行器不能查询旧进程沙盒，明确交接，不重放未知动作。"""
        with self.store.lock, self.store.connection:
            for task in self.store.list():
                if task.status in {"RUNNING", "PAUSING", "CANCELLING"}:
                    task.status = "HANDED_OFF"
                    task.stop_reason = (
                        "上次进程在执行中退出；需核查实际任务和环境后继续，未自动重放"
                    )
                    self._report(task)
                    self.store.save(task)
