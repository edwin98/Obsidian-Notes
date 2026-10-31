"""【学习定位：统一数据字典】
Conditions是完整条件，ConditionPatch是增量；Task关联Experiment、Run和Evidence。PlanDecision只是候选。
对应01、03、05、14；编号和执行事实由程序生成。

统一业务记录：任务、实验和实际运行分别保存，不以聊天代替执行事实。
"""

from datetime import UTC, datetime
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator


def identifier(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def timestamp() -> str:
    return datetime.now(UTC).isoformat()


TaskType = Literal["TEST", "INVESTIGATE", "VERIFY_FIX"]
Action = Literal["CREATE", "UPDATE", "QUERY", "PAUSE", "RESUME", "CANCEL"]
Goal = Literal["TEST", "REPRODUCE", "NARROW", "CHECK_CAUSE", "VERIFY_FIX"]
Version = Literal["old", "new", "fixed", "slow_fix"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Conditions(Record):
    """完整条件：可以执行的状态；模型补丁不能用默认值重建它。"""

    env: Literal["sandbox"] = "sandbox"
    source_cell: str = Field(default="A", min_length=1, max_length=40)
    target_cell: str = Field(default="B", min_length=1, max_length=40)
    version: Version = "new"
    load: Literal["normal", "high"] = "normal"
    handovers: int = Field(default=1, ge=1, le=10)
    background_traffic: bool = True
    band: str = Field(default="n78", pattern=r"^n\d{1,3}$")
    bandwidth_mhz: int = Field(default=100, ge=5, le=100)
    speed_kmh: int = Field(default=120, ge=0, le=350)


class ConditionPatch(Record):
    """增量修改：None 表示本条消息没有修改该字段，而不是清空原条件。"""

    source_cell: str | None = Field(default=None, min_length=1, max_length=40)
    target_cell: str | None = Field(default=None, min_length=1, max_length=40)
    version: Version | None = None
    load: Literal["normal", "high"] | None = None
    handovers: int | None = Field(default=None, ge=1, le=10)
    background_traffic: bool | None = None
    band: str | None = Field(default=None, pattern=r"^n\d{1,3}$")
    bandwidth_mhz: int | None = Field(default=None, ge=5, le=100)
    speed_kmh: int | None = Field(default=None, ge=0, le=350)


class IntentDecision(Record):
    actions: list[Action] = Field(default_factory=lambda: ["CREATE"], min_length=1)
    task_type: TaskType | None = None
    goal: Goal | None = None
    patch: ConditionPatch = Field(default_factory=ConditionPatch)
    question: str = ""
    reason: str = ""
    read_only: bool | None = None
    investigate_on_failure: bool | None = None
    max_runs: int | None = Field(default=None, ge=1, le=200)
    unsupported_environment: bool = False


class MessageRequest(Record):
    message_id: str = Field(min_length=1, max_length=100)
    conversation_id: str = Field(default="local", min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=8000)
    target_task_id: str | None = None
    expected_task_version: int | None = Field(default=None, ge=1)


class ExperimentPlan(Record):
    """模型提出的单实验臂。编号、观测与判定均由程序另行登记。"""

    question: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    conditions: Conditions
    repeats: int = Field(default=3, ge=1, le=100)
    kind: Literal["test", "screen", "control", "minimize", "confirm", "fix"]
    comparison_id: str | None = None
    rule_id: str = "SANDBOX-BUSINESS-1"


class PlanDecision(Record):
    """统一计划器输出。WAIT 结束本轮调度，不代表整个任务完成。"""

    action: Literal["EXPERIMENT", "REPORT", "HANDOFF", "WAIT"]
    plan: ExperimentPlan | None = None
    explanation: str = ""

    @model_validator(mode="after")
    def validate_action(self):
        # schema 合法仍不够；这里仅保证动作与载荷一致。
        # 对象、规则、预算和版本校验在 TaskService.commit_step 中完成。
        if (self.action == "EXPERIMENT") != (self.plan is not None):
            raise ValueError("只有 EXPERIMENT 必须且可以携带实验计划")
        return self


class PreparedStep(Record):
    """计划阶段的轻量输出：不复制所有历史，提交时重新读取数据库。"""

    task_id: str
    task_version: int
    plan_version: int
    continuation_id: str | None = None
    decision: PlanDecision
    context_snapshot_id: str
    context: dict


class ExecutionBatch(Record):
    """执行阶段只接收已登记引用，不能拿一段自然语言直接操作平台。"""

    task_id: str
    experiment_id: str
    plan_version: int


class RunRecord(Record):
    """四种信息分开：执行状态、观测有效性、产品判定和环境恢复。"""

    run_id: str
    session_id: str | None = None
    conditions: Conditions
    status: str = "RUNNING"
    validity: str = "INCOMPLETE"
    metrics: dict = Field(default_factory=dict)
    verdict: str = "INCONCLUSIVE"
    evidence: list[dict] = Field(default_factory=list)
    cleanup_status: str = "pending"
    error: str = ""
    started_at: str = Field(default_factory=timestamp)
    finished_at: str | None = None


class ExperimentRecord(Record):
    """一组为回答同一个问题安排的运行；属于确定的计划版本。"""

    experiment_id: str = Field(default_factory=lambda: identifier("exp"))
    plan_version: int
    plan: ExperimentPlan
    status: str = "RUNNING"
    runs: list[RunRecord] = Field(default_factory=list)
    interpretation: str = ""


class TaskRecord(Record):
    """业务事实的聚合根，聊天消息、图状态和模型摘要不能替代它。"""

    task_id: str = Field(default_factory=lambda: identifier("task"))
    conversation_id: str
    issue_id: str | None = None
    related_task_id: str | None = None
    task_type: TaskType
    goal: Goal
    query: str
    conditions: Conditions = Field(default_factory=Conditions)
    fixed_fields: list[str] = Field(default_factory=list)
    status: str = "READY"
    task_version: int = 1  # 用户修改与控制的并发版本，挡住迟到的模型计划。
    plan_version: int = 1  # 实验条件版本，决定哪些历史结果仍属于当前计划。
    read_only: bool = False
    investigate_on_failure: bool = False
    max_runs: int = 36
    max_experiments: int = 12
    # 累计额度占用：已使用 + 尚未结算的预留次数。
    # 已开始的运行不可因异常/暂停而免费重跑，未开始的预留才在收尾时释放。
    reserved_runs: int = 0
    experiments: list[ExperimentRecord] = Field(default_factory=list)
    reference_experiments: list[ExperimentRecord] = Field(default_factory=list)
    hypotheses: list[dict] = Field(default_factory=list)
    reproduction: Conditions | None = None
    report: dict = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)
    messages: list[dict] = Field(default_factory=list)
    question: str = ""
    stop_reason: str = ""
    created_at: str = Field(default_factory=timestamp)

    def event(self, name: str, **details) -> None:
        """追加审计事件；序号用于本地顺序，事件 ID 用于跨层关联。"""
        self.events.append(
            {
                "event_id": identifier("evt"),
                "task_id": self.task_id,
                "plan_version": self.plan_version,
                "sequence": len(self.events),
                "time": timestamp(),
                "event": name,
                **details,
            }
        )

    def current_experiments(self) -> list[ExperimentRecord]:
        """旧结果仍保留，但不能自动参与新条件的结论。"""
        return [e for e in self.experiments if e.plan_version == self.plan_version]
