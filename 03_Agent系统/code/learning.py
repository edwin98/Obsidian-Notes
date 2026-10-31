"""【学习定位：运行记录到专家数据】
只导出明确审核的记录；实验前快照成为输入，PlanDecision成为目标，SFT与DPO按同一原问题分组。
对应08；导出文件不更新模型参数。

离线学习飞轮的最小数据入口：只有专家明确确认的修正才能进入训练集。

这里只导出 SFT/DPO 数据，不启动昂贵训练，不把模型自己的结论当专家标签。
输出格式可交给现有 training/ 脚本进一步清洗、划分训练/验证集并转换。
"""

import json
from dataclasses import dataclass
from pathlib import Path

from task_models import ExperimentPlan, PlanDecision, TaskRecord


@dataclass(frozen=True)
class ExpertReview:
    task_id: str
    query: str
    original_answer: str
    corrected_answer: str
    reviewer: str
    confirmed: bool = False
    source_group: str = ""
    sample_type: str = "report"
    target_schema: str = "report"
    context_snapshot_id: str = ""


def export_reviews(
    reviews: list[ExpertReview],
    output_dir: Path,
    *,
    system_prompt="你是 5G 测试验证助手，必须依据工具证据给出结论。",
):
    """导出已确认样本；按 task_id 去重，冲突标注报错，空字段拒绝。

    SFT 学习正确答案；DPO 同一问题下比较 chosen 与 rejected。
    两者相同时只输出 SFT。结果文件用 x 模式创建，避免覆盖已有训练资产。
    """
    seen = {}
    for review in reviews:
        if type(review.confirmed) is not bool:
            raise ValueError("confirmed 必须是明确的布尔值")
        if not review.confirmed:
            continue
        if any(
            not v.strip()
            for v in (
                review.task_id,
                review.query,
                review.original_answer,
                review.corrected_answer,
                review.reviewer,
            )
        ):
            raise ValueError("已确认反馈的所有文本字段必须非空")
        if review.task_id in seen and seen[review.task_id] != review:
            raise ValueError(f"任务 {review.task_id} 存在冲突反馈")
        seen[review.task_id] = review
    sft, dpo = [], []
    for review in seen.values():
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": review.query},
        ]
        chosen = [{"role": "assistant", "content": review.corrected_answer}]
        metadata = {
            "task_id": review.task_id,
            "reviewer": review.reviewer,
            "source_group": review.source_group or review.task_id,
            "sample_type": review.sample_type,
            "target_schema": review.target_schema,
            "context_snapshot_id": review.context_snapshot_id,
        }
        sft.append({"messages": prompt + chosen, "metadata": metadata})
        if review.original_answer != review.corrected_answer:
            dpo.append(
                {
                    # 保留角色边界，由目标模型的模板序列化。
                    "prompt": prompt,
                    "chosen": review.corrected_answer,
                    "rejected": review.original_answer,
                    "metadata": metadata,
                }
            )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / "sft.jsonl", output_dir / "dpo.jsonl"]
    if any(path.exists() for path in paths):
        raise FileExistsError("输出目录已有训练数据，请使用新目录")
    for path, rows in zip(paths, [sft, dpo]):
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"sft_count": len(sft), "dpo_count": len(dpo), "paths": paths}


@dataclass(frozen=True)
class DecisionReview:
    experiment_id: str
    reviewer: str
    corrected_plan: dict
    confirmed: bool = False


def export_decision_reviews(
    task: TaskRecord, reviews: list[DecisionReview], output_dir: Path
):
    # 读取冻结的决策前输入，绝不从最终原因反向补写上下文；专家修改是目标，不是新的测量值。
    """导出专家审核的实验决策；只使用执行前快照，不泄漏本轮或未来结果。"""
    events = {
        e["experiment_id"]: e for e in task.events if e["event"] == "experiment_started"
    }
    converted = []
    for review in reviews:
        if not review.confirmed:
            continue
        event = events.get(review.experiment_id)
        if event is None or "context" not in event:
            raise ValueError("缺少该实验的执行前快照，不能用最终报告补写训练输入")

        # 部署计划器返回 PlanDecision，训练也输出同样的包装，避免只学会内层计划。
        def decision(value):
            if "action" in value:
                return PlanDecision.model_validate(value)
            plan = ExperimentPlan.model_validate(value)
            return PlanDecision(action="EXPERIMENT", plan=plan, explanation=plan.reason)

        chosen = decision(review.corrected_plan)
        rejected = decision(event["plan"])
        converted.append(
            ExpertReview(
                task_id=f"{task.task_id}:{review.experiment_id}",
                query=json.dumps(event["context"], ensure_ascii=False, sort_keys=True),
                original_answer=rejected.model_dump_json(),
                corrected_answer=chosen.model_dump_json(),
                reviewer=review.reviewer,
                confirmed=review.confirmed,
                source_group=task.issue_id or task.task_id,
                sample_type="experiment_decision",
                target_schema="PlanDecision",
                context_snapshot_id=event.get(
                    "context_snapshot_id", event.get("event_id", "")
                ),
            )
        )
    return export_reviews(
        converted,
        output_dir,
        system_prompt="你负责安排下一组沙盒实验。只根据当前信息决策，保持用户条件，不伪造结果。",
    )
