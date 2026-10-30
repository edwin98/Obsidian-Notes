"""完整 episode 评测：公开任务进入 Agent，私有验收条件留在评测端。

对应文档 07。区分 regression_passed（程序符合预期）和 strict_success（目标达成）。
监测缺失场景正确停止，可以通过回归，却不能算成功复现。
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from experiments import SandboxExecutor
from task_grading import grade_task
from task_models import MessageRequest, identifier
from task_service import TaskService


@dataclass(frozen=True)
class EvaluationCase:
    name: str
    request: str  # 唯一交给 Agent 的任务描述。
    expected_status: str  # 以下字段均由评测器使用，不能拼进模型上下文。
    expected_verdict: str | None = None
    environment_options: dict = field(default_factory=dict)
    followup: str | None = None
    source_group: str = "sandbox-mobility-family"


def run_task_evaluation(*, repetitions=1, include_traces=False):
    """每个 episode 使用独立存储和执行器，避免前一题残留影响后一题。"""
    if repetitions < 1:
        raise ValueError("重复次数必须大于零")
    cases = [
        EvaluationCase("按需测试", "验证新版本切换", "COMPLETED", "PASS"),
        EvaluationCase("产品失败", "验证新版本高负载连续2次切换", "COMPLETED", "FAIL"),
        EvaluationCase("异常复现", "复现升级后移动卡顿", "COMPLETED", "FAIL"),
        EvaluationCase("次数上限", "复现卡顿，最多5次", "HANDED_OFF"),
        EvaluationCase(
            "监测缺失", "复现卡顿", "ERROR", "INCONCLUSIVE", {"missing_monitor": True}
        ),
        EvaluationCase(
            "恢复失败", "复现卡顿", "ERROR", environment_options={"fail_cleanup": True}
        ),
        EvaluationCase(
            "修复验证",
            "复现视频卡顿",
            "COMPLETED",
            "PASS",
            followup="这是修复版，按上次方法验证",
        ),
        EvaluationCase(
            "修复副作用",
            "复现视频卡顿",
            "COMPLETED",
            "FAIL",
            followup="这是修复变慢版，按上次方法验证",
        ),
    ]
    results = []
    for case in cases:
        for repeat in range(repetitions):
            service = TaskService(executor=SandboxExecutor(**case.environment_options))
            try:
                response = service.message(
                    MessageRequest(message_id=identifier("msg"), text=case.request)
                )
                task = service.run(response["task_id"])
                if case.followup:
                    response = service.message(
                        MessageRequest(
                            message_id=identifier("msg"),
                            text=case.followup,
                            target_task_id=task.task_id,
                        )
                    )
                    task = service.run(response["task_id"])
                grading = grade_task(task)
                passed = task.status == case.expected_status and (
                    case.expected_verdict is None
                    or task.report.get("test_verdict") == case.expected_verdict
                )
                record = {
                    "episode_id": identifier("episode"),
                    "source_group": case.source_group,
                    "repeat": repeat,
                    "name": case.name,
                    "status": task.status,
                    "verdict": task.report.get("test_verdict"),
                    "passed": passed,
                    "regression_passed": passed,
                    "runs": len([r for e in task.experiments for r in e.runs]),
                    "grading": grading,
                }
                if include_traces:
                    record.update(trace=task.events, report=task.report)
                results.append(record)
            finally:
                service.store.close()
    return {
        "scope": "规则替身与确定性沙盒的工程回归，不是模型或生产效果",
        "units": {
            "source_groups": len({case.source_group for case in cases}),
            "task_cases": len(cases),
            "episodes": len(results),
            "repetitions": repetitions,
        },
        "cases": results,
        "all_passed": all(row["passed"] for row in results),
        "strict_task_success_rate": sum(r["grading"]["strict_success"] for r in results)
        / len(results),
        "note": "重复的是同一确定性任务，不把重复轮次当独立新问题或模型效果置信区间。",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output", type=Path, help="保存含轨迹的完整评测记录")
    args = parser.parse_args()
    result = run_task_evaluation(
        repetitions=args.repetitions, include_traces=bool(args.output)
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    # 终端只展示摘要；轨迹写文件后可以从 episode 一路追到每次运行。
    compact = {
        **result,
        "cases": [
            {k: v for k, v in r.items() if k not in {"trace", "report"}}
            for r in result["cases"]
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)
