"""离线学习飞轮的最小数据入口：只有专家明确确认的修正才能进入训练集。

这里只导出 SFT/DPO 数据，不启动昂贵训练，不把模型自己的结论当专家标签。
输出格式可交给现有 training/ 脚本进一步清洗、划分训练/验证集并转换。
"""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExpertReview:
    task_id: str
    query: str
    original_answer: str
    corrected_answer: str
    reviewer: str
    confirmed: bool = False


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
        metadata = {"task_id": review.task_id, "reviewer": review.reviewer}
        sft.append({"messages": prompt + chosen, "metadata": metadata})
        if review.original_answer != review.corrected_answer:
            dpo.append(
                {
                    "prompt": f"{system_prompt}\n\n用户：{review.query}\n助手：",
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
