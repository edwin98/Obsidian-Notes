"""【学习定位：新版数据准备】
统一验证SFT/DPO目标并按source_group哈希拆分，训练入口不得再次按行随机打散。
对应08；相同故障及修复跟进属于同一组。

验证新版决策数据，并按原始问题统一拆分 SFT/DPO，避免同源泄漏。

从 code 目录运行 python -m training.prepare_investigation_data --input-dir ... --output-dir ...
本脚本不调用模型，不启动训练，也不把旧结论型清洗规则用于实验计划。
"""

import argparse
import hashlib
import json
from pathlib import Path

from task_models import ExperimentPlan, PlanDecision


def split_for_group(group: str, seed: str = "investigation-v1") -> str:
    """来源组做稳定哈希，不依赖输入行序；SFT/DPO共用同一分配。"""
    value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest()[:8], 16) % 100
    return "train" if value < 80 else "validation" if value < 90 else "test"


def read_training_splits(train_path: Path, validation_path: Path):
    """训练脚本显式读取已拆分文件，禁止再次随机拆散同一个问题。"""

    def read(path):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(
                f"数据为空：{path}。需要更多独立问题，不能拿测试集补验证集。"
            )
        groups = {row.get("metadata", {}).get("source_group") for row in rows}
        if None in groups or "" in groups:
            raise ValueError("训练数据缺少原问题分组 source_group")
        return rows, groups

    train, train_groups = read(train_path)
    validation, validation_groups = read(validation_path)
    if train_groups & validation_groups:
        raise ValueError("同一原始问题同时出现在训练集和验证集")
    return train, validation


def normalize_target(text: str, schema_name: str | None) -> str:
    """旧内层计划显式升级为部署使用的PlanDecision，输出统一序列化格式。"""
    if schema_name == "PlanDecision":
        return PlanDecision.model_validate_json(text).model_dump_json()
    if schema_name not in {None, "ExperimentPlan"}:
        raise ValueError("不支持的训练目标schema")
    plan = ExperimentPlan.model_validate_json(text)
    return PlanDecision(
        action="EXPERIMENT", plan=plan, explanation=plan.reason
    ).model_dump_json()


def prepare(input_dir: Path, output_dir: Path, *, seed="investigation-v1"):
    """先完成全部验证再写文件，避免坏样本产生半套可误用的训练集。"""
    rows = {
        f"{kind}_{split}": []
        for kind in ("sft", "dpo")
        for split in ("train", "validation", "test")
    }
    groups = {}
    for kind in ("sft", "dpo"):
        path = input_dir / f"{kind}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            metadata = sample.get("metadata", {})
            group = metadata.get("source_group")
            if (
                metadata.get("sample_type") != "experiment_decision"
                or not group
                or not metadata.get("reviewer")
            ):
                raise ValueError("只接收带专家与原问题分组的实验决策数据")
            if kind == "sft":
                messages = sample["messages"]
                if messages[-1]["role"] != "assistant":
                    raise ValueError("SFT 需要完整的助手目标")
                messages[-1]["content"] = normalize_target(
                    messages[-1]["content"], metadata.get("target_schema")
                )
            else:
                # prompt可用角色列表；候选仍是完整的计划JSON。
                sample["chosen"] = normalize_target(
                    sample["chosen"], metadata.get("target_schema")
                )
                sample["rejected"] = normalize_target(
                    sample["rejected"], metadata.get("target_schema")
                )
                if sample["chosen"] == sample["rejected"]:
                    raise ValueError("相同候选不作为 DPO 样本")
            metadata["target_schema"] = "PlanDecision"
            split = split_for_group(group, seed)
            groups[group] = split
            rows[f"{kind}_{split}"].append(sample)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in rows.items():
        with (output_dir / f"{name}.jsonl").open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "groups.json").write_text(
        json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {name: len(records) for name, records in rows.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default="investigation-v1")
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.input_dir, args.output_dir, seed=args.seed), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
