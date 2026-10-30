"""【学习定位：可选框架格式转换】
转换已拆分的数据；保留来源元数据；不支持的工具角色显式报错，不能静默丢弃观测。
对应08；默认不做上采样，转换不负责创造监督真值。

prepare_sft_data.py — 将原始 SFT 数据转换为 LlamaFactory ShareGPT 格式

输入格式（JSONL，每行一条）：
  {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}],
   "scenario": "handover"}

输出格式（JSON 数组，LlamaFactory ShareGPT 格式）：
  [{"conversations": [{"from": "system", "value": "..."},
                      {"from": "human", "value": "..."},
                      {"from": "gpt",   "value": "..."}]}]

核心处理：
  保留来源与角色；默认不改变样本分布。只有已拆分训练集才可显式上采样，
  验证与最终测试不使用重复采样。工具交互暂不支持时明确报错，不丢弃。

运行示例：
  python data/prepare_sft_data.py     --input  ../../training/data/sft_clean.jsonl     --output data/sft_train.json     --upweight 2
"""

import argparse
import json
import random


ROLE_MAP = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
}

# 历史示例类别；是否真的需要提高权重，取决于独立验证结果。
HARD_SCENARIOS = {"error_recovery", "auth_bearer"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 SFT JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument(
        "--upweight",
        type=int,
        default=1,
        help="显式选择的整数重复倍率；默认不改变分布，仅用于已拆分训练集",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_sharegpt(sample: dict) -> dict:
    """将 messages 格式的样本转换为 LlamaFactory ShareGPT 格式。"""
    conversations = []
    for msg in sample["messages"]:
        role = ROLE_MAP.get(msg["role"])
        if role is None or msg.get("tool_calls"):
            raise ValueError("此简化转换器不序列化工具交互，不能静默丢弃消息")
        conversations.append({"from": role, "value": msg["content"]})
    return {"conversations": conversations, "metadata": sample.get("metadata", {})}


def main():
    args = parse_args()
    if args.upweight < 1:
        raise ValueError("上采样倍率必须至少为1")
    rng = random.Random(args.seed)

    raw: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    hard = [s for s in raw if s.get("scenario") in HARD_SCENARIOS]
    normal = [s for s in raw if s.get("scenario") not in HARD_SCENARIOS]

    # 难例重复 (upweight - 1) 份，总出现次数 = upweight 倍
    extra = hard * int(args.upweight - 1)
    all_samples = normal + hard + extra
    rng.shuffle(all_samples)

    output = [to_sharegpt(s) for s in all_samples]

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    total_hard = len(hard) + len(extra)
    print(
        f"完成：共 {len(output)} 条（原始 {len(raw)}，难例 {len(hard)} → {total_hard}）"
    )
    print(f"输出：{args.output}")


if __name__ == "__main__":
    main()
