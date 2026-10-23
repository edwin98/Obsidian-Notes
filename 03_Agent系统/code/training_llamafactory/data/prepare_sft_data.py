"""
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
  放弃分阶段上采样，改为在数据准备阶段一次性对难例上采样（upweight=2.0），
  训练时直接用这份数据跑完全部 3 个 Epoch。
  难例类型（error_recovery、auth_bearer）来自文档 §5.3。

运行示例：
  python data/prepare_sft_data.py \
    --input  ../../training/data/sft_clean.jsonl \
    --output data/sft_train.json \
    --upweight 2.0
"""

import argparse
import json
import random


ROLE_MAP = {
    "system": "system",
    "user": "human",
    "assistant": "gpt",
}

# 文档 §5.3：这两类场景模型最容易做错，上采样强化
HARD_SCENARIOS = {"error_recovery", "auth_bearer"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 SFT JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    parser.add_argument(
        "--upweight",
        type=float,
        default=2.0,
        help="难例上采样倍率（默认 2.0，即难例重复 2 次）",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def to_sharegpt(sample: dict) -> dict:
    """将 messages 格式的样本转换为 LlamaFactory ShareGPT 格式。"""
    conversations = []
    for msg in sample["messages"]:
        role = ROLE_MAP.get(msg["role"])
        if role is None:
            continue
        conversations.append({"from": role, "value": msg["content"]})
    return {"conversations": conversations}


def main():
    args = parse_args()
    random.seed(args.seed)

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
    random.shuffle(all_samples)

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
