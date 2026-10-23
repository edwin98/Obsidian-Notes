"""
prepare_dpo_data.py — 将原始 DPO 偏好数据转换为 LlamaFactory ShareGPT 格式

输入格式（JSONL，每行一条）：
  {
    "prompt":    [{"role": "user", "content": "..."}, ...],  // 对话上下文（messages 列表）
    "chosen":    "专家修改后的安全回复",
    "rejected":  "Agent 生成的危险回复"
  }

  注意：若 prompt 为字符串而非列表，本脚本会自动包装为 [{"role": "user", "content": prompt}]。

输出格式（JSON 数组，LlamaFactory ShareGPT DPO 格式）：
  [
    {
      "conversations": [{"from": "human", "value": "..."}],
      "chosen":   {"from": "gpt", "value": "安全回复"},
      "rejected": {"from": "gpt", "value": "危险回复"}
    }
  ]

运行示例：
  python data/prepare_dpo_data.py \
    --input  ../../training/data/dpo_clean.jsonl \
    --output data/dpo_train.json
"""

import argparse
import json


ROLE_MAP = {"system": "system", "user": "human", "assistant": "gpt"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="原始 DPO JSONL 路径")
    parser.add_argument("--output", required=True, help="输出 JSON 路径")
    return parser.parse_args()


def convert_sample(sample: dict) -> dict | None:
    """将一条 DPO 样本转换为 LlamaFactory 格式，字段缺失时返回 None。"""
    prompt = sample.get("prompt")
    chosen = sample.get("chosen")
    rejected = sample.get("rejected")

    if not prompt or not chosen or not rejected:
        return None

    # prompt 可能是字符串，也可能是 messages 列表
    if isinstance(prompt, str):
        conversations = [{"from": "human", "value": prompt}]
    else:
        conversations = []
        for msg in prompt:
            role = ROLE_MAP.get(msg.get("role", ""))
            if role and role != "gpt":  # DPO prompt 中不含 assistant 轮
                conversations.append({"from": role, "value": msg["content"]})

    if not conversations:
        return None

    return {
        "conversations": conversations,
        "chosen": {"from": "gpt", "value": chosen},
        "rejected": {"from": "gpt", "value": rejected},
    }


def main():
    args = parse_args()

    raw: list[dict] = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw.append(json.loads(line))

    output = []
    skipped = 0
    for s in raw:
        converted = convert_sample(s)
        if converted:
            output.append(converted)
        else:
            skipped += 1

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"完成：共 {len(output)} 条（跳过 {skipped} 条字段缺失样本）")
    print(f"输出：{args.output}")


if __name__ == "__main__":
    main()
