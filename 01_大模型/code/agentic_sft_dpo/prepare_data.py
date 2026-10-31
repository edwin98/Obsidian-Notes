"""第 1 步：把开源 GSM8K 改造成“多轮工具调用轨迹”，作为 agentic SFT 数据。

GSM8K（openai/gsm8k）的参考解答里自带计算器标注，例如：
    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
    #### 72

每个 <<表达式=结果>> 就是一次“该调工具的地方”。我们把它改写成标准的 tool-calling 多轮对话：
    assistant: (调用 calculator("48/2"))
    tool:      24                      ← 由我们的计算器真实算出，和评测时环境返回完全一致
    assistant: Natalia sold 48/2 = 24 clips in May.  (调用 calculator("48+24"))
    tool:      72
    assistant: Natalia sold 48+24 = 72 clips ... \n Final answer: 72

输出（data/ 目录）：
    sft_train.jsonl / sft_val.jsonl   多轮轨迹（messages），用于 SFT
    dpo_prompts.jsonl                 另一批训练题，只含题目和答案，DPO 阶段让模型自己去做
    eval.jsonl                        测试集题目，评测用（训练全程不碰）

用法：python -X utf8 prepare_data.py [--n_sft 2000 --n_dpo 400 --n_eval 300]
"""

from __future__ import annotations

import argparse
import json
import random
import re

from datasets import load_dataset

from common import DATA_DIR, SYSTEM_PROMPT, calculate, format_number, write_jsonl

ANN_RE = re.compile(r"<<([^<>]*?)=([^<>=]*?)>>")


def gold_answer(solution: str) -> float:
    return float(solution.split("####")[-1].strip().replace(",", ""))


def solution_to_messages(question: str, solution: str) -> list[dict] | None:
    """返回多轮 messages；标注无法被我们的计算器复现时返回 None（丢弃这条）。"""
    body, final = solution.split("####")
    final = final.strip().replace(",", "")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]

    buffer: list[str] = []  # 已经“想好”但还没说出口的解释句
    for line in body.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for expr, result in ANN_RE.findall(line):
            try:
                value = calculate(expr)
                if abs(value - float(result.replace(",", ""))) > 1e-4 * max(1, abs(value)):
                    return None  # 标注本身算错了
            except Exception:  # noqa: BLE001
                return None  # 表达式里有单位、变量等，计算器解析不了
            messages.append(
                {
                    "role": "assistant",
                    "content": "\n".join(buffer),
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "calculator", "arguments": {"expression": expr.strip()}},
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "content": format_number(value)})
            buffer = []
        buffer.append(ANN_RE.sub("", line))  # 去掉标注，保留自然语言

    buffer.append(f"Final answer: {final}")
    messages.append({"role": "assistant", "content": "\n".join(buffer)})
    return messages


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sft", type=int, default=2000, help="SFT 训练轨迹条数")
    ap.add_argument("--n_val", type=int, default=150, help="SFT 验证集条数")
    ap.add_argument("--n_dpo", type=int, default=400, help="DPO 阶段用于采样的题目数")
    ap.add_argument("--n_eval", type=int, default=300, help="评测题目数（来自 test split）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset("openai/gsm8k", "main")
    rng = random.Random(args.seed)

    train = list(ds["train"])
    rng.shuffle(train)

    converted, dropped = [], 0
    for ex in train:
        msgs = solution_to_messages(ex["question"], ex["answer"])
        if msgs is None:
            dropped += 1
            continue
        converted.append({"messages": msgs, "answer": gold_answer(ex["answer"]), "question": ex["question"]})
    print(f"GSM8K train: {len(train)} 条，可转换 {len(converted)} 条，丢弃 {dropped} 条")

    n_sft, n_val, n_dpo = args.n_sft, args.n_val, args.n_dpo
    assert n_sft + n_val + n_dpo <= len(converted), "数量超过可用数据"
    sft_train = converted[:n_sft]
    sft_val = converted[n_sft : n_sft + n_val]
    # DPO 用的是 SFT 没见过的题：让模型在“新题”上自己探索、产生对错不同的轨迹
    dpo_prompts = [
        {"question": r["question"], "answer": r["answer"]} for r in converted[n_sft + n_val : n_sft + n_val + n_dpo]
    ]

    test = list(ds["test"])
    random.Random(args.seed).shuffle(test)
    eval_rows = [{"question": ex["question"], "answer": gold_answer(ex["answer"])} for ex in test[: args.n_eval]]

    write_jsonl(DATA_DIR / "sft_train.jsonl", sft_train)
    write_jsonl(DATA_DIR / "sft_val.jsonl", sft_val)
    write_jsonl(DATA_DIR / "dpo_prompts.jsonl", dpo_prompts)
    write_jsonl(DATA_DIR / "eval.jsonl", eval_rows)

    n_calls = [sum(1 for m in r["messages"] if m["role"] == "tool") for r in sft_train]
    print(
        f"写出 sft_train={len(sft_train)} sft_val={len(sft_val)} dpo_prompts={len(dpo_prompts)} eval={len(eval_rows)}"
    )
    print(f"每条轨迹平均工具调用 {sum(n_calls) / len(n_calls):.2f} 次，最多 {max(n_calls)} 次")
    print("\n===== 一条 SFT 轨迹示例 =====")
    for m in sft_train[0]["messages"][1:]:
        print(
            f"[{m['role']}]",
            m.get("content", ""),
            json.dumps(m["tool_calls"][0]["function"]["arguments"]) if "tool_calls" in m else "",
        )


if __name__ == "__main__":
    main()
