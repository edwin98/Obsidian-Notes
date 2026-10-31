"""对比演示：同一道题，分别让 原模型 / SFT / SFT+DPO 当 Agent 去做，逐轮打印轨迹。

python -X utf8 demo.py                    # 测试集第 0 题
python -X utf8 demo.py --index 7
python -X utf8 demo.py --question "A shop sells 37 boxes of 24 pens at $1.35 each. How much money?" --answer 1198.8
"""

from __future__ import annotations

import argparse
import gc
import json

import torch

from agent import run_agent
from common import DATA_DIR, DEFAULT_MODEL, OUT_DIR, load_model, load_tokenizer, read_jsonl

STAGES = {"base": [], "sft": [OUT_DIR / "sft"], "dpo": [OUT_DIR / "sft", OUT_DIR / "dpo"]}


def show(name: str, t) -> None:
    flag = "✅" if t.correct else "❌"
    print(f"\n{'=' * 30} {name} {flag}  预测={t.pred}  标准答案={t.answer}  工具调用={t.n_tool_calls} {'=' * 30}")
    for m in t.messages[2:]:
        if m["role"] == "assistant":
            if m["content"]:
                print(f"🤖 {m['content']}")
            for c in m.get("tool_calls", []):
                print(f"   🔧 calculator({json.dumps(c['function']['arguments']['expression'])})")
        elif m["role"] == "tool":
            print(f"   📟 → {m['content']}")
        else:
            print(f"   ⚠️ {m['content']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--question")
    ap.add_argument("--answer", type=float, default=float("nan"))
    ap.add_argument("--stages", nargs="*", default=["base", "sft", "dpo"])
    args = ap.parse_args()

    problem = (
        {"question": args.question, "answer": args.answer}
        if args.question
        else read_jsonl(DATA_DIR / "eval.jsonl")[args.index]
    )
    print("题目:", problem["question"])
    tok = load_tokenizer(args.model)
    for stage in args.stages:
        adapters = STAGES[stage]
        if any(not p.exists() for p in adapters):
            print(f"\n(跳过 {stage}：还没有 {[str(p) for p in adapters]})")
            continue
        model = load_model(args.model, adapters)
        show(stage, run_agent(model, tok, [problem], desc=stage)[0])
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
