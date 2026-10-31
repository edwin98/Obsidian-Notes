"""在测试题上真实运行 Agent，统计效果；也可以把几次评测结果排成对比表。

    python -X utf8 evaluate.py --name base
    python -X utf8 evaluate.py --name sft --adapters outputs/sft
    python -X utf8 evaluate.py --name dpo --adapters outputs/sft outputs/dpo
    python -X utf8 evaluate.py --compare base sft dpo

结果写入 outputs/eval_<name>.json（汇总）和 outputs/eval_<name>_trajs.jsonl（每条轨迹，方便翻看失败案例）。
"""

from __future__ import annotations

import argparse
import json

from agent import run_agent, summarize
from common import DATA_DIR, DEFAULT_MODEL, OUT_DIR, load_model, load_tokenizer, read_jsonl, set_seed, write_jsonl

METRICS = [
    ("accuracy", "答案正确率"),
    ("used_tool_rate", "使用了工具的比例"),
    ("avg_tool_calls", "平均工具调用次数"),
    ("invalid_call_rate", "出现非法调用的比例"),
    ("tool_error_rate", "工具报错的比例"),
    ("final_format_rate", "按 Final answer 格式收尾"),
    ("no_answer_rate", "没给出答案"),
]


def compare(names: list[str]) -> None:
    rows = {n: json.loads((OUT_DIR / f"eval_{n}.json").read_text(encoding="utf-8")) for n in names}
    header = f"{'指标':<22}" + "".join(f"{n:>10}" for n in names)
    print(header)
    print("-" * len(header.encode("gbk", "ignore")))
    for key, label in METRICS:
        vals = "".join(
            f"{rows[n][key]:>10.2f}" if key == "avg_tool_calls" else f"{rows[n][key] * 100:>9.1f}%" for n in names
        )
        print(f"{label:<{22 - sum(1 for c in label if ord(c) > 127)}}{vals}")
    n = rows[names[0]]["n"]
    se = (0.5 * 0.5 / n) ** 0.5 * 100
    print(f"\n题目数 n={n}；正确率的标准误约 ±{se:.1f} 个百分点，差距小于约 2 倍标准误时别急着下结论。")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="这次评测的名字，例如 base / sft / dpo")
    ap.add_argument("--adapters", nargs="*", default=[], help="按顺序合并的 LoRA 目录")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=None, help="只评前 N 题（调试用）")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--compare", nargs="*", help="只打印已有评测结果的对比表")
    args = ap.parse_args()

    if args.compare:
        compare(args.compare)
        return

    set_seed(0)
    problems = read_jsonl(DATA_DIR / "eval.jsonl")[: args.limit]
    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.adapters)
    trajs = run_agent(model, tok, problems, temperature=0.0, batch_size=args.batch_size, desc=f"eval-{args.name}")

    summary = summarize(trajs) | {"name": args.name, "adapters": args.adapters, "model": args.model}
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"eval_{args.name}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_jsonl(OUT_DIR / f"eval_{args.name}_trajs.jsonl", [t.to_dict() for t in trajs])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
