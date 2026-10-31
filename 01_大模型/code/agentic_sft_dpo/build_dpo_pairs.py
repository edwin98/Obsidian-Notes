"""第 3 步：让 SFT 后的模型在“没见过的新题”上自己做题，用环境结果构造偏好对。

为什么不直接下载现成的偏好数据集？
    DPO 学的是“同一个 prompt 下，chosen 比 rejected 更好”。如果 rejected 不是当前模型会犯的错，
    模型就在压低一些它本来就不会说的话，信号很弱。用当前模型自己的采样（on-policy）构造的偏好对，
    能精准地压低它“真的会犯”的错误：心算不调工具、表达式写错、答案没按格式写、算到一半放弃……

流程：
    每道题采样 K 条轨迹（temperature>0，计算器真实执行）
    → 用标准答案判对错（这就是环境奖励）
    → 同一道题里既有对又有错时，配成 (chosen=对的轨迹, rejected=错的轨迹)
    → 写入 data/dpo_pairs.jsonl

    python -X utf8 build_dpo_pairs.py --adapters outputs/sft
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict

from agent import run_agent, summarize
from common import DATA_DIR, DEFAULT_MODEL, load_model, load_tokenizer, read_jsonl, set_seed, write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--adapters", nargs="*", default=["outputs/sft"])
    ap.add_argument("--k", type=int, default=6, help="每道题采样几条轨迹")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max_pairs_per_question", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)

    problems = read_jsonl(DATA_DIR / "dpo_prompts.jsonl")[: args.limit]
    tok = load_tokenizer(args.model)
    model = load_model(args.model, args.adapters)
    trajs = run_agent(
        model, tok, problems, n_samples=args.k, temperature=args.temperature, batch_size=args.batch_size, desc="sample"
    )
    write_jsonl(DATA_DIR / "dpo_samples.jsonl", [t.to_dict() for t in trajs])
    print("采样整体统计:", json.dumps(summarize(trajs), ensure_ascii=False))

    by_q = defaultdict(list)
    for t in trajs:
        by_q[t.question].append(t)

    pairs, all_right, all_wrong = [], 0, 0
    reasons = defaultdict(int)
    for q, ts in by_q.items():
        good = [t for t in ts if t.correct and t.n_invalid_calls == 0]
        bad = [t for t in ts if not t.correct]
        if not bad:
            all_right += 1
        if not good:
            all_wrong += 1
        if not good or not bad:
            continue  # 全对或全错：这道题没有对比信号
        rng.shuffle(good)
        rng.shuffle(bad)
        for g, b in list(zip(good, bad))[: args.max_pairs_per_question]:
            reason = (
                "非法工具调用"
                if b.n_invalid_calls
                else "没调用工具"
                if b.n_tool_calls == 0
                else "回合用完未作答"
                if b.pred is None
                else "答案错误"
            )
            reasons[reason] += 1
            pairs.append(
                {
                    "prompt": g.messages[:2],  # system + user，两条轨迹完全相同
                    "chosen": g.messages[2:],
                    "rejected": b.messages[2:],
                    "answer": g.answer,
                    "rejected_reason": reason,
                }
            )

    write_jsonl(DATA_DIR / "dpo_pairs.jsonl", pairs)
    print(f"\n题目 {len(by_q)} 道：全对 {all_right}，全错 {all_wrong}，可配对 {len(by_q) - all_right - all_wrong}")
    print(f"共 {len(pairs)} 个偏好对 → data/dpo_pairs.jsonl")
    print("rejected 的错误类型:", dict(reasons))
    if pairs:
        p = pairs[0]
        print("\n===== 示例偏好对 =====\n题目:", p["prompt"][1]["content"][:200])
        for side in ("chosen", "rejected"):
            print(f"\n--- {side} ---")
            for m in p[side]:
                tc = m.get("tool_calls")
                print(f"[{m['role']}]", m["content"][:200], json.dumps(tc[0]["function"]["arguments"]) if tc else "")


if __name__ == "__main__":
    main()
