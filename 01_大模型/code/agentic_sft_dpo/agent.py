"""Agent 运行时：模型生成 → 解析工具调用 → 真实执行计算器 → 回填结果 → 继续生成。

评测（evaluate.py）和 DPO 采样（build_dpo_pairs.py）都调用这里的 run_agent，
所以“训练时见到的格式”和“真实运行时的格式”是同一套。

为了在单卡上跑得动，多条对话按“回合”批量推进：
每一回合把所有还没结束的对话一起左填充送进 generate，生成到 <|im_end|> 为止。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from common import SYSTEM_PROMPT, extract_answer, is_correct, parse_assistant, render, run_calculator


@dataclass
class Trajectory:
    question: str
    answer: float
    messages: list[dict]
    n_tool_calls: int = 0
    n_invalid_calls: int = 0
    n_tool_errors: int = 0
    finished: bool = False
    pred: float | None = None
    formatted: bool = False  # 是否按 "Final answer:" 格式收尾
    correct: bool = False
    raw_turns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@torch.no_grad()
def run_agent(
    model,
    tokenizer,
    problems: list[dict],
    n_samples: int = 1,
    temperature: float = 0.0,
    max_turns: int = 8,
    max_new_tokens: int = 512,
    batch_size: int = 32,
    desc: str = "agent",
) -> list[Trajectory]:
    """对每道题跑 n_samples 条轨迹。temperature=0 为贪心（评测）；>0 为采样（构造偏好数据）。"""
    trajs = [
        Trajectory(
            question=p["question"],
            answer=p["answer"],
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p["question"]}],
        )
        for p in problems
        for _ in range(n_samples)
    ]
    tokenizer.padding_side = "left"
    eos_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id]
    device = next(model.parameters()).device
    model.eval()

    for turn in range(max_turns):
        active = [t for t in trajs if not t.finished]
        if not active:
            break
        print(f"[{desc}] 第 {turn + 1} 回合，进行中的轨迹 {len(active)}/{len(trajs)}", flush=True)
        for i in range(0, len(active), batch_size):
            chunk = active[i : i + batch_size]
            prompts = [render(tokenizer, t.messages, add_generation_prompt=True) for t in chunk]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            gen_kwargs = dict(max_new_tokens=max_new_tokens, eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id)
            if temperature > 0:
                gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.95, top_k=0)
            else:
                gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            out = model.generate(**enc, **gen_kwargs)
            new_tokens = out[:, enc["input_ids"].shape[1] :]
            texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for t, text in zip(chunk, texts):
                _step(t, text)

    for t in trajs:
        if not t.finished:  # 回合用完还在调工具：判错
            t.finished = True
    return trajs


def _step(t: Trajectory, text: str) -> None:
    t.raw_turns.append(text)
    msg, calls, ok = parse_assistant(text)
    t.messages.append(msg)
    if not ok:
        # 工具调用写坏了：告诉模型格式错误，让它有机会改正
        t.n_invalid_calls += 1
        t.messages.append(
            {
                "role": "user",
                "content": "Your tool call could not be parsed. Use valid JSON inside <tool_call></tool_call>.",
            }
        )
        return
    if calls:
        for c in calls:
            result = run_calculator(c["arguments"])
            t.n_tool_calls += 1
            t.n_tool_errors += result.startswith("Error")
            t.messages.append({"role": "tool", "content": result})
        return
    # 没有工具调用 → 这一轮是最终回答
    t.finished = True
    t.pred, t.formatted = extract_answer(text)
    t.correct = is_correct(t.pred, t.answer)


def summarize(trajs: list[Trajectory]) -> dict:
    n = len(trajs)
    return {
        "n": n,
        "accuracy": sum(t.correct for t in trajs) / n,
        "used_tool_rate": sum(t.n_tool_calls > 0 for t in trajs) / n,
        "avg_tool_calls": sum(t.n_tool_calls for t in trajs) / n,
        "invalid_call_rate": sum(t.n_invalid_calls > 0 for t in trajs) / n,
        "tool_error_rate": sum(t.n_tool_errors > 0 for t in trajs) / n,
        "final_format_rate": sum(t.formatted for t in trajs) / n,
        "no_answer_rate": sum(t.pred is None for t in trajs) / n,
    }
