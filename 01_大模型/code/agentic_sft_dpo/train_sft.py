"""第 2 步：LoRA SFT —— 让模型模仿“边算边调工具”的多轮轨迹。

为了看清楚每一步，这里没有用 TRL 的 SFTTrainer，而是手写了一个很短的训练循环：
    多轮对话 → encode_conversation（只在 assistant 段打 loss_mask）→ 右填充成 batch
    → 前向得到 logits → 交叉熵只在 labels != -100 的位置计算 → 反向只更新 LoRA 参数

    python -X utf8 train_sft.py --inspect     # 只看一个 batch 的张量和被学习的 token，不训练
    python -X utf8 train_sft.py               # 正式训练，适配器保存到 outputs/sft
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
from peft import get_peft_model

from common import (
    DATA_DIR,
    DEFAULT_MODEL,
    OUT_DIR,
    encode_conversation,
    get_device_dtype,
    load_tokenizer,
    lora_config,
    pad_batch,
    read_jsonl,
    set_seed,
)


def sft_loss(model, batch) -> tuple[torch.Tensor, int]:
    """因果语言模型：位置 t 的 logits 预测 t+1 的 token，所以 logits 去掉最后一位，labels 去掉第一位。"""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    # 只取需要算损失的位置再转 float32：词表有 15 万维，整段 logits 转 float 很吃显存
    logits = out.logits[:, :-1, :][mask].float()
    loss = torch.nn.functional.cross_entropy(logits, labels[mask])
    return loss, int(mask.sum())


@torch.no_grad()
def evaluate_loss(model, data, tok, args, device) -> float:
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(data), args.micro_batch):
        batch = {k: v.to(device) for k, v in pad_batch(data[i : i + args.micro_batch], tok.pad_token_id).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss, n = sft_loss(model, batch)
        total += loss.item() * n
        count += n
    model.train()
    return total / max(count, 1)


def inspect(tok, rows, max_len) -> None:
    feats = [encode_conversation(tok, r["messages"], max_len) for r in rows[:2]]
    batch = pad_batch(feats, tok.pad_token_id)
    print("input_ids 形状:", tuple(batch["input_ids"].shape))
    for name, f in zip(["样本0", "样本1"], feats):
        print(
            f"{name}: 总 token {len(f['input_ids'])}，参与损失 {sum(f['loss_mask'])} 个 ({sum(f['loss_mask']) / len(f['input_ids']):.0%})"
        )
    print("\n===== 样本0：被学习的片段（=== LEARN ===）与不学的片段（--- skip ---）=====")
    ids, mask = feats[0]["input_ids"], feats[0]["loss_mask"]
    seg, cur = [], mask[0]
    for t, m in zip(ids + [None], mask + [None]):
        if m != cur:
            print("=== LEARN ===" if cur else "--- skip  ---")
            print(tok.decode(seg))
            seg, cur = [], m
        if t is not None:
            seg.append(t)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--output_dir", default=str(OUT_DIR / "sft"))
    ap.add_argument("--limit", type=int, default=None, help="只用前 N 条训练数据（调试用）")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--micro_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4, help="有效 batch = micro_batch * grad_accum")
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect", action="store_true")
    args = ap.parse_args()
    set_seed(args.seed)

    tok = load_tokenizer(args.model)
    tok.padding_side = "right"
    train_rows = read_jsonl(DATA_DIR / "sft_train.jsonl")[: args.limit]
    val_rows = read_jsonl(DATA_DIR / "sft_val.jsonl")[: (args.limit // 5 if args.limit else None)]
    if args.inspect:
        inspect(tok, train_rows, args.max_len)
        return

    train = [encode_conversation(tok, r["messages"], args.max_len) for r in train_rows]
    val = [encode_conversation(tok, r["messages"], args.max_len) for r in val_rows]
    print(f"训练 {len(train)} 条，验证 {len(val)} 条，截断 {sum(f['truncated'] for f in train)} 条")

    from transformers import AutoModelForCausalLM

    device, dtype = get_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, lora_config(r=args.lora_r, alpha=2 * args.lora_r))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(train) / (args.micro_batch * args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, total_steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))
    )

    log = []
    v0 = evaluate_loss(model, val, tok, args, device)
    print(f"训练前验证损失 {v0:.4f}")
    log.append({"step": 0, "val_loss": v0})

    model.train()
    step, t0 = 0, time.time()
    order = []
    while step < total_steps:
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        running, ntok = 0.0, 0
        for _ in range(args.grad_accum):
            idx, order = order[: args.micro_batch], order[args.micro_batch :]
            if not idx:
                break
            batch = {k: v.to(device) for k, v in pad_batch([train[i] for i in idx], tok.pad_token_id).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                loss, n = sft_loss(model, batch)
            (loss / args.grad_accum).backward()
            running += loss.item() * n
            ntok += n
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1

        entry = {"step": step, "train_loss": running / max(ntok, 1), "lr": sched.get_last_lr()[0]}
        if step % 10 == 0 or step == 1:
            print(
                f"step {step}/{total_steps}  loss {entry['train_loss']:.4f}  lr {entry['lr']:.2e}  {time.time() - t0:.0f}s",
                flush=True,
            )
        if step % args.eval_every == 0 or step == total_steps:
            entry["val_loss"] = evaluate_loss(model, val, tok, args, device)
            print(f"  └ 验证损失 {entry['val_loss']:.4f}")
        log.append(entry)

    model.save_pretrained(args.output_dir)
    with open(f"{args.output_dir}/train_log.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "log": log}, f, ensure_ascii=False, indent=2)
    print(f"LoRA 适配器已保存到 {args.output_dir}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
