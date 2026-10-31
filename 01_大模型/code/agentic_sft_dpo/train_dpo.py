"""第 4 步：在 SFT 模型上做多轮轨迹级 DPO（LoRA）。

模型结构：
    基座 + SFT LoRA（合并进权重，冻结）  ← 这就是参考模型 π_ref
    再挂一层新的 DPO LoRA（可训练）       ← 策略模型 π_θ
    参考模型不需要第二份显存：用 `with model.disable_adapter():` 临时关掉 DPO LoRA 即可。

DPO 损失（Rafailov et al., 2023）：
    log π(y|x) = 轨迹中所有 assistant token 的对数概率之和（工具返回、user 消息不算，和 SFT 掩码一致）
    r(y)       = β · [log π_θ(y|x) − log π_ref(y|x)]          # 隐式奖励
    loss       = −log σ( r(chosen) − r(rejected) )

训练时重点看：
    reward_acc   r(chosen) > r(rejected) 的比例，应明显高于 0.5
    margin       r(chosen) − r(rejected)，应逐渐增大
    chosen_logp  chosen 的对数概率是否也在下降——下降太多说明在“两边一起压”，可调小 lr 或加 --sft_alpha

    python -X utf8 train_dpo.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time

import torch
import torch.nn.functional as F
from peft import PeftModel, get_peft_model

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


def sequence_logps(model, batch) -> torch.Tensor:
    """每条序列在 loss_mask 位置上的 token 对数概率之和，形状 [B]。"""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    # 只在 assistant token 上取 logits（省显存），算出每个 token 的 log p，再按所属序列加总
    logits = out.logits[:, :-1, :][mask].float()  # [N, V]
    token_logps = F.log_softmax(logits, dim=-1).gather(1, labels[mask].unsqueeze(1)).squeeze(1)  # [N]
    seq_index = torch.arange(labels.size(0), device=labels.device).unsqueeze(1).expand_as(labels)[mask]
    seq_logps = torch.zeros(labels.size(0), device=labels.device, dtype=token_logps.dtype)
    seq_logps = seq_logps.index_add(0, seq_index, token_logps)
    return seq_logps, mask.sum(-1)


def dpo_step(model, batch, n_pairs: int, beta: float, sft_alpha: float, device):
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    with torch.autocast(**amp):
        pol, ntok = sequence_logps(model, batch)
        with torch.no_grad(), model.disable_adapter():
            ref, _ = sequence_logps(model, batch)
    pc, pr = pol[:n_pairs], pol[n_pairs:]
    rc, rr = ref[:n_pairs], ref[n_pairs:]
    r_chosen, r_rejected = beta * (pc - rc), beta * (pr - rr)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    if sft_alpha > 0:  # 可选：在 chosen 上加一点 NLL（RPO 思路），防止 chosen 概率被一起压低
        loss = loss + sft_alpha * (-(pc / ntok[:n_pairs]).mean())
    stats = {
        "loss": loss.item(),
        "reward_acc": (r_chosen > r_rejected).float().mean().item(),
        "margin": (r_chosen - r_rejected).mean().item(),
        "r_chosen": r_chosen.mean().item(),
        "r_rejected": r_rejected.mean().item(),
        "chosen_logp": pc.mean().item(),
        "rejected_logp": pr.mean().item(),
    }
    return loss, stats


def make_batch(pairs, tok, max_len, device):
    feats = [encode_conversation(tok, p["prompt"] + p["chosen"], max_len) for p in pairs]
    feats += [encode_conversation(tok, p["prompt"] + p["rejected"], max_len) for p in pairs]
    return {k: v.to(device) for k, v in pad_batch(feats, tok.pad_token_id).items()}


@torch.no_grad()
def evaluate(model, val, tok, args, device) -> dict:
    model.eval()
    agg = []
    for i in range(0, len(val), args.micro_batch):
        chunk = val[i : i + args.micro_batch]
        _, s = dpo_step(model, make_batch(chunk, tok, args.max_len, device), len(chunk), args.beta, 0.0, device)
        agg.append((s, len(chunk)))
    model.train()
    n = sum(c for _, c in agg)
    return {k: sum(s[k] * c for s, c in agg) / n for k in agg[0][0]} if agg else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--sft_adapter", default=str(OUT_DIR / "sft"))
    ap.add_argument("--output_dir", default=str(OUT_DIR / "dpo"))
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--micro_batch", type=int, default=2, help="每个 micro batch 的偏好对数（实际序列数 ×2）")
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1536)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--sft_alpha", type=float, default=0.0)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_seed(args.seed)

    pairs = read_jsonl(DATA_DIR / "dpo_pairs.jsonl")
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val_frac))
    val, train = pairs[:n_val], pairs[n_val:]
    print(f"偏好对：训练 {len(train)}，验证 {len(val)}")

    from transformers import AutoModelForCausalLM

    tok = load_tokenizer(args.model)
    tok.padding_side = "right"
    device, dtype = get_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    model = PeftModel.from_pretrained(model, args.sft_adapter).merge_and_unload()  # π_ref = SFT 模型
    model = model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model = get_peft_model(model, lora_config(r=args.lora_r, alpha=2 * args.lora_r, dropout=0.0))
    model.print_trainable_parameters()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    total_steps = max(1, int(math.ceil(len(train) / (args.micro_batch * args.grad_accum)) * args.epochs))
    warmup = max(1, total_steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))

    log = [{"step": 0, "val": evaluate(model, val, tok, args, device)}]
    print(
        "训练前验证（此时策略=参考，奖励全为 0，reward_acc 约为 0）:",
        {k: round(v, 3) for k, v in log[0]["val"].items()},
    )

    model.train()
    step, order, t0 = 0, [], time.time()
    while step < total_steps:
        accum = []
        for _ in range(args.grad_accum):
            if not order:
                order = list(range(len(train)))
                random.shuffle(order)
            idx, order = order[: args.micro_batch], order[args.micro_batch :]
            chunk = [train[i] for i in idx]
            loss, s = dpo_step(
                model, make_batch(chunk, tok, args.max_len, device), len(chunk), args.beta, args.sft_alpha, device
            )
            (loss / args.grad_accum).backward()
            accum.append(s)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        s = {k: sum(a[k] for a in accum) / len(accum) for k in accum[0]}
        entry = {"step": step, "train": s}
        if step % 5 == 0 or step == 1:
            print(
                f"step {step}/{total_steps}  loss {s['loss']:.4f}  reward_acc {s['reward_acc']:.2f}  "
                f"margin {s['margin']:.3f}  chosen_logp {s['chosen_logp']:.1f}  rejected_logp {s['rejected_logp']:.1f}  "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )
        if step % 20 == 0 or step == total_steps:
            entry["val"] = evaluate(model, val, tok, args, device)
            print("  └ 验证:", {k: round(v, 3) for k, v in entry["val"].items()})
        log.append(entry)

    model.save_pretrained(args.output_dir)
    with open(f"{args.output_dir}/train_log.json", "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "log": log}, f, ensure_ascii=False, indent=2)
    print(f"DPO 适配器已保存到 {args.output_dir}，用时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
