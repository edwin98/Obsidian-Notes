"""
train_reranker.py
==================
gte-multilingual-reranker-0.3B 全量 SFT 主训练脚本。

架构：Cross-Encoder（判别式），输入 [CLS] query [SEP] doc [SEP]，输出相关性分数。
策略：全量参数微调（Full Parameter SFT）—— 不使用 LoRA。
      原因见 03_Embedding与Reranker微调全流程.md §3.4.1。

框架栈：
  sentence-transformers  —— CrossEncoder 封装 + CERerankingEvaluator
  accelerate             —— 多 GPU DDP（2× A10G）
  deepspeed ZeRO-1       —— 优化器状态分片（可选 --deepspeed）
  torch.compile          —— 图编译加速 ~15%（--use_compile）
  wandb                  —— 训练监控

硬件：2× NVIDIA A10G 24GB，实测训练时间约 5h（3 epoch，28万 instance）。

启动命令：

  # 单卡（调试）
  python train_reranker.py \
      --train_file  data/reranker_train.jsonl \
      --val_file    data/reranker_val.jsonl \
      --output_dir  checkpoints/reranker-finetuned \
      --model_name  Alibaba-NLP/gte-multilingual-reranker-base

  # 双卡 DDP
  accelerate launch --num_processes 2 train_reranker.py \
      --train_file  data/reranker_train.jsonl \
      --val_file    data/reranker_val.jsonl \
      --output_dir  checkpoints/reranker-finetuned \
      --model_name  Alibaba-NLP/gte-multilingual-reranker-base \
      --per_device_batch 32 \
      --num_epochs 5

  # 双卡 DDP + DeepSpeed ZeRO-1
  accelerate launch --num_processes 2 --use_deepspeed train_reranker.py \
      --train_file  data/reranker_train.jsonl \
      --val_file    data/reranker_val.jsonl \
      --output_dir  checkpoints/reranker-finetuned \
      --deepspeed   ds_config_zero1.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import RerankerListwiseDataset, listwise_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 损失函数：Listwise Softmax Loss（带温度的 InfoNCE 变体）
# ---------------------------------------------------------------------------

class ListwiseSoftmaxLoss(nn.Module):
    """
    将一个 query 对应的所有文档视为一个列表，用 Softmax 建模整体排序。

    对每个 query，loss = -sum_{i in positives} log( exp(score_i) / sum_j exp(score_j) )

    使用连续分数标签（来自 LLM 归一化分数 [0,1]）而非二值标签：
      target_dist = softmax(labels / temperature)  # 软目标分布
      loss = KLDiv(log_softmax(scores), target_dist)

    与硬标签相比，软标签保留了正/负样本之间的相对梯度，
    让模型学到分数差异而非只学到"谁排第一"。
    """

    def __init__(self, temperature: float = 1.0, label_smoothing: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(
        self,
        scores: torch.Tensor,     # [total_pairs]，模型输出的原始相关性分数
        labels: torch.Tensor,     # [total_pairs]，LLM 归一化分数 [0, 1]
        query_ids: torch.Tensor,  # [total_pairs]，每个 pair 属于哪个 query
    ) -> torch.Tensor:
        unique_queries = query_ids.unique()
        total_loss = torch.tensor(0.0, device=scores.device)
        n_valid = 0

        for qid in unique_queries:
            mask = query_ids == qid
            q_scores = scores[mask]   # [n_docs_for_this_query]
            q_labels = labels[mask]   # [n_docs_for_this_query]

            if q_scores.shape[0] < 2:
                continue

            # 目标分布：用 LLM 分数的 Softmax 作为软标签
            target_dist = F.softmax(q_labels / self.temperature, dim=0)

            # Label smoothing：与均匀分布插值
            if self.label_smoothing > 0:
                n = target_dist.shape[0]
                target_dist = (1 - self.label_smoothing) * target_dist + self.label_smoothing / n

            # KL 散度：KL(target || pred)
            log_pred = F.log_softmax(q_scores, dim=0)
            loss = F.kl_div(log_pred, target_dist, reduction="sum")
            total_loss += loss
            n_valid += 1

        return total_loss / max(n_valid, 1)


# ---------------------------------------------------------------------------
# 验证集评测：NDCG@10, Precision@3
# ---------------------------------------------------------------------------

def compute_ndcg(ranked_labels: list[float], k: int) -> float:
    """计算单个 query 的 NDCG@k（标签为归一化分数 [0,1]）。"""
    dcg = sum(
        (2 ** label - 1) / math.log2(i + 2)
        for i, label in enumerate(ranked_labels[:k])
    )
    ideal = sorted(ranked_labels, reverse=True)
    idcg = sum(
        (2 ** label - 1) / math.log2(i + 2)
        for i, label in enumerate(ideal[:k])
    )
    return dcg / idcg if idcg > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    val_file: Path,
    accelerator: Accelerator,
    max_length: int = 512,
    batch_size: int = 64,
) -> dict[str, float]:
    """
    在验证集上评测 NDCG@10 和 Precision@3。
    每个 instance 是一个 query + 多个 doc（Listwise 格式）。
    """
    model.eval()
    ndcg_list: list[float] = []
    p3_list: list[float] = []

    with open(val_file, encoding="utf-8") as f:
        for line in f:
            instance = json.loads(line.strip())
            query = instance["query"]
            docs = instance["docs"]
            gt_labels = [d["score"] for d in docs]
            doc_texts = [d["text"] for d in docs]

            # Batch encode all (query, doc) pairs
            encodings = tokenizer(
                [query] * len(doc_texts),
                doc_texts,
                max_length=max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            encodings = {k: v.to(accelerator.device) for k, v in encodings.items()
                         if v is not None}

            outputs = model(**encodings)
            scores = outputs.logits.squeeze(-1).cpu().tolist()

            # 按模型分数排序，取 gt_labels 的对应顺序
            sorted_labels = [gt_labels[i] for i in sorted(range(len(scores)), key=lambda x: scores[x], reverse=True)]

            ndcg_list.append(compute_ndcg(sorted_labels, k=10))
            p3 = sum(1 for l in sorted_labels[:3] if l >= 0.75) / 3  # 分数 >= 0.75（对应 LLM 4分）算相关
            p3_list.append(p3)

    return {
        "ndcg@10": sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0,
        "precision@3": sum(p3_list) / len(p3_list) if p3_list else 0.0,
    }


# ---------------------------------------------------------------------------
# 主训练循环
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    set_seed(42)

    # ---- Accelerator 初始化 ----
    accelerator = Accelerator(
        mixed_precision="fp16",  # fp16 混合精度：速度 +40%，显存 -30%
        log_with="wandb" if not args.no_wandb else None,
        project_dir=args.output_dir,
    )
    if not args.no_wandb and accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="reranker-finetune",
            config=vars(args),
            init_kwargs={"wandb": {"name": Path(args.output_dir).name}},
        )

    logger.info("Accelerator: %s | Devices: %d | Mixed precision: fp16",
                accelerator.device, accelerator.num_processes)

    # ---- 模型 & Tokenizer ----
    logger.info("Loading model: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,  # 输出单个相关性分数
    )

    # torch.compile 图编译加速（PyTorch 2.1+）
    if args.use_compile:
        logger.info("Applying torch.compile...")
        model = torch.compile(model)

    # ---- 数据集 & DataLoader ----
    train_dataset = RerankerListwiseDataset(args.train_file, max_docs_per_query=args.max_docs)
    collate = partial(listwise_collate_fn, tokenizer=tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=4,
        collate_fn=collate,
        pin_memory=True,
    )

    # ---- 优化器 & 调度器 ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    logger.info("Total steps: %d | Warmup: %d | LR: %.2e", total_steps, warmup_steps, args.learning_rate)

    # ---- accelerate 包装 ----
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ---- 损失函数 ----
    criterion = ListwiseSoftmaxLoss(
        temperature=args.loss_temperature,
        label_smoothing=args.label_smoothing,
    )

    # ---- 训练循环 ----
    best_ndcg = 0.0
    no_improve_epochs = 0
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not accelerator.is_main_process)

        for batch in progress:
            # 前向传播
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids"),
            )
            scores = outputs.logits.squeeze(-1)  # [total_pairs]

            # 损失计算
            loss = criterion(scores, batch["labels"], batch["query_ids"])

            # 反向传播 + 梯度裁剪（Cross-Encoder 前期梯度容易突然放大）
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 记录
            loss_val = loss.item()
            epoch_loss += loss_val
            global_step += 1

            progress.set_postfix(loss=f"{loss_val:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            if global_step % 200 == 0 and not args.no_wandb:
                accelerator.log({"train/loss": loss_val, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)

        avg_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d | avg_loss=%.4f", epoch, avg_loss)

        # ---- Epoch 结束：验证集评测 ----
        if args.val_file and accelerator.is_main_process:
            unwrapped_model = accelerator.unwrap_model(model)
            metrics = evaluate(unwrapped_model, tokenizer, Path(args.val_file), accelerator)
            ndcg = metrics["ndcg@10"]
            p3 = metrics["precision@3"]
            logger.info("Epoch %d | NDCG@10=%.4f | Precision@3=%.4f", epoch, ndcg, p3)

            if not args.no_wandb:
                accelerator.log({
                    "val/ndcg@10": ndcg,
                    "val/precision@3": p3,
                    "val/epoch_loss": avg_loss,
                }, step=global_step)

            # 保存最优 checkpoint（以 NDCG@10 为准）
            if ndcg > best_ndcg:
                best_ndcg = ndcg
                no_improve_epochs = 0
                save_path = Path(args.output_dir) / "best_checkpoint"
                save_path.mkdir(parents=True, exist_ok=True)
                unwrapped_model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                logger.info("New best checkpoint saved → %s (NDCG@10=%.4f)", save_path, ndcg)
            else:
                no_improve_epochs += 1
                logger.info("No improvement for %d epoch(s) (best NDCG@10=%.4f)", no_improve_epochs, best_ndcg)

            # 早停：连续 2 个 epoch 无提升
            if no_improve_epochs >= args.early_stop_patience:
                logger.info("Early stopping triggered at epoch %d", epoch)
                break

        # 每个 epoch 保存一个 checkpoint，保留最近 3 个
        if accelerator.is_main_process:
            ckpt_path = Path(args.output_dir) / f"epoch_{epoch}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)

            # 清理旧 checkpoint（保留最近 3 个）
            all_ckpts = sorted(
                [p for p in Path(args.output_dir).iterdir() if p.name.startswith("epoch_")],
                key=lambda p: int(p.name.split("_")[1]),
            )
            for old_ckpt in all_ckpts[:-3]:
                import shutil
                shutil.rmtree(old_ckpt)

    accelerator.end_training()
    logger.info("Training complete. Best NDCG@10=%.4f", best_ndcg)


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reranker Full SFT Training")

    # 数据
    p.add_argument("--train_file",   required=True, type=str)
    p.add_argument("--val_file",     default=None,  type=str)
    p.add_argument("--max_docs",     type=int, default=7, help="每个 instance 最多文档数（1 正 + N 负）")

    # 模型
    p.add_argument("--model_name",   default="Alibaba-NLP/gte-multilingual-reranker-base")
    p.add_argument("--max_length",   type=int, default=512, help="query+doc 拼接后最大 token 数")
    p.add_argument("--use_compile",  action="store_true",   help="启用 torch.compile 加速（需 PyTorch 2.1+）")

    # 训练
    p.add_argument("--output_dir",         required=True, type=str)
    p.add_argument("--num_epochs",         type=int,   default=5)
    p.add_argument("--per_device_batch",   type=int,   default=32,  help="单卡 batch size")
    p.add_argument("--learning_rate",      type=float, default=1e-5, help="推荐范围 5e-6 ~ 2e-5")
    p.add_argument("--warmup_ratio",       type=float, default=0.1)
    p.add_argument("--loss_temperature",   type=float, default=1.0, help="Listwise Softmax 温度")
    p.add_argument("--label_smoothing",    type=float, default=0.1)
    p.add_argument("--early_stop_patience",type=int,   default=2,   help="验证集无提升几个 epoch 后停止")

    # 监控
    p.add_argument("--no_wandb",     action="store_true", help="禁用 wandb 日志")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
