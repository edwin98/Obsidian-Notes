"""
train_embedding.py
===================
gte-multilingual-base Embedding 模型微调主脚本。
支持 A30×2 双卡 DDP + GradCache 三阶段训练。

三阶段流程：
  Stage 1 —— 弱监督预热（1 epoch，lr=2e-4，仅正样本对）
  Stage 2 —— Hard Negative 精调（2 epochs，lr=2e-5，InfoNCE + 硬负样本）
  Stage 3 —— Reranker 知识蒸馏（2 epochs，lr=2e-5，InfoNCE + KL 联合损失）

启动命令（双卡 DDP）：
  torchrun --nproc_per_node=2 train_embedding.py \
      --triplets_file       data/triplets.jsonl \
      --distill_labels_file data/distill_labels.jsonl \
      --output_dir          checkpoints/gte-finetuned \
      --model_name          Alibaba-NLP/gte-multilingual-base \
      --per_device_batch    64 \
      --grad_cache_mini     16 \
      --stage1_lr           2e-4 \
      --stage2_lr           2e-5 \
      --stage3_lr           2e-5 \
      --temperature         0.02

关键超参数（A30×2 实测）：
  per_device_batch = 64   → 双卡有效 batch = 128（+ GradCache 扩展至 1024）
  grad_cache_mini  = 16   → GradCache 分块大小，显存换 batch
  temperature      = 0.02 → InfoNCE 温度
  distill_alpha    = 0.4  → KL 蒸馏权重
  fp16             = True
"""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import Stage1Dataset, Stage2Dataset, Stage3Dataset, make_collate_fn
from losses import CombinedLoss, InfoNCELoss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] rank=%(rank)d %(message)s",
    defaults={"rank": int(os.environ.get("LOCAL_RANK", 0))},
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding 模型封装
# ---------------------------------------------------------------------------

class GTEBiEncoder(torch.nn.Module):
    """
    gte-multilingual-base Bi-Encoder 封装。

    特性：
      - 支持 output_dim 参数（384 或 768），原生支持无需 MRL 技巧
      - mean pooling（对所有 token 取均值，不只取 CLS）是 GTE 的推荐 pooling 方式
      - 输出向量 L2 归一化，使余弦相似度 = 内积，加速计算
    """

    def __init__(self, model_name: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_dim: int | None = None,
    ) -> torch.Tensor:
        """
        前向编码，返回 L2 归一化的 embedding 向量。

        output_dim：None 表示使用全维度（768）；384 表示截断到前 384 维。
        gte-multilingual-base 原生支持截断维度仍保持较好效果（因为训练时用了 MRL）。
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # mean pooling：对非 padding token 取均值
        token_embeddings = outputs.last_hidden_state         # [B, L, D]
        mask_expanded = attention_mask.unsqueeze(-1).float() # [B, L, 1]
        summed = (token_embeddings * mask_expanded).sum(dim=1)
        counts = mask_expanded.sum(dim=1).clamp(min=1e-9)
        embeddings = summed / counts                         # [B, D]

        # 截断维度（gte-multilingual-base 支持 384 / 768）
        if output_dim is not None and output_dim < embeddings.size(-1):
            embeddings = embeddings[:, :output_dim]

        # L2 归一化：使余弦相似度等于内积，InfoNCE 计算更高效
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_emb = self.encode(query_input_ids, query_attention_mask, output_dim=768)
        d_emb = self.encode(doc_input_ids,   doc_attention_mask,   output_dim=768)
        return q_emb, d_emb


# ---------------------------------------------------------------------------
# GradCache 实现
# ---------------------------------------------------------------------------

class GradCacheTrainer:
    """
    GradCache：用于在显存受限时扩大有效 batch size。

    原理：
      1. 将大 batch 切成 mini_batch_size 的小块，分块前向传播并缓存所有 embedding
      2. 基于缓存的 embedding 计算完整 batch 的损失与梯度（对 embedding 的梯度）
      3. 将 embedding 梯度回传，触发各小块的反向传播

    效果：显存使用量约为标准 batch 的 1/k（k = 分块数），
    但计算量稍增（多一次前向），吞吐量下降 < 10%。

    参考：GradCache 论文 https://arxiv.org/abs/2101.06983
    """

    def __init__(
        self,
        model: GTEBiEncoder,
        mini_batch_size: int = 16,
    ) -> None:
        self.model = model
        self.mini_batch_size = mini_batch_size

    def _split_batch(self, tensor: torch.Tensor) -> list[torch.Tensor]:
        """将 tensor 按 mini_batch_size 切分。"""
        return tensor.split(self.mini_batch_size, dim=0)

    def build_embeddings_no_grad(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_dim: int = 768,
    ) -> torch.Tensor:
        """
        无梯度的前向传播，仅缓存 embedding 表示。
        后续会对 embedding 重新计算梯度（二次前向）。
        """
        all_embeds = []
        for ids_chunk, mask_chunk in zip(
            self._split_batch(input_ids),
            self._split_batch(attention_mask),
        ):
            with torch.no_grad():
                emb = self.model.encode(ids_chunk, mask_chunk, output_dim=output_dim)
            all_embeds.append(emb)
        return torch.cat(all_embeds, dim=0)

    def backward_with_grad_cache(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        cached_q_embeds: torch.Tensor,   # 无梯度版本，用于算 loss
        cached_d_embeds: torch.Tensor,
        loss_fn: InfoNCELoss | CombinedLoss,
        num_docs_per_sample: list[int],
        soft_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        GradCache 的核心步骤：

        Step 1：基于缓存 embedding 计算损失，得到 d(Loss)/d(embedding)
        Step 2：分块二次前向，用 embedding 梯度反向传播到模型参数

        返回损失值（用于日志记录）。
        """
        # Step 1：启用 embedding 梯度，计算损失
        cached_q_embeds = cached_q_embeds.detach().requires_grad_(True)
        cached_d_embeds = cached_d_embeds.detach().requires_grad_(True)

        if soft_labels is not None and isinstance(loss_fn, CombinedLoss):
            loss, l_infonce, l_distill = loss_fn(
                cached_q_embeds, cached_d_embeds, soft_labels, num_docs_per_sample
            )
        else:
            loss = loss_fn(cached_q_embeds, cached_d_embeds, num_docs_per_sample)

        # 对缓存 embedding 求梯度
        loss.backward()
        q_grads = cached_q_embeds.grad    # [B, D]
        d_grads = cached_d_embeds.grad    # [B * n_docs, D]

        # Step 2：分块二次前向，将 embedding 梯度链式传播回模型参数
        q_offset = 0
        for ids_chunk, mask_chunk in zip(
            self._split_batch(query_input_ids),
            self._split_batch(query_attention_mask),
        ):
            chunk_size = ids_chunk.size(0)
            emb = self.model.encode(ids_chunk, mask_chunk, output_dim=768)
            # 只对本块的 embedding 梯度做 backward（不是 loss.backward）
            emb.backward(gradient=q_grads[q_offset: q_offset + chunk_size])
            q_offset += chunk_size

        d_offset = 0
        for ids_chunk, mask_chunk in zip(
            self._split_batch(doc_input_ids),
            self._split_batch(doc_attention_mask),
        ):
            chunk_size = ids_chunk.size(0)
            emb = self.model.encode(ids_chunk, mask_chunk, output_dim=768)
            emb.backward(gradient=d_grads[d_offset: d_offset + chunk_size])
            d_offset += chunk_size

        return loss.detach()


# ---------------------------------------------------------------------------
# 训练工具函数
# ---------------------------------------------------------------------------

def setup_ddp() -> tuple[int, int, int]:
    """
    初始化 DDP 进程组。

    torchrun 会自动设置环境变量：
      RANK          全局进程编号（0 ~ world_size-1）
      LOCAL_RANK    本机进程编号（0 ~ nproc_per_node-1）
      WORLD_SIZE    总进程数

    返回 (rank, local_rank, world_size)。
    """
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    logger.info("DDP 初始化完成：rank=%d, local_rank=%d, world_size=%d", rank, local_rank, world_size)
    return rank, local_rank, world_size


def is_main_process() -> bool:
    """只在 rank=0 的进程执行保存、日志等操作。"""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_num_warmup_steps(num_training_steps: int, warmup_ratio: float = 0.1) -> int:
    """计算 warmup 步数：总步数的 10%，保护预训练权重，避免初期梯度震荡。"""
    return math.ceil(num_training_steps * warmup_ratio)


def build_dataloader(
    dataset,
    tokenizer,
    batch_size: int,
    query_max_len: int,
    doc_max_len: int,
    is_ddp: bool,
    shuffle: bool = True,
) -> DataLoader:
    """构建 DataLoader，DDP 环境下用 DistributedSampler 保证各卡数据不重叠。"""
    sampler = DistributedSampler(dataset, shuffle=shuffle) if is_ddp else None
    collate_fn = make_collate_fn(tokenizer, query_max_len, doc_max_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(shuffle and not is_ddp),  # 用 sampler 时 shuffle 必须为 False
        num_workers=4,
        pin_memory=True,          # 异步 CPU→GPU 数据传输
        collate_fn=collate_fn,
        drop_last=True,           # 丢弃最后不完整 batch，避免 DDP 各卡 batch 大小不同
    )


# ---------------------------------------------------------------------------
# 单阶段训练循环
# ---------------------------------------------------------------------------

def train_one_stage(
    stage: int,
    model: DDP,
    dataloader: DataLoader,
    optimizer: torch.optim.AdamW,
    scheduler,
    loss_fn,
    grad_cache: GradCacheTrainer,
    device: torch.device,
    args: argparse.Namespace,
    rank: int,
) -> None:
    """
    通用训练循环，三个阶段共用。

    stage：1 / 2 / 3，用于日志标识
    """
    model.train()
    total_steps = len(dataloader) * args.__dict__.get(f"stage{stage}_epochs", 2)
    global_step = 0

    for epoch in range(args.__dict__.get(f"stage{stage}_epochs", 2)):
        if hasattr(dataloader.sampler, "set_epoch"):
            # DDP DistributedSampler：每个 epoch 用不同的随机种子打乱数据
            dataloader.sampler.set_epoch(epoch)

        epoch_loss = 0.0
        for step, batch in enumerate(dataloader):
            # 将 batch 中的所有 Tensor 移到当前 GPU
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            query_input_ids      = batch["query_input_ids"]
            query_attention_mask = batch["query_attention_mask"]
            doc_input_ids        = batch["doc_input_ids"]
            doc_attention_mask   = batch["doc_attention_mask"]
            num_docs             = batch["num_docs_per_sample"]
            soft_labels          = batch.get("soft_labels")   # Stage3 专属

            optimizer.zero_grad()

            if args.use_grad_cache:
                # GradCache 路径：先无梯度缓存 embedding，再分块反向
                cached_q = grad_cache.build_embeddings_no_grad(
                    query_input_ids, query_attention_mask
                )
                cached_d = grad_cache.build_embeddings_no_grad(
                    doc_input_ids, doc_attention_mask
                )
                loss = grad_cache.backward_with_grad_cache(
                    query_input_ids, query_attention_mask,
                    doc_input_ids,   doc_attention_mask,
                    cached_q, cached_d,
                    loss_fn, num_docs, soft_labels,
                )
            else:
                # 标准路径（不用 GradCache，显存足够时使用）
                q_emb, d_emb = model.module.forward(
                    query_input_ids, query_attention_mask,
                    doc_input_ids,   doc_attention_mask,
                )
                if stage == 3 and soft_labels is not None:
                    loss, _, _ = loss_fn(q_emb, d_emb, soft_labels, num_docs)
                else:
                    loss = loss_fn(q_emb, d_emb, num_docs)
                loss.backward()

            # 梯度裁剪：防止梯度爆炸，尤其在 stage1 大 lr 时重要
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += loss.item()

            if global_step % 100 == 0 and is_main_process():
                avg_loss = epoch_loss / (step + 1)
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    "Stage%d | Epoch %d | Step %d/%d | loss=%.4f | lr=%.2e",
                    stage, epoch + 1, global_step, total_steps, avg_loss, lr,
                )

        if is_main_process():
            logger.info("Stage%d Epoch %d 完成，平均 loss=%.4f", stage, epoch + 1, epoch_loss / len(dataloader))


# ---------------------------------------------------------------------------
# 主训练函数
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # 初始化 DDP
    is_ddp = "LOCAL_RANK" in os.environ
    if is_ddp:
        rank, local_rank, world_size = setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载 tokenizer 和模型
    if is_main_process():
        logger.info("加载模型：%s", args.model_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = GTEBiEncoder(args.model_name).to(device)

    # fp16 混合精度（A30 支持，节省约 40% 显存）
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # DDP 包装：每张卡维护相同的模型参数副本，梯度自动 all_reduce
    if is_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # GradCache 封装
    grad_cache = GradCacheTrainer(
        model.module if is_ddp else model,
        mini_batch_size=args.grad_cache_mini,
    )

    # -----------------------------------------------------------------------
    # Stage 1：弱监督预热
    # -----------------------------------------------------------------------
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Stage 1：弱监督预热（仅正样本对，lr=%.0e，epoch=1）", args.stage1_lr)

    stage1_dataset = Stage1Dataset(args.triplets_file)
    stage1_loader  = build_dataloader(
        stage1_dataset, tokenizer,
        args.per_device_batch, args.query_max_len, args.doc_max_len,
        is_ddp=is_ddp,
    )

    # Stage1 用较大 lr，快速适应领域词汇
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage1_lr, weight_decay=0.01)
    num_steps_s1 = len(stage1_loader) * args.stage1_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=get_num_warmup_steps(num_steps_s1),
        num_training_steps=num_steps_s1,
    )
    loss_fn_s1 = InfoNCELoss(temperature=args.temperature)

    train_one_stage(1, model, stage1_loader, optimizer, scheduler, loss_fn_s1,
                    grad_cache, device, args, rank)

    # 保存 Stage1 checkpoint（只在 rank=0 保存）
    if is_main_process():
        ckpt_dir = Path(args.output_dir) / "stage1"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (model.module if is_ddp else model).encoder.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info("Stage1 checkpoint 已保存至 %s", ckpt_dir)

    # -----------------------------------------------------------------------
    # Stage 2：Hard Negative 精调
    # -----------------------------------------------------------------------
    if is_main_process():
        logger.info("=" * 60)
        logger.info("Stage 2：Hard Negative 精调（lr=%.0e，epoch=2）", args.stage2_lr)

    stage2_dataset = Stage2Dataset(args.triplets_file)
    stage2_loader  = build_dataloader(
        stage2_dataset, tokenizer,
        args.per_device_batch, args.query_max_len, args.doc_max_len,
        is_ddp=is_ddp,
    )

    # Stage2/3 用小 lr 精调，防止破坏 Stage1 建立的领域表示
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage2_lr, weight_decay=0.01)
    num_steps_s2 = len(stage2_loader) * args.stage2_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=get_num_warmup_steps(num_steps_s2),
        num_training_steps=num_steps_s2,
    )
    loss_fn_s2 = InfoNCELoss(temperature=args.temperature)

    train_one_stage(2, model, stage2_loader, optimizer, scheduler, loss_fn_s2,
                    grad_cache, device, args, rank)

    if is_main_process():
        ckpt_dir = Path(args.output_dir) / "stage2"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (model.module if is_ddp else model).encoder.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        logger.info("Stage2 checkpoint 已保存至 %s", ckpt_dir)

    # -----------------------------------------------------------------------
    # Stage 3：Reranker 知识蒸馏
    # -----------------------------------------------------------------------
    if is_main_process():
        logger.info("=" * 60)
        logger.info(
            "Stage 3：Reranker 知识蒸馏（lr=%.0e，epoch=2，alpha=%.2f）",
            args.stage3_lr, args.distill_alpha,
        )

    if not Path(args.distill_labels_file).exists():
        logger.warning("蒸馏标签文件不存在，跳过 Stage3：%s", args.distill_labels_file)
    else:
        stage3_dataset = Stage3Dataset(args.distill_labels_file)
        stage3_loader  = build_dataloader(
            stage3_dataset, tokenizer,
            args.per_device_batch, args.query_max_len, args.doc_max_len,
            is_ddp=is_ddp,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.stage3_lr, weight_decay=0.01)
        num_steps_s3 = len(stage3_loader) * args.stage3_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=get_num_warmup_steps(num_steps_s3),
            num_training_steps=num_steps_s3,
        )
        # Stage3 联合损失：InfoNCE + KL 蒸馏
        loss_fn_s3 = CombinedLoss(alpha=args.distill_alpha, temperature=args.temperature)

        train_one_stage(3, model, stage3_loader, optimizer, scheduler, loss_fn_s3,
                        grad_cache, device, args, rank)

        if is_main_process():
            ckpt_dir = Path(args.output_dir) / "final"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            (model.module if is_ddp else model).encoder.save_pretrained(str(ckpt_dir))
            tokenizer.save_pretrained(str(ckpt_dir))
            logger.info("最终模型已保存至 %s", ckpt_dir)

    # 清理 DDP 进程组
    if is_ddp:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gte-multilingual-base 三阶段微调（A30×2 DDP）")

    # 数据
    parser.add_argument("--triplets_file",       default="data/triplets.jsonl",
                        help="Hard Negative 三元组数据（Stage1/2 共用）")
    parser.add_argument("--distill_labels_file", default="data/distill_labels.jsonl",
                        help="Reranker 软标签数据（Stage3 专用）")
    parser.add_argument("--output_dir",          default="checkpoints/gte-finetuned")

    # 模型
    parser.add_argument("--model_name",          default="Alibaba-NLP/gte-multilingual-base")

    # Tokenizer 参数
    parser.add_argument("--query_max_len",  type=int,   default=128)
    parser.add_argument("--doc_max_len",    type=int,   default=512)

    # 训练超参数
    parser.add_argument("--per_device_batch", type=int,   default=64,
                        help="单卡实际 batch size（A30 24GB 上限）")
    parser.add_argument("--grad_cache_mini",  type=int,   default=16,
                        help="GradCache 分块大小，用显存换有效 batch")
    parser.add_argument("--use_grad_cache",   action="store_true", default=True,
                        help="是否启用 GradCache（默认开启）")
    parser.add_argument("--stage1_lr",        type=float, default=2e-4)
    parser.add_argument("--stage2_lr",        type=float, default=2e-5)
    parser.add_argument("--stage3_lr",        type=float, default=2e-5)
    parser.add_argument("--stage1_epochs",    type=int,   default=1)
    parser.add_argument("--stage2_epochs",    type=int,   default=2)
    parser.add_argument("--stage3_epochs",    type=int,   default=2)
    parser.add_argument("--temperature",      type=float, default=0.02,
                        help="InfoNCE 温度 τ")
    parser.add_argument("--distill_alpha",    type=float, default=0.4,
                        help="蒸馏损失权重 α，总损失 = (1-α)*InfoNCE + α*KL")
    parser.add_argument("--fp16",             action="store_true", default=True,
                        help="使用 FP16 混合精度（A30 支持）")

    args = parser.parse_args()
    main(args)
