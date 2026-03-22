"""
losses.py
==========
训练损失函数。

  InfoNCELoss     —— 对比学习主损失（Stage1 / Stage2）
  DistillLoss     —— KL 散度蒸馏损失（Stage3）
  CombinedLoss    —— InfoNCE + KL 联合损失（Stage3）

所有损失均支持 DDP（DistributedDataParallel）环境，
跨卡 In-batch Negatives 通过 GatherLayer 在 all_gather 后统一计算。
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 跨卡梯度同步工具
# ---------------------------------------------------------------------------

class GatherLayer(torch.autograd.Function):
    """
    All-gather + 梯度回传。

    DDP 下每张卡只有本地 batch 的 embedding，
    用 all_gather 收集所有卡的 embedding 后，
    才能计算跨卡的 In-batch Negatives，使有效负样本数 = global_batch_size - 1。

    关键：前向 all_gather，反向 all_reduce 梯度（等价于只对本卡的梯度负责）。
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> tuple[torch.Tensor, ...]:
        ctx.save_for_backward(tensor)
        # 收集所有进程的 tensor，返回 tuple[Tensor] 长度 = world_size
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        return tuple(gathered)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        (tensor,) = ctx.saved_tensors
        # 将所有卡的梯度求和，归还给当前卡的 tensor
        grad = torch.stack(grads, dim=0).sum(dim=0)
        return grad


def gather_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """
    在 DDP 下收集所有卡的 embedding 并拼接。
    单卡（非 DDP）环境直接返回原 tensor。
    """
    if not dist.is_available() or not dist.is_initialized():
        return embeddings

    gathered = GatherLayer.apply(embeddings)      # tuple of Tensor
    return torch.cat(gathered, dim=0)             # [world_size * B, D]


# ---------------------------------------------------------------------------
# InfoNCE 对比学习损失
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    多正样本 InfoNCE 损失（也称 NT-Xent / SimCSE 损失）。

    原理：
      对每个 query，最大化其与对应正样本文档的相似度，
      同时最小化与同 batch 内所有其他文档（in-batch negatives + hard negatives）的相似度。

      Loss = -log( exp(sim(q, d+) / τ) / Σ exp(sim(q, di) / τ) )

    参数：
      temperature τ = 0.02（本系统实际取值）
        越小梯度越集中在难分样本上，但训练越不稳定
        越大则等同于退化为 MSE，学不到对比关系

    DDP 支持：
      通过 GatherLayer 收集所有卡的 doc embeddings，
      使 effective in-batch negatives = global_batch_size - 1 = 1023（双卡×512）
    """

    def __init__(self, temperature: float = 0.02) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeds: torch.Tensor,    # [B, D]，query 向量（已 L2 归一化）
        doc_embeds: torch.Tensor,      # [B * num_docs, D]，所有文档向量
        num_docs_per_sample: list[int],  # 每条样本的文档数（第一个为正样本）
    ) -> torch.Tensor:
        """
        计算 InfoNCE 损失。

        当 num_docs_per_sample 全为 1（Stage1）时，
        退化为标准的 in-batch negatives 对比损失。

        当 num_docs_per_sample > 1（Stage2）时，
        每条样本的第一个文档为正样本，其余为 hard negatives。
        """
        # DDP：收集所有卡的文档向量，扩大负样本池
        all_doc_embeds = gather_embeddings(doc_embeds)   # [G * B * num_docs, D]

        batch_size = query_embeds.size(0)
        total_loss = torch.tensor(0.0, device=query_embeds.device)

        doc_offset = 0
        all_doc_offset = 0  # 全局文档偏移（用于定位正样本在 all_doc_embeds 中的位置）

        # 计算全局偏移量（DDP 下需要知道本卡正样本在全局中的位置）
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        global_doc_offset_base = local_rank * sum(num_docs_per_sample)

        for i in range(batch_size):
            n_docs = num_docs_per_sample[i]
            q = query_embeds[i]                          # [D]
            local_docs = doc_embeds[doc_offset: doc_offset + n_docs]  # [n_docs, D]

            # 正样本：每条样本的第一个文档
            pos_doc = local_docs[0]                      # [D]

            # 计算 query 与全局所有文档的相似度（用于分母）
            # 全局文档 = 本 batch 所有样本的 positive + hard negatives（所有卡）
            sim_all = (q @ all_doc_embeds.T) / self.temperature  # [N_all]

            # 正样本在全局文档中的位置
            pos_global_idx = global_doc_offset_base + doc_offset
            loss_i = -sim_all[pos_global_idx] + torch.logsumexp(sim_all, dim=0)

            total_loss = total_loss + loss_i
            doc_offset += n_docs
            all_doc_offset += n_docs

        return total_loss / batch_size


# ---------------------------------------------------------------------------
# KL 散度蒸馏损失
# ---------------------------------------------------------------------------

class DistillLoss(nn.Module):
    """
    Reranker → Bi-Encoder 知识蒸馏损失（KL 散度）。

    原理：
      教师（Reranker）对 [positive, neg1, neg2, ...] 打分后经 softmax(score/T) 得软标签，
      学生（Bi-Encoder）对同一组文档计算余弦相似度后经 softmax(sim/τ) 得预测分布，
      最小化两者的 KL 散度：

        L_distill = KL(p_teacher || p_student)
                  = Σ p_teacher * log(p_teacher / p_student)

    注意：soft_labels 可能存在 padding（0 值），需要 mask 掉。
    """

    def forward(
        self,
        query_embeds: torch.Tensor,      # [B, D]
        doc_embeds: torch.Tensor,        # [B * n_docs, D]（flatten）
        soft_labels: torch.Tensor,       # [B, max_n_docs]（含 padding）
        num_docs_per_sample: list[int],
        student_temperature: float = 0.02,
    ) -> torch.Tensor:
        batch_size = query_embeds.size(0)
        total_loss = torch.tensor(0.0, device=query_embeds.device)

        doc_offset = 0
        for i in range(batch_size):
            n_docs = num_docs_per_sample[i]
            q = query_embeds[i]                        # [D]
            docs = doc_embeds[doc_offset: doc_offset + n_docs]  # [n_docs, D]

            # 学生分布：Bi-Encoder 余弦相似度经 softmax
            sim = (q @ docs.T) / student_temperature   # [n_docs]
            p_student = F.log_softmax(sim, dim=-1)     # log 概率，用于 KLDivLoss

            # 教师分布（从预计算软标签取前 n_docs 列，去掉 padding）
            p_teacher = soft_labels[i, :n_docs]        # [n_docs]
            # 确保教师分布归一化（数值安全）
            p_teacher = p_teacher / (p_teacher.sum() + 1e-9)

            # KL(p_teacher || p_student) = Σ p_t * (log p_t - log p_s)
            # PyTorch KLDivLoss 期望输入为 log_softmax(student)，目标为 p_teacher
            loss_i = F.kl_div(p_student, p_teacher, reduction="sum")
            total_loss = total_loss + loss_i
            doc_offset += n_docs

        return total_loss / batch_size


# ---------------------------------------------------------------------------
# Stage 3 联合损失
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Stage3 联合损失 = (1 - alpha) * InfoNCE + alpha * KL_Distill

    alpha 控制蒸馏信号的强度：
      - InfoNCE 保证正负样本的基本区分能力
      - KL 蒸馏注入 Reranker 的细粒度排序知识（否定词、数字匹配等）
      - alpha 通常取 0.3~0.5，推荐 0.4（在验证集上搜索确定）
    """

    def __init__(self, alpha: float = 0.4, temperature: float = 0.02) -> None:
        super().__init__()
        self.alpha = alpha
        self.infonce = InfoNCELoss(temperature=temperature)
        self.distill = DistillLoss()

    def forward(
        self,
        query_embeds: torch.Tensor,
        doc_embeds: torch.Tensor,
        soft_labels: torch.Tensor,
        num_docs_per_sample: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        返回：(total_loss, infonce_loss, distill_loss)
        分开返回便于 TensorBoard 监控各分量。
        """
        loss_infonce = self.infonce(query_embeds, doc_embeds, num_docs_per_sample)
        loss_distill = self.distill(
            query_embeds, doc_embeds, soft_labels, num_docs_per_sample
        )
        total = (1 - self.alpha) * loss_infonce + self.alpha * loss_distill
        return total, loss_infonce, loss_distill
