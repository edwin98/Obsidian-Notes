"""训练数学的教学实现：先用 CPU 小张量理解，再读真实训练入口。

对应文档 08 的张量、因果位移与 DPO 章节。这里只计算，不加载模型，
不下载权重，也不会把工具返回当作助手监督目标。
"""

import torch.nn.functional as F


def build_labels(
    input_ids: list[int], assistant_mask: list[int], attention_mask: list[int]
) -> list[int]:
    """保留助手目标；prompt、工具返回和真正的 padding 都置为 -100。"""
    if not (len(input_ids) == len(assistant_mask) == len(attention_mask)):
        raise ValueError("token、助手掩码和输入掩码必须等长")
    if any(value not in (0, 1) for value in assistant_mask + attention_mask):
        raise ValueError("掩码只能包含0或1")
    labels = [
        token if assistant and actual else -100
        for token, assistant, actual in zip(
            input_ids, assistant_mask, attention_mask, strict=True
        )
    ]
    if not any(token != -100 for token in labels[1:]):
        raise ValueError("样本没有可用于下一token预测的助手目标")
    return labels


def _shift(logits, labels):
    """logits[t] 预测 labels[t+1]，只在这个函数里移动一次。"""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("需要 logits[B,L,V] 与 labels[B,L]")
    targets = labels[:, 1:].contiguous()
    mask = targets != -100
    if (mask.sum(dim=-1) == 0).any().item():
        raise ValueError("每条序列都必须有有效目标")
    if ((targets[mask] < 0) | (targets[mask] >= logits.shape[-1])).any().item():
        raise ValueError("目标token超出词表")
    return logits[:, :-1, :].contiguous(), targets, mask


def causal_token_loss(logits, labels):
    """SFT：按整批有效目标 token 平均，不把各微批次平均误当同一目标。"""
    shifted, targets, mask = _shift(logits, labels)
    loss_sum = F.cross_entropy(
        shifted.reshape(-1, shifted.shape[-1]),
        targets.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    count = mask.sum()
    return loss_sum / count, count


def sequence_log_probs(logits, labels):
    """DPO：只对候选位置求和；prompt 只是条件，不加入候选似然。"""
    shifted, targets, mask = _shift(logits, labels)
    # gather 不接受 -100，因此先将不参与评分的位置临时改为合法ID。
    safe_targets = targets.masked_fill(~mask, 0)
    logps = F.log_softmax(shifted.float(), dim=-1)
    selected = logps.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
    return selected.masked_fill(~mask, 0).sum(dim=-1)


def dpo_pair_loss(
    policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta=0.1
):
    """标准 sigmoid DPO；参考分布冻结，梯度只流向 policy。"""
    if beta <= 0:
        raise ValueError("beta必须大于0")
    values = (policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    if any(value.shape != values[0].shape for value in values):
        raise ValueError("每对候选的四个序列log probability必须同形")
    preferred_gain = policy_chosen - reference_chosen.detach()
    rejected_gain = policy_rejected - reference_rejected.detach()
    return -F.logsigmoid(beta * (preferred_gain - rejected_gain)).mean()


def inspect_training_batch(batch, tokenizer=None):
    """检查训练器整理后的真实批次，而不是只检查原始JSON看起来正确。"""
    labels, attention = batch["labels"], batch["attention_mask"]
    if labels.shape != batch["input_ids"].shape or labels.shape != attention.shape:
        raise ValueError("批次张量形状不匹配")
    if (labels[attention == 0] != -100).any().item():
        raise ValueError("padding位置不应参与损失")
    counts = (labels[:, 1:] != -100).sum(dim=-1)
    if (counts == 0).any().item():
        raise ValueError("截断或模板处理后出现零有效目标")
    result = {"shape": list(labels.shape), "target_tokens_per_sample": counts.tolist()}
    if tokenizer:
        # 只在用户主动检查样本时解码。它可暴露误把工具结果作为目标的问题。
        result["first_target"] = tokenizer.decode(labels[0][labels[0] != -100].tolist())
    return result
