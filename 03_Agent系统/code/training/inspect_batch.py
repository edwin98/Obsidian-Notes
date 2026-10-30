"""无需模型、密钥或GPU的微调数学演练：python -m training.inspect_batch。"""

import json
import math

import torch

from training.objectives import causal_token_loss, dpo_pair_loss, sequence_log_probs


def demonstrate():
    # 两个目标概率为0.8与0.5，最后一行logits没有“下一个token”，会被位移丢弃。
    logits = torch.tensor(
        [[[math.log(0.8), math.log(0.2)], [math.log(0.5), math.log(0.5)], [0.0, 0.0]]],
        requires_grad=True,
    )
    labels = torch.tensor([[-100, 0, 1]])
    loss, count = causal_token_loss(logits, labels)
    sequence = sequence_log_probs(logits, labels)
    loss.backward()  # 演示梯度；没有优化器，也没有更新任何真实模型。
    dpo = dpo_pair_loss(
        torch.tensor([-80.0]),
        torch.tensor([-100.0]),
        torch.tensor([-82.0]),
        torch.tensor([-95.0]),
    )
    return {
        "scope": "CPU小张量演练，不下载模型、不执行GPU训练",
        "logits_shape": list(logits.shape),
        "labels": labels.tolist(),
        "valid_target_tokens": count.item(),
        "sft_loss": loss.item(),
        "sequence_log_probability": sequence.item(),
        "dpo_loss": dpo.item(),
        "gradient_finite": torch.isfinite(logits.grad).all().item(),
    }


if __name__ == "__main__":
    print(json.dumps(demonstrate(), ensure_ascii=False, indent=2))
