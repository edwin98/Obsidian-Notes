"""随机生成变长整数序列，训练模型按原顺序复制，并观察真正的自回归结果。"""

import argparse
import time

import torch
from torch import Tensor, nn

from model import BOS, EOS, PAD, Config, Transformer, greedy_decode


def make_batch(
    batch_size: int,
    max_tokens: int,
    vocab_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor]:
    """内容 token 从 3 开始；0/1/2 分别留给 PAD/BOS/EOS。

    src:      [a, b, c, EOS, PAD]
    full_tgt: [BOS, a, b, c, EOS, PAD]
    tgt_in:   [BOS, a, b, c, EOS]     full_tgt[:, :-1]
    labels:   [a, b, c, EOS, PAD]     full_tgt[:, 1:]
    labels 中的 PAD 不计算损失，其余位置（包括 EOS）都要学习。
    """
    if batch_size < 1 or max_tokens < 1 or vocab_size <= EOS + 1:
        raise ValueError("批量、内容长度必须为正数，词表需含内容 token")
    lengths = torch.randint(1, max_tokens + 1, (batch_size,), generator=generator)
    src = torch.full((batch_size, max_tokens + 1), PAD, dtype=torch.long)
    full_target = torch.full((batch_size, max_tokens + 2), PAD, dtype=torch.long)
    full_target[:, 0] = BOS
    for row, length in enumerate(lengths.tolist()):
        content = torch.randint(3, vocab_size, (length,), generator=generator)
        src[row, :length] = content
        src[row, length] = EOS
        full_target[row, 1 : length + 1] = content
        full_target[row, length + 1] = EOS
    return src, full_target[:, :-1], full_target[:, 1:]


def sequence_tokens(row: Tensor) -> list[int]:
    """保留首个 EOS，去掉 BOS 与填充，便于比较实际输出。"""
    result = []
    for token in row.tolist():
        if token == EOS:
            result.append(token)
            break
        if token not in (PAD, BOS):
            result.append(token)
    return result


@torch.no_grad()
def evaluate(
    model: Transformer, batch: tuple[Tensor, Tensor, Tensor]
) -> tuple[float, float, float, Tensor]:
    """固定验证集：同时报告 Teacher Forcing 与自回归的差异。"""
    src, tgt, labels = batch
    model.eval()
    logits = model(src, tgt)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=PAD
    )
    valid = labels != PAD
    correct = (logits.argmax(-1) == labels) & valid
    token_accuracy = correct.sum().item() / valid.sum().item()
    predictions = greedy_decode(model, src, max_new_tokens=labels.size(1))
    # 比较原始 token 序列，包括 EOS；多余 BOS/PAD 不能被当作正确预测。
    exact = 0
    for predicted, expected in zip(predictions[:, 1:].tolist(), labels.tolist()):
        expected = [token for token in expected if token != PAD]
        exact += predicted[: len(expected)] == expected
    return loss.item(), token_accuracy, exact / src.size(0), predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=6)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()
    if (
        min(
            args.steps,
            args.batch_size,
            args.max_tokens,
            args.eval_size,
            args.threads,
            args.log_every,
        )
        < 1
        or args.lr <= 0
    ):
        parser.error("步数、批量、长度、线程、日志间隔和学习率必须为正数")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    config = Config(max_len=args.max_tokens + 1)
    model = Transformer(config)  # 默认 CPU，不下载模型或数据，不写检查点
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    train_generator = torch.Generator().manual_seed(args.seed + 1)
    eval_generator = torch.Generator().manual_seed(args.seed + 2)
    validation = make_batch(
        args.eval_size, args.max_tokens, config.vocab_size, eval_generator
    )
    print(f"参数量：{sum(p.numel() for p in model.parameters()):,}；设备：CPU")
    print("特殊 token：PAD=0, BOS=1, EOS=2；内容 token=3..15")
    initial_loss, initial_accuracy, initial_exact, _ = evaluate(model, validation)
    print(
        f"训练前：验证 loss={initial_loss:.4f}，"
        f"TF token 准确率={initial_accuracy:.1%}，自回归整句正确率={initial_exact:.1%}"
    )
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        src, tgt_in, labels = make_batch(
            args.batch_size, args.max_tokens, config.vocab_size, train_generator
        )
        # Teacher Forcing：一次前向同时预测所有位置，因果 mask 阻止偷看未来。
        logits = model(src, tgt_in)  # [B,T,V]，无需提前 softmax
        loss = criterion(logits.reshape(-1, config.vocab_size), labels.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # 梯度从 LM Head 一直传播到可训练 Embedding
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if step % args.log_every == 0 or step == args.steps:
            val_loss, accuracy, exact, _ = evaluate(model, validation)
            print(
                f"step={step:4d} train_loss={loss.item():.4f} "
                f"val_loss={val_loss:.4f} TF_token_acc={accuracy:.1%} "
                f"AR_exact_acc={exact:.1%}"
            )
    print(f"训练耗时：{time.perf_counter() - started:.1f} 秒")
    _, _, _, predictions = evaluate(model, validation)
    for src, predicted in zip(validation[0][:5], predictions[:5]):
        expected = sequence_tokens(src)
        actual = sequence_tokens(predicted[1:])
        print(f"输入/期望={expected}，生成={actual}")
    print("TF=给定真实前缀；AR=从 BOS 开始，使用模型自己的预测作为下一步输入。")
    print("整句指标以原始 token（包含 EOS）比较；显示行只为阅读去掉 BOS/PAD。")


if __name__ == "__main__":
    main()
