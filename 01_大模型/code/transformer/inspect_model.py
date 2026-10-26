"""无需训练：观察张量形状、注意力矩阵，以及 mask 是否真的阻止信息泄漏。"""

import torch
from torch import nn

from model import (
    BOS,
    EOS,
    PAD,
    Config,
    DecoderLayer,
    EncoderLayer,
    MultiHeadAttention,
    SinusoidalPosition,
    TokenEmbedding,
    Transformer,
    source_mask,
    target_mask,
)


def shape_hook(name):
    """返回一个只打印输入输出形状的 hook，不修改模型数据。"""

    def hook(module, inputs, output):
        input_shapes = [list(item.shape) for item in inputs if torch.is_tensor(item)]
        print(f"{name:44s} {input_shapes} -> {list(output.shape)}")

    return hook


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=100)
    config = Config(d_model=8, n_heads=2, d_ff=16, n_layers=1)
    model = Transformer(config).eval()
    src = torch.tensor([[5, 7, EOS, PAD]])
    tgt = torch.tensor([[BOS, 5, 7, EOS]])

    print("1. 编号、参数表与掩码")
    print("源序列：", src)
    print("目标输入：", tgt)
    print("Embedding 参数表：", list(model.src_embedding.table.weight.shape))
    print("源端有效 Key（1=允许，0=屏蔽）：", source_mask(src)[0, 0].int())
    print("目标端因果 mask（行=Query，列=Key）：\n", target_mask(tgt)[0, 0].int())

    print("\n2. 一次完整前向的数据流；注意源长度与目标长度可不同")
    tracked_types = (
        TokenEmbedding,
        SinusoidalPosition,
        MultiHeadAttention,
        EncoderLayer,
        DecoderLayer,
        nn.Linear,
        nn.LayerNorm,
    )
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, MultiHeadAttention):
            module.capture_attention = True
        if isinstance(module, tracked_types):
            handles.append(module.register_forward_hook(shape_hook(name)))
    try:
        logits = model(src, tgt)
    finally:
        for handle in handles:
            handle.remove()
    print("最终 logits：", list(logits.shape), "；最后一维是词表大小")
    print("Q/K/V 拆头：[B,T,D] -> [B,H,T,Dh]；合头时执行相反变换。")

    print("\n3. 观察第一个头；模型尚未训练，只观察形状和掩码，不解释语义")
    attention_modules = {
        "Encoder Self-Attention": model.encoder_layers[0].self_attention,
        "Decoder Self-Attention": model.decoder_layers[0].self_attention,
        "Decoder Cross-Attention": model.decoder_layers[0].cross_attention,
    }
    for name, attention in attention_modules.items():
        weights = attention.last_attention[0, 0]
        print(f"{name}（行=Query，列=Key）：\n{weights}")
        print("每行概率和：", weights.sum(dim=-1))
    print("Encoder/Cross 的最后一列为零；Decoder Self 的右上三角为零。")
    print("PAD 的 Query 行可以非零，mask 主要防止读取 PAD 的 Key。")

    print("\n4. 改动未来 token，检查较早位置是否受到影响")
    changed_tgt = tgt.clone()
    changed_tgt[:, 2:] = torch.tensor([9, 10])
    changed_logits = model(src, changed_tgt)
    difference = (logits[:, :2] - changed_logits[:, :2]).abs().max().item()
    print(f"前两个位置 logits 的最大差异：{difference:.8f}（应接近 0）")

    print("\n5. 把源端 PAD 位置的 memory 改成大数，检查 Cross-Attention")
    src_mask = source_mask(src)
    memory = model.encode(src, src_mask)
    noisy_memory = memory.clone()
    noisy_memory[src == PAD] = 10000
    reference = model.decode(tgt, memory, src_mask, target_mask(tgt))
    masked = model.decode(tgt, noisy_memory, src_mask, target_mask(tgt))
    difference = (reference - masked).abs().max().item()
    print(f"输出 logits 的最大差异：{difference:.8f}（应接近 0）")

    print("\n6. 参数与位置表")
    print("模型总参数：", sum(p.numel() for p in model.parameters()))
    print("源 Embedding 参数：", model.src_embedding.table.weight.numel())
    print("正弦位置表是否求梯度：", model.src_position.table.requires_grad)
    print("下一步运行 train_copy.py，观察参数如何从损失中学会复制。")


if __name__ == "__main__":
    main()
