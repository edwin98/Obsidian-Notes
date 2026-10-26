"""从张量运算搭建 Encoder–Decoder Transformer。

阅读顺序：Embedding → Attention → 多头 → FFN → Encoder → Decoder → 生成。
符号：B=批量，S=源长度，T=目标长度，D=隐藏维度，H=头数，Dh=D/H。
所有布尔 mask 均约定 True=允许关注，与某些 PyTorch 接口的语义不同。
这是独立编写的教学实现，讲解路线参考 Harvard Annotated Transformer。
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

PAD, BOS, EOS = 0, 1, 2


@dataclass(frozen=True)
class Config:
    """源端和目标端使用相同词表，但 Embedding 和输出头不共享权重。"""

    vocab_size: int = 16
    d_model: int = 64
    n_heads: int = 4
    d_ff: int = 128
    n_layers: int = 2
    dropout: float = 0.0
    max_len: int = 64

    def __post_init__(self):
        if self.vocab_size <= EOS + 1:
            raise ValueError("词表至少要容纳 PAD、BOS、EOS 和一个内容 token")
        if min(self.d_model, self.n_heads, self.d_ff, self.n_layers, self.max_len) < 1:
            raise ValueError("模型维度、头数、层数及最大长度必须为正数")
        if self.d_model % self.n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout 必须在 [0, 1) 内")


# %% 1. 掩码：屏蔽的是注意力分数矩阵的列，即不能读取的 Key。
def source_mask(src: Tensor) -> Tensor:
    """[B,S] → [B,1,1,S]，广播到各头、各 Query 位置。

    Encoder Self-Attention 和 Decoder Cross-Attention 共用这一源端 mask。
    PAD 对应的 Query 输出不一定为零，但不会作为有效 Key 被读取。
    """
    return (src != PAD)[:, None, None, :]


def target_mask(tgt: Tensor) -> Tensor:
    """[B,T] → [B,1,T,T]：目标端 Key 非 PAD，且列位置 j <= 行位置 i。"""
    length = tgt.size(1)
    causal = torch.ones(length, length, dtype=torch.bool, device=tgt.device).tril()
    return (tgt != PAD)[:, None, None, :] & causal[None, None, :, :]


# %% 2. Token 编号查表，并注入固定的正弦/余弦位置信息。
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.table = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.scale = math.sqrt(d_model)

    def forward(self, ids: Tensor) -> Tensor:
        # 权重表 [V,D]；输入 [B,T]；查表结果 [B,T,D]，没有输入长度维的参数。
        return self.table(ids) * self.scale


class SinusoidalPosition(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        table = torch.zeros(max_len, d_model)
        table[:, 0::2] = torch.sin(position * frequencies)
        # 切片兼容奇数隐藏维度；本例默认使用偶数维度。
        table[:, 1::2] = torch.cos(position * frequencies[: d_model // 2])
        self.register_buffer("table", table.unsqueeze(0))  # [1,max_len,D]，不求梯度
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        if x.size(1) > self.table.size(1):
            raise ValueError("序列超过 max_len，请增大配置中的位置表长度")
        return self.dropout(x + self.table[:, : x.size(1)].to(dtype=x.dtype))


# %% 3. 缩放点积注意力：softmax(QK^T / sqrt(Dh) + mask) V。
def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor
) -> tuple[Tensor, Tensor]:
    """Q=[B,H,Tq,Dh]，K/V=[B,H,Tk,Dh]；返回加权结果和概率。

    保留完整概率矩阵便于观察，因而有 O(Tq*Tk) 的显存开销。
    没有调用 nn.MultiheadAttention 或融合注意力内核。
    """
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    # 使用有限负极值，再把被屏蔽项归零。全屏蔽行因此输出零而非 NaN。
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probabilities = scores.softmax(dim=-1).masked_fill(~mask, 0.0)
    context = probabilities @ v
    return context, probabilities


class MultiHeadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model)
        self.k_proj = nn.Linear(config.d_model, config.d_model)
        self.v_proj = nn.Linear(config.d_model, config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self.capture_attention = False
        self.last_attention: Tensor | None = None

    def split_heads(self, x: Tensor) -> Tensor:
        batch, length, _ = x.shape
        # [B,T,D] → [B,T,H,Dh] → [B,H,T,Dh]
        return x.reshape(batch, length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor
    ) -> Tensor:
        q = self.split_heads(self.q_proj(query))
        k = self.split_heads(self.k_proj(key))
        v = self.split_heads(self.v_proj(value))
        context, probabilities = scaled_dot_product_attention(q, k, v, mask)
        # 仅观察实验打开记录；detach 避免保留反向图。
        self.last_attention = probabilities.detach() if self.capture_attention else None
        batch, _, length, _ = context.shape
        joined = context.transpose(1, 2).contiguous().view(batch, length, -1)
        return self.out_proj(joined)  # [B,T,D]，混合各个头的信息


# %% 4. FFN：每个位置使用同一组参数，不在不同 token 之间交换信息。
class FeedForward(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.up = nn.Linear(config.d_model, config.d_ff)
        self.down = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.dropout(torch.relu(self.up(x))))


# %% 5. Encoder：双向 Self-Attention + FFN，各子层都有归一化和残差。
class EncoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # Pre-Norm：先归一化再计算子层；残差保留原始 x。
        normalized = self.attention_norm(x)
        x = x + self.dropout(
            self.self_attention(normalized, normalized, normalized, mask)
        )
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


# %% 6. Decoder：因果 Self-Attention + Cross-Attention + FFN。
class DecoderLayer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.self_attention = MultiHeadAttention(config)
        self.cross_attention = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.self_norm = nn.LayerNorm(config.d_model)
        self.cross_norm = nn.LayerNorm(config.d_model)
        self.ffn_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        normalized = self.self_norm(x)
        x = x + self.dropout(
            self.self_attention(normalized, normalized, normalized, tgt_mask)
        )
        # Cross-Attention 的 Q 来自 Decoder，K/V 来自 Encoder 的输出 memory。
        x = x + self.dropout(
            self.cross_attention(self.cross_norm(x), memory, memory, src_mask)
        )
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


# %% 7. 完整模型：Encoder 输出 memory，Decoder 隐状态经 LM Head 变成 logits。
class Transformer(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.src_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.tgt_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.src_position = SinusoidalPosition(
            config.d_model, config.max_len, config.dropout
        )
        self.tgt_position = SinusoidalPosition(
            config.d_model, config.max_len, config.dropout
        )
        # 每层独立创建，避免列表乘法导致多个层共享同一组参数。
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(config) for _ in range(config.n_layers)]
        )
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.encoder_norm = nn.LayerNorm(config.d_model)
        self.decoder_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        # Xavier 会覆盖 Embedding 初始化；明确恢复 PAD 行为零。
        with torch.no_grad():
            self.src_embedding.table.weight[PAD].zero_()
            self.tgt_embedding.table.weight[PAD].zero_()

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        x = self.src_position(self.src_embedding(src))
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return self.encoder_norm(x)  # memory: [B,S,D]

    def decode(
        self, tgt: Tensor, memory: Tensor, src_mask: Tensor, tgt_mask: Tensor
    ) -> Tensor:
        x = self.tgt_position(self.tgt_embedding(tgt))
        for layer in self.decoder_layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.lm_head(self.decoder_norm(x))  # logits: [B,T,V]

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        src_mask = source_mask(src)
        memory = self.encode(src, src_mask)
        return self.decode(tgt, memory, src_mask, target_mask(tgt))


# %% 8. 自回归生成：只提供 BOS，后续输入全部取自模型自己的预测。
@torch.no_grad()
def greedy_decode(model: Transformer, src: Tensor, max_new_tokens: int) -> Tensor:
    """返回 [B,生成长度]，包含 BOS/EOS；先结束的样本后续位置用 PAD 填充。

    本例复用 Encoder memory，每一步重新计算完整目标前缀；没有 KV Cache。
    函数保留调用前的 train/eval 状态，推理内部关闭 dropout。
    """
    if not 1 <= max_new_tokens <= model.config.max_len:
        raise ValueError("max_new_tokens 必须在 [1, max_len] 内")
    was_training = model.training
    model.eval()
    try:
        src_mask = source_mask(src)
        memory = model.encode(src, src_mask)
        generated = torch.full(
            (src.size(0), 1), BOS, dtype=torch.long, device=src.device
        )
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_new_tokens):
            logits = model.decode(generated, memory, src_mask, target_mask(generated))
            next_token = logits[:, -1].argmax(dim=-1)
            next_token = next_token.masked_fill(finished, PAD)
            generated = torch.cat([generated, next_token[:, None]], dim=1)
            finished |= next_token == EOS
            if finished.all():
                break
        return generated
    finally:
        model.train(was_training)
