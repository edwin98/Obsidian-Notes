# Transformer 知识实践

从“编号如何变成向量”开始，亲手追踪注意力、Encoder、Decoder、损失与生成。讲解路线参考 [Harvard Annotated Transformer](https://github.com/harvardnlp/annotated-transformer)，这里独立编写中文注释和小实验，将重点放在张量形状、信息能否流动，以及模型能否真正学会一个任务。

对应笔记：[[01_大模型/01_模型结构/01_Transformer与生成机制]]、[[01_大模型/01_模型结构/02_位置编码与上下文扩展]]、[[01_大模型/02_训练与对齐/03_DeepSpeed与显存估算]]。

## 先运行，再阅读

在仓库根目录打开 PowerShell。可以直接使用笔记项目已有的环境：

```powershell
$practice = '01_大模型/code/transformer'
& '.venv/Scripts/python.exe' -X utf8 "$practice/inspect_model.py"
& '.venv/Scripts/python.exe' -X utf8 "$practice/train_copy.py"
& '.venv/Scripts/python.exe' -m pytest "$practice/tests" -q
```

如果需要独立环境，进入本目录再执行：

```powershell
cd '01_大模型/code/transformer'
uv sync --group dev
uv run python -X utf8 inspect_model.py
uv run python -X utf8 train_copy.py
uv run pytest tests -q
uv run ruff check .
uv run ruff format .
```

需要 Python 3.12 或更新版本。运行依赖 PyTorch 和 NumPy，开发检查另外使用 pytest 和 Ruff。首次安装依赖需要联网；实验本身在 CPU 上生成数据，不下载语料或预训练权重，也不写模型检查点。默认小模型只有 **170,768 个参数**，适合先理解结构。

## 文件与阅读顺序

| 文件 | 阅读重点 | 建议断点 |
| ---- | ---- | ---- |
| [model.py](model.py) | 从底层运算组装完整 Transformer | `TokenEmbedding.forward` → `scaled_dot_product_attention` → `MultiHeadAttention.forward` → `DecoderLayer.forward` |
| [inspect_model.py](inspect_model.py) | 形状追踪、三个注意力矩阵、信息隔离实验 | `logits = model(src, tgt)`，随后逐层进入 |
| [train_copy.py](train_copy.py) | 输入右移、训练循环、固定验证集、逐 token 生成 | `make_batch` → `loss.backward` → `evaluate` |
| [tests/test_transformer.py](tests/test_transformer.py) | 公式对照、掩码隔离、梯度与实际学习 | 从测试名称选择想验证的假设 |

`model.py` 中用 `# %%` 标出八个知识段落。推荐先完整运行脚本，再按段落阅读；在 IDE 中单独运行后面的段落时，需要先执行其依赖的导入和定义。

## 第一步：观察数据如何经过模型

运行 `inspect_model.py`。记住这些符号：

| 符号 | 含义 | 观察脚本中的值 |
| ---- | ---- | ---- |
| B | 批量大小 | 1 |
| S | 源序列长度 | 4 |
| T | 目标输入长度 | 4 |
| V | 词表大小 | 16 |
| D | 每个 token 的隐藏维度 | 8 |
| H | 注意力头数 | 2 |
| Dh | 每头维度，D/H | 4 |

观察脚本使用一层的小配置，训练脚本使用两层、D=64 的默认配置。

```text
src [B,S] → 源 Embedding [B,S,D] → 加位置编码 → Encoder → memory [B,S,D]
                                                                │
tgt [B,T] → 目标 Embedding [B,T,D] → 加位置编码 → Decoder ←────────┘
                                                       ↓
                                                隐状态 [B,T,D]
                                                       ↓ LM Head
                                                 logits [B,T,V]
```

源长度 S 和目标长度 T 可以不同。Cross-Attention 将每个目标位置与源序列位置关联，其概率矩阵是 `[B,H,T,S]`。

### Embedding 的表与输出

`nn.Embedding` 的参数表是 `[V,D]`。输入 `[B,T]` 只是在表中挑选 B×T 行，得到 `[B,T,D]`。

本例沿用正弦位置编码方案，将查表结果乘以 `sqrt(D)`，再加上位置表。位置表由固定公式生成，通过 `register_buffer` 保存并随模型迁移设备；它不参与梯度更新。即使 PAD 的查表向量为零，加上位置编码后该位置也可能非零，因此仍然需要 padding mask。

### Q、K、V 与多头

对一个注意力头：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{D_h}}+M\right),\qquad O=AV
$$

`model.py` 使用布尔 mask 实现 M：不允许的位置在 softmax 前填入很小的负数，再将其概率置零。

```text
线性投影： [B,T,D] → Q/K/V 各一个 [B,T,D]
拆成多头： [B,T,D] → [B,T,H,Dh] → [B,H,T,Dh]
打分：     [B,H,Tq,Dh] @ [B,H,Dh,Tk] → [B,H,Tq,Tk]
加权：     [B,H,Tq,Tk] @ [B,H,Tk,Dh] → [B,H,Tq,Dh]
合并多头： [B,H,Tq,Dh] → [B,Tq,D] → 输出投影
```

Self-Attention 的 Q/K/V 来自同一条隐状态序列，但三个投影矩阵分别学习。Cross-Attention 的 Q 来自 Decoder，K/V 来自 Encoder memory。

## 第二步：看清三种注意力的可见范围

本项目统一约定 **mask 的 True 表示允许关注**。阅读其他库时要检查其约定，部分接口的 True 表示屏蔽。

| 注意力 | 谁提供 Q | 谁提供 K/V | 哪些位置不可见 |
| ---- | ---- | ---- | ---- |
| Encoder Self-Attention | Encoder 隐状态 | Encoder 隐状态 | 源端 PAD Key |
| Decoder Self-Attention | Decoder 隐状态 | Decoder 隐状态 | 未来 Key、目标端 PAD Key |
| Decoder Cross-Attention | Decoder 隐状态 | Encoder memory | 源端 PAD Key |

注意力矩阵的**行是 Query，列是 Key**。正常有效行的概率和为 1；目标端 Self-Attention 的右上三角为零，源端有 PAD 时 Encoder/Cross 相应列为零。本实现还将全屏蔽行的概率定义为零，避免产生 NaN；实际复制任务每条输入都有有效 token。

`inspect_model.py` 做两个干预实验：

1. 修改目标输入的后两个 token，较早位置的 logits 应保持不变，验证因果隔离。
2. 将源端 PAD 位置的 memory 改为大数，Decoder 输出应保持不变，验证 Cross-Attention 的 padding mask。

PAD 的 Query 行可能有非零输出，这不表示 mask 失败。这里屏蔽 PAD 的 Key，并在损失中忽略 PAD 标签，防止这些填充位置影响有效 token。

观察脚本会暂存每个头的概率，使用 `last_attention[样本下标, 头下标]` 查看。模型尚未训练，此时注意力只用于理解运算和可见范围，不能直接解释成已经学会的语言关系。

## 第三步：理解层结构、残差和归一化

Encoder 的每层包含两个子层，Decoder 多一个读取 memory 的子层：

```text
Encoder：Self-Attention → FFN
Decoder：Causal Self-Attention → Cross-Attention → FFN
```

每个子层使用 Pre-Norm 残差结构：`x = x + dropout(sublayer(norm(x)))`。LayerNorm 对每个位置最后一维 D 归一化；残差让信息和梯度沿加法路径传播。

FFN 是 `D → Dff → D` 的两层网络，中间使用 ReLU；同一组 FFN 参数应用于每个位置。它改变单个位置的特征，跨位置的信息交换由注意力完成。

这一学习版本采用与 Annotated Transformer 代码相近的 Pre-Norm 组织方式，2017 年论文的结构采用 Post-Norm。这里使用 PyTorch LayerNorm，并只在位置输入、FFN 中间及残差分支设置 dropout；注意力概率本身不加 dropout。默认 dropout=0，方便稳定观察。

## 第四步：训练模型学会复制

运行 `train_copy.py`。数据是随机生成的变长整数序列，内容 token 是 3～15，保留三个特殊编号：

| Token | 编号 | 作用 |
| ---- | ---- | ---- |
| PAD | 0 | 补齐长度，在标签损失中忽略 |
| BOS | 1 | 告诉 Decoder 开始生成 |
| EOS | 2 | 告诉 Decoder 输出已经结束，也需要计算损失 |

例如：

```text
源输入 src：         [5,   7,   9, EOS, PAD]
完整目标 full_tgt：  [BOS, 5,   7,   9, EOS, PAD]
Decoder 输入 tgt_in：[BOS, 5,   7,   9, EOS]
监督标签 labels：    [5,   7,   9, EOS, PAD]
```

输入与标签错开一个位置，模型在读到 BOS 时预测 5，读到真实前缀 BOS、5 时预测 7。复制任务中，Encoder 允许看到完整源输入；目标端仍需因果 mask，以免从目标输入的后一个位置直接读出标签。

```text
logits = model(src, tgt_in)                  # [B,T,V]
loss = cross_entropy(logits, labels)         # 实际计算前展平 B、T
optimizer.zero_grad()                        # 清除上一步累积的梯度
loss.backward()                             # 求各可训练参数的梯度
clip_grad_norm_(...)                        # 限制梯度范数
optimizer.step()                            # 根据梯度更新参数
```

交叉熵接收原始 logits，内部完成 log-softmax 等计算，不需要手动提前 softmax。梯度会经过注意力和 FFN，传播到源端及目标端的可训练 Embedding。

训练使用每步新生成的随机批次；验证使用独立随机数生成器产生的固定 64 条序列。固定验证集方便比较不同步骤的进展，重复查看它仍不等于完成独立测试集评估。

### Teacher Forcing 与自回归的区别

训练时，真实目标前缀已经提供，所有位置能通过一次前向并行计算；因果 mask 限制每个位置能看到的范围。生成时先给 BOS，之后把模型自己的预测追加回输入，一步一步生成，直到 EOS 或长度上限。

脚本同时报告：

- `val_loss`：固定验证集上的交叉熵。
- `TF_token_acc`：给定真实前缀时，有效位置（包括 EOS）的预测准确率。
- `AR_exact_acc`：从 BOS 自回归生成后，整条序列与标签完全一致的比例，包括 EOS。

token 准确率高并不保证整句生成正确；一次错误可能影响后续前缀。这个实验可以直接观察两项指标的差异。

`greedy_decode` 复用 Encoder memory，但每次重新计算整个目标前缀，便于理解自回归过程。它不实现 KV Cache；可结合笔记的 KV Cache 章节思考哪些结果可以缓存。

### 本机验证记录

2026-10-26，Windows、Python 3.12 环境、PyTorch 2.10.0+cpu，默认随机种子 42，CPU 单线程，600 步训练：

| 阶段 | 验证 loss | TF token 准确率 | 自回归整句正确率 |
| ---- | ---- | ---- | ---- |
| 训练前 | 3.3270 | 7.4% | 0.0% |
| 100 步 | 0.3245 | 89.9% | 67.2% |
| 300 步 | 0.0372 | 98.6% | 95.3% |
| 600 步 | 0.0066 | 100.0% | 100.0% |

本次训练循环耗时约 6.9 秒，不含导入和初始化；其他硬件、库版本、种子可能得到不同耗时和数值。该结果只说明模型在这个小复制任务和固定验证集上的表现，不能据此推断语言理解或任意长度泛化能力。

可调整实验规模：

```powershell
uv run python -X utf8 train_copy.py --steps 100 --log-every 25
uv run python -X utf8 train_copy.py --steps 1000 --max-tokens 10
uv run python -X utf8 train_copy.py --seed 7 --batch-size 16
```

## 第五步：动手修改，验证自己的理解

每次只改一个地方，先预测结果，再运行观察或测试。

| 练习 | 修改位置 | 应观察什么 |
| ---- | ---- | ---- |
| 从一个头改为多个头 | `Config.n_heads`，保持 D 能整除 H | 每头维度变小，合头后的 D 不变 |
| 让源与目标长度不同 | `inspect_model.py` 的 src/tgt | Cross-Attention 是 T×S，Self-Attention 是 T×T；同步调整后续修改 token 的实验切片 |
| 临时移除因果限制 | `target_mask`，保留 padding 约束 | 修改未来 token 会影响过去 logits，因果隔离测试应失败 |
| 临时去掉源 padding mask | `source_mask` | 修改 PAD memory 的实验和相关测试应失败 |
| 不加位置编码 | `SinusoidalPosition.forward` | 观察复制是否变难；结合源端序列重排思考顺序信息从何而来 |
| 复制改成逆序输出 | `make_batch` 构造 full_target 时反转内容 | Encoder 输入保持原序；监督目标和验证标签需要一致变化 |
| 开启注意力观察 | 将相应 `MultiHeadAttention.capture_attention` 设为 True | 对比训练前后的各头；注意图案不自动等价于因果解释 |

逆序练习只改变内容 token 的顺序，BOS/EOS 位置和输入标签错位规则保持一致。实验后恢复代码，再运行测试。

## 验证范围与后续阅读

测试涵盖显式注意力与 PyTorch SDPA 的数值对照、手算加权均值、全屏蔽行、因果及 padding 信息隔离、输入标签右移、梯度传播、EOS 停止和生成上限，以及一个小批次上的实际训练与自回归复现。小批次过拟合测试验证训练链路，随机批次实验用于观察更广的复制表现。

此处实现 Encoder–Decoder、正弦位置编码、ReLU FFN 和普通多头注意力。继续阅读现代 Decoder-only 模型时，可以依次比较移除 Encoder/Cross-Attention 后的数据流，以及 RoPE、RMSNorm、SwiGLU、GQA 和 KV Cache 的作用。

参考资料：

- [Harvard Annotated Transformer 项目](https://github.com/harvardnlp/annotated-transformer)
- [Annotated Transformer Python 源码](https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
