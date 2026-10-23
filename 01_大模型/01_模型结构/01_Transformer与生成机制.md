---
title: "Transformer 与生成机制"
updated: 2026-10-23
---

# Transformer 与生成机制

本页集中维护生成目标和模型层结构的实例，避免只在面试回答中出现完整原理。位置编码见 [[01_大模型/01_模型结构/02_位置编码与上下文扩展]]，稀疏专家见 [[01_大模型/01_模型结构/03_MoE混合专家模型]]，推测解码与服务性能见 [[01_大模型/03_推理与接口/03_vLLM与推理优化]]。不同模型的 MTP 结构应分别理解，通用多预测头示意不代表所有实现。

## Multi Token Prediction

Multi Token Prediction（MTP）是 Meta 和 DeepSeek 等在研究中使用的技术，让模型在单次前向传播中同时预测未来多个 token，而不仅仅是下一个 token。

### 标准 Next Token Prediction 的局限

标准 LM 在位置 $t$ 只预测 $t+1$ 的 token：

```
输入：[x1, x2, x3, x4]
目标：[x2, x3, x4, x5]     ← 只预测 next 1 token
```

### MTP 的实现方式

在主预测头之外，额外添加 $D$ 个独立的预测头，每个头预测未来第 $k$ 个 token：

```python
# 伪代码结构
hidden_states = transformer_backbone(input_ids)   # 主干网络共享

# 主预测头（next-1 token）
logits_1 = lm_head_1(hidden_states)   # 预测 t+1

# 额外预测头
logits_2 = lm_head_2(hidden_states)   # 预测 t+2
logits_3 = lm_head_3(hidden_states)   # 预测 t+3

# 训练 loss = λ1*CE(logits_1, target_1) + λ2*CE(logits_2, target_2) + ...
```

关键设计：主干网络只跑一次（共享计算），额外的预测头参数量很小，训练额外成本可控。

### MTP 的两个作用

**训练阶段**：提供更丰富的训练信号，让模型每个位置同时学习短程和长程依赖，相当于数据增强，实验显示能提升模型在代码和推理任务上的准确率。

**推理阶段（结合 Speculative Decoding）**：
```
1. 用 MTP 头同时"草稿"预测 t+1, t+2, t+3
2. 主模型用一次前向验证这 3 个 token 是否可接受
3. 若全部接受，直接输出 3 个 token（相当于 1 次前向生成了 3 个）
4. 若第 2 个 token 被拒绝，只接受第 1 个，重新从第 2 个位置开始
```

理论上接受率越高，吞吐量提升越大（DeepSeek-V3 报告约 1.8~2x 吞吐提升）。

---

## Qwen3 的归一化与层结构

Qwen3 在归一化上的设计与当前主流大模型（Llama、Mistral）保持一致，核心有以下几点：

### 使用 RMSNorm 而非 LayerNorm

**LayerNorm** 计算每个 token 的均值和方差，再做标准化：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

**RMSNorm** 只计算 RMS（均方根），去掉均值计算，去掉 bias 项：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \cdot \gamma$$

优点：计算量更小（省去均值统计），效果与 LayerNorm 相当，是现代大模型的标配。

### Pre-Norm 架构

Qwen3 采用 **Pre-RMSNorm**（在 Attention 和 FFN 之前归一化），而不是 Post-Norm：

```
Pre-Norm（Qwen3、Llama 等）：
  x_out = x + Attention(RMSNorm(x))   ← 先 Norm，再加残差

Post-Norm（原始 Transformer、BERT 等；GPT-2 使用 Pre-Norm）：
  x_out = LayerNorm(x + Attention(x))   ← 先加残差，再 Norm
```

Pre-Norm 通常更容易优化，但最终效果依赖深度、初始化与训练配置，不能普遍断言 Post-Norm 更好。

### Qwen3 的 RMSNorm 位置

标准 Qwen3 将 token embedding 直接送入 decoder，没有独立的 Embedding 后额外 RMSNorm。每个 block 含 Attention 前和 MLP 前的 Pre-RMSNorm，Attention 内含 Q/K Norm，所有 block 后还有输出 RMSNorm。第一层 block 的 input_layernorm 属于该 block。[实现源码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py)。

### 无 Bias（Weight-Only）

所有 Linear 层和 RMSNorm 均去掉 bias 项，减少参数量，同时在量化时更友好（有 bias 的 Linear 层在 INT8 量化时需要额外处理 bias 的精度）。

## Tokenize 与 BBPE

**常见分词算法**：

| 算法 | 核心思想 | 代表模型 |
|------|---------|---------|
| BPE（Byte Pair Encoding） | 从字符级迭代合并频率最高的相邻 pair | GPT-2 |
| BBPE（Byte-level BPE） | 以字节（byte）而非字符为最小单元做 BPE | GPT-3、LLaMA |
| WordPiece | 类似 BPE，但合并策略是最大化语料似然而非频率 | BERT |
| SentencePiece | 与语言无关，支持 BPE 和 Unigram LM，无需预分词 | T5、LLaMA2、Qwen |
| Unigram Language Model | 从大词表开始剪枝，保留最大化语料似然的最小词表 | AlBERT |

**BBPE 为什么使用字节级（Unicode）编码**：

1. **消除 OOV 问题**：词汇表基础单元是 256 个字节（覆盖所有可能字节值），任何字符序列都能被分解为字节序列，永远不会出现未知词（OOV/`<UNK>`），对罕见语言、emoji、特殊符号完全兼容
2. **词表规模极小**：基础词表只需 256 个 token，BPE 合并后总词表通常在 32K~128K，远小于逐字符词表（中文字符就有数万个）
3. **UTF-8 天然兼容**：UTF-8 将任何 Unicode 字符编码为 1~4 个字节，BBPE 在字节序列上操作，无需对每种语言单独维护字符级词表，一个词表处理全语言
4. **对比普通 BPE 的优势**：普通 BPE 以字符为最小单元，遇到词表外字符（如生僻汉字、小语种字母）直接报错；BBPE 退化到单字节表示，仍能正常处理，不丢失信息

---

## 自回归生成与结束条件

**完整流程**：

1. **Tokenization**：Query 文本 → 分词器（BPE/SentencePiece）→ Token ID 序列
2. **Embedding 查表**：Token ID → Token Embedding。位置机制按架构处理：RoPE 在 Attention 中旋转 Q/K，ALiBi 向注意力分数添加偏置，均不是在 Embedding 上直接相加。
3. **Transformer Decoder 前向传播**：经过 N 层 Decoder Block，每层：
   - Masked Multi-Head Self-Attention（含 KV Cache 复用历史 K/V 计算结果）
   - FFN（前馈网络，通常是 SwiGLU 激活）
   - RMSNorm + 残差连接
4. **LM Head**：最后一层隐状态 → 线性投影层 → 词汇表大小的 Logits
5. **采样/解码**：对 Logits 做 Temperature 缩放 → Top-p / Top-k 过滤 → 采样得到下一个 Token
6. **自回归循环**：将新 Token 追加到序列末尾，带入步骤 3 重新前向（借助 KV Cache 跳过已算部分），直到输出终止条件

**如何判断输出结束**：
- 模型输出特殊 Token `<EOS>`（End of Sequence），解码器识别后停止循环
- 或达到 `max_new_tokens` 硬截断上限
- 或遇到自定义 `stop` 词列表中的字符串

**自回归生成**：常规逐 token 解码依赖前序 token。KV Cache 保存历史 K/V，下一步只计算新 token 的 Q/K/V，再让新 Q 关注历史 K/V。固定模型维度时，第 t 个缓存解码步的全注意力成本为 O(t)，生成 n 个 token 的累计 Attention 成本仍为 O(n²)；若每步重算整个前缀，累计可为 O(n³)。prefill 成本单独计入，不能笼统说整个生成从 O(n²) 降为 O(n)。

---

## 生成与采样参数

| 参数                          | 作用                                   | 典型值      | 实践建议                          |
| --------------------------- | ------------------------------------ | -------- | ----------------------------- |
| `temperature`               | 控制 Logits 缩放，越低越确定，越高越随机；设为 0 近似贪心解码 | 0.0~2.0  | 事实性生成用 0.0~0.1；创意/改写用 0.7~1.0 |
| `top_p`（核采样）                | 只从累积概率达到 p 的最小候选集中采样，过滤长尾低概率 token   | 0.9~0.95 | 与 temperature 配合，不建议两者都设为极端值  |
| `top_k`                     | 只从概率最高的 k 个 token 中采样                | 20~50    | top_p 和 top_k 通常二选一，top_p 更常用 |
| `max_tokens`                | 最大输出 token 数，防止无限生成                  | 按业务定     | RAG 回答约 512，用例生成约 2048        |
| `frequency_penalty`         | 按已出现频率惩罚 token，出现越多惩罚越重，降低重复率        | 0.0~2.0  | 0.3~0.7 可有效减少啰嗦               |
| `presence_penalty`          | 对任何已出现过的 token 施加固定惩罚（与频率无关），鼓励话题多样性 | 0.0~2.0  | 长文生成时用于增加词汇多样性                |
| `repetition_penalty`（HF 格式） | 将已出现 token 的 logit 除以该系数（>1 降低重复）    | 1.1~1.3  | Transformers 库参数，与上两者不通用      |
| `stop`                      | 遇到指定字符串立即截断，不等 EOS                   | 按业务定     | 避免用 `}` 作 JSON stop，可能在嵌套对象处截断       |

**实践**：RAG 系统生成答案时 temperature=0.1 用于降低采样随机性，但不保证事实正确；Query 改写时 temperature=0.7 增加问法多样性；结构化 JSON 输出时配合 Function Calling 原生支持比纯靠参数控制更可靠。

---
