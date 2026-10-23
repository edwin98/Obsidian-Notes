---
title: "视觉语言模型原理与 Qwen 实现"
updated: 2026-10-23
---

# 视觉语言模型原理与 Qwen 实现

按视觉编码、连接器、语言融合和训练理解 VLM，并以 Qwen2.5-VL 为主要实现实例。Qwen 各代配置保持各自来源与边界。

大模型理解图片的底层原理，本质上是将“非结构化的图像信号”编码成“结构化的数学特征”，经过特征对齐后，再送入大语言模型（LLM）进行理解与生成。整个体系可从**系统流程**、**训练范式**、**核心组件**三个维度来拆解。

---

## 一、 系统架构与推理过程

多模态大模型的推理流水线主要分为三大步：**感知 (Perception)**、**对齐 (Alignment)** 和 **融合 (Fusion)**。

### 1. 视觉感知 (Perception)
主要通过 ViT (Vision Transformer) 或 CLIP 等视觉编码器提取图像特征。
- **图像切片 (Patching)**：将原始图像切割成网格状的固定大小方块（Patch）。
- **线性投影 (Linear Projection)**：将每个小图块展平，并映射为高维的 Embedding 向量。
- **位置编码 (Positional Encoding)**：为每个无序的方块打上绝对或相对位置的标签，保留图像的空间几何结构。
- **特征提取 (Feature Extraction)**：通过多层 Self-Attention 结构，让各个 Patch 相互交换信息，最终输出具备高阶语义的图像特征矩阵。

### 2. 语义对齐 (Alignment)
视觉编码器输出的高维向量与 LLM 原生的文本向量空间是不互通的，需要通过**适配器（Connector / Projector）**进行“翻译”和空间对齐。常见对齐方案：

| 方案 | 做法 | 特点 | 代表模型 |
| :--- | :--- | :--- | :--- |
| **线性网络 (Linear/MLP)** | 用线性层或多层感知机直接把视觉特征映射到文本维度。 | 简单粗暴，不损失视觉细节，但如果图像分辨率高，会有巨大的 Token 负担。 | LLaVA 系列 |
| **Q-Former (查询变换器)** | 使用一组设定好的“可学习 Query”去视觉特征里“交叉提取”信息。 | 压缩率极高，能过滤无关背景，降低 LLM 运算量。 | BLIP-2, InstructBLIP |
| **感知重采样 (Perceiver)** | 将任意长度的视觉特征固定压缩为几十个 Token。 | 适合处理极高分辨率图像或超长视频帧。 | Flamingo, IDEFICS |

### 3. 模态融合 (Fusion)
对齐后的图像特征已经被翻译成了 LLM 能听懂的“视觉 Token”。系统会直接将其与用户的“文本 Token”序列进行拼接（Concatenation）：

`[视觉 Token 1, 视觉 Token 2, ...] + [文本 Token: "请问图中有什么？"]`

最终统一送入 LLM 进行自回归推理。

---

## 二、 经典两阶段训练范式

要让独立的视觉模块和语言模型配合默契，通常需要分步骤训练。多模态模型的训练往往建立在强大的预训练 LLM 和预训练视觉编码器（如 CLIP ViT）之上。

### 阶段一：视觉-语言特征对齐 (Alignment Pre-training)
- **核心目标**：让模型“认识物体”，建立视觉概念与文本概念的强绑定。
- **训练数据**：海量且简单的“图像 - 较短描述”对（Image-Caption Pairs，例如数亿级别的 LAION 数据集：一张橘猫照片 + "a ginger cat"）。
- **模型状态**：通常**冻结（Freeze）** LLM 本身和视觉编码器的权重，仅仅训练中间的**对齐层（Connector）**。这一步是让对齐层变成一名“翻译官”，学会把视觉特征准确翻译成 LLM 理解的词汇空间。

### 阶段二：多模态指令微调 (Visual Instruction Tuning)
- **核心目标**：让模型具备复杂的视觉推理、视觉问答和遵守人类指令的能力。
- **训练数据**：经过专门构造的复杂图文对话数据（如：“这张图里左边的人在干什么？”、“这盘菜的菜谱是什么？”）。这些数据往往由 GPT-4V 或人工进行高质量构造。
- **模型状态**：保持视觉编码器冻结，**解冻对齐层并结合 LLM 的全量微调（Full Fine-tuning）或高效微调（如 LoRA）**，让 LLM 适应多模态 Token 的输入逻辑，从而学会根据图像内容进行逻辑判断与详细输出。

---

## 三、 核心组件底层细节

### 1. ViT (Vision Transformer)：提取图像特征
ViT 使用 Transformer 处理图像，将图像切分为 Patch 后编码为特征，在部分视觉任务中可替代卷积网络（CNN）。效果取决于训练数据、模型和任务。

**底层核心机制：**
- **序列化 (Patch Embedding)**：假设输入图像尺寸为 $H \times W \times C$。ViT 将图像划分为 $N$ 个固定大小的 Patch（如 $16 \times 16$），则有：
  
  $$N = \frac{H \times W}{P^2}$$
  
  每个 Patch 展平为一个 $P^2 \cdot C$ 维的向量，经过线性层 $E$ 映射到 $D$ 维，并加入可训练的位置编码矩阵 $E_{pos}$：
  
  $$z_0 = [x^1_p E; x^2_p E; \dots; x^N_p E] + E_{pos}$$
  
  *(注：Transformer 注意力机制天然是输入置换不变的，没有 $E_{pos}$，模型就完全无法分辨物体在图里的空间相对位置。)*

- **注意力汇聚 (Self-Attention)**：通过注意力公式计算各 Patch 间的上下文相关性：
  
  $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
  
  这让模型理解“左上角的猫耳朵”和“右下角的猫尾巴”同属于一只猫。

- **特征输出策略**：
  - **CLIP 模式**：由于原本受限于 Image-Text 任务更关注全局的高层语义，通常只提取头部的 `[CLS]` Token 被用作图像的整图特征。
  - **LLaVA 模式**：直接摒弃 `[CLS]`，提取出所有的 $N$ 个 Patch 的空间特征矩阵（Grid Features），最大程度保留细粒度局部特征。

---

### 2. 线性层 / MLP (Linear Projector)：映射视觉特征
在 LLaVA 等架构中，投影层将视觉特征映射到 LLM 的输入维度，使模型可以接收图像信息。

**底层核心机制：**
- **数学本质**：若视觉特征 $V \in \mathbb{R}^{N \times D_{vis}}$，LLM 自身输入维度为 $D_{llm}$，映射就是一个带有偏差权重的矩阵乘法 $W \in \mathbb{R}^{D_{vis} \times D_{llm}}$：
  
  $$X_{visual\_tokens} = V \cdot W + b$$

- **常见结构**：
  - **单层 Linear**：早期的 LLaVA-1.0 只做纯线性变换，表达能力有限。
  - **两层 MLP（多层感知机）**：LLaVA-1.5 使用 `Linear -> GELU激活层 -> Linear` 结构。相比单层线性变换，非线性映射可以学习更复杂的视觉与语言特征对应关系。
  
- **局限性**：逐个投影通常不减少视觉 Token 数量。例如，输入 576 个视觉 Token，投影后仍为 576 个。更高分辨率可能产生更多 Token，增加显存和计算开销。

---

### 3. Q-Former (Query Transformer)：提取并压缩视觉信息
Q-Former 是 BLIP-2、InstructBLIP 等模型连接视觉编码器与语言模型的中间组件，用少量可学习 Query 提取视觉特征。

**底层核心机制：**
Q-Former 使用双向与跨模态注意力，并设置固定数量的**可学习查询向量（Learned Queries）**，例如 32 个。

- **工作流与双重提取机制**：
  1. **Self-Attention 自注意力**：查询向量之间交换信息，形成联合表示。
  2. **Cross-Attention 交叉注意力**：
     - **Query（Q）**：32 个可学习查询向量
     - **Key (K) / Value (V)**: ViT 输出的大量图像 Patch 实体特征
     
     $$Output = \text{Softmax}\left(\frac{Q_{query} \cdot K_{visual}^T}{\sqrt{d}}\right) V_{visual}$$
     
     这一步根据查询向量对图像特征加权汇总，提取与训练目标相关的视觉信息。

- **Query 如何训练？**
  查询向量通常随机初始化，在图文预训练中通过反向传播更新，逐步学习如何提取有用的视觉信息。不能直接假定每个 Query 都对应“颜色”或“人脸”等固定含义。

- **作用：减少输入语言模型的视觉 Token**：
  - **固定输出数量**：在此 32 个 Query 的配置下，输出 32 个查询表示，减少传给 LLM 的视觉 Token。视觉编码器和交叉注意力的开销仍会随输入变化，整个模型的计算成本并非固定。
  - **学习筛选信息**：通过训练调整对不同图像区域的关注程度，减少部分无关背景的影响；实际提取质量需要验证。
---

## 四、 Qwen 视觉语言模型是怎么实现的？

这里的“视觉模型”指 **Qwen-VL 系列视觉语言模型**，主要做图像/视频理解与文本生成。下面以 **Qwen2.5-VL-7B** 为例，再对比 Qwen2-VL 与 Qwen3-VL；各代的编码器和训练方案不能直接混用。

### 1. 总体架构：ViT + Patch Merger + Qwen LLM

Qwen2.5-VL 的主干仍是前面的“感知 → 对齐 → 融合”：图像经过 ViT 得到局部特征，Patch Merger 压缩并投影特征，最后送入 Qwen2.5 的 Decoder-only 语言模型。[官方技术报告](https://arxiv.org/html/2502.13923v1)

**关键理解：视觉 Token 是连续特征向量，不必先变成文字，也不必对应词表中的某个词。** 文本通过词表 Embedding 查表得到向量，图像通过视觉网络得到向量；只要维度一致，训练后便能共同参与 LLM 的注意力计算。

| 模块 | Qwen2.5-VL-7B 中的职责 | 示例形状（单张 448×448 图像） |
| --- | --- | --- |
| Patch Embedding | 将局部像素映射为特征 | 1024 个 Patch，各 1280 维 |
| ViT | 在 Patch 之间交换信息 | `1024 × 1280` |
| Patch Merger | 合并相邻 2×2 特征并投影 | `256 × 3584` |
| Qwen2.5 LLM | 联合处理视觉向量与文本向量 | 每个输入位置均为 3584 维 |

表中的维度来自 [7B 官方配置](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)，Token 数是按该配置推算的，不包含视觉边界标记和文本。

### 2. 图像如何变成 Token：动态分辨率与切块

Qwen2-VL 引入 **Naive Dynamic Resolution**：不同尺寸的图像可以产生不同数量的视觉 Token，而不是全部压缩到固定方形尺寸。这样能为小字、表格和小目标保留更多细节。[Qwen2-VL 技术报告](https://arxiv.org/html/2409.12191v1)

对 Qwen2.5-VL，空间 Patch 大小为 **14×14**，随后每 **2×2** 个相邻 Patch 合并成一个 LLM 视觉 Token。因此，对于预处理后尺寸为 $H' \times W'$ 的单张图像：

$$
N_{patch}=\frac{H'}{14}\frac{W'}{14},\qquad
N_{visual}=\frac{N_{patch}}{4}=\frac{H'W'}{28^2}
$$

例如 448×448 得到 $32×32=1024$ 个 Patch，合并后得到 $16×16=256$ 个视觉 Token。若调整为 896×896，视觉 Token 变为 1024 个。

这里 **28×28 是合并后对应的空间采样尺度**；由于 ViT 已经交换上下文信息，不能把一个输出向量理解为只包含这块像素的信息。[配置参数](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)

“动态分辨率”也不代表原图永不缩放。实际预处理会受像素预算和网格整除约束影响。提高分辨率能增加细节，也会增加视觉编码、LLM Prefill 和 KV Cache 的开销。[官方图像分辨率用法](https://qwenlm.github.io/blog/qwen2.5-vl/)

### 3. ViT 如何兼顾局部细节和全局信息？

Qwen2.5-VL 的视觉编码器使用 **2D-RoPE、窗口注意力，以及少量全局注意力层**。7B 配置中有 32 层，索引 `7、15、23、31` 使用全局注意力，其余使用窗口注意力，窗口大小为 112×112 像素，即 8×8 个 Patch。[官方配置](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)

- **窗口层**：主要在局部区域交换特征，控制高分辨率输入的计算成本。
- **全局层**：让不同窗口的信息交流，建立整张图的关系。
- **2D-RoPE**：在视觉注意力中表达行列位置，保留二维空间关系。

这是对“全部 Patch 两两计算注意力”的折中。固定窗口大小时，窗口注意力部分的成本随 Patch 数近似线性增加，但**整个模型仍包含全局注意力，不能笼统说整套模型是线性复杂度**。

### 4. Patch Merger：压缩序列，同时对齐维度

Qwen2.5-VL 的连接器不是 BLIP-2 的 Q-Former。它将空间相邻的四个特征拼接，再用两层 MLP 投影；输出数量随图像尺寸变化，不是固定 32 个 Query。

对 7B 模型，一组特征的计算可写成：

$$
u=\operatorname{Concat}(\operatorname{RMSNorm}(v_1),\ldots,\operatorname{RMSNorm}(v_4))
\in\mathbb{R}^{5120}
$$

$$
z=W_2\operatorname{GELU}(W_1u+b_1)+b_2\in\mathbb{R}^{3584}
$$

这同时完成 **4:1 的序列压缩**和 **5120→3584 的维度映射**。MLP 投影本身并不保证“不损失细节”；是否保留有用信息取决于压缩方式和训练。[Transformers 中的 Patch Merger 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)

### 5. 视觉向量怎么送进 LLM？

输入模板里使用 `<|vision_start|>`、`<|image_pad|>`、`<|vision_end|>` 等特殊标记。Processor 根据视觉网格，将图像占位位置展开到相应数量；模型再把这些位置的 Embedding 替换为 Merger 产生的视觉向量。

因此，示意输入为：

`用户角色标记 + 视觉开始标记 + 256个视觉向量 + 视觉结束标记 + 问题文本向量`

之后由同一个 Decoder 的自注意力处理。对于位于图像之后的回答 Token，其 Query 可以关注前面的视觉 Key/Value，从而根据图像生成文本。**融合发生在向量序列及后续注意力中，不需要先接一个独立 OCR 模型把整张图翻译成文字。**

下面是单张图像的概念伪代码，省略批处理、视频、边界处理和缓存逻辑，不能直接运行：

```python
# Processor 负责缩放、切块、展开图像占位符，并记录 image_grid_thw
inputs = processor(image=image, text=prompt)

# 以预处理后 448×448、Qwen2.5-VL-7B 为例
patches = patch_embed(inputs.pixel_values)          # [1024, 1280]
features = vision_transformer(patches, spatial_rope=True)
visual = patch_merger(features)                    # [256, 3584]

embeds = text_embedding(inputs.input_ids)
embeds[image_placeholder_mask] = visual            # 替换图像位置的向量
positions = build_mrope_positions(inputs.input_ids, inputs.image_grid_thw)
hidden = llm(inputs_embeds=embeds, position_ids=positions)
logits = lm_head(hidden)                           # 预测后续文本 Token
```

相关标记和模块见 [官方模型配置](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json) 与 [Transformers 模型实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py)。

### 6. MRoPE 与视频：从一维词序扩展到时空位置

前文的 RoPE 主要处理文本位置 $p$。Qwen2-VL / Qwen2.5-VL 的 **MRoPE** 则把旋转维度分配给时间、高度、宽度三个轴，对每个位置使用 $(t,h,w)$：

| 输入 | 位置表达方式 |
| --- | --- |
| 文本 | 三个轴使用同一个位置编号，等效于一维 RoPE |
| 单张图像 | 图像内部时间位置固定，行列位置随网格变化 |
| 视频 | 在空间网格之外增加时间位置 |

注意区分：**ViT 中的 2D-RoPE**负责视觉编码阶段的空间关系，**LLM 中的 MRoPE**负责混合输入序列的位置关系。[Qwen2-VL 技术报告](https://arxiv.org/html/2409.12191v1)

Qwen2.5-VL 视频处理进一步使用 **动态 FPS 训练**与**绝对时间对齐的 MRoPE**：时间编号与实际视频时间关联，使相同采样序号在不同 FPS 下不会被误当成相同时间。[官方说明](https://qwenlm.github.io/blog/qwen2.5-vl/)

其 Patch Embedding 的时间块大小为 2，空间块为 14×14，即把相邻两帧的对应区域共同编码；静态图片可通过重复帧适配同一接口。空间合并仍为 2×2。若采样后有偶数帧 $T$ 且各帧尺寸一致，则可推算：

$$
N_{visual,video}=\frac{T}{2}\frac{H'}{28}\frac{W'}{28}
$$

例如 16 帧、每帧 448×448，对应 $8×16×16=2048$ 个视觉 Token，不包含其他标记。增加帧数主要提高时间采样密度，提高分辨率主要增加空间细节。[时间与空间块参数](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/config.json)

### 7. 模型怎么学会“看图回答”？

视觉向量能够对齐语言语义，依赖训练数据和梯度更新。图像问答的生成目标可以概括为：

$$
\mathcal{L}_{answer}=-\sum_{i\in\text{回答位置}}\log P(y_i\mid I,q,y_{<i})
$$

这里 $I$ 是图像、$q$ 是问题、$y_i$ 是答案 Token。这是回答监督的概念表达；具体样本的 loss mask 由训练方案决定。预测答案的损失可以通过 LLM 回传到 Merger，若视觉编码器未冻结，还能继续更新 ViT。

**前面的“两阶段范式”是常见入门框架，不能直接当作 Qwen2.5-VL 的完整训练配方。** 其报告描述了视觉预训练、多模态预训练、长上下文预训练，随后进行 SFT 和 DPO。多模态训练包含全参数解冻阶段，数据覆盖 OCR、定位、文档、视频、Agent 和纯文本等任务。[训练流程与后训练说明](https://arxiv.org/html/2502.13923v1#S2)

例如定位任务用坐标形式的文本答案进行监督，模型学习生成坐标或 JSON；不能仅因为它会输出检测框，就推断它带有 YOLO 式检测头。

### 8. Qwen3-VL 相比之下改了什么？

Qwen3-VL 延续“视觉编码器 + MLP Merger + LLM”，但主要增加以下变化：[Qwen3-VL 官方技术报告，第 2 节](https://arxiv.org/html/2511.21631v1#S2)

| 改动 | 实现与作用 |
| --- | --- |
| 视觉编码器 | 从预训练 SigLIP-2 初始化，继续进行动态分辨率训练；不要套用 Qwen2.5-VL 从头训练 ViT 的描述 |
| Interleaved MRoPE | 在旋转频率维度交错分配时间、高度、宽度，使各轴都覆盖高低频，改善原有频率分配失衡 |
| DeepStack | 从 ViT 的三个层级提取特征，分别经 Merger 投影，残差加入 LLM 前三层的对应视觉位置；无需把这些特征全部追加成更长序列 |
| 显式视频时间戳 | 为视频时间块加入文本时间戳，替代 Qwen2.5-VL 仅靠绝对时间位置编号表达时间的方案 |

理解各代时，可以记住：**Qwen2-VL 的重点是动态分辨率与 MRoPE；Qwen2.5-VL 进一步优化视觉编码效率与视频时间建模；Qwen3-VL 增加多层视觉特征注入，并调整位置与时间表达。** [Qwen3-VL 官方架构概述](https://github.com/QwenLM/Qwen3-VL#introduction)

---
