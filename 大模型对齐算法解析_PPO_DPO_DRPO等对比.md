# 大模型对齐算法解析：PPO、DPO、GRPO 等核心算法深度对比

在大语言模型（LLM）的训练流水线中，**对齐（Alignment）** 是使得模型输出符合人类偏好、价值观并具备安全性的关键步骤。传统的 SFT（有监督微调）教会模型"如何回复"，而基于偏好优化的对齐阶段则教会模型"如何更好地回复"。

本文将详细对比目前主流及前沿的对齐优化算法：**PPO**、**DPO**、**GRPO**，并延伸介绍部分衍生的高效算法（如 **ORPO**、**KTO** 等），探讨它们的异同与适用场景。

---

## 1. PPO (Proximal Policy Optimization)

PPO 即近端策略优化，是 RLHF（基于人类反馈的强化学习）框架中最经典、也是 ChatGPT 早期使用的核心对齐算法。

### 原理简述

PPO 将 LLM 视为在环境中执行"动作"（生成 token）的智能体（Agent）。训练通常分为几个独立阶段：

1. 训练一个独立的 **奖励模型 (Reward Model, RM)**：通过人类偏好数据集（如对于同一 Prompt 的好坏回答对）训练 RM，使其能够对回复进行打分。
2. 使用强化学习迭代：PPO 利用 RM 的打分作为强化学习的奖励信号（Reward），通过 **Actor-Critic 架构**去优化生成策略（Actor Model）。为了防止模型一味追求高分而偏离原语言分布，PPO 会引入参考模型（Reference Model），通过计算 KL 散度（KL Penalty）来限制策略更新幅度。

PPO 的完整流水线需要同时维护 **4 个模型**：

| 模型 | 作用 | 是否可训练 |
| :--- | :--- | :---: |
| **Actor Model** | 当前策略，负责生成回复 | ✅ 是 |
| **Critic Model** | 评估当前状态的价值函数（Value Function），用于计算优势函数 | ✅ 是 |
| **Reward Model** | 对生成的回复打分 | ❌ 冻结 |
| **Reference Model** | 提供 KL 约束基准（通常是 SFT 后的模型） | ❌ 冻结 |

**数学公式表达**：

PPO 对齐阶段的核心目标函数，通过优化以下目标来最大化人类偏好（同时通过 KL 散度惩罚防止偏离）：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) - \beta \mathbb{D}_{\text{KL}} \left[\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right] \right]$$

其中 Actor 的实际更新采用了截断（Clip）机制，以限制单步更新幅度：

$$\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t,\ \text{clip}(r_t(\theta),\ 1-\varepsilon,\ 1+\varepsilon) \hat{A}_t \right) \right]$$

- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：新旧策略的概率比
- $\hat{A}_t$：由 Critic 网络计算的优势函数估计（GAE）
- $\varepsilon$：截断系数（通常取 0.1～0.2），防止策略更新过大

### 优缺点

- **优点**：泛化上限高，理论上通过 RL 的主动探索能在复杂的开放论述或长文本推理中取得极高的效果；对抗鲁棒性较好。
- **缺点**：系统复杂度极高，**极度吃显存**。一条 PPO 流水线需要同时驻留 4 个模型（Actor、Critic、Reward、Reference），导致训练时资源开销巨大；超参数繁杂（如 KL 惩罚系数、学习率、PPO 截断系数等），极容易发生崩溃或由于奖励模型缺陷导致"**Reward Hacking（欺骗奖励）**"。

### 适用场景

- **顶尖基础大模型的攻坚对齐**：当具有极强算力、完善的安全对齐团队时（如 OpenAI、Anthropic），常选用 PPO 去挖掘模型最高上限。
- **复杂推理与代码对齐**：适合那些不只是单次偏好判定，还需要进行多步探索、推理链路优化的场景（如带有验证器的复杂数学推理环境）。

---

## 2. DPO (Direct Preference Optimization)

DPO 是目前开源社区中最受欢迎、使用最广泛的偏好对齐算法方案，被视为"干翻 PPO"的轻量化革命性工作。

### 原理简述

DPO 彻底抛弃了显式的奖励模型（Reward Model）以及繁杂的强化学习层。作者在数学上证明了：通过 KL 散度约束的奖励最大化目标，可以通过贝叶斯定理直接映射回策略（模型权重本身）的最优化目标上。

简单来说，DPO 使用同一对（Chosen $y_w$ / Rejected $y_l$）数据，直接去优化当前模型，目标是**拉大"好回复"相对于参考概率的比值，并降低"差回复"相对于参考概率的比值**。

DPO 只需要同时维护 **2 个模型**：

| 模型 | 作用 | 是否可训练 |
| :--- | :--- | :---: |
| **Policy Model** | 正在优化的当前策略 | ✅ 是 |
| **Reference Model** | 提供对数概率基准 | ❌ 冻结 |

**数学公式表达**：

DPO 的对齐损失函数（Loss）构建如下，可以直接作为类似于交叉熵的模型微调损失：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

- $y_w$：Preferred response（标注中的"好回复"）
- $y_l$：Dispreferred response（标注中的"差回复"）
- $\sigma$：Sigmoid 激活函数
- $\beta$：控制与参考模型偏离程度的惩罚系数

由此可见，DPO 的训练本质是对这组差异化的对数概率建立二元交叉熵损失，极大地简化了训练过程。

### 优缺点

- **优点**：极其简单、高效。只需要加载当前模型（Policy）和参考模型（Reference），显存要求直接减半；训练与常规的监督微调（SFT）无异，不需要做 RL 的探索（On-policy Sampling），训练过程极其稳定；几乎不发生 Reward Hacking 问题。
- **缺点**：完全受限于标注数据的质量，因为缺乏强化学习的自主探索特性，模型不能生成超出数据覆盖情况的"新奇优质答案"；在长序列和极端复杂逻辑推理中的对齐上限，可能略逊色于完美调优的 PPO 或 GRPO。

### 适用场景

- **绝大多数开源微调场景与企业内落地**：计算资源有限、期望追求稳定产出的普通开发者或中小型 AI 团队。
- **快速验证与偏好注入**：将特定的垂域偏好（例如让客服大模型学会特定口吻、避免答非所问）快速通过 paired data 注入到预训练模型中。
- 大多数 Llama 3、Qwen、Mistral 等开源模型的官方 Instruct 版本，均重度使用了 DPO。

---

## 3. GRPO (Group Relative Policy Optimization)

GRPO 是由 **DeepSeek 团队**在 2024 年底提出并在 **DeepSeek-R1** 中大放异彩的新型强化学习对齐算法。它是目前最受瞩目的前沿对齐方法之一，被认为是在 PPO 基础上的重大工程性突破——在**保留 RL 探索能力的同时，大幅降低了对 Critic 网络的依赖和显存消耗**。

### 核心思路：用"组内相对评分"替代 Critic 网络

PPO 最大的工程性痛点在于需要一个与 Actor 同等规模的 **Critic 网络**来估计每个状态的价值函数（Value Function）。这个 Critic 网络既占显存，又难以训练。

GRPO 的核心创新是：**将 Critic 网络彻底去掉，转而用同一批"组内采样"的结果来估计基准线（Baseline）**。

具体做法是：
1. 对于每一个输入 Prompt $q$，从当前策略（旧策略 $\pi_{\theta_{old}}$）中**采样一组（Group）响应** $\{o_1, o_2, ..., o_G\}$（论文中典型值 $G = 8$ 或 $G = 16$）。
2. 用奖励模型（或基于规则的验证器）对这 $G$ 个回复分别打分，得到 $\{r_1, r_2, ..., r_G\}$。
3. **以组内所有奖励的均值和标准差对每个奖励做归一化**，得到相对优势（Relative Advantage）：
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$
4. 用这个归一化后的 $\hat{A}_i$ 替代 PPO 中由 Critic 产生的 GAE 优势估计，推动 Actor 的更新。

GRPO 只需要同时维护 **3 个模型**（相比 PPO 少了 Critic）：

| 模型 | 作用 | 是否可训练 |
| :--- | :--- | :---: |
| **Actor Model** | 当前策略，负责生成回复 | ✅ 是 |
| **Reward Model** | 对组内各回复打分（也可用规则验证器代替） | ❌ 冻结 |
| **Reference Model** | 提供 KL 约束基准，防止偏离原分布 | ❌ 冻结 |

### 数学公式表达

GRPO 的训练目标函数为：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q) \\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_{i,t}(\theta)\,\hat{A}_i,\; \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}_i \right) - \beta\, \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right) \right]$$

其中：

- $q$：输入的 Prompt（从训练集中采样）
- $o_i$：第 $i$ 个从旧策略 $\pi_{\theta_{old}}$ 采样出的完整回复（共 $G$ 条）
- $r_{i,t}(\theta) = \dfrac{\pi_\theta(o_{i,t} \mid q,\, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q,\, o_{i,<t})}$：新旧策略在第 $t$ 个 token 上的概率比
- $\hat{A}_i = \dfrac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$：**组内归一化的相对优势**，由奖励组统计量计算，完全替代 Critic 网络
- $\varepsilon$：PPO 截断系数，防止策略单步更新过大
- $\beta$：KL 散度惩罚系数

> **关键对比**：在 PPO 中，$\hat{A}_t$ 由 Critic 网络通过 GAE（广义优势估计）算出，需要额外的 Critic 模型；而在 GRPO 中，$\hat{A}_i$ 完全由**组内统计量**（组均值和组方差）给出，无需任何 Critic 参数。

### GRPO 与可验证奖励（Verifiable Reward）的结合

GRPO 之所以在 DeepSeek-R1 中大放异彩，还有一个重要原因：它可以配合**基于规则的可验证奖励（Rule-based Verifiable Reward）**使用，从而完全跳过训练奖励模型的步骤：

- **数学题**：验证最终答案是否正确（正确 +1，错误 0）
- **代码题**：运行生成代码并检查测试用例是否通过
- **格式规范**：检查是否包含正确的 `<think>` 和 `<answer>` 标签

这种"**无 Reward Model 的 GRPO**"使得整个 RL pipeline 只需 Actor + Reference 两个模型，系统复杂度进一步大幅降低，同时完全避免了 Reward Hacking（因为奖励是真实计算结果，无法被欺骗）。

### 优缺点

- **优点**：
  - 相比 PPO，**去掉了 Critic 模型**，显存消耗显著降低（对于 7B 以上大模型，节省约 25% 显存）；
  - **保留了 RL 的在线探索（On-policy Sampling）能力**，可以产生训练数据分布之外的"超水平"回复，性能上限远超 DPO；
  - 配合可验证奖励使用时，**完全无需奖励模型**，避免了奖励模型本身的偏差和 Reward Hacking 问题；
  - 组内对比的方式使奖励基线估计自然、稳定，训练崩溃风险较 PPO 明显降低；
  - 在数学推理、代码生成、逻辑链路 (Chain-of-Thought) 等可验证任务上表现极其出色（DeepSeek-R1 SOTA 佐证）。
- **缺点**：
  - 相比 DPO，仍然需要进行在线采样（每个 prompt 要采 G 条回复），**推理（Sampling）开销较大**，对吞吐量有一定压力；
  - 在无法设计精确可验证奖励的主观任务（如写作风格、情感表达）上，仍需要奖励模型，复杂度与 PPO 类似；
  - $G$ 的选择（组内采样数量）对显存和效果有较大影响，需要仔细调节。

### 适用场景

- **数学、代码等可验证推理任务**：GRPO + 规则验证器是目前训练推理模型的最优实践（DeepSeek-R1、Qwen-2.5-Math 等均使用此方案）。
- **中等算力下追求 RL 探索上限**：希望超越 DPO 的上限，又不具备 PPO 完整 4 模型流水线算力的团队。
- **需要激发长链推理（Chain-of-Thought）能力的场景**：GRPO 的 RL 探索特性使模型能自发学会更完整的推理步骤。

---

## 4. 其他主流衍生对齐算法（ORPO、KTO）

除了上述三者，DPO 流行之后，学术界诞生了大批提升训练效率的变体算法，同样具有极高热度：

### 4.1 ORPO (Monolithic Preference Optimization without Reference Model)

- **特点**：连 DPO 所需要的那个 Reference Model（冻结的旧模型）也干掉了，只需 **1 个模型**。
- **原理**：直接将偏好损失项以"附加罚项（Odds Ratio）"的形式加到 SFT 损失函数的后面。这意味着**不需要先做 SFT 再做 DPO，而是一步到位直接基于带有偏好的数据完成 SFT 与对齐**。

  **数学公式表达**：
  $$\mathcal{L}_{\text{ORPO}} = \mathbb{E}\left[\mathcal{L}_{\text{SFT}}\right] + \lambda\, \mathbb{E}\left[\mathcal{L}_{\text{OR}}\right]$$

  对于其中的几率比损失 $\mathcal{L}_{\text{OR}}$：
  $$\mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)$$

  其中定义几率（Odds）为：$\text{odds}_\theta(y|x) = \dfrac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}$

- **适用场景**：极度缺乏显存算力的场景（例如单卡 24G/40G 微调 7B / 14B 甚至 32B 模型时资源受限的开发者）。

### 4.2 KTO (Kahneman-Tversky Optimization)

- **特点**：无需成对数据（Pairwise Data），只需单个正/负反馈。
- **原理**：基于前景理论（Prospect Theory）中人类避险心理对效用的刻画，只需要给一个回答点赞（👍，$y_{\text{desirable}}$）或踩（👎，$y_{\text{undesirable}}$），就可以进行单侧优化，不需要强行构造对碰的（Chosen ↔ Rejected）回答对。

  **数学公式表达**：
  KTO 构建了评估价值 $v_\theta(x,y) = \beta \log \dfrac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$，并设计了非对称损失：
  $$\mathcal{L}_{\text{KTO}}(\pi_\theta, \pi_{\text{ref}}) = \mathbb{E}_{x,y \sim \mathcal{D}} \left[ w(y) \log \sigma \big(z_{\text{ref}} - v_\theta(x, y) \big) \right]$$
  这能够针对 $y$ 是理想回答还是非理想回答，自动应用差异化的价值偏好与更新权重 $w(y)$。

- **适用场景**：现实世界中极其难以获取同源的成对偏好数据集，但很容易收集单一条目的点赞/点踩数据时（例如直接在现有 App 中埋点收集真实用户反馈后对齐）。

---

## 5. 对比总结表格

为了更直观地展示各算法的选型区别，总结如下：

| 对比维度 | PPO | DPO | GRPO | ORPO | KTO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **所需模型数量** | 4 个（Actor + Critic + Reward + Ref） | 2 个（Policy + Ref） | 3 个（Actor + Reward/验证器 + Ref） | 1 个 | 2 个（Policy + Ref） |
| **基础原理** | 奖励模型 + Actor-Critic 强化学习 | 隐式奖励最大化（数学等效 RL） | 组内相对优势估计替代 Critic，On-policy RL | 基于 Odds Ratio 的 SFT 合并改进 | 基于前景理论的单向偏好损失 |
| **是否需要 Reward Model** | ✅ 需要 | ❌ 不需要 | ⚠️ 可选（可用规则验证器替代） | ❌ 不需要 | ❌ 不需要 |
| **是否有 On-policy 采样** | ✅ 有（RL 探索） | ❌ 无（Off-policy） | ✅ 有（组内采样 G 个回复） | ❌ 无 | ❌ 无 |
| **显存消耗** | 🔴🔴🔴 极高 | 🔴🔴 中等 | 🔴🔴 中高（少一个 Critic） | 🟢 极低 | 🔴🔴 中等 |
| **训练数据格式** | 成对偏好数据（Chosen / Rejected） | 成对偏好数据 | Prompt 即可（自动在线采样） | 成对偏好数据 | 单个正/负反馈 |
| **稳定性与调参难度** | 🔴🔴 难，易崩溃，需精调 KL 系数 | 🟢 极稳定，只需调 $\beta$ | 🟡 较稳定，需调 $G$、$\varepsilon$、$\beta$ | 🟢 极稳定，几乎无痛 | 🟢 稳定 |
| **性能上限** | ⭐⭐⭐⭐⭐ 理论最高，可超过数据分布 | ⭐⭐⭐ 受数据质量上限限制 | ⭐⭐⭐⭐⭐ 高，RL 探索能力强 | ⭐⭐⭐ 依赖 SFT 数据质量 | ⭐⭐⭐ 较高，释放标注成本 |
| **Reward Hacking 风险** | 🔴 高 | 🟢 极低 | ⚠️ 使用规则验证器时为零，使用 RM 时中等 | 🟢 极低 | 🟢 极低 |
| **典型应用场景** | 闭源大厂底座训练、ChatGPT 等 | 90% 的开源微调团队、业务对齐 | **DeepSeek-R1**、Qwen-Math、推理模型训练 | 低算力单卡微调 | 产品埋点用户反馈驱动的对齐 |

### 算法演进关系示意

```
                              RLHF 对齐算法演进路线
                              
     [PPO] ──── 去掉Critic，用组内统计基线替代 ────▶ [GRPO]
       │                                               │
       │ 去掉RM和RL层，                                │ 用规则验证器
       │ 映射为监督目标                                │ 替代 Reward Model
       ▼                                               ▼
     [DPO] ──── 去掉Reference Model ────▶ [ORPO]    [GRPO + Verifiable Reward]
       │                                             (DeepSeek-R1 核心方案)
       │ 去掉成对数据需求
       ▼
     [KTO]
```

### 💡 给开发者的落地建议

1. **初期探索与常规偏好对齐**：如果拥有成对标注数据，毫不犹豫**首选 DPO**。生态最成熟，社区支持最好，Hugging Face TRL 库原生支持。

2. **数学 / 代码 / 逻辑推理任务**：优先考虑 **GRPO + 规则验证器**，无需奖励模型，性能上限极高。这是当前训练推理模型（Reasoning Model）的最佳实践。

3. **显存极度拮据**：如果在单卡消费级显卡上训练，且希望一步到位跳过 SFT，可以选择 **ORPO**。

4. **真实用户产品反馈收集**：如果产品已经上线且数据只有点赞点踩，使用 **KTO** 极其方便，不需要配对数据。

5. **追逐最强天花板或多维复杂对话对齐**：搭建成熟的 RL 团队去碰 **PPO**，需要极强基建（如 vLLM 加速 RL 采样、Ray 分布式框架等）。

6. **GRPO vs PPO 的选择原则**：如果任务**可以设计验证器**（如答案可判对错），优先用 GRPO；如果任务是主观开放式生成（如写作、对话）且有足够算力，可以考虑 PPO。
