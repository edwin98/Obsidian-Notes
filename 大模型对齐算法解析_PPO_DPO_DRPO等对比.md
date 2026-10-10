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

DPO 是一种常用的离线偏好对齐方法，省去显式奖励模型训练和在线强化学习采样，降低了工程复杂度。

### 原理简述

DPO 从 KL 正则化奖励最大化问题的最优策略关系出发，用策略与参考策略的对数概率比表示奖励，再代入 Bradley–Terry 偏好模型，得到直接优化策略的损失；这一推导不是直接应用贝叶斯定理。

简单来说，DPO 使用同一对（Chosen $y_w$ / Rejected $y_l$）数据，直接去优化当前模型，目标是**拉大"好回复"相对于参考概率的比值，并降低"差回复"相对于参考概率的比值**。

DPO 包含当前策略与参考策略两个角色。参考对数概率可预计算，PEFT 场景也可通过禁用适配器使用共享底座；不一定需要两份完整模型同时驻留。

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

- **优点**：省去独立奖励模型、Critic 和在线采样，通常比 PPO 更容易实现。具体显存节省取决于模型部署、优化器与参考策略实现，不能固定说减半。
- **局限**：依赖偏好数据质量与覆盖范围，缺少在线探索；仍能泛化到未见样本。与 PPO、GRPO 的性能差异需要在相同底座、数据和预算下评测，不能预设算法上限。

### 适用场景

- **绝大多数开源微调场景与企业内落地**：计算资源有限、期望追求稳定产出的普通开发者或中小型 AI 团队。
- **快速验证与偏好注入**：将特定的垂域偏好（例如让客服大模型学会特定口吻、避免答非所问）快速通过 paired data 注入到预训练模型中。
- 大多数 Llama 3、Qwen、Mistral 等开源模型的官方 Instruct 版本，均重度使用了 DPO。

---

## 3. GRPO (Group Relative Policy Optimization)

GRPO 由 DeepSeek 团队在 **2024 年 2 月的 DeepSeekMath 论文**中提出，随后用于 DeepSeek-R1。它用组内奖励统计量估计优势，省去独立 Critic。[DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300)。

### 核心思路：用"组内相对评分"替代 Critic 网络

常见 PPO 实现使用 Critic 估计状态价值，它增加参数、优化器与训练开销；Critic 的规模和与 Actor 的参数共享方式依具体实现而定。

GRPO 的核心创新是：**将 Critic 网络彻底去掉，转而用同一批"组内采样"的结果来估计基准线（Baseline）**。

具体做法是：
1. 对于每一个输入 Prompt $q$，从当前策略（旧策略 $\pi_{\theta_{old}}$）中**采样一组（Group）响应** $\{o_1, o_2, ..., o_G\}$（论文中典型值 $G = 8$ 或 $G = 16$）。
2. 用奖励模型（或基于规则的验证器）对这 $G$ 个回复分别打分，得到 $\{r_1, r_2, ..., r_G\}$。
3. **以组内所有奖励的均值和标准差对每个奖励做归一化**，得到相对优势（Relative Advantage）：
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$
4. 用这个归一化后的 $\hat{A}_i$ 替代 PPO 中由 Critic 产生的 GAE 优势估计，推动 Actor 的更新。

GRPO 包含 Actor、参考策略和奖励来源三个角色；规则验证器不是神经网络模型，也不一定需要三份模型同时驻留。

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

使用规则验证器时，GRPO 可不训练独立 Reward Model，但仍有 rollout 和参考策略计算开销。可验证奖励能减少学习型奖励模型的偏差，**不能消除 Reward Hacking**：验证器漏洞、测试覆盖不足或环境缺陷仍可能被利用。

### 优缺点

- **优点**：
  - 省去独立 Critic，可降低训练开销；显存节省比例随实现与 rollout 配置而变。
  - 保留在线采样与探索能力，实际效果依赖奖励质量、数据和训练预算。
  - 使用可验证奖励时可省去奖励模型训练，但需要防范验证器被利用。
  - 组内奖励标准差为零时需稳定处理，常用 epsilon 或跳过无区分度组。
  - 已在 DeepSeekMath、DeepSeek-R1 等数学和代码推理任务中得到应用；不构成对所有任务的性能保证。
- **缺点**：
  - 相比 DPO，仍然需要进行在线采样（每个 prompt 要采 G 条回复），**推理（Sampling）开销较大**，对吞吐量有一定压力；
  - 在无法设计精确可验证奖励的主观任务（如写作风格、情感表达）上，仍需要奖励模型，复杂度与 PPO 类似；
  - $G$ 的选择（组内采样数量）对显存和效果有较大影响，需要仔细调节。

### 适用场景

- **数学、代码等可验证推理任务**：可考虑 GRPO + 规则验证器，并独立评估奖励可靠性。
- **希望开展在线 RL 的团队**：省去 Critic 后仍需承担组内采样、参考策略与优化开销。
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
- **原理**：基于前景理论（Prospect Theory）中人类避险心理对效用的刻画，只需要给一个回答正反馈（$y_{\text{desirable}}$）或负反馈（$y_{\text{undesirable}}$），就可以进行单侧优化，不需要强行构造对碰的（Chosen ↔ Rejected）回答对。

  **数学公式表达**：
  令 $r_\theta(x,y)=\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$，KL 基准为 $z_0=\mathbb{E}_x[\mathrm{KL}(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))]$。效用按正负反馈采用相反方向：
  $$v_\theta(x,y)=\begin{cases}\lambda_D\sigma(\beta(r_\theta(x,y)-z_0)), & y\text{ 为理想回答}\\\lambda_U\sigma(\beta(z_0-r_\theta(x,y))), & y\text{ 为非理想回答}\end{cases}$$
  $$\mathcal{L}_{\mathrm{KTO}}=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\lambda_y-v_\theta(x,y)]$$
  其中 $\lambda_y$ 对应 $\lambda_D$ 或 $\lambda_U$；实现中用批内不匹配的输入/输出估计 KL 基准，并停止对该基准反向传播。[KTO 原论文第 4.1 节](https://arxiv.org/html/2402.01306v4#S4.SS1)。

- **适用场景**：现实世界中极其难以获取同源的成对偏好数据集，但很容易收集单一条目的点赞/点踩数据时（例如直接在现有 App 中埋点收集真实用户反馈后对齐）。

---

## 5. 对比总结表格

为了更直观地展示各算法的选型区别，总结如下：

| 对比维度 | PPO | DPO | GRPO | ORPO | KTO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **模型角色** | Actor、Critic、奖励来源、Ref | Policy、Ref（可共享或预计算） | Actor、奖励来源、Ref（无 Critic） | Policy | Policy、Ref |
| **基础原理** | 奖励模型 + Actor-Critic 强化学习 | 隐式奖励最大化（数学等效 RL） | 组内相对优势估计替代 Critic，On-policy RL | 基于 Odds Ratio 的 SFT 合并改进 | 基于前景理论的单向偏好损失 |
| **是否需要 Reward Model** | ✅ 需要 | ❌ 不需要 | ⚠️ 可选（可用规则验证器替代） | ❌ 不需要 | ❌ 不需要 |
| **是否有 On-policy 采样** | ✅ 有（RL 探索） | ❌ 无（Off-policy） | ✅ 有（组内采样 G 个回复） | ❌ 无 | ❌ 无 |
| **显存消耗** | 🔴🔴🔴 极高 | 🔴🔴 中等 | 🔴🔴 中高（少一个 Critic） | 🟢 极低 | 🔴🔴 中等 |
| **训练数据格式** | Prompt + 奖励信号；学习型 RM 可另用偏好数据训练 | 成对偏好数据 | Prompt + 奖励信号 | 成对偏好数据 | 单个正/负反馈 |
| **稳定性与调参** | 需调 KL、裁剪、学习率等 | 需调 beta、数据与学习率 | 需调组大小、裁剪、KL 等 | 需调偏好权重与学习率 | 需调正负权重、beta 等 |
| **性能比较** | 依赖底座、奖励与预算 | 依赖底座、偏好数据与预算 | 依赖底座、奖励与预算 | 依赖底座、偏好数据与预算 | 依赖底座、反馈数据与预算 |
| **奖励被利用风险** | 可能利用奖励模型或验证器 | 仍需评估代理目标偏差 | 使用规则时仍可能利用验证器；使用 RM 时仍有 RM 偏差 | 仍需评估代理目标偏差 | 仍需评估反馈及代理目标偏差 |
| **典型应用场景** | 在线 RLHF | 离线业务偏好对齐 | DeepSeek-R1 等推理训练 | 资源受限的偏好微调 | 单条正负反馈对齐 |

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
