---
aliases:
  - "大模型对齐算法解析_PPO_DPO_DRPO等对比"
---

# 大模型对齐算法解析：PPO、DPO、GRPO 等核心算法深度对比

在大语言模型（LLM）的训练流水线中，**对齐（Alignment）** 是使得模型输出符合人类偏好、价值观并具备安全性的关键步骤。传统的 SFT（有监督微调）教会模型"如何回复"，而基于偏好优化的对齐阶段则教会模型"如何更好地回复"。

本文将详细对比目前主流及前沿的对齐优化算法：**PPO**、**DPO**、**GRPO**，并延伸介绍部分衍生的高效算法（如 **ORPO**、**KTO** 等），探讨它们的异同与适用场景。

---

## 1. PPO (Proximal Policy Optimization)

PPO 即近端策略优化，是 RLHF（基于人类反馈的强化学习）框架中最经典、也是 ChatGPT 早期使用的核心对齐算法。

### 原理简述

PPO 将 LLM 视为在环境中执行"动作"（生成 token）的智能体（Agent）。训练通常分为几个独立阶段：

1. 在学习型奖励的经典 RLHF 配方中，训练一个独立的 **奖励模型 (Reward Model, RM)**：通过人类偏好数据集（如对于同一 Prompt 的好坏回答对）训练 RM，使其能够对回复进行打分。
2. 使用强化学习迭代：PPO 利用 RM 的打分作为强化学习的奖励信号（Reward），通过 **Actor-Critic 架构**去优化生成策略（Actor Model）。为了防止模型一味追求高分而偏离原语言分布，PPO 会引入参考模型（Reference Model），通过计算 KL 散度（KL Penalty）来限制策略更新幅度。

经典 PPO-RLHF 常包含以下四个角色。奖励也可来自规则或环境，价值头可共享参数，参考概率可另行计算，因此不一定是四份完整模型同时驻留：

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

- **优点**：支持在线采样和基于奖励的策略优化，可用于多步交互与复杂任务。
- **代价**：Actor、Critic、奖励计算、参考策略和 rollout 增加系统开销，资源需求取决于共享、分片和部署配置。需要调节裁剪、学习率、KL 等，奖励模型或验证器缺陷仍可能导致 Reward Hacking。

### 适用场景

适合具备可靠奖励、交互或采样环境及相应训练基础设施的在线策略优化任务。PPO 可以使用偏好奖励或可验证奖励，不能预设其在所有任务上的性能上限。

---

## 2. DPO (Direct Preference Optimization)

DPO 是一种常用的离线偏好对齐方法，省去显式奖励模型训练和在线强化学习采样，降低了工程复杂度。

### 原理简述

DPO 从 KL 正则化奖励最大化问题的最优策略关系出发，用策略与参考策略的对数概率比表示奖励，再代入 Bradley–Terry 偏好模型，得到直接优化策略的损失；这一推导不是直接应用贝叶斯定理。

简单来说，DPO 使用同一问题下的偏好对（Chosen $y_w$ / Rejected $y_l$）直接优化当前模型，目标是**相较参考模型，提高 Chosen 对 Rejected 的相对概率比**。这不保证 Chosen 的绝对概率一定提高，也不要求 Rejected 在所有方面都错误。

DPO 包含当前策略与参考策略两个角色。参考对数概率可预计算，PEFT 场景也可通过禁用适配器使用共享基础模型；不一定需要两份完整模型同时驻留。

| 模型 | 作用 | 是否可训练 |
| :--- | :--- | :---: |
| **Policy Model** | 正在优化的当前策略 | 是 |
| **Reference Model** | 提供对数概率基准 | 冻结 |

**数学公式表达**：

DPO 的对齐损失函数（Loss）构建如下，可以直接作为类似于交叉熵的模型微调损失：

$$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

- $y_w$：Preferred response（标注中的"好回复"）
- $y_l$：Dispreferred response（标注中的"差回复"）
- $\sigma$：Sigmoid 激活函数
- $\beta$：控制与参考模型偏离程度的惩罚系数

由此可见，DPO 的训练本质是对这组差异化的对数概率建立二元交叉熵损失，极大地简化了训练过程。

### 2.1 从 SFT 到偏好分差：损失函数在比较什么

以下讨论原始 sigmoid DPO。SFT 使用问题与示范回答 $(x,y)$，通过最小化 $-\log\pi_\theta(y\mid x)$ 提高示范回答的概率；DPO 使用三元组 $(x,y_w,y_l)$，监督信号是“同一问题下，$y_w$ 比 $y_l$ 更好”。

例如，“如何重置设备”对应两个回答：Chosen 引用了手册中的正确步骤，Rejected 编造了步骤。DPO 学习这个相对偏好，不需要标注两条回答各自的绝对奖励分数。

定义相对于参考模型的偏好分差：

$$
\Delta_\theta
=\log\frac{\pi_\theta(y_w\mid x)}{\pi_\theta(y_l\mid x)}
-\log\frac{\pi_{\mathrm{ref}}(y_w\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}
$$

单条样本的损失就可以写为：

$$
\ell(\theta)=-\log\sigma(\beta\Delta_\theta)
$$

这里 $\pi_\theta$ 是当前模型，$\pi_{\mathrm{ref}}$ 是冻结的参考模型，通常取 DPO 开始前的 SFT 模型。$\sigma(z)=1/(1+e^{-z})$，且 $\beta>0$。

如果参考模型的 Chosen/Rejected 概率比为 2，当前模型的概率比变为 6，则：

$$
\Delta_\theta=\log 6-\log 2=\log 3
$$

这表示相较参考模型，当前模型对 Chosen 的相对倾向提高到原来的 3 倍。参考模型为回答原本就存在的概率差异提供基准；它不要求训练开始前两条回答等概率。

当当前模型与参考模型完全相同时，$\Delta_\theta=0$，损失为 $-\log 0.5\approx0.693$。这是正常初始状态，此时梯度并不为零。

令 $z=\beta\Delta_\theta$：

| $z$ | $\sigma(z)$ | 损失 $-\log\sigma(z)$ |
| ---: | ---: | ---: |
| -2 | 0.119 | 2.127 |
| 0 | 0.500 | 0.693 |
| 2 | 0.881 | 0.127 |

分差越大，损失越小。**$\sigma(z)$ 是隐含偏好模型给出的“Chosen 胜过 Rejected”的概率，不是生成 Chosen 全文的概率。**

### 2.2 从 KL 正则化奖励目标推导 DPO

DPO 从下面的奖励最大化目标出发：

$$
\max_\pi\;\mathbb E_{x\sim\mathcal D}
\left[
\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}\bigl(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\bigr)
\right]
$$

它要求提高期望奖励，同时惩罚偏离参考模型的程度。在相应支持集与可表示性等假设下，最优策略满足：

$$
\pi^*(y\mid x)=\frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
$$

其中 $Z(x)$ 是使回答概率之和为 1 的归一化项。整理可得：

$$
r(x,y)=\beta\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+\beta\log Z(x)
$$

接着采用 Bradley–Terry 偏好模型：

$$
P(y_w\succ y_l\mid x)=\sigma\bigl(r(x,y_w)-r(x,y_l)\bigr)
$$

两个回答共享同一个问题 $x$，所以奖励相减时，$\beta\log Z(x)$ 抵消。用当前策略 $\pi_\theta$ 参数化上述策略，得到：

$$
P_\theta(y_w\succ y_l\mid x)=\sigma(\beta\Delta_\theta)
$$

对标注偏好做最大似然估计，即最小化其负对数概率，就得到 DPO 损失。也可以定义隐式奖励 $\hat r_\theta(x,y)=\beta\log[\pi_\theta(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]$，从而把 DPO 看成直接拟合隐式奖励差。

**关键是把奖励重新参数化为策略概率比，省去独立奖励模型及后续 PPO 更新。** 理论关系依赖上述建模假设，不代表有限偏好数据上的优化一定得到理想策略。

### 2.3 梯度逐步求导：为什么错误偏好的纠正信号更强

把 $\Delta_\theta$ 暂时写成 $\Delta$，将 $\beta$ 视为常数：

$$
\ell=-\log\sigma(\beta\Delta)
$$

第一步，利用链式法则，依次对负对数、sigmoid 和 $\beta\Delta$ 求导：

$$
\frac{\partial\ell}{\partial\Delta}
=-\frac{1}{\sigma(\beta\Delta)}\cdot\sigma'(\beta\Delta)\cdot\beta
$$

第二步，代入 sigmoid 导数 $\sigma'(z)=\sigma(z)(1-\sigma(z))$：

$$
\begin{aligned}
\frac{\partial\ell}{\partial\Delta}
&=-\frac{\beta\sigma(\beta\Delta)(1-\sigma(\beta\Delta))}{\sigma(\beta\Delta)}\\
&=-\beta(1-\sigma(\beta\Delta))
\end{aligned}
$$

第三步，由 $1-\sigma(z)=\sigma(-z)$，得到：

$$
\boxed{\frac{\partial\ell}{\partial\Delta}=-\beta\sigma(-\beta\Delta)}
$$

当 $\beta>0$ 时，这个导数始终为负，说明增大偏好分差能降低损失；但模型参数的实际更新还要继续应用链式法则：

$$
\nabla_\theta\ell
=-\beta\sigma(-\beta\Delta_\theta)
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-\nabla_\theta\log\pi_\theta(y_l\mid x)
\right]
$$

参考模型冻结，因此它的对数概率不对 $\theta$ 求导，但仍影响分差及样本权重。

| 分差状态 | $\partial\ell/\partial\Delta$ | 含义 |
| --- | --- | --- |
| $\beta\Delta\ll0$ | 接近 $-\beta$ | 相对参考模型的变化与标注偏好相反，纠正权重较大 |
| $\Delta=0$ | $-\beta/2$ | 相对参考模型尚未形成额外偏好 |
| $\beta\Delta\gg0$ | 接近 0 | 已形成足够大的正向分差，更新权重逐渐减小 |

这里“信号更强”指前面的标量权重更大。真实参数更新还取决于两个回答的对数概率梯度、学习率和优化器，不能仅由分差判断更新幅度，也不能保证每条样本的分差在每次批量更新后都增加。

### 2.4 β、概率计算与训练中的常见误解

**β 的两个作用角度**：在固定奖励的原始目标中，β 越大，KL 惩罚越强，最优策略越接近参考模型；在 DPO 损失中，β 同时缩放分差，影响梯度大小和 sigmoid 饱和速度。因此不能只凭 β 判断实际训练是否更激进，还要结合学习率、训练步数和数据。标准 DPO 没有单独计算的 KL 损失项，也不提供实际 KL 的硬上限。

**回答概率如何计算**：在已知回答上使用 teacher forcing，把回答 token 的条件对数概率求和：

$$
\log\pi_\theta(y\mid x)=\sum_{t=1}^{T}\log\pi_\theta(y_t\mid x,y_{<t})
$$

问题提供上下文，不计入回答得分，padding 也应排除。原始 DPO 使用完整回答的求和对数概率；改成按长度平均会改变目标，不能视为完全相同的实现。

每条偏好样本需要当前模型与参考模型分别对 Chosen、Rejected 打分，共四个序列对数概率。训练流程为：**前向计算概率 → 计算偏好损失 → 反向传播得到梯度 → 优化器更新当前模型参数**。标准离线 DPO 不需要每步现场生成回答；参考概率在模型与数据处理固定时可预计算。

**相对概率改善不等于绝对概率提高**：Chosen 的概率可能下降，只要 Rejected 下降得更多，损失仍可能降低。因此损失下降不能单独证明回答正确性、安全性或整体能力提高，需要独立评测。

参考：[DPO 原论文第 4 节及附录 A 的目标与梯度推导](https://arxiv.org/html/2305.18290v3#S4)。相关实践见 [[Agent系统/04_Agent模型后训练SFT_DPO#4.3 DPO 训练过程]]。

### 优缺点

- **优点**：省去独立奖励模型、Critic 和在线采样，通常比 PPO 更容易实现。具体显存节省取决于模型部署、优化器与参考策略实现，不能固定说减半。
- **局限**：依赖偏好数据质量与覆盖范围，缺少在线探索；仍能泛化到未见样本。与 PPO、GRPO 的性能差异需要在相同基础模型、数据和预算下评测，不能预设算法上限。

### 适用场景

- **离线偏好微调与企业内落地**：计算资源有限、期望追求稳定产出的普通开发者或中小型 AI 团队。
- **快速验证与偏好注入**：将特定的垂域偏好（例如让客服大模型学会特定口吻、避免答非所问）快速通过 paired data 注入到预训练模型中。
- Instruct 版本可能组合 SFT、离线偏好优化和在线 RL，具体配方应以相应论文或模型卡为准，不能由名称推断都采用 DPO。

---

## 3. GRPO (Group Relative Policy Optimization)

GRPO 由 DeepSeek 团队在 **2024 年 2 月的 DeepSeekMath 论文**中提出，随后用于 DeepSeek-R1。它用组内奖励统计量估计优势，省去独立 Critic。[DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300)。

### 核心思路：用"组内相对评分"替代 Critic 网络

常见 PPO 实现使用 Critic 估计状态价值，它增加参数、优化器与训练开销；Critic 的规模和与 Actor 的参数共享方式依具体实现而定。

GRPO 的主要特点是：**不训练独立的 Critic 网络，而是利用同一问题的一组采样结果估计基准线（Baseline）**。

具体做法是：
1. 对于每一个输入 Prompt $q$，从当前策略（旧策略 $\pi_{\theta_{old}}$）中**采样一组（Group）响应** $\{o_1, o_2, ..., o_G\}$（论文中典型值 $G = 8$ 或 $G = 16$）。
2. 用奖励模型（或基于规则的验证器）对这 $G$ 个回复分别打分，得到 $\{r_1, r_2, ..., r_G\}$。
3. **以组内所有奖励的均值和标准差对每个奖励做归一化**，得到相对优势（Relative Advantage）：
$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, ..., r_G\})}{\text{std}(\{r_1, ..., r_G\})}$$
4. 用这个归一化后的 $\hat{A}_i$ 替代 PPO 中由 Critic 产生的 GAE 优势估计，推动 Actor 的更新。

GRPO 包含 Actor、参考策略和奖励来源三个角色；规则验证器不是神经网络模型，也不一定需要三份模型同时驻留。

当前 TRL 支持多个损失形式，并可将 KL 系数默认设为 0；这是具体实现的配置，不代表全部 GRPO 都没有 KL 或参考策略。[TRL 实现](https://huggingface.co/docs/trl/en/grpo_trainer)

| 模型 | 作用 | 是否可训练 |
| :--- | :--- | :---: |
| **Actor Model** | 当前策略，负责生成回复 | ✅ 是 |
| **Reward Model** | 对组内各回复打分（也可用规则验证器代替） | ❌ 冻结 |
| **Reference Model** | 提供 KL 约束基准，防止偏离原分布 | ❌ 冻结 |

### 数学公式表达

以下是原始 GRPO 的一种目标形式；后续变体会改变归一化、重要性比率、裁剪或 KL 配置：

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q) \\ \{o_i\}_{i=1}^{G} \sim \pi_{\theta_{old}}(\cdot|q)}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( r_{i,t}(\theta)\,\hat{A}_i,\; \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\,\hat{A}_i \right) - \beta\, \mathbb{D}_{\text{KL}}\left[\pi_\theta \| \pi_{\text{ref}}\right] \right) \right]$$

其中：

- $q$：输入的 Prompt（从训练集中采样）
- $o_i$：第 $i$ 个从旧策略 $\pi_{\theta_{old}}$ 采样出的完整回复（共 $G$ 条）
- $r_{i,t}(\theta) = \dfrac{\pi_\theta(o_{i,t} \mid q,\, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t} \mid q,\, o_{i,<t})}$：新旧策略在第 $t$ 个 token 上的概率比
- $\hat{A}_i = \dfrac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$：**组内归一化的相对优势**，由奖励组统计量计算，完全替代 Critic 网络
- $\varepsilon$：PPO 截断系数，防止策略单步更新过大
- $\beta$：KL 散度惩罚系数

> **关键对比**：在 PPO 中，$\hat{A}_t$ 由 Critic 网络通过 GAE（广义优势估计）算出，需要额外的 Critic 模型；而在 GRPO 中，$\hat{A}_i$ 完全由**组内统计量**（组均值和组方差）给出，无需任何 Critic 参数。

### 长度与难度偏差

原始 GRPO 的按回答长度归一化和组内奖励标准差缩放，会改变样本的相对梯度权重。Dr.GRPO 分析了由此产生的长度、题目难度偏差。因此，观察训练时应同时看正确率、输出长度和消耗，不能只把长 CoT 当成能力进步。[Dr.GRPO 分析](https://arxiv.org/abs/2503.20783)

DAPO 的动态采样会过滤缺少组内奖励差异的样本组，再补充有学习信号的组；其 token 级损失只是完整训练配方的一部分。库中选择一个同名 loss，不代表动态采样等全部机制都已启用。[DAPO 论文](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf)

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
  - 主观任务仍需可靠的偏好或评审奖励，省去 Critic 不会自动解决奖励质量问题；
  - $G$ 的选择（组内采样数量）对显存和效果有较大影响，需要仔细调节。

### 适用场景

- **数学、代码等可验证推理任务**：可考虑 GRPO + 规则验证器，并独立评估奖励可靠性。
- **希望开展在线 RL 的团队**：省去 Critic 后仍需承担组内采样、参考策略与优化开销。
- **研究推理行为的在线训练**：同时衡量正确率、长度和消耗，不能把更长 CoT 直接视为能力提高。

---

## 4. GRPO 后续算法：采样、偏差与稳定性

| 方法      | 时间                | 主要变化                          | 理解重点                                       |
| ------- | ----------------- | ----------------------------- | ------------------------------------------ |
| DAPO    | 2025              | 非对称裁剪、动态采样、token 级损失聚合、超长奖励处理 | 改善长推理训练的探索和有效训练信号。                         |
| Dr.GRPO | 2025              | 调整长度归一化与奖励标准化                 | 避免优化目标人为鼓励错误回答变长等偏差。                       |
| GSPO    | 2025-07           | 序列级重要性比率和裁剪                   | 从 token 级转向序列级优化，研究大规模 MoE RL 的稳定性。        |
| SAPO    | 2025-11           | 平滑、温度控制的软门控                   | 对偏离采样策略的更新连续降权，保留更多有效学习信号。                 |
| LUSPO   | 2026-02           | 修正 GSPO 的长度偏差                 | 序列级优化也可能出现回答长度坍缩。                          |
| VESPO   | 2026-02 首版，05 月修订 | 序列级重要性权重的变分软重塑                | 处理异步 rollout 陈旧及训练/推理不一致带来的 off-policy 问题。 |

来源：[DAPO 原论文](https://papers.nips.cc/paper_files/paper/2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf)、[Dr.GRPO 原论文](https://arxiv.org/abs/2503.20783)、[GSPO 官方介绍](https://qwenlm.github.io/blog/gspo/)、[SAPO 原论文](https://arxiv.org/abs/2511.20347)、[LUSPO 原论文](https://arxiv.org/abs/2602.05261)、[VESPO 原论文](https://arxiv.org/abs/2602.10693)。

论文算法、同名软件 loss 与实际启用的采样机制应分开核查。奖励来源与 PPO/GRPO 的选择并非一一对应，不能仅依据发布时间认定全面替代。关联：[[Agent系统/04_Agent模型后训练SFT_DPO]]。

## 5. 其他主流衍生对齐算法（ORPO、KTO）

除上述三类方法外，还有 ORPO、KTO 等偏好优化方法，它们对训练流程和数据形式有不同要求：

### 5.1 ORPO (Monolithic Preference Optimization without Reference Model)

- **特点**：连 DPO 所需要的那个 Reference Model（冻结的旧模型）也干掉了，只需 **1 个模型**。
- **原理**：直接将偏好损失项以"附加罚项（Odds Ratio）"的形式加到 SFT 损失函数的后面。这意味着**不需要先做 SFT 再做 DPO，而是一步到位直接基于带有偏好的数据完成 SFT 与对齐**。

  **数学公式表达**：
  $$\mathcal{L}_{\text{ORPO}} = \mathbb{E}\left[\mathcal{L}_{\text{SFT}}\right] + \lambda\, \mathbb{E}\left[\mathcal{L}_{\text{OR}}\right]$$

  对于其中的几率比损失 $\mathcal{L}_{\text{OR}}$：
  $$\mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)} \right)$$

  其中定义几率（Odds）为：$\text{odds}_\theta(y|x) = \dfrac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}$

- **适用场景**：显存和计算资源有限，需要减少训练组件或模型副本的场景。实际能训练多大模型，仍取决于精度、微调方式和序列长度。

### 5.2 KTO (Kahneman-Tversky Optimization)

- **特点**：无需成对数据（Pairwise Data），只需单个正/负反馈。
- **原理**：基于前景理论（Prospect Theory）构造效用目标，只需将单条回答标记为正反馈（$y_{\text{desirable}}$）或负反馈（$y_{\text{undesirable}}$），不要求为同一输入提供 Chosen/Rejected 成对回答。

  **数学公式表达**：
  令 $r_\theta(x,y)=\log\frac{\pi_\theta(y|x)}{\pi_{\mathrm{ref}}(y|x)}$，KL 基准为 $z_0=\mathbb{E}_x[\mathrm{KL}(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x))]$。效用按正负反馈采用相反方向：
  $$v_\theta(x,y)=\begin{cases}\lambda_D\sigma(\beta(r_\theta(x,y)-z_0)), & y\text{ 为理想回答}\\\lambda_U\sigma(\beta(z_0-r_\theta(x,y))), & y\text{ 为非理想回答}\end{cases}$$
  $$\mathcal{L}_{\mathrm{KTO}}=\mathbb{E}_{(x,y)\sim\mathcal{D}}[\lambda_y-v_\theta(x,y)]$$
  其中 $\lambda_y$ 对应 $\lambda_D$ 或 $\lambda_U$；实现中用批内不匹配的输入/输出估计 KL 基准，并停止对该基准反向传播。[KTO 原论文第 4.1 节](https://arxiv.org/html/2402.01306v4#S4.SS1)。

- **适用场景**：难以获得同一输入的成对偏好回答，但可以收集单条回答的点赞、点踩等反馈。

---

## 6. 对比总结表格

为了更直观地展示各算法的选型区别，总结如下：

| 对比维度 | PPO | DPO | GRPO | ORPO | KTO |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **模型角色** | Actor、Critic、奖励来源、Ref | Policy、Ref（可共享或预计算） | Actor、奖励来源、Ref（无 Critic） | Policy | Policy、Ref |
| **基础原理** | 奖励模型 + Actor-Critic 强化学习 | 隐式奖励最大化（数学等效 RL） | 组内相对优势估计替代 Critic，On-policy RL | 基于 Odds Ratio 的 SFT 合并改进 | 基于前景理论的单向偏好损失 |
| **是否需要 Reward Model** | 可选（规则或环境也可提供奖励） | ❌ 不需要 | ⚠️ 可选（可用规则验证器替代） | ❌ 不需要 | ❌ 不需要 |
| **是否有 On-policy 采样** | ✅ 有（RL 探索） | ❌ 无（Off-policy） | ✅ 有（组内采样 G 个回复） | ❌ 无 | ❌ 无 |
| **资源开销** | Actor/Critic、采样与奖励计算 | 偏好训练与参考概率计算 | 组内采样，无独立 Critic | SFT 与偏好联合训练，无 Ref | 反馈训练与参考概率计算 |
| **训练数据格式** | Prompt + 奖励信号；学习型 RM 可另用偏好数据训练 | 成对偏好数据 | Prompt + 奖励信号 | 成对偏好数据 | 单个正/负反馈 |
| **稳定性与调参** | 需调 KL、裁剪、学习率等 | 需调 beta、数据与学习率 | 需调组大小、裁剪、KL 等 | 需调偏好权重与学习率 | 需调正负权重、beta 等 |
| **性能比较** | 依赖基础模型、奖励与预算 | 依赖基础模型、偏好数据与预算 | 依赖基础模型、奖励与预算 | 依赖基础模型、偏好数据与预算 | 依赖基础模型、反馈数据与预算 |
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

### 给开发者的落地建议

1. 有可靠成对偏好数据、预算有限时，可先评估 DPO。
2. 可验证任务可评估 GRPO 及后续变体，同时核查验证器、采样成本和长度偏差。
3. 希望联合 SFT 与偏好目标且不使用参考模型时，可评估 ORPO；它保留 SFT 目标，并非取消监督学习。
4. 只有单条正负反馈时，可评估 KTO，同时处理反馈噪声和正负分布。
5. 有在线环境、奖励体系和 Actor-Critic 基础设施时，可评估 PPO。
6. 奖励来源与算法选择分别判断，最终按相同基础模型、数据、预算和评测口径比较。

## DeepSeek 的 RL 训练 Step

以 DeepSeek-R1 的 GRPO 训练为例，一个训练 Step 的完整流程：

**① 采样阶段（Rollout）**
对一批 Prompt，策略模型（Policy）并行自回归生成 G 个完整回答（Group），通常 G = 8~16。

**② 奖励计算（Reward）**
对每个 (Prompt, Response) 对，规则打分器评分：
- **格式奖励**：思考过程是否在 `<think>...</think>` 标签内
- **答案正确性奖励**：数学题对比最终答案，代码题运行测试用例

**③ 优势估计（Advantage Estimation，GRPO 特有）**
不训练独立的 Critic/Value 网络，而是用组内相对奖励：

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\}_{j=1}^G)}{\text{std}(\{r_j\}_{j=1}^G)}$$

组内得分高于平均值的回答获得正优势，反之获得负优势。

**④ Policy 梯度更新**
用 PPO-clip 目标函数 + KL 散度约束：

$$\mathcal{L} = \mathbb{E}\left[\min\left(\frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)} \hat{A}, \;\text{clip}\!\left(\frac{\pi_\theta}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right)\hat{A}\right)\right] - \beta\, D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

KL 项防止策略偏离参考模型（原始 SFT 模型）太远。

**⑤ 梯度同步与权重更新**
多卡（数据并行 + 张量并行 + 流水线并行）All-Reduce 梯度，优化器（AdamW）更新 Policy 权重，完成一个 Step。

---

## DeepSeek-R1 的架构与训练流程

**架构层面**（基于 DeepSeek-V3）：

1. **MoE（Mixture of Experts）架构**：FFN 层替换为多个专家网络，每个 token 通过 Top-K 路由只激活部分专家。总参数 671B，每个 token 激活参数约 37B，推理计算量大幅降低，实现"大参数量 + 低推理成本"的平衡。

2. **MLA（Multi-head Latent Attention，多头潜在注意力）**：将 K/V 投影到低维潜空间（Latent Space）再做 Attention，大幅压缩 KV Cache 的显存占用。长文本推理时内存瓶颈显著缓解，相比标准 MHA 节省约 93% 的 KV Cache。

3. **无辅助损失负载均衡**：通过为每个专家维护动态偏置项（Bias）控制路由均衡，而非引入额外辅助 Loss 项。避免辅助 Loss 与主任务梯度相互干扰，负载均衡效果更稳定。

**算法层面**：

1. **GRPO**：省去 Critic，用组内奖励统计量估计优势；训练成本节省依赖 rollout 和模型部署配置，不能固定说减半。

2. **直接从基础模型进行 RL 训练（R1-Zero）**：不先进行 SFT，而是使用强化学习训练。实验观察到反思、较长推理和回溯等行为，说明在该实验条件下可以通过 RL 改善推理表现；不能据此断言所有推理能力都只来自 RL。

3. **DeepSeek-R1 完整训练流程**：冷启动 SFT → 推理导向 RL → 拒绝采样并混合非推理数据再次 SFT → 面向全场景的后续 RL。前两步不是完整流程。

4. **奖励设计**：R1-Zero 使用准确性和格式奖励；正式 R1 的后续全场景 RL 还结合学习型奖励评估帮助性等目标。规则奖励可减少 RM 训练需求，但验证器漏洞、测试不充分和环境缺陷仍会产生 Reward Hacking 风险。[DeepSeek-R1 原论文](https://arxiv.org/html/2501.12948v1)。

---
