---
tags:
  - Agent
  - SFT
  - DPO
  - 模型微调
  - 后训练
status: active
---

# Agent 主力模型后训练全流程：SFT + DPO

> 本文档详细描述 5G 测试验证 Agent 系统中 Qwen3-32B 主力模型的后训练过程，包括为什么要做、数据如何产生、训练细节和方案选型逻辑。

---

## 一、为什么通用基座模型不够用

在开始任何训练之前，首先要诚实地回答：**到底遇到了什么问题，让我们决定做后训练？**

### 1.1 零样本基座模型的三个核心缺陷

**缺陷一：领域术语理解不足（OOV 语义漂移）**

通用模型（即使是 Qwen3-32B）的训练语料以通用互联网文本为主，5G 通信领域的专有术语密度极高：`RSRP`、`SINR`、`PUSCH`、`BWP`、`NR PDCP`、`EPS Fallback`、`A3 Event`……

典型失败案例：
```
用户：请测试 gNB 的 Xn 接口在 A3 事件触发时的切换延迟

零样本输出（错误）：
{
  "test_type": "interface_test",
  "target": "gNB",
  "event": "A3",
  "metric": "switch_delay",   ← 字段名不符合平台规范
  "frequency_band": "700MHz"  ← 捏造的频段，A3事件与频段无直接关系
}
```

**缺陷二：结构化格式输出不稳定**

下游执行器（仿真平台 API）对 JSON/YAML 格式有严格要求：字段名必须完全匹配、必填字段不能缺失、类型必须正确。零样本模型在多轮对话后格式稳定性显著下降：

- 第 1 轮：输出标准 JSON
- 第 3 轮：开始在 JSON 外加解释性文字（破坏解析）
- 第 5 轮：出现字段名拼写变化（`test_case_id` → `testCaseId`）
- 第 7 轮：丢失必填字段 `bearer_id`

**缺陷三：危险逻辑幻觉**

最严重的问题。模型有时会"一本正经"地生成包含高危操作的测试用例：

```json
{
  "test_case": "极限压测-全网并发",
  "actions": [
    {"op": "set_concurrency", "value": 99999},    ← 超大并发，可能击垮仿真器
    {"op": "reset_counters", "type": "all"},       ← 清除所有历史基线数据
    {"op": "force_reboot", "target": "all_gnb"}    ← 重启所有基站
  ]
}
```

这类输出在逻辑上自洽（"当然要在极限并发下测才能发现问题"），但在工业场景中是灾难性的。通用基座模型缺乏对"什么是工业级安全边界"的概念。

### 1.2 RAG + Prompt Engineering 的局限

RAG 和精心设计的 System Prompt 可以缓解上述问题，但有固有局限：

| 方法 | 局限性 |
|:---|:---|
| RAG | 知识注入量有限，超长上下文"Lost in the Middle"效应严重；无法改变模型的基础行为模式 |
| System Prompt | 在多轮对话后稳定性下降；复杂约束超过模型的指令遵循能力上限 |
| Few-shot 示例 | 占用大量 Token，每次请求成本上升；无法覆盖所有边界情况 |

**结论**：RAG 和 Prompt 解决的是"给模型提供信息"的问题，后训练解决的是"改变模型的行为模式"的问题。二者互补，不可互相替代。

---

## 二、训练路线图

```
基座模型（Qwen3-32B）
       │
       ▼
[阶段一] SFT 监督微调
  目标：格式规范 + 术语理解 + 通用领域能力
  数据：约5万条高质量对话（合成 + Golden Dataset）
  硬件：8×A100 80G，约14小时
       │
       ▼
SFT 模型（Qwen3-32B-5G-SFT）
       │
       ▼
[阶段二] DPO 偏好对齐
  目标：安全边界对齐 + 危险幻觉抑制
  数据：HITL 沉淀的偏好对（约1.2万对）
  硬件：8×A100 80G，约6小时
       │
       ▼
生产模型（Qwen3-32B-5G-Aligned）
       │
       ▼
[持续] 在线学习飞轮
  每周：新 HITL 数据 → 增量 SFT/DPO → 评测回归
```

---

## 三、SFT 阶段：监督微调

### 3.1 训练数据：来源与构建

**目标**：构建约 5 万条覆盖以下能力的高质量对话数据：
1. 5G 通信专有术语的正确理解与使用
2. 严格 JSON/YAML 格式的稳定输出
3. ReAct 推理范式（Thought → Action → Observation 的正确循环）
4. 面对非理想环境（工具失败、模糊诉求）的容错行为

**数据来源一：Self-Instruct + Magpie 合成流水线（约3.5万条）**

纯人工标注进度太慢，且容易遗漏长尾场景。采用强模型自动合成：

```
Step 1：种子数据准备
  ├── 100条专家手写的高质量示例（覆盖核心场景）
  ├── 3GPP TS 38.331、38.211等核心协议文档片段
  └── 企业内部测试规范文档（脱敏处理）

Step 2：指令生成（Self-Instruct 变体）
  输入给强LLM（如GPT-4o）：
  "基于以下5G测试背景，生成50条不同类型的测试请求指令，
   要求涵盖：正向测试、边界场景、模糊诉求、异常恢复等类型"
  → 生成多样化的用户指令

Step 3：响应生成（Magpie 思路）
  将生成的指令 + 5G专业背景 + 工具定义注入LLM，
  生成完整的多轮对话（含Tool Calling序列）：
  [用户] → [模型思考] → [工具调用] → [工具返回] → [继续推理] → [最终结论]

Step 4：质量过滤
  ├── 格式校验：Pydantic自动过滤格式错误的样本（过滤~15%）
  ├── 一致性校验：最终结论与工具返回数据是否逻辑自洽
  └── 人工抽检：随机抽取5%样本，专家审核逻辑正确性
```

**关键点：加入失败案例数据**

这是容易被忽视但极其重要的设计决策。如果训练数据全是"顺畅成功"的案例，模型在遇到工具故障时会不知所措。

专门构造的失败恢复数据示例：
```
[用户] 请测试基站A的切换成功率
[模型] Thought: 需要先查询相关测试用例
       Action: test_case_query(feature="handover")
[工具] 返回: HTTP 500 - Service temporarily unavailable
[模型] Thought: 工具调用失败，等待10秒后重试
       Action: test_case_query(feature="handover")  ← 重试
[工具] 返回: {"cases": [...], "total": 12}
[模型] Thought: 重试成功，继续下一步...
```

训练数据中故意加入工具失败、超时、返回空数据等情况，并标注正确的恢复行为（重试、降级、告知用户），使模型学会在非理想环境下优雅降级。

**数据来源二：HITL 沉淀的 Golden Dataset（约1.5万条）**

来自生产系统中经过专家确认的高质量用例，可信度最高：

```python
# HITL 数据转化为SFT训练样本的流程

def convert_hitl_to_sft(hitl_record: HITLRecord) -> SFTSample:
    """
    HITLRecord 包含：
    - original_agent_output: Agent最初生成的用例（可能有问题）
    - expert_modification: 专家修改后的版本
    - expert_rationale: 专家说明修改原因
    - final_execution_result: 最终执行结果（PASS/FAIL + 根因）
    """
    return SFTSample(
        # 用修改后的正确版本作为期望输出
        instruction=hitl_record.user_input,
        expected_reasoning=build_cot_from_expert(hitl_record),
        expected_tool_calls=hitl_record.expert_modification.tool_calls,
        expected_conclusion=hitl_record.final_execution_result
    )
```

Golden Dataset 的价值：这些数据已经过真实执行验证，不是合成的——模型学习的是"真正在仿真环境中跑通过"的用例。

### 3.2 数据格式设计（ChatML）

所有数据统一打包为 ChatML 格式，包含完整的系统提示、多轮对话和工具调用序列：

```
<|im_start|>system
You are a 5G test verification agent. Follow the ReAct framework:
- Think step by step before each action
- Always start with test_case_query before simulation_runner
- Never invent tool parameters — use values from previous tool outputs
- Set confidence_score < 0.65 if uncertain — this triggers human review
[工具定义的JSON Schema...]
<|im_end|>

<|im_start|>user
近期对基站A做了RRC配置变更，请验证切换到基站B的Xn接口是否正常
<|im_end|>

<|im_start|>assistant
Thought: 需要先查询Xn切换相关测试用例，再执行仿真
Action: test_case_query
{"feature": "Xn_handover", "category": "regression"}
<|im_end|>

<|im_tool_response|>
{"feature": "Xn_handover", "cases": [{"id": "TC-0042", ...}], "total": 14}
<|im_end|>

<|im_start|>assistant
Thought: 获取到14条切换测试用例，下一步启动仿真执行
Action: simulation_runner
{"test_case_ids": ["TC-0042", "TC-0043", ...], "env": "sandbox"}
<|im_end|>

... （多轮工具调用）

<|im_start|>assistant
```json
{
  "confidence_score": 0.91,
  "verdict": "FAIL",
  "summary": "切换成功率85%低于基线99%，根因：Xn回传链路SN Status Transfer超时"
}
```
<|im_end|>
```

**格式设计的关键考量**：
- 工具调用和工具返回都在训练序列中明确标注，模型学会区分"自己的输出"和"环境反馈"
- 最终结论强制要求 JSON 格式（用代码块包裹），而不是自然语言，保证下游系统可解析
- `confidence_score` 是必填字段，强制模型进行自我置信度评估

### 3.3 模型选型：为什么是 Qwen3-32B + QLoRA

**为什么是 Qwen3-32B 而不是更大的模型**：

| 因素 | 分析 |
|:---|:---|
| **推理能力** | 32B 在通信领域逻辑推理上已足够，更大模型的边际收益有限 |
| **延迟** | 生产环境要求 TTFT < 5s，72B 模型在当前硬件上难以满足 |
| **成本** | 32B vLLM 部署在 4×A100 即可，72B 至少需要 8×A100 |
| **微调代价** | 32B QLoRA 约14小时，72B 需要 40+ 小时，迭代速度差异显著 |

**为什么选 QLoRA 而不是全参微调（Full Fine-tuning）**：

```
全参微调：
  优点：更充分利用所有参数，性能上限更高
  缺点：
  ├── 显存需求：32B模型全参微调需要约 2TB 显存（BF16 + 优化器状态）
  ├── 训练时间：单次 Epoch 需要数天
  └── 风险：容易灾难性遗忘（Catastrophic Forgetting）通用能力

QLoRA（Quantized LoRA）：
  优点：
  ├── 显存：nf4 量化后仅需约 80GB（恰好 1×A100 可以加载）
  ├── 效果：在充分数据下，效果接近全参（通常 95%+ 的性能）
  └── 安全：原始权重冻结，通用能力不受损
  缺点：性能上限低于全参（可接受）
```

**为什么选 LoRA 而不是其他 PEFT 方法（Prefix Tuning、Adapter 等）**：

LoRA 的可解释性和稳定性更好：梯度直接更新低秩矩阵，不引入额外的前缀 Token 或中间层，与 LangChain/LangGraph 的工具调用兼容性更好。Prefix Tuning 在长上下文下稳定性差，不适合多轮工具调用场景。

### 3.4 训练配置详解

```python
# LoRA 超参数
lora_config = LoraConfig(
    r=64,                    # 秩（rank）：越大表达能力越强，但参数量增加
    lora_alpha=128,          # 缩放因子：通常设为 2*r，控制 LoRA 更新的学习率缩放
    lora_dropout=0.05,       # 防过拟合，5% 是经验值
    target_modules=[         # 关键插入位置
        "q_proj",            # Query 投影：影响注意力模式，对工具选择最关键
        "v_proj",            # Value 投影：影响信息提取
        "o_proj",            # Output 投影：影响最终输出
        "gate_proj",         # MLP Gate：Qwen 的 SwiGLU 激活门控
        "up_proj",           # MLP Up：扩展维度
    ],
    # 为什么不插入 k_proj？
    # Key 投影主要影响注意力的键空间，在工具调用格式化任务中
    # 改动它反而容易扰乱模型的位置编码，实验中发现插入k_proj收益有限
    bias="none",
)

# 量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # NormalFloat4：比 int4 更适合正态分布的权重
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算时反量化为 BF16
    bnb_4bit_use_double_quant=True,   # 对量化参数本身再量化，进一步节省显存
)

# 训练超参数
training_args = TrainingArguments(
    num_train_epochs=3,               # 3个Epoch：实验发现2个Epoch欠拟合，4个开始过拟合
    per_device_train_batch_size=2,    # 受限于显存
    gradient_accumulation_steps=8,   # 等效batch_size = 2×8×8卡 = 128
    learning_rate=5e-5,               # Cosine退火的初始学习率
    lr_scheduler_type="cosine",       # Cosine退火：避免最后阶段学习率过高扰乱权重
    warmup_ratio=0.05,               # 5%的步数用于warmup
    fp16=False,
    bf16=True,                        # A100支持BF16，比FP16数值更稳定
    gradient_checkpointing=True,      # 以计算换显存，减少约40%显存占用
    dataloader_num_workers=4,
    # FlashAttention-2：重新计算 attention 而不存储，显存节省约 50%
    attn_implementation="flash_attention_2",
)
```

**`r=64` 的选择逻辑**：

通信测试场景的任务多样性高（切换、干扰、容量、信道等），需要较高的 LoRA 秩来捕捉足够的任务变化。实验对比：

| r 值 | 格式合规率 | 术语准确率 | 显存增量 |
|:---|:---|:---|:---|
| 16 | 91.2% | 83.4% | +0.5GB |
| 32 | 94.8% | 88.1% | +1.0GB |
| **64** | **96.7%** | **91.3%** | **+2.0GB** |
| 128 | 96.9% | 91.5% | +4.0GB |

`r=64` 是性能和显存的最优平衡点，从 64 到 128 的提升微乎其微但显存翻倍。

### 3.5 DeepSpeed ZeRO-3 配置

8 卡 A100 上的多机分布式训练需要显存优化：

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",    // 优化器状态卸载到 CPU，节省约 30GB 显存
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",    // 非活跃参数卸载到 CPU
      "pin_memory": true
    },
    "overlap_comm": true,  // 通信与计算重叠，减少等待时间
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "fp16": {"enabled": false},
  "bf16": {"enabled": true}
}
```

**ZeRO-3 vs ZeRO-2 的选择**：ZeRO-2 只分片优化器状态和梯度，ZeRO-3 还分片模型参数本身。对于 32B 模型，ZeRO-2 每卡仍需 ~40GB 显存存放参数，而 ZeRO-3 将参数也分片，每卡约 10GB 用于参数，大幅降低峰值显存需求。

### 3.6 训练监控与早停

```python
# 监控指标
metrics_to_watch = {
    "train_loss": "应持续下降，若在第2个Epoch后不再下降，可能需要调整lr",
    "eval_format_compliance": "格式合规率，目标>95%",
    "eval_tool_selection_accuracy": "工具选择准确率，目标>90%",
    "grad_norm": "梯度范数，若持续>10.0说明学习率过高",
}

# 早停条件（防止过拟合）
# 验证集格式合规率连续5个evaluation步骤不再提升 → 停止训练
# 训练损失与验证损失的差距 > 0.3 → 停止训练（过拟合信号）
```

**第三个 Epoch 的必要性**：在本项目数据集上，第 2 个 Epoch 结束时，模型对常见场景已表现良好，但对低频的边界场景（工具失败恢复、模糊诉求处理）仍不稳定。第 3 个 Epoch 专门对这部分数据做了上采样（难例 ×2 权重），显著提升了稳健性。

### 3.7 SFT 效果评估

SFT 完成后，在评测集上运行对比：

| 指标 | 基座模型 | SFT 后 | 提升 |
|:---|:---|:---|:---|
| 格式合规率 | 71.3% | 96.7% | +25.4pp |
| 工具选择准确率 | 67.8% | 91.2% | +23.4pp |
| 参数 F1-Score | 59.4% | 88.4% | +29.0pp |
| 任务完成率 | 52.1% | 83.6% | +31.5pp |
| 高危场景阻断率 | 38.7% | 72.4% | +33.7pp |

注意：**高危场景阻断率 SFT 后仍只有 72.4%**，远未达到 100% 的生产要求。这正是 DPO 阶段的核心任务。

---

## 四、DPO 阶段：偏好对齐

### 4.1 为什么 SFT 不够，必须做 DPO

SFT 的本质是**最大似然估计**：给模型看"正确答案"，让它学会生成相似的输出。但这有一个根本性的局限：

**SFT 只告诉模型"什么是好的"，无法直接告诉它"什么是不能做的"。**

对于危险操作的问题：

```
SFT 数据中没有任何"生成 reset_all 然后被拒绝"的样本
→ 模型没学到"这样做会有什么后果"
→ 即使做了 SFT，遇到极端场景时仍可能产生危险幻觉

DPO 数据中有：
  Y_rejected: {"op": "reset_all", "scope": "all_gnb"}   ← 被专家打回
  Y_chosen:   {"op": "restart", "scope": "single_gnb"}  ← 专家修正后
→ 模型直接学习"打回版本"和"正确版本"的对比
→ 学到了安全边界的判断依据
```

更精确地说，DPO 通过偏好对学习的是一个**隐式奖励函数**：模型知道什么样的输出会被专家打回，因此在推理时会主动回避。

### 4.2 偏好数据的采集：HITL 作为数据采集器

HITL 机制的设计有一个重要的战略考量：**它不只是安全机制，同时也是 DPO 数据的天然采集系统**。

```
Agent 生成测试用例（含高危操作）
         │
         ▼
Guardrail 拦截 → HITL 触发
         │
         ▼
专家 Review：
  ├── 若判定为危险：标注为 Y_rejected（负样本）
  │        │
  │        ▼
  │   专家修改为安全版本 → 标注为 Y_chosen（正样本）
  │
  └── 若判定为 Novel Case（新颖但非危险）：
           │
           ▼
       专家确认后执行 → 执行结果 + 专家批注 → 正样本

最终形成三元组：(Prompt, Y_chosen, Y_rejected)
```

**关键设计：偏好对的多样性**

不同类型的安全边界需要不同类型的偏好对：

| 类型 | Y_rejected 特征 | Y_chosen 特征 | 学习目标 |
|:---|:---|:---|:---|
| **直接高危** | 含 `reset_all`、`force_reboot` 等关键词 | 使用细粒度、可回滚的等价操作 | 避免破坏性操作 |
| **隐性高危** | 无明显关键词，但逻辑链导致危险后果（如无限循环） | 添加迭代上限和超时保护 | 识别隐性风险 |
| **参数越界** | 并发数超过平台限制（如 99999）| 使用平台允许的最大并发（100）| 参数合理性判断 |
| **频段错误** | 使用受限/军用频段 | 使用合法的实验频段 | 域知识遵守 |

### 4.3 DPO 训练过程

**为什么选 DPO 而不是 RLHF**：

```
RLHF 流程：
  Step 1: 训练 Reward Model（需要额外的标注数据 + 训练算力）
  Step 2: 用 PPO 算法优化 Actor Model（训练不稳定，超参数难调）
  Step 3: 维护 Reference Model 防止 KL 爆炸
  总计：约 3个独立训练阶段，工程复杂度极高

DPO 流程：
  Step 1: 直接用偏好对 (Y_chosen, Y_rejected) 计算对比损失
  Step 2: 单阶段训练，无需独立 Reward Model
  总计：1个训练阶段，工程复杂度低约 70%

DPO 的数学等价性保证：
  DPO 的优化目标与 RLHF 在理论上等价（Bradley-Terry 偏好模型）
  在数据量充足时，实际效果接近
```

**DPO 损失函数直觉**：

$$\mathcal{L}_{DPO} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]$$

直觉理解：
- 增大 `Y_chosen`（合法安全操作）的生成概率
- 减小 `Y_rejected`（危险操作）的生成概率
- `β` 控制对 Reference Model 的偏离程度（防止遗忘 SFT 能力）

**DPO 训练配置**：

```python
dpo_config = DPOConfig(
    beta=0.1,                   # KL 惩罚系数：越大越保守（不偏离 SFT 模型）
                                # 0.1 是经验起点，太大会导致 DPO 效果不明显
    loss_type="sigmoid",        # 标准 DPO，也可以尝试 ipo（Identity Preference）
    learning_rate=1e-6,         # 比 SFT 小一个数量级，防止过拟合偏好数据
    num_train_epochs=2,         # 偏好数据量少，2个Epoch足够
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    max_prompt_length=2048,
    max_length=4096,
    gradient_checkpointing=True,
)
```

**`beta=0.1` 的选择逻辑**：

| beta 值 | 效果 | 风险 |
|:---|:---|:---|
| 0.01 | DPO 效果强，安全边界提升明显 | 可能遗忘 SFT 学到的格式和术语 |
| **0.1** | **平衡：安全提升 + 保留 SFT 能力** | **（推荐值）** |
| 0.5 | 非常保守，几乎不偏离 SFT 模型 | DPO 效果微弱，安全对齐不充分 |

**Reference Model 的选取**：

使用 SFT 完成后的模型（Qwen3-32B-5G-SFT）作为 Reference Model，而不是基座模型。原因：DPO 的 KL 惩罚防止模型"太偏离 Reference Model"，如果 Reference 是基座，则 DPO 会同时对抗基座和 SFT 的影响，导致 SFT 学到的格式能力退化。

### 4.4 DPO 效果评估

DPO 前后对比（在安全评测集上）：

| 指标 | SFT 后 | DPO 后 | 提升 |
|:---|:---|:---|:---|
| 高危场景阻断率 | 72.4% | 96.8% | +24.4pp |
| 格式合规率 | 96.7% | 96.1% | -0.6pp（可接受退化）|
| 任务完成率 | 83.6% | 88.0% | +4.4pp |
| 低置信度正确触发率 | 61.3% | 84.7% | +23.4pp |

**高危场景阻断率 96.8% → 结合 Guardrail 双缝检测达到生产要求的 100%**：

DPO 让模型主动回避高危操作（模型层防护），Guardrail 节点做关键词二次过滤（系统层防护），两层叠加保证 100% 阻断率。不能只靠其中一层——DPO 防"聪明绕过"，Guardrail 防"意外漏网"。

**为什么格式合规率略微下降（-0.6pp）**：

DPO 训练中偏好数据的文本风格与 SFT 数据略有差异（专家修改后的用例文风更简洁），导致模型在格式上有轻微偏移。这是可接受的退化，可以通过在 DPO 数据中混入少量高质量 SFT 格式数据来缓解（Rehearsal 技术）。

---

## 五、训练数据的质量保证

### 5.1 数据污染检测

评测集和训练集的构建由不同人员负责，构建完成后做 n-gram 重叠检测：

```python
def check_data_contamination(train_set, eval_set, threshold=0.8):
    """检测训练集样本是否在评测集中有高度相似的样本"""
    for train_sample in train_set:
        for eval_sample in eval_set:
            similarity = compute_ngram_overlap(
                train_sample.instruction,
                eval_sample.instruction,
                n=4  # 4-gram 重叠
            )
            if similarity > threshold:
                # 从训练集中剔除该样本
                train_set.remove(train_sample)
                log_contamination(train_sample, eval_sample, similarity)
```

实际运行中，约 2.3% 的合成数据被检测为与评测集高度相似，全部从训练集剔除。

### 5.2 数据分布监控

训练数据应覆盖各类测试场景，避免某类场景过拟合：

```python
SCENARIO_DISTRIBUTION_TARGET = {
    "handover": 0.25,       # 切换场景：最常见
    "interference": 0.20,   # 干扰场景
    "capacity": 0.20,       # 容量场景
    "channel": 0.15,        # 信道场景
    "auth_bearer": 0.10,    # 认证/承载场景
    "error_recovery": 0.10, # 工具失败恢复（容错）
}

# 实际数据分布偏离目标超过 5% 时，通过过采样/欠采样纠正
```

### 5.3 难例挖掘与上采样

第 3 个 Epoch 对模型当前还做错的样本进行上采样（难例挖掘）：

```python
# 在第 2 个 Epoch 结束后，用当前模型跑评测集
current_model_errors = evaluate(model, eval_set)

# 找到对应的训练样本（覆盖相同场景类型）
hard_samples = find_matching_train_samples(current_model_errors, train_set)

# 在第 3 个 Epoch 中，对难例样本设置更高的采样权重
train_with_sample_weights(
    train_set,
    hard_sample_weight=2.0,  # 难例2倍采样
    easy_sample_weight=1.0
)
```

---

## 六、持续在线学习飞轮

SFT + DPO 不是一次性的工作，而是持续演进的飞轮：

```
每周运行：
  ├── 收集本周 HITL 数据（新的偏好对）
  ├── 合成针对近期失败场景的 SFT 数据
  ├── 运行增量微调（基于上次模型 checkpoint，而不是从头开始）
  └── 评测回归：所有 1258 题必须通过质量门禁

增量微调策略：
  ├── 增量 SFT：只在新数据上微调 1 个 Epoch（Continual Learning）
  │   风险：灾难性遗忘 → 缓解：在新数据中混入 10% 历史 replay 数据
  └── 增量 DPO：用本周 HITL 新偏好对更新安全边界
      风险：偏好数据量小 → 缓解：与历史偏好对混合训练
```

**灾难性遗忘的量化监控**：

每次增量微调后，在历史"核心能力集"（100 道固定题）上评估，若核心能力下降 > 3%，触发回滚并调整 Replay 数据比例。

---

## 七、常见问题与经验教训

### Q1：LoRA 的 r 值越大越好吗？

不是。r 值控制低秩矩阵的表达能力，但存在收益递减和显存膨胀的权衡。在本项目中实验发现 r=64 是最优点，r=128 性能几乎没有提升但显存翻倍。建议从 r=16 开始实验，逐步增大直到验证集性能不再提升。

### Q2：DPO 训练后为什么有时候模型变笨了？

`beta` 值设置不当。beta 过小时，模型为了最大化偏好对的奖励差异，可能过度修改自己的行为，导致 SFT 学到的格式和推理能力退化。解决方案：增大 beta（更保守），或在 DPO 数据中混入 SFT 正样本做 Rehearsal。

### Q3：合成数据的质量如何保证？

核心是质量过滤，而不是减少合成。合成流水线产出的数据中，约 15% 因格式错误被 Pydantic 自动过滤，约 5% 因逻辑不自洽被人工抽检发现并删除。最终进入训练集的数据合格率约 80%。宁缺毋滥——低质量数据的危害远超数量不足。

### Q4：训练中 Loss 下降了但评测集指标没变化怎么办？

可能是训练数据和评测数据分布不匹配。检查步骤：
1. 验证训练数据场景分布是否覆盖评测集的所有场景类型
2. 检查是否有数据污染（训练集记住了答案但没有真正泛化）
3. 检查评测集是否过于集中在单一场景类型

### Q5：HITL 数据量不够时怎么做 DPO？

初期 HITL 数据稀少时，可以用强模型（GPT-4o）对同一任务生成多个不同安全级别的输出，由专家标注优劣，构建初始偏好数据集。等 HITL 积累到足够数量后再切换到真实偏好数据。
