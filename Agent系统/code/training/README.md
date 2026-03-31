# Agent 模型后训练：SFT + DPO 完整流水线

对应笔记：`Agent系统/04_Agent模型后训练SFT_DPO.md`

---

## 流程总览

```
基座模型（Qwen3-32B）
        │
        ▼
Step 1: 数据生成      generate_sft_data.py
        Self-Instruct + Magpie → 合成 3.5 万条 SFT 对话
        │
        ▼
Step 2: 数据清洗      clean_data.py
        格式校验 + 质量过滤 + 去重 + 污染检测 + 分布校正
        │
        ▼
Step 3: SFT 微调      train_sft.py + deepspeed_config.json
        Qwen3-32B + QLoRA (r=64) + DeepSpeed ZeRO-3
        → Qwen3-32B-5G-SFT（格式合规率 71% → 97%）
        │
        ▼
Step 4: 偏好数据准备  prepare_dpo_data.py
        HITL 记录 → (prompt, chosen, rejected) 三元组
        │
        ▼
Step 5: DPO 训练      train_dpo.py
        β=0.1，SFT 模型作为 Reference
        → Qwen3-32B-5G-Aligned（高危阻断率 72% → 97%）
```

---

## 文件说明

| 文件 | 作用 |
|:---|:---|
| `generate_sft_data.py` | Self-Instruct 指令扩增 + Magpie 对话生成 |
| `clean_data.py` | 格式校验、质量过滤、去重、污染检测、分布校正 |
| `train_sft.py` | Qwen3-32B QLoRA SFT 训练主脚本 |
| `prepare_dpo_data.py` | HITL 数据库 → DPO 三元组（或合成偏好数据） |
| `train_dpo.py` | DPO 偏好对齐训练主脚本 |
| `deepspeed_config.json` | DeepSpeed ZeRO-3 + CPU 卸载配置 |

---

## 快速开始

```bash
# 安装依赖
pip install transformers peft trl bitsandbytes datasets openai
pip install deepspeed flash-attn --no-build-isolation  # 多卡训练

# Step 1: 生成 SFT 数据（需要 OpenAI API Key）
python generate_sft_data.py

# Step 2: 清洗数据
python clean_data.py

# Step 3: SFT 训练（8 卡 A100）
deepspeed --num_gpus=8 train_sft.py \
  --deepspeed deepspeed_config.json \
  --data_path data/sft_clean.jsonl \
  --output_dir checkpoints/sft

# Step 4: 准备 DPO 数据（演示用合成数据）
python prepare_dpo_data.py --use_synthetic

# Step 5: DPO 训练
deepspeed --num_gpus=8 train_dpo.py \
  --deepspeed deepspeed_config.json \
  --sft_model_path checkpoints/sft/final \
  --data_path data/dpo_clean.jsonl \
  --output_dir checkpoints/dpo
```

---

## 关键超参数一览

### SFT（train_sft.py）

| 超参数 | 值 | 原因 |
|:---|:---|:---|
| LoRA r | 64 | r=64 → 96.7% 格式合规率（r=128 仅 +0.2%，显存翻倍） |
| LoRA alpha | 128 | = 2×r，标准缩放因子 |
| target_modules | q/v/o/gate/up proj | 覆盖注意力+MLP，不加 k_proj（扰乱位置编码） |
| epochs | 3 | 第 3 Epoch 对难例上采样（×2），提升边界场景鲁棒性 |
| lr | 5e-5 | Cosine 退火，warmup 5% |
| batch size | 128（等效） | 2 × 8 accum × 8 GPU |

### DPO（train_dpo.py）

| 超参数 | 值 | 原因 |
|:---|:---|:---|
| beta | 0.1 | KL 惩罚：太小→遗忘 SFT 能力，太大→DPO 效果微弱 |
| loss_type | sigmoid | 标准 DPO |
| lr | 1e-6 | 比 SFT 小 50 倍，防过拟合偏好数据 |
| epochs | 2 | 偏好数据少（~1.2 万对），2 个 Epoch 足够 |
| LoRA r | 32 | 比 SFT 更保守（对齐任务，非新能力学习） |
| Reference Model | SFT 模型 | 防止 SFT 格式能力退化 |

---

## 预期效果

| 指标 | 基座模型 | SFT 后 | DPO 后 |
|:---|:---|:---|:---|
| 格式合规率 | 71.3% | 96.7% | 96.1% |
| 工具选择准确率 | 67.8% | 91.2% | — |
| 任务完成率 | 52.1% | 83.6% | 88.0% |
| **高危场景阻断率** | 38.7% | 72.4% | **96.8%** |

结合 `nodes.py` 中的 Guardrail 节点（关键词二次过滤），最终实现 **100% 高危场景阻断**。
