"""
train_dpo.py — DPO（直接偏好优化）偏好对齐训练

为什么做 DPO（笔记 §4.1）：
  SFT 的本质是最大似然估计，只告诉模型"什么是好的"，无法告诉它"什么是不能做的"。

  SFT 后高危场景阻断率：72.4%（远未达到生产要求）
  DPO 后高危场景阻断率：96.8%（结合 Guardrail 达到 100%）

为什么选 DPO 而不是 RLHF（笔记 §4.3）：
  - RLHF：3 个独立训练阶段（Reward Model + PPO + KL 约束），工程复杂度极高
  - DPO：1 个训练阶段，数学上等价于 RLHF（Bradley-Terry 偏好模型），稳定可靠

DPO 损失函数直觉：
  L = -E[log σ(β·(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
  → 增大 Y_chosen（安全操作）的生成概率
  → 减小 Y_rejected（危险操作）的生成概率
  → β 控制对 Reference Model 的偏离程度（防止遗忘 SFT 能力）

Reference Model 选取（笔记 §4.3）：
  使用 SFT merge 后的完整模型（checkpoints/sft/merged）作为 Reference，而非基座模型。
  原因：Reference 若是基座，DPO 会同时对抗基座和 SFT 的影响，导致 SFT 格式能力退化。

运行方式（8 卡 A100 80G）：
  deepspeed --num_gpus=8 train_dpo.py \
    --deepspeed deepspeed_config.json \
    --sft_model_path checkpoints/sft/merged \
    --data_path data/dpo_clean.jsonl \
    --output_dir checkpoints/dpo

训练完成后会自动 merge LoRA 权重并保存完整模型至 checkpoints/dpo/merged，
可直接用 vLLM 部署。
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DPO 偏好对齐训练")
    parser.add_argument("--sft_model_path", default="checkpoints/sft/final",
                        help="SFT 训练后的模型路径（同时作为 Policy 和 Reference 的起点）")
    parser.add_argument("--data_path", default="data/dpo_clean.jsonl",
                        help="清洗后的 DPO 偏好数据（JSONL 格式）")
    parser.add_argument("--output_dir", default="checkpoints/dpo",
                        help="DPO 模型保存路径")
    parser.add_argument("--deepspeed", default=None)

    # DPO 超参数（笔记 §4.3）
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL 惩罚系数：越大越保守（不偏离 SFT 模型）"
                             "笔记建议 0.1：平衡安全提升+保留 SFT 能力")
    parser.add_argument("--learning_rate", type=float, default=1e-6,
                        help="比 SFT 小一个数量级，防止过拟合偏好数据")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="偏好数据量少，2 个 Epoch 足够")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--loss_type", default="sigmoid",
                        choices=["sigmoid", "ipo", "kto_pair"],
                        help="DPO 变体：sigmoid=标准DPO，ipo=IPO，kto_pair=KTO对比版")
    return parser.parse_args()


# ── 模型加载 ─────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str):
    """
    加载 SFT merged 模型作为 DPO 的 Policy Model 起点。

    重要细节：
    - 输入为 SFT 阶段 merge_and_unload() 后的完整 BF16 模型（无 LoRA adapter）
    - DPO 需要同时维护 Policy Model（可训）和 Reference Model（冻结）
    - TRL DPOTrainer 在内部自动创建 Reference Model（深拷贝 Policy 并冻结）
    - 因此我们只需要加载一个模型实例，DPO 阶段再套新的 LoRA adapter
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="left",   # DPO 推荐左 padding（生成任务的惯例）
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DeepSpeed ZeRO-3 与 device_map 不兼容，同 train_sft.py 的原因。
    # DPO 阶段同样由 DeepSpeed 接管分片，不设置 device_map。
    # 单卡调试时取消注释 device_map="cuda:0"。
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        # device_map="cuda:0",  # 单卡调试时取消注释
    )
    return model, tokenizer


def add_dpo_lora(model, r: int = 32):
    """
    为 DPO 训练添加 LoRA 适配器。

    注意：DPO 阶段的 LoRA 配置比 SFT 更保守（r=32 而非 r=64）：
    - DPO 的目标是微调"行为模式"（对齐），而非学习新能力
    - 较小的 r 降低过拟合偏好数据的风险
    - 偏好数据量（~1.2 万条）远少于 SFT 数据（~5 万条），过大的 r 容易过拟合

    如果是从 SFT LoRA 权重继续训练（推荐），应加载已有的 LoRA adapter，
    而非重新初始化：
      from peft import PeftModel
      model = PeftModel.from_pretrained(model, sft_lora_path, is_trainable=True)
    """
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=32,          # DPO 用更小的 r（比 SFT 的 64 更保守）
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
        inference_mode=False,
    )
    return get_peft_model(model, lora_config)


# ── 数据集加载 ────────────────────────────────────────────────────────────────

def load_dpo_dataset(data_path: str) -> tuple[Dataset, Dataset]:
    """
    加载 DPO 偏好数据集。

    TRL DPOTrainer 期望数据集包含 3 个字段：
      - "prompt"：对话上下文（系统提示 + 用户输入，到 assistant 开始之前）
      - "chosen"：偏好回复（Y_chosen，专家修改的安全版本）
      - "rejected"：拒绝回复（Y_rejected，Agent 生成的危险版本）

    注意：DPOTrainer 会自动对 prompt+chosen 和 prompt+rejected 进行
    tokenize 和损失计算，不需要手动处理。
    """
    raw_samples = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))

    # 验证字段完整性
    valid_samples = []
    for s in raw_samples:
        if all(k in s for k in ["prompt", "chosen", "rejected"]):
            valid_samples.append({
                "prompt": s["prompt"],
                "chosen": s["chosen"],
                "rejected": s["rejected"],
            })

    print(f"[数据] 加载 DPO 数据: {len(valid_samples)} 条（原始 {len(raw_samples)} 条）")

    dataset = Dataset.from_list(valid_samples)
    # 分割：90% 训练，10% 验证
    split = dataset.train_test_split(test_size=0.1, seed=42)
    return split["train"], split["test"]


# ── DPO 训练配置 ──────────────────────────────────────────────────────────────

def get_dpo_config(args) -> DPOConfig:
    """
    DPO 训练配置（笔记 §4.3）：

    beta=0.1（KL 惩罚系数）：
      - beta 控制 Policy Model 对 Reference Model 的偏离程度
      - β 小（0.01）：DPO 效果强，但可能遗忘 SFT 学到的格式和术语
      - β 大（0.5）：非常保守，几乎不偏离 SFT，但 DPO 效果微弱
      - β=0.1：平衡点（推荐值），安全提升 + 保留 SFT 能力

    loss_type="sigmoid"（标准 DPO）：
      - sigmoid：原始 DPO 损失（Rafailov et al., 2023）
      - ipo：IPO（Azar et al., 2023），对偏好强度更鲁棒
      - kto_pair：KTO 的成对版本，适合非对称偏好强度的场景

    learning_rate=1e-6：
      - 比 SFT（5e-5）小 50 倍，防止过度偏离 SFT 学到的能力
      - 偏好数据量少，过高的学习率容易过拟合

    max_prompt_length=2048 / max_length=4096：
      - prompt 部分最长 2048 tokens（系统提示 + 用户输入）
      - 完整序列（prompt + response）最长 4096 tokens
      - 超出截断（truncation_mode="keep_start" 保留 prompt 头部）
    """
    return DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_prompt_length=args.max_prompt_length,
        max_length=args.max_length,
        gradient_checkpointing=True,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=3,
        warmup_ratio=0.05,
        deepspeed=args.deepspeed,
        # truncation_mode：当 prompt+response 超过 max_length 时，
        # "keep_start" 保留 prompt 头部（确保任务描述不被截断）
        truncation_mode="keep_start",
        report_to=["tensorboard"],
    )


# ── DPO 监控指标（理解训练状态）────────────────────────────────────────────────

def explain_dpo_metrics():
    """
    DPO 训练过程中的关键监控指标说明。
    这些指标由 TRL DPOTrainer 自动记录到 TensorBoard/WandB。

    理解这些指标对判断训练是否正常至关重要：
    """
    print("""
DPO 关键监控指标说明：

1. rewards/chosen（期望上升）：
   Policy Model 对 Y_chosen 的隐式奖励。
   公式：β × (log π_θ(y_w|x) - log π_ref(y_w|x))
   → 应该随训练持续上升（模型越来越偏好安全输出）

2. rewards/rejected（期望下降）：
   Policy Model 对 Y_rejected 的隐式奖励。
   公式：β × (log π_θ(y_l|x) - log π_ref(y_l|x))
   → 应该随训练持续下降（模型越来越回避危险输出）

3. rewards/margins（期望上升）：
   = rewards/chosen - rewards/rejected
   → 偏好间距。越大说明模型对好/坏输出的区分度越强。
   → 如果 margins 不上升或为负，说明 DPO 训练没有效果。

4. rewards/accuracies（期望 > 0.7）：
   在当前 batch 中，chosen 奖励 > rejected 奖励的比例。
   → 类似于"准确率"，越高越好，目标 > 70%。

5. logps/chosen 和 logps/rejected：
   Policy 对 chosen/rejected 序列的对数概率。
   正常状态：chosen log probs 上升，rejected log probs 下降。
   异常信号：两者都下降 → 模型在"遗忘"（beta 可能太小）。

6. beta（超参数 β）：
   控制 KL 散度惩罚强度。固定值，不随训练变化。
   如果 rewards/margins 不上升，可尝试降低 beta（如 0.05）。
""")


# ── 主训练流程 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"DPO 偏好对齐训练")
    print(f"SFT 模型: {args.sft_model_path}")
    print(f"数据: {args.data_path}")
    print(f"输出: {args.output_dir}")
    print(f"beta: {args.beta}，loss_type: {args.loss_type}")
    print(f"{'='*60}\n")

    explain_dpo_metrics()

    # Step 1：加载 SFT 模型（同时作为 Policy 和 Reference 的初始点）
    print("[1/4] 加载 SFT 模型...")
    model, tokenizer = load_model_and_tokenizer(args.sft_model_path)

    # Step 2：添加 DPO LoRA 适配器
    print("[2/4] 添加 DPO LoRA 适配器（r=32）...")
    model = add_dpo_lora(model, r=32)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.3f}%)")

    # Step 3：加载偏好数据集
    print("[3/4] 加载 DPO 偏好数据...")
    train_dataset, eval_dataset = load_dpo_dataset(args.data_path)

    # Step 4：配置并启动 DPO 训练
    print("[4/4] 开始 DPO 训练...")
    dpo_config = get_dpo_config(args)

    trainer = DPOTrainer(
        model=model,
        ref_model=None,   # None 表示 TRL 自动创建（深拷贝当前模型并冻结）
                          # 如果显存不足，可以传入单独加载的 reference model
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # peft_config=None 是因为我们已经手动 get_peft_model
        # 如果要让 TRL 自动管理 LoRA，可以传入 peft_config
    )

    trainer.train()

    # 保存 LoRA adapter 权重
    lora_output = Path(args.output_dir) / "lora_adapter"
    model.save_pretrained(str(lora_output))
    tokenizer.save_pretrained(str(lora_output))
    print(f"\n[完成] DPO LoRA adapter 已保存至 {lora_output}")

    # Merge LoRA 权重到基础模型，输出完整 BF16 模型供 vLLM 部署
    print("[Merge] 将 DPO LoRA 权重 merge 到模型...")
    merged_model = model.merge_and_unload()
    merged_output = Path(args.output_dir) / "merged"
    merged_model.save_pretrained(str(merged_output))
    tokenizer.save_pretrained(str(merged_output))

    print(f"[完成] DPO Merged 模型已保存至 {merged_output}")
    print(f"\n预期效果（笔记 §4.4）：")
    print(f"  高危场景阻断率: 72.4% → 96.8%（+24.4pp）")
    print(f"  任务完成率:     83.6% → 88.0%（+4.4pp）")
    print(f"  格式合规率:     96.7% → 96.1%（-0.6pp，可接受退化）")
    print(f"\n[提示] 下一步：")
    print(f"  1. 用 vLLM 部署: vllm serve {merged_output}")
    print(f"  2. 结合 Guardrail 节点（nodes.py）达到 100% 高危场景阻断")
    print(f"  3. 新的 HITL 数据持续积累 → 下周增量 DPO → 学习飞轮")


if __name__ == "__main__":
    main()
