"""
train_sft.py — Qwen3-32B QLoRA 监督微调（SFT）

训练目标（笔记 §1.2 的结论）：
  RAG 和 Prompt 解决"给模型提供信息"，SFT 解决"改变模型行为模式"。
  具体目标：
    1. 5G 专有术语的正确理解（RSRP/SINR/PUSCH/BWP/A3 Event 等）
    2. ReAct 推理范式的稳定输出（Thought → Action → Observation → 结论）
    3. 严格 JSON 格式的持久稳定（多轮对话后仍保持格式合规）
    4. 面对工具失败时的容错行为（重试、降级、触发 HITL）

模型选型（笔记 §3.3）：
  Qwen3-32B + QLoRA（而非 Full Fine-Tuning）
  原因：
    - 32B 在通信领域逻辑推理上已足够，更大模型边际收益有限
    - 全参微调需要 ~2TB 显存；QLoRA 仅需 ~80GB（1×A100 可加载）
    - QLoRA 通过冻结基础权重，通用能力不受损（防止灾难性遗忘）

训练配置（笔记 §3.4）：
  LoRA: r=64, alpha=128, dropout=0.05
  优化器: AdamW + Cosine LR Decay (5e-5)
  批次: per_device=2, gradient_accum=8 → 等效 batch=128（8×A100）
  DeepSpeed: ZeRO-3 + CPU 卸载

运行方式（8 卡 A100）：
  deepspeed --num_gpus=8 train_sft.py \
    --deepspeed deepspeed_config.json \
    --data_path data/sft_clean.jsonl \
    --output_dir checkpoints/sft
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

# ── 参数解析 ─────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-32B QLoRA SFT 训练")
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-32B",
                        help="基础模型路径（HuggingFace Hub ID 或本地路径）")
    parser.add_argument("--data_path", default="data/sft_clean.jsonl",
                        help="清洗后的 SFT 训练数据（JSONL 格式）")
    parser.add_argument("--output_dir", default="checkpoints/sft",
                        help="模型保存路径")
    parser.add_argument("--deepspeed", default=None,
                        help="DeepSpeed 配置文件路径（多卡训练时使用）")
    # 以下参数均有默认值，单卡 Debug 时可以直接运行
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--hard_sample_upweight", type=float, default=2.0,
                        help="第 3 个 Epoch 对难例的上采样倍率（笔记 §3.6）")
    return parser.parse_args()


# ── 量化配置 ─────────────────────────────────────────────────────────────────

def get_bnb_config() -> BitsAndBytesConfig:
    """
    QLoRA 量化配置（笔记 §3.3）：

    load_in_4bit=True：将基础模型加载为 4-bit 量化形式，大幅节省显存。
    nf4（NormalFloat4）：比 int4 更适合正态分布的模型权重（LLM 权重近似正态分布），
      精度损失更小。
    bfloat16：量化时（反量化计算）使用 BF16，相比 FP16 数值范围更大，不容易溢出。
    double_quant：对量化参数本身再量化，进一步节省约 0.4 bits/参数。

    整体效果：32B 模型原本 BF16 需要 64GB，4-bit 量化后约 18GB，
    加上激活值和优化器状态，A100 80G 可以舒适地运行。
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,   # 双重量化：量化参数也量化一次
    )


# ── LoRA 配置 ─────────────────────────────────────────────────────────────────

def get_lora_config(r: int, alpha: int) -> LoraConfig:
    """
    LoRA 超参数配置（笔记 §3.4）：

    r（秩）：控制 LoRA 矩阵的表达能力。
      r=16：格式合规率 91.2%，r=64：96.7%，r=128：96.9%（显存翻倍但提升微乎其微）
      → 选 r=64 为最优平衡点。
      原理：原始权重矩阵 W ∈ R^{d×k}，LoRA 用 W = W₀ + BA 近似，
      其中 B ∈ R^{d×r}，A ∈ R^{r×k}，r 越小参数量越少。

    lora_alpha（缩放因子）：等效学习率缩放 = alpha/r。
      通常设为 2*r（即 lora_alpha=128），让 LoRA 的更新幅度与 r 无关。

    target_modules（注入位置）：
      - q_proj / v_proj：注意力的 Query/Value，影响工具选择的注意力模式
      - o_proj：注意力输出投影，影响推理结果的表达
      - gate_proj / up_proj：SwiGLU 激活的门控和扩展部分（Qwen3 使用 SwiGLU）
      - 为什么不加 k_proj：Key 投影主要影响键空间，改动容易扰乱位置编码，
        实验中插入 k_proj 收益有限

    lora_dropout=0.05：防过拟合，5% 是经验值（数据量大时可降为 0.01）。
    """
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",     # Attention: Query
            "v_proj",     # Attention: Value
            "o_proj",     # Attention: Output
            "gate_proj",  # MLP: SwiGLU Gate
            "up_proj",    # MLP: SwiGLU Up
            # 不加 k_proj：实验发现收益有限且扰乱位置编码
            # 不加 down_proj：MLP 下投影，影响通用能力较大，不建议加
        ],
        # inference_mode=False 必须设置，否则 forward 时不更新参数
        inference_mode=False,
    )


# ── 训练参数 ─────────────────────────────────────────────────────────────────

def get_training_args(args, output_dir: str) -> TrainingArguments:
    """
    训练超参数（笔记 §3.4）：

    等效 batch size = per_device × gradient_accum × num_gpus
                   = 2 × 8 × 8 = 128（8 卡 A100）
    为什么 batch_size=128：
      - 太小（< 32）：梯度噪声大，loss 震荡，难以收敛
      - 太大（> 256）：对指令微调任务帮助有限，显存浪费
      - 128 是工业界 SFT 的经验值

    学习率 5e-5（Cosine 退火）：
      - 比预训练（1e-4）小一个数量级（防止破坏预训练权重）
      - Cosine 退火：避免最后阶段学习率过高扰乱权重

    gradient_checkpointing=True：
      - 以重新计算 attention 换取显存（牺牲 30% 速度，节省 40% 显存）

    attn_implementation="flash_attention_2"：
      - FlashAttention-2 重新计算 attention 而非存储，节省约 50% 显存
      - 对长序列（>2048 token）效果尤为显著
    """
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,          # 5% 步数用于 warmup：防止训练初期梯度爆炸
        fp16=False,
        bf16=True,                  # A100 原生支持 BF16，数值更稳定
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,             # 每 200 步做一次评估
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,         # 只保留最近 3 个 checkpoint，节省磁盘
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        deepspeed=args.deepspeed,
        # FlashAttention-2：重新计算而非存储 attention，节省约 50% 显存
        # 需要安装 flash-attn：pip install flash-attn --no-build-isolation
        # attn_implementation="flash_attention_2",  # 取消注释时需要确保已安装 flash-attn
        report_to=["tensorboard"],  # 同时支持 wandb，改为 ["wandb"] 即可
        remove_unused_columns=False,
    )


# ── 数据加载与格式化 ──────────────────────────────────────────────────────────

def load_and_format_dataset(data_path: str, tokenizer, max_seq_length: int) -> Dataset:
    """
    加载 SFT 数据并格式化为 tokenizer 可处理的格式。

    关键设计：SFT 训练时只对 assistant 部分的 token 计算损失（Cross Entropy Loss）。
    system 和 user 的 token 被 mask 掉（labels 设为 -100）。

    这样模型学到的是：在给定 system+user context 下，如何生成 assistant 的回复。
    而不是"背诵"用户问了什么。

    ChatML 格式（Qwen 原生）：
      <|im_start|>system
      {system content}<|im_end|>
      <|im_start|>user
      {user content}<|im_end|>
      <|im_start|>assistant
      {assistant content}<|im_end|>
    """
    raw_data = []
    with open(data_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    def format_sample(sample: dict) -> dict:
        """将 messages 格式的样本转换为训练所需的 input_ids 和 labels"""
        messages = sample["messages"]

        # 使用 tokenizer 的 apply_chat_template 生成符合 ChatML 格式的文本
        # tokenize=False：先生成文本，后续再 tokenize（方便 Debug 查看原始文本）
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,   # SFT 训练时不加 generation prompt
        )

        # Tokenize 整个对话
        tokenized = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]

        # 构建 labels：只计算 assistant 部分的损失
        # 方法：找到 assistant 回复的起始 token，之前的都设为 -100（忽略）
        labels = [-100] * len(input_ids)
        assistant_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        # 找到所有 <|im_start|> 的位置
        # 注意：Qwen 的 ChatML 格式每个角色都以 <|im_start|> 开头
        assistant_starts = []
        for i, tid in enumerate(input_ids):
            if tid == assistant_token_id:
                # 检查下一个 token 是否是 "assistant"
                # 实际实现需要解码后检查，这里用简化逻辑
                # 生产中应使用 trl 的 DataCollatorForCompletionOnlyLM
                assistant_starts.append(i)

        # 简化：将最后一段 assistant 回复的标签设为实际 token ID
        # 生产中推荐使用 TRL 的 DataCollatorForCompletionOnlyLM，
        # 它能准确地只 mask non-assistant 部分
        # 这里为了教学清晰，用简化方式表示
        labels = input_ids.copy()  # 简化：全部计算损失（真实训练应 mask system/user）

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
        }

    dataset = Dataset.from_list(raw_data)
    return dataset


# ── 早停回调（笔记 §3.6）────────────────────────────────────────────────────────

class EarlyStoppingCallback(TrainerCallback):
    """
    自定义早停逻辑（笔记 §3.6 的早停条件）：

    条件1：验证集格式合规率连续 5 个 evaluation 步骤不再提升 → 停止
    条件2：train_loss 和 eval_loss 的差距 > 0.3 → 停止（过拟合信号）

    注意：HuggingFace Trainer 内置了 EarlyStoppingCallback（基于 eval_loss），
    这里实现基于自定义指标（过拟合差距）的早停，作为额外保障。
    """
    def __init__(self, overfit_threshold: float = 0.3):
        self.overfit_threshold = overfit_threshold
        self.best_eval_format_compliance = 0.0
        self.no_improve_count = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        train_loss = metrics.get("train_loss", 0)
        eval_loss = metrics.get("eval_loss", 0)

        # 条件2：过拟合检测
        if train_loss > 0 and eval_loss > 0:
            gap = eval_loss - train_loss
            if gap > self.overfit_threshold:
                print(f"\n[EarlyStopping] 训练集与验证集 loss 差距 {gap:.3f} > {self.overfit_threshold}，停止训练")
                control.should_training_stop = True
                return

        # 条件1：格式合规率不再提升（需要在 compute_metrics 中实现，这里仅作示意）
        eval_format = metrics.get("eval_format_compliance", None)
        if eval_format is not None:
            if eval_format > self.best_eval_format_compliance:
                self.best_eval_format_compliance = eval_format
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= 5:
                    print(f"\n[EarlyStopping] 格式合规率连续 5 步未提升，停止训练")
                    control.should_training_stop = True


# ── 难例上采样（笔记 §3.6、§5.3）────────────────────────────────────────────────

def upsample_hard_examples(
    dataset: Dataset,
    model,
    tokenizer,
    upweight: float = 2.0,
) -> Dataset:
    """
    难例挖掘与上采样（笔记 §5.3）：

    在第 2 个 Epoch 结束后，用当前模型跑评测集，找出模型还做错的样本对应的
    训练集场景类型，对这些类型的训练样本进行上采样（难例 × upweight 权重）。

    为什么在第 3 个 Epoch 做上采样：
    - 第 1-2 Epoch：让模型学习所有场景的基础能力
    - 第 3 Epoch：专门攻克低频的边界场景（工具失败恢复、模糊诉求处理）
    - 如果一开始就上采样，可能导致模型过度关注难例而忽略常规场景

    upweight=2.0：难例出现频率变为原来的 2 倍（重复添加到数据集）。
    """
    # 识别错误恢复类场景（最容易做错的类型）
    hard_scenarios = {"error_recovery", "auth_bearer"}

    hard_samples = [
        s for s in dataset
        if s.get("scenario") in hard_scenarios
    ]
    easy_samples = [
        s for s in dataset
        if s.get("scenario") not in hard_scenarios
    ]

    # 对难例按 upweight 重复
    extra_hard = hard_samples * int(upweight - 1)
    upsampled_list = easy_samples + hard_samples + extra_hard

    print(f"[难例上采样] 原始: {len(dataset)} → 上采样后: {len(upsampled_list)} 条")
    print(f"  难例数量: {len(hard_samples)} → {len(hard_samples) + len(extra_hard)} 条")

    return Dataset.from_list(upsampled_list)


# ── 主训练流程 ────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Qwen3-32B QLoRA SFT 训练")
    print(f"模型: {args.model_name_or_path}")
    print(f"数据: {args.data_path}")
    print(f"输出: {args.output_dir}")
    print(f"{'='*60}\n")

    # Step 1：加载 Tokenizer
    print("[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",   # Causal LM 需要右侧 padding
    )
    if tokenizer.pad_token is None:
        # Qwen3 没有 pad_token，使用 eos_token 代替（训练时 pad 位置会被 mask）
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2：加载量化模型（QLoRA 核心）
    print("[2/5] 加载量化基础模型（QLoRA 4-bit）...")
    bnb_config = get_bnb_config()
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",         # 自动跨多卡分配（单卡时 = "cuda:0"）
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",  # 需要 flash-attn
    )

    # prepare_model_for_kbit_training：
    # 1. 禁用 4-bit 模型的 layer norm 中的输入梯度（防止精度损失）
    # 2. 将所有 non-quantized 参数（如 LayerNorm）转为 BF16
    # 3. 启用 gradient checkpointing
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )
    print(f"  量化模型加载完成，GPU 显存: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    # Step 3：插入 LoRA 适配器
    print("[3/5] 插入 LoRA 适配器...")
    lora_config = get_lora_config(r=args.lora_r, alpha=args.lora_alpha)
    model = get_peft_model(model, lora_config)

    # 打印可训练参数量（LoRA 的参数只有模型总参数的 ~0.1-1%）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"  总参数: {total_params:,}")

    # Step 4：加载数据集
    print("[4/5] 加载训练数据...")
    full_dataset = load_and_format_dataset(
        args.data_path,
        tokenizer,
        args.max_seq_length,
    )

    # 分割训练集和验证集（95% train，5% eval）
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  训练集: {len(train_dataset)} 条，验证集: {len(eval_dataset)} 条")

    # Step 5：配置训练参数并开始训练
    print("[5/5] 开始训练...")
    training_args = get_training_args(args, args.output_dir)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # TRL SFTTrainer 的关键参数：
        # max_seq_length：超过长度的样本会被截断而非丢弃
        # dataset_text_field：若数据是纯文本格式，指定字段名（此处用 messages 格式）
        callbacks=[EarlyStoppingCallback(overfit_threshold=0.3)],
    )

    # 第 1-2 Epoch：正常训练
    print("\n[Epoch 1-2] 正常训练...")
    # 注意：这里的分阶段逻辑是概念展示。
    # 实际实现中，可以在 TrainerCallback 的 on_epoch_end 中动态替换 train_dataset
    trainer.train()

    # 第 3 Epoch：难例上采样（笔记 §5.3）
    # 在第 2 个 Epoch 结束后，识别当前模型的薄弱点，对对应训练样本上采样
    print("\n[Epoch 3] 难例上采样，专项强化边界场景...")
    hard_upsampled_dataset = upsample_hard_examples(
        train_dataset,
        model,
        tokenizer,
        upweight=args.hard_sample_upweight,
    )
    # 替换训练集后重新初始化 Trainer（仅训练 1 个 Epoch）
    training_args_epoch3 = get_training_args(args, args.output_dir + "_epoch3")
    training_args_epoch3 = TrainingArguments(
        **{
            **training_args_epoch3.to_dict(),
            "num_train_epochs": 1,
            "output_dir": args.output_dir + "_epoch3",
        }
    )
    trainer_epoch3 = SFTTrainer(
        model=model,
        args=training_args_epoch3,
        train_dataset=hard_upsampled_dataset,
        eval_dataset=eval_dataset,
    )
    trainer_epoch3.train()

    # 保存最终模型（LoRA 权重 + 训练配置）
    final_output = Path(args.output_dir) / "final"
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    print(f"\n[完成] SFT 模型已保存至 {final_output}")
    print(f"[提示] 下一步:")
    print(f"  1. 运行 prepare_dpo_data.py 构建 DPO 偏好数据")
    print(f"  2. 运行 train_dpo.py 进行偏好对齐（高危场景阻断率 72.4% → 96.8%）")
    print(f"  3. 使用 vLLM 部署: vllm serve {final_output} --enable-lora")


if __name__ == "__main__":
    main()
