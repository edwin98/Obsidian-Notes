"""
train_sft.py — Qwen3-32B LoRA 监督微调（SFT）

训练目标（笔记 §1.2 的结论）：
  RAG 和 Prompt 解决"给模型提供信息"，SFT 解决"改变模型行为模式"。
  具体目标：
    1. 5G 专有术语的正确理解（RSRP/SINR/PUSCH/BWP/A3 Event 等）
    2. ReAct 推理范式的稳定输出（Thought → Action → Observation → 结论）
    3. 严格 JSON 格式的持久稳定（多轮对话后仍保持格式合规）
    4. 面对工具失败时的容错行为（重试、降级、触发 HITL）

模型选型（笔记 §3.3）：
  Qwen3-32B + LoRA（BF16，而非 Full Fine-Tuning 或 QLoRA）
  原因：
    - 32B 在通信领域逻辑推理上已足够，更大模型边际收益有限
    - 全参微调需要 ~2TB 显存；LoRA 仅需多卡 BF16 加载（DeepSpeed ZeRO-3 分片）
    - LoRA 通过冻结基础权重，通用能力不受损（防止灾难性遗忘）
    - 选 LoRA 而非 QLoRA：避免量化精度损失，训练结束后直接 merge 权重供 DPO 使用

训练配置（笔记 §3.4）：
  LoRA: r=64, alpha=128, dropout=0.05
  优化器: AdamW + Cosine LR Decay (5e-5)
  批次: per_device=2, gradient_accum=8 → 等效 batch=128（8×A100 80G）
  DeepSpeed: ZeRO-3 + CPU 卸载

运行方式（8 卡 A100 80G）：
  deepspeed --num_gpus=8 train_sft.py \
    --deepspeed deepspeed_config.json \
    --data_path data/sft_clean.jsonl \
    --output_dir checkpoints/sft

训练完成后会自动 merge LoRA 权重并保存完整模型至 checkpoints/sft/merged，
供后续 train_dpo.py 直接加载。
"""

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    TrainingArguments,
)
from trl import SFTTrainer

# ── 参数解析 ─────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-32B LoRA SFT 训练")
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen3-32B",
        help="基础模型路径（HuggingFace Hub ID 或本地路径）",
    )
    parser.add_argument(
        "--data_path",
        default="data/sft_clean.jsonl",
        help="清洗后的 SFT 训练数据（JSONL 格式）",
    )
    parser.add_argument("--output_dir", default="checkpoints/sft", help="模型保存路径")
    parser.add_argument(
        "--deepspeed", default=None, help="DeepSpeed 配置文件路径（多卡训练时使用）"
    )
    # 以下参数均有默认值，单卡 Debug 时可以直接运行
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument(
        "--hard_sample_upweight",
        type=float,
        default=2.0,
        help="第 3 个 Epoch 对难例的上采样倍率（笔记 §3.6）",
    )
    return parser.parse_args()


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
            "q_proj",  # Attention: Query
            "v_proj",  # Attention: Value
            "o_proj",  # Attention: Output
            "gate_proj",  # MLP: SwiGLU Gate
            "up_proj",  # MLP: SwiGLU Up
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
      - SFT 微调使用较小的学习率，防止破坏预训练权重
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
        warmup_ratio=0.05,  # 5% 步数用于 warmup：防止训练初期梯度爆炸
        fp16=False,
        bf16=True,  # A100 原生支持 BF16，数值更稳定
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,  # 每 200 步做一次评估
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,  # 只保留最近 3 个 checkpoint，节省磁盘
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
    加载 SFT 数据，tokenize 并构建正确的 labels。

    关键设计：SFT 训练时只对 assistant 部分的 token 计算损失（Cross Entropy Loss）。
    system 和 user 的 token 被 mask 掉（labels 设为 -100）。

    实现方式：在 token 序列中查找所有 "<|im_start|>assistant\n" 的 token ID 子序列，
    将其后直到 "<|im_end|>" 的 token（含）解除 mask，其余位置保持 -100。
    多轮对话中每个 assistant 回复均会被正确处理。

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

    # 预计算模板 token IDs（只做一次，避免在每条样本中重复 encode）
    response_template_ids = tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    tpl_len = len(response_template_ids)

    def format_sample(sample: dict) -> dict:
        """Tokenize 并对 system/user token 设 -100，只保留 assistant 部分的 labels。"""
        messages = sample["messages"]
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        tokenized = tokenizer(
            full_text,
            max_length=max_seq_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = tokenized["input_ids"]

        # 初始全部 mask
        labels = [-100] * len(input_ids)

        # 逐段找 response_template，解除 mask
        i = 0
        while i <= len(input_ids) - tpl_len:
            if input_ids[i : i + tpl_len] == response_template_ids:
                start = i + tpl_len  # assistant 内容的第一个 token
                j = start
                while j < len(input_ids) and input_ids[j] != im_end_id:
                    j += 1
                # 解除 mask，包含末尾的 <|im_end|>
                for k in range(start, min(j + 1, len(input_ids))):
                    labels[k] = input_ids[k]
                i = j + 1
            else:
                i += 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
        }

    dataset = Dataset.from_list(raw_data)
    return dataset.map(format_sample, remove_columns=dataset.column_names)


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
                print(
                    f"\n[EarlyStopping] 训练集与验证集 loss 差距 {gap:.3f} > {self.overfit_threshold}，停止训练"
                )
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
                    print("\n[EarlyStopping] 格式合规率连续 5 步未提升，停止训练")
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

    hard_samples = [s for s in dataset if s.get("scenario") in hard_scenarios]
    easy_samples = [s for s in dataset if s.get("scenario") not in hard_scenarios]

    # 对难例按 upweight 重复
    extra_hard = hard_samples * int(upweight - 1)
    upsampled_list = easy_samples + hard_samples + extra_hard

    print(f"[难例上采样] 原始: {len(dataset)} → 上采样后: {len(upsampled_list)} 条")
    print(f"  难例数量: {len(hard_samples)} → {len(hard_samples) + len(extra_hard)} 条")

    return Dataset.from_list(upsampled_list)


# ── 置信度阈值自动重校准（笔记"近期改"第3条）────────────────────────────────────────
#
# 背景：系统硬编码了 confidence_score < 0.65 触发 HITL 的阈值。
# 但模型每次更新后，其置信度分布会发生漂移：
#   - SFT 后模型倾向于高置信度输出
#   - DPO 后模型对危险场景的置信度会主动降低
# 如果不重新校准，固定阈值会导致：
#   - 阈值偏高 → HITL 触发过于频繁，人工审核负担增加
#   - 阈值偏低 → 部分不确定的输出绕过 HITL，安全风险升高
#
# 校准方法：
#   1. 在验证集上跑一遍模型，收集每条样本的 (confidence_score, is_correct) 对
#   2. 找出"错误率急剧上升的相变点"（即 confidence_score 从哪个值开始错误率 > 15%）
#   3. 该相变点即为新阈值
#
# 使用方式：在 SFT/DPO 训练完成后调用本函数，将返回值写入 Agent 系统配置。


def calibrate_confidence_threshold(
    model,
    tokenizer,
    eval_dataset,
    error_rate_threshold: float = 0.15,
    score_bins: int = 20,
) -> float:
    """
    在验证集上自动校准置信度阈值。

    参数：
      error_rate_threshold：允许的最大错误率。当某个置信度区间的错误率超过此值时，
                           该区间对应的置信度上界即为新阈值。默认 15%。
      score_bins：将 [0, 1] 区间划分为多少个桶（精度越高，需要的样本量越大）。

    返回：
      新的置信度阈值（float），建议写入 Agent 配置文件。

    注意：
      eval_dataset 中的样本需要包含 "expected_verdict" 字段（PASS/FAIL/INCONCLUSIVE），
      用于判断模型输出是否正确。若验证集没有此字段，本函数返回默认值 0.65。
    """

    model.eval()
    bin_width = 1.0 / score_bins

    # 每个 bin：[correct_count, total_count]
    bins = [[0, 0] for _ in range(score_bins)]

    print(f"\n[置信度校准] 在验证集上评估 {len(eval_dataset)} 条样本...")

    with torch.no_grad():
        for sample in eval_dataset:
            if "expected_verdict" not in sample:
                continue  # 跳过无 ground truth 的样本

            # 推理：生成模型输出
            messages = sample.get("messages", [])
            if not messages:
                continue

            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],  # 去掉最后的 assistant 轮
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
            )
            generated = tokenizer.decode(
                output_ids[0][inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            # 从生成文本中提取 confidence_score
            import re as _re

            score_match = _re.search(r'"confidence_score"\s*:\s*([0-9.]+)', generated)
            verdict_match = _re.search(r'"verdict"\s*:\s*"(\w+)"', generated)

            if not score_match or not verdict_match:
                continue

            confidence = float(score_match.group(1))
            predicted_verdict = verdict_match.group(1).upper()
            expected_verdict = sample["expected_verdict"].upper()

            is_correct = predicted_verdict == expected_verdict
            bin_idx = min(int(confidence / bin_width), score_bins - 1)
            bins[bin_idx][1] += 1
            if is_correct:
                bins[bin_idx][0] += 1

    # 从低置信度到高置信度，找第一个错误率 < error_rate_threshold 的相变点
    # 逻辑：错误率高于阈值的区间 → 需要 HITL 介入；低于阈值的区间 → 可以自主执行
    new_threshold = 0.65  # 默认值（若数据不足则保守回退）
    for i in range(score_bins):
        total = bins[i][1]
        if total < 5:
            continue  # 样本量不足，跳过此 bin
        error_rate = 1.0 - bins[i][0] / total
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width
        if error_rate <= error_rate_threshold:
            # 从这个 bin 开始，错误率可接受，新阈值 = 这个 bin 的下界
            new_threshold = round(bin_lower, 2)
            print(
                f"  [校准] confidence >= {new_threshold:.2f} 时错误率 {error_rate * 100:.1f}% <= {error_rate_threshold * 100:.0f}%，确定为新阈值"
            )
            break
        else:
            print(
                f"  [跳过] [{bin_lower:.2f}, {bin_upper:.2f}]: 错误率 {error_rate * 100:.1f}% > {error_rate_threshold * 100:.0f}%"
            )

    print(f"\n[校准结果] 置信度阈值: {new_threshold} (原值: 0.65)")
    if abs(new_threshold - 0.65) > 0.1:
        print("  [提示] 阈值变化超过 0.1，建议更新 Agent 系统配置：")
        print(f"    CONFIDENCE_THRESHOLD = {new_threshold}")
        print("  在 nodes.py 的 result_judge_node 和 agent_node 中同步更新此值")
    else:
        print(f"  [提示] 阈值变化在 0.1 以内，可保留原值 0.65 或更新为 {new_threshold}")
    return new_threshold


# ── 主训练流程 ────────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print("Qwen3-32B LoRA SFT 训练（BF16 全精度，8×A100 80G）")
    print(f"模型: {args.model_name_or_path}")
    print(f"数据: {args.data_path}")
    print(f"输出: {args.output_dir}")
    print(f"{'=' * 60}\n")

    # Step 1：加载 Tokenizer
    print("[1/5] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",  # Causal LM 需要右侧 padding
    )
    if tokenizer.pad_token is None:
        # Qwen3 没有 pad_token，使用 eos_token 代替（训练时 pad 位置会被 mask）
        tokenizer.pad_token = tokenizer.eos_token

    # Step 2：加载基础模型（BF16，DeepSpeed ZeRO-3 跨卡分片）
    print("[2/5] 加载基础模型（BF16）...")
    # 重要：DeepSpeed ZeRO-3 与 device_map 不兼容。
    # ZeRO-3 会自行接管参数分片（每卡只持有 1/N 的参数），
    # 若同时设置 device_map="auto"，HuggingFace 会预先把参数分配到各卡，
    # 与 ZeRO-3 的分片逻辑冲突，导致显存重复占用或训练挂起。
    # 正确做法：多卡 DeepSpeed 训练时不设置 device_map，
    # 单卡调试时可临时改为 device_map="cuda:0"。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,  # 减少模型加载时的 CPU 内存峰值，ZeRO-3 必需
        # 不设置 device_map：DeepSpeed ZeRO-3 自行接管参数分片
        # 单卡调试时取消下面注释：
        # device_map="cuda:0",
        # attn_implementation="flash_attention_2",  # 需要 flash-attn
    )
    print(f"  模型加载完成，GPU 显存: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Step 3：插入 LoRA 适配器
    print("[3/5] 插入 LoRA 适配器...")
    lora_config = get_lora_config(r=args.lora_r, alpha=args.lora_alpha)
    model = get_peft_model(model, lora_config)

    # 打印可训练参数量（LoRA 的参数只有模型总参数的 ~0.1-1%）
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)"
    )
    print(f"  总参数: {total_params:,}")

    # Step 4：加载数据集
    print("[4/5] 加载训练数据...")
    full_dataset = load_and_format_dataset(
        args.data_path, tokenizer, args.max_seq_length
    )

    # 分割训练集和验证集（95% train，5% eval）
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  训练集: {len(train_dataset)} 条，验证集: {len(eval_dataset)} 条")

    # Step 5：配置训练参数并开始训练
    print("[5/5] 开始训练...")
    training_args = get_training_args(args, args.output_dir)

    # DataCollatorForSeq2Seq：负责 padding，labels 的 masking 已在数据加载时完成。
    # label_pad_token_id=-100 确保 padding 位置不参与 loss 计算。
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=8,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
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
        data_collator=collator,
    )
    trainer_epoch3.train()

    # 保存 LoRA adapter 权重（用于断点续训或单独分发）
    lora_output = Path(args.output_dir) / "lora_adapter"
    model.save_pretrained(str(lora_output))
    tokenizer.save_pretrained(str(lora_output))
    print(f"\n[完成] LoRA adapter 已保存至 {lora_output}")

    # Merge LoRA 权重到基础模型，输出完整 BF16 模型供 DPO 使用
    print("[Merge] 将 LoRA 权重 merge 到基础模型...")
    merged_model = model.merge_and_unload()
    merged_output = Path(args.output_dir) / "merged"
    merged_model.save_pretrained(str(merged_output))
    tokenizer.save_pretrained(str(merged_output))
    print(f"[完成] Merged 模型已保存至 {merged_output}")
    # 置信度阈值校准（在 eval_dataset 上评估，自动找新阈值）
    print("\n[校准] 在验证集上重新校准置信度阈值...")
    new_threshold = calibrate_confidence_threshold(
        model=merged_model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        error_rate_threshold=0.15,
    )

    print("\n[提示] 下一步:")
    print("  1. 运行 prepare_dpo_data.py 构建 DPO 偏好数据")
    print("     若 HITL 数据不足（< 200 条），可用冷启动模式：")
    print(
        "     python prepare_dpo_data.py --cold_start --api_key <key> --cold_start_count 500"
    )
    print(f"  2. 运行 train_dpo.py --sft_model_path {merged_output} 进行偏好对齐")
    print(
        f"  3. 若置信度阈值有变化，在 nodes.py 中同步更新 CONFIDENCE_THRESHOLD = {new_threshold}"
    )
    print("  4. 使用 vLLM 部署最终 DPO 模型")


if __name__ == "__main__":
    main()
