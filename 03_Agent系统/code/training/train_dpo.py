"""DPO 教学入口：同一上下文的偏好对 + 明确固定的 SFT 参考。

对照正文08：候选序列log probability → 相对参考的变化 → sigmoid偏好损失。
本例显式加载独立reference，概念清楚但占用更多内存；不假设关闭SFT adapter
就一定得到正确参考。输入应为SFT合并模型，而不是只有adapter的目录。
"""

import argparse
import json

from training.runtime import (
    add_training_arguments,
    save_training_outputs,
    validate_runtime,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    add_training_arguments(parser)
    parser.add_argument(
        "--sft_model_path", required=True, help="SFT合并后的完整模型目录"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.1)
    return parser.parse_args()


def conversational_sample(row):
    """统一成对话式偏好数据，不把system/user手工拼成另一套部署模板。"""
    prompt = row["prompt"]
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    return {
        "prompt": prompt,
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def build_trainer(args, train_rows, validation_rows):
    """同一 SFT 起点分别加载可训练策略和固定参考，让比较基准保持不变。"""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.sft_model_path, dtype=dtype)
    reference = AutoModelForCausalLM.from_pretrained(args.sft_model_path, dtype=dtype)
    # requires_grad=False 阻止参数更新；eval() 关闭 dropout 等训练态随机性。
    # 固定参考表示“相对 SFT 改变了多少”，不能跟着 policy 一起学习。
    reference.requires_grad_(False)
    reference.eval()
    model.config.use_cache = False
    config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        # sigmoid 是标准成对偏好目标；可对照 objectives.dpo_pair_loss 的小张量演算。
        # 两个候选共用同一 prompt，偏好来自专家确认，不能把不同任务硬拼成一对。
        loss_type="sigmoid",
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        use_cpu=args.cpu,
        gradient_checkpointing=True,
        warmup_steps=args.warmup_steps,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        logging_steps=5,
        report_to="none",
        deepspeed=args.deepspeed,
    )
    adapter = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    # 由 Trainer 把 LoRA 装到 policy；独立的 reference 保持完整 SFT 权重。
    # 因此此阶段 adapter 必须与 SFT 合并后的基座一起使用。
    return DPOTrainer(
        model=model,
        ref_model=reference,
        processing_class=tokenizer,
        args=config,
        train_dataset=Dataset.from_list([conversational_sample(r) for r in train_rows]),
        eval_dataset=Dataset.from_list(
            [conversational_sample(r) for r in validation_rows]
        ),
        peft_config=adapter,
    ), tokenizer


def main():
    """检查成对数据与实际批次，再训练；loss 降低后仍需完整 Agent 任务评测。"""
    args = parse_args()
    train_rows, validation_rows = validate_runtime(args)
    if args.beta <= 0:
        raise ValueError("beta必须大于0")
    trainer, tokenizer = build_trainer(args, train_rows, validation_rows)
    # 输出实际整理后的张量结构，不打印虚构的阻断率、准确率等收益。
    batch = next(iter(trainer.get_train_dataloader()))
    shapes = {
        key: list(value.shape)
        for key, value in batch.items()
        if hasattr(value, "shape")
    }
    print(
        json.dumps(
            {"batch_shapes": shapes, "reference": "独立冻结SFT模型"},
            ensure_ascii=False,
            indent=2,
        )
    )
    if args.inspect_only:
        return
    trainer.train()
    print(json.dumps(trainer.evaluate(), ensure_ascii=False, indent=2))
    save_training_outputs(trainer, tokenizer, args, "dpo")


if __name__ == "__main__":
    main()
