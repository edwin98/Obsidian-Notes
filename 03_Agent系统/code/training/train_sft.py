"""SFT 教学入口：先分组验证，再检查真实批次，最后训练与完整任务验证。

对照正文08：messages → 聊天模板 → 助手掩码 → labels → loss → LoRA更新。
本入口不固定ChatML头、不自动追加难例轮次、不编造准确率或置信度阈值。
仅运行 --help 不需要加载训练库。先用 training.inspect_batch 学习CPU小张量。
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
    parser.add_argument("--model_name_or_path", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    return parser.parse_args()


def build_trainer(args, train_rows, validation_rows):
    """延迟导入与加载权重，数据错误应在昂贵操作前暴露。"""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    # 模型先按所选精度加载；真正迁移到训练设备由 Trainer/Accelerate 处理。
    # use_cache 服务于逐 token 推理，训练时关闭，避免与梯度检查点冲突。
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    model.config.use_cache = False
    config = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        # 自动助手掩码依赖模板支持。不要叠加一份手工labels让两套逻辑互相覆盖。
        assistant_only_loss=True,
        # 暂不把多条对话打包，先让每条样本的掩码和截断边界易于核验。
        packing=False,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # 累积多次小批次再更新；有效批量还要乘上参与训练的设备数量。
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        use_cpu=args.cpu,
        gradient_checkpointing=True,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
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
        # 只给注意力的 Q/V 投影加低秩增量；不是解冻整个模型。
        # r 控制增量矩阵的秩，alpha/r 控制缩放，目标模块名要与模型结构一致。
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )

    # source_group 已用于统一拆分，不把审核元数据直接作为模型输入。
    def dataset(rows):
        return Dataset.from_list([{"messages": row["messages"]} for row in rows])

    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=config,
        train_dataset=dataset(train_rows),
        eval_dataset=dataset(validation_rows),
        peft_config=adapter,
    ), tokenizer


def main():
    """验证来源隔离 → 构造训练器 → 检查真实目标 → 更新 → 验证并保存。"""
    args = parse_args()
    train_rows, validation_rows = validate_runtime(args)
    trainer, tokenizer = build_trainer(args, train_rows, validation_rows)
    from training.objectives import inspect_training_batch

    # 检查的是整理器真正产生的labels，能发现模板、padding和截断错误。
    batch = next(iter(trainer.get_train_dataloader()))
    print(
        json.dumps(
            inspect_training_batch(batch, tokenizer), ensure_ascii=False, indent=2
        )
    )
    if args.inspect_only:
        return
    trainer.train()
    # 小数据可能不足50步，也必须做最后一次独立验证。
    print(json.dumps(trainer.evaluate(), ensure_ascii=False, indent=2))
    save_training_outputs(trainer, tokenizer, args, "sft")


if __name__ == "__main__":
    main()
