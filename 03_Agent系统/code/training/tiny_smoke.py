"""从零构造极小随机模型，CPU跑通SFT与DPO各一步；不下载权重。

这是文档08的结构自检，不是模型质量评估：toy训练/验证材料不构成独立业务任务。
运行 python -X utf8 -m training.tiny_smoke，可观察模板、标签、adapter与固定参考。
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace


def run_tiny_training(output_dir):
    # 只在主动运行这个演练时启用离线模式，导入模块不会修改调用方环境。
    os.environ.update(
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from training.objectives import inspect_training_batch
    from training.runtime import save_training_outputs
    from training.train_dpo import build_trainer as build_dpo
    from training.train_sft import build_trainer as build_sft

    root = Path(output_dir)
    model_dir = root / "initial"
    model_dir.mkdir(parents=True, exist_ok=True)
    (root / "toy-source.txt").write_text(
        "结构自检：内联toy数据，不用于模型质量评估", encoding="utf-8"
    )
    vocab = {
        word: index
        for index, word in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "[EOS]",
                "system",
                "user",
                "assistant",
                "question",
                "good",
                "bad",
                "plan",
            ]
        )
    }
    tokenizer_impl = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer_impl.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_impl,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ' ' }}"
        "{% if message['role'] == 'assistant' %}{% generation %}"
        "{{ message['content'] + ' [EOS] ' }}{% endgeneration %}"
        "{% else %}{{ message['content'] + ' [EOS] ' }}{% endif %}{% endfor %}"
    )
    tokenizer.save_pretrained(model_dir)
    model = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    model.save_pretrained(model_dir)

    def args(stage):
        return SimpleNamespace(
            model_name_or_path=str(model_dir),
            sft_model_path=str(root / "sft" / "merged"),
            data_path=str(root / "toy-source.txt"),
            validation_path=str(root / "toy-source.txt"),
            output_dir=str(root / stage),
            max_length=64,
            lora_r=2,
            lora_alpha=4,
            learning_rate=1e-4,
            beta=0.1,
            warmup_steps=0,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            num_train_epochs=1,
            bf16=False,
            cpu=True,
            deepspeed=None,
            merge=True,
        )

    messages = [
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "good plan"},
    ]
    sft_args = args("sft")
    trainer, tokenizer = build_sft(
        sft_args, [{"messages": messages}], [{"messages": messages}]
    )
    inspection = inspect_training_batch(
        next(iter(trainer.get_train_dataloader())), tokenizer
    )
    trainer.args.max_steps = 1
    trainer.args.disable_tqdm = True
    trainer.train()
    save_training_outputs(trainer, tokenizer, sft_args, "sft")

    dpo_args = args("dpo")
    sample = {"prompt": messages[:1], "chosen": "good plan", "rejected": "bad plan"}
    trainer, tokenizer = build_dpo(dpo_args, [sample], [sample])
    assert all(
        not parameter.requires_grad for parameter in trainer.ref_model.parameters()
    )
    trainer.args.max_steps = 1
    trainer.args.disable_tqdm = True
    trainer.train()
    save_training_outputs(trainer, tokenizer, dpo_args, "dpo")
    return {
        "scope": "极小随机模型的结构自检，不评估模型质量",
        "sft_batch": inspection,
        "sft_steps": 1,
        "dpo_steps": 1,
        "reference_frozen": True,
        "output": str(root),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, help="保存并查看toy权重；默认使用临时目录"
    )
    args = parser.parse_args()
    if args.output_dir:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        result = run_tiny_training(args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="agent-tiny-training-") as directory:
            result = run_tiny_training(directory)
            result["output"] = "临时目录已自动清理"
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
