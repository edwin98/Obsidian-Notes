"""SFT/DPO 共用的教学配置与记录，避免两个入口采用不同的数据拆分口径。"""

import hashlib
import json
import os
import sys
from pathlib import Path

from training.prepare_investigation_data import read_training_splits


def add_training_arguments(parser):
    """参数不是经验最优值，只提供可检查和可复现的学习起点。"""
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--validation_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=float, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--max_length", "--max_seq_length", type=int, default=4096)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--bf16", action="store_true", help="硬件支持时显式启用BF16")
    parser.add_argument(
        "--cpu", action="store_true", help="小模型CPU教学，速度可能很慢"
    )
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=0,
        help="按实际更新步设置；极小演练默认不预热",
    )
    parser.add_argument(
        "--inspect-only", action="store_true", help="构造并核对一个真实批次，不训练"
    )
    parser.add_argument("--merge", action="store_true", help="单进程训练后合并适配器")


def validate_runtime(args):
    """先拒绝不可复现的参数与来源重叠，随后才允许加载昂贵的模型权重。"""
    # TRL 部分版本读取模板时未显式指定编码。Windows 的 GBK 默认会导入失败。
    # 这里给出可以复现的运行方式，不修改第三方库，也不悄悄重启用户进程。
    if os.name == "nt" and not sys.flags.utf8_mode:
        raise ValueError(
            "Windows训练入口请用 python -X utf8 -m training.train_sft/train_dpo ..."
        )
    if args.cpu and args.bf16:
        raise ValueError("本教学入口的CPU模式使用FP32；请去掉--bf16")
    if args.merge and int(os.getenv("WORLD_SIZE", "1")) > 1:
        raise ValueError("此处仅演示单进程合并；分布式权重请使用对应框架的聚合流程")
    if (
        min(
            args.lora_r,
            args.lora_alpha,
            args.max_length,
            args.per_device_train_batch_size,
            args.gradient_accumulation_steps,
        )
        < 1
    ):
        raise ValueError("长度、批次和适配器维度必须为正数")
    return read_training_splits(Path(args.data_path), Path(args.validation_path))


def save_training_outputs(trainer, tokenizer, args, stage):
    """区分可恢复checkpoint、adapter和合并模型；不输出未经评测的效果数字。"""
    output = Path(args.output_dir)
    # save_model 让 Trainer 处理分布式保存；不要各进程任意直接覆盖权重。
    trainer.save_model(str(output / "adapter"))
    trainer.save_state()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(output / "adapter")
        manifest = {
            "stage": stage,
            "arguments": vars(args),
            "data_hashes": {
                str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for path in (args.data_path, args.validation_path)
            },
            "note": "训练loss不是完整任务收益，采用模型前仍需运行Agent评测",
        }
        (output / "training_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if args.merge:
            merged = trainer.model.merge_and_unload()
            merged.save_pretrained(output / "merged")
            tokenizer.save_pretrained(output / "merged")
