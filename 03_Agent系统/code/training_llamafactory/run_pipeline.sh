#!/usr/bin/env bash
# run_pipeline.sh — SFT + DPO 完整训练流水线（LlamaFactory，8×A100 80G）
#
# 用法：
#   bash run_pipeline.sh
#
# 前置条件：
#   pip install llamafactory deepspeed
#   原始数据已放置：
#     ../training/data/sft_clean.jsonl
#     ../training/data/dpo_clean.jsonl

set -euo pipefail

# ─── 路径配置 ────────────────────────────────────────────────────────────────

RAW_SFT=../training/data/sft_clean.jsonl
RAW_DPO=../training/data/dpo_clean.jsonl
SFT_TRAIN=data/sft_train.json
DPO_TRAIN=data/dpo_train.json

# ─── 步骤 1：准备 SFT 数据（含一次性难例上采样） ────────────────────────────

echo "====== [1/5] 准备 SFT 数据 ======"
python data/prepare_sft_data.py \
  --input   "$RAW_SFT" \
  --output  "$SFT_TRAIN" \
  --upweight 2.0

# ─── 步骤 2：SFT 训练（8 卡，约 18 小时） ───────────────────────────────────

echo "====== [2/5] SFT 训练 ======"
# FORCE_TORCHRUN=1 让 LlamaFactory 用 torchrun 启动多卡训练
# 等效于：torchrun --nproc_per_node=8 ... llamafactory train ...
FORCE_TORCHRUN=1 llamafactory-cli train configs/sft_qwen3_32b.yaml

# ─── 步骤 3：合并 SFT LoRA 权重 ─────────────────────────────────────────────

echo "====== [3/5] 合并 SFT LoRA ======"
llamafactory-cli export \
  model_name_or_path=Qwen/Qwen3-32B \
  adapter_name_or_path=checkpoints/sft \
  template=qwen3 \
  finetuning_type=lora \
  export_dir=checkpoints/sft/merged \
  export_size=2 \
  export_dtype=bfloat16 \
  export_legacy_format=false

echo "SFT merged 模型已保存至 checkpoints/sft/merged"

# ─── 步骤 4：准备 DPO 数据 ──────────────────────────────────────────────────

echo "====== [4/5] 准备 DPO 数据 ======"
python data/prepare_dpo_data.py \
  --input  "$RAW_DPO" \
  --output "$DPO_TRAIN"

# ─── 步骤 5：DPO 训练（8 卡，约 6 小时） ────────────────────────────────────

echo "====== [5/5] DPO 训练 ======"
FORCE_TORCHRUN=1 llamafactory-cli train configs/dpo_qwen3_32b.yaml

# ─── 步骤 6：合并 DPO LoRA 权重 ─────────────────────────────────────────────

echo "====== [6/6] 合并 DPO LoRA ======"
llamafactory-cli export \
  model_name_or_path=checkpoints/sft/merged \
  adapter_name_or_path=checkpoints/dpo \
  template=qwen3 \
  finetuning_type=lora \
  export_dir=checkpoints/dpo/merged \
  export_size=2 \
  export_dtype=bfloat16 \
  export_legacy_format=false

echo ""
echo "====== 训练完成 ======"
echo "最终模型：checkpoints/dpo/merged"
echo "预期效果（文档 §4.4）："
echo "  高危场景阻断率：72.4% → 96.8%（+24.4pp）"
echo "  任务完成率：    83.6% → 88.0%（+4.4pp）"
echo "  格式合规率：    96.7% → 96.1%（-0.6pp，可接受退化）"
echo ""
echo "下一步："
echo "  vllm serve checkpoints/dpo/merged --dtype bfloat16"
