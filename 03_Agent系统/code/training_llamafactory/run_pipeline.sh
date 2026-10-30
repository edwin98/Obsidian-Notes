#!/usr/bin/env bash
# 可选LlamaFactory路径。先准备独立来源数据；本脚本不重新随机拆分。
# 用法：bash run_pipeline.sh ../prepared
# 需要DPO时：RUN_DPO=1 bash run_pipeline.sh ../prepared
# 不固定8卡、耗时或效果；先核对数据、模板、资源与reference配置。
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PREPARED_DIR="${1:-../prepared}"
RUN_DPO="${RUN_DPO:-0}"

# 这里检查的是已经拆分的输入；实际分组隔离由prepare_investigation_data验证。
for file in sft_train.jsonl sft_validation.jsonl groups.json; do
  test -s "$PREPARED_DIR/$file" || { echo "缺少非空输入：$PREPARED_DIR/$file" >&2; exit 1; }
done
python -X utf8 data/prepare_sft_data.py --input "$PREPARED_DIR/sft_train.jsonl" --output data/sft.json
python -X utf8 data/prepare_sft_data.py --input "$PREPARED_DIR/sft_validation.jsonl" --output data/sft_validation.json
llamafactory-cli train configs/sft_qwen3_32b.yaml
llamafactory-cli export configs/export_sft.yaml

if [[ "$RUN_DPO" == "1" ]]; then
  for file in dpo_train.jsonl dpo_validation.jsonl; do
    test -s "$PREPARED_DIR/$file" || { echo "缺少非空偏好数据：$file" >&2; exit 1; }
  done
  python -X utf8 data/prepare_dpo_data.py --input "$PREPARED_DIR/dpo_train.jsonl" --output data/dpo.json
  python -X utf8 data/prepare_dpo_data.py --input "$PREPARED_DIR/dpo_validation.jsonl" --output data/dpo_validation.json
  llamafactory-cli train configs/dpo_qwen3_32b.yaml
  llamafactory-cli export configs/export_dpo.yaml
fi
printf '%s\n' '训练与导出已完成；实际收益请使用独立Agent任务集评测。'
