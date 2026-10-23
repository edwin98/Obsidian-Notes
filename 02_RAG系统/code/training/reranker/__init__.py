# Reranker 微调模块（gte-multilingual-reranker-0.3B，全量 SFT）
#
# 数据准备流程：
#   score_with_llm.py          → 用 Qwen2.5-72B 对 (query, doc) 打 1~5 分
#   build_reranker_dataset.py  → 挖掘硬负样本，组装 Listwise 训练格式
#
# 训练流程：
#   dataset.py                 → RerankerListwiseDataset / collate_fn
#   train_reranker.py          → 全量 SFT，sentence-transformers + accelerate DDP
#
# 后处理：
#   calibrate_scores.py        → Platt Scaling 概率校准
