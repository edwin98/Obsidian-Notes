"""
calibrate_scores.py
====================
对训练好的 Reranker 进行 Platt Scaling 概率校准。

背景：
  Reranker 输出的原始分数只有相对排序意义，不具备概率含义。
  例如，score=0.8 不代表"80% 概率相关"。
  校准后，score=0.7 可以被解读为"该文档有 70% 概率与 query 相关"，
  从而支持设置过滤阈值（如 score < 0.3 时触发兜底降级策略）。

方法：Platt Scaling
  在一个独立的校准集（不参与训练）上：
  1. 用训练好的 Reranker 计算所有 (query, doc) pair 的原始分数
  2. 用 Logistic Regression 拟合 raw_score → P(相关) 的映射
  3. 保存 calibrator（sklearn 模型）到 checkpoint 目录

使用：
  python calibrate_scores.py \
      --model_dir   checkpoints/reranker-finetuned/best_checkpoint \
      --calib_file  data/calibration_set.jsonl \
      --output_dir  checkpoints/reranker-finetuned/best_checkpoint

校准后推理示例：
  calibrated_score = calibrator.predict_proba([[raw_score]])[0][1]
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 收集原始分数
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_raw_scores(
    model_dir: str,
    calib_file: Path,
    max_length: int = 512,
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    用训练好的 Reranker 在校准集上计算原始分数。

    校准集格式（JSONL，每行一条）：
      {"query": "...", "doc": "...", "label": 1}  # label: 1=相关, 0=不相关
      或带连续分数：
      {"query": "...", "doc": "...", "llm_score": 4}

    Returns:
        raw_scores: shape [N]，Reranker 输出的原始 logit
        binary_labels: shape [N]，0 或 1（llm_score >= 4 视为相关）
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)

    queries, docs, labels = [], [], []
    with open(calib_file, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            queries.append(item["query"])
            docs.append(item["doc"])
            # 支持二值 label 或 LLM 分数
            if "label" in item:
                labels.append(int(item["label"]))
            elif "llm_score" in item:
                labels.append(1 if item["llm_score"] >= 4 else 0)
            elif "normalized_score" in item:
                labels.append(1 if item["normalized_score"] >= 0.75 else 0)
            else:
                raise ValueError(f"Cannot determine label from: {item}")

    raw_scores = []
    for i in range(0, len(queries), batch_size):
        batch_q = queries[i : i + batch_size]
        batch_d = docs[i : i + batch_size]
        enc = tokenizer(
            batch_q, batch_d,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc)
        scores = out.logits.squeeze(-1).cpu().numpy()
        raw_scores.extend(scores.tolist())

        if (i // batch_size + 1) % 20 == 0:
            logger.info("Scored %d / %d", i + len(batch_q), len(queries))

    return np.array(raw_scores, dtype=np.float32), np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# Platt Scaling 拟合
# ---------------------------------------------------------------------------

def fit_platt_scaling(
    raw_scores: np.ndarray,
    labels: np.ndarray,
) -> LogisticRegression:
    """
    用 Logistic Regression 拟合 raw_score → P(相关) 的映射（Platt Scaling）。
    """
    calibrator = LogisticRegression(max_iter=1000)
    calibrator.fit(raw_scores.reshape(-1, 1), labels)

    # 评估校准质量
    prob_pred = calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
    bs = brier_score_loss(labels, prob_pred)
    ll = log_loss(labels, prob_pred)
    logger.info("Calibration quality → Brier Score: %.4f | Log Loss: %.4f", bs, ll)

    # 校准曲线（理想情况下接近对角线）
    fraction_pos, mean_pred = calibration_curve(labels, prob_pred, n_bins=10)
    logger.info("Calibration curve (mean_pred → fraction_positive):")
    for mp, fp in zip(mean_pred, fraction_pos):
        logger.info("  %.2f → %.2f", mp, fp)

    return calibrator


# ---------------------------------------------------------------------------
# 保存 & 加载
# ---------------------------------------------------------------------------

def save_calibrator(calibrator: LogisticRegression, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "platt_calibrator.pkl"
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)
    logger.info("Calibrator saved → %s", path)

    # 同时保存一个可读的参数文件，方便其他语言（Java/Go）复现
    params = {
        "coef": calibrator.coef_[0][0],
        "intercept": calibrator.intercept_[0],
        "formula": "P(relevant) = sigmoid(coef * raw_score + intercept)",
    }
    with open(output_dir / "platt_params.json", "w") as f:
        json.dump(params, f, indent=2)
    logger.info("Platt params: coef=%.4f, intercept=%.4f", params["coef"], params["intercept"])


def load_calibrator(model_dir: str | Path) -> LogisticRegression:
    path = Path(model_dir) / "platt_calibrator.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 推理示例（校准后）
# ---------------------------------------------------------------------------

class CalibratedReranker:
    """
    已校准的 Reranker 推理封装。

    Usage:
        reranker = CalibratedReranker("checkpoints/reranker-finetuned/best_checkpoint")
        scores = reranker.score(query, docs)
        # scores[i] = P(doc_i 与 query 相关) ∈ [0, 1]
    """

    def __init__(self, model_dir: str, device: str = "cuda") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval().to(device)
        self.device = device
        self.calibrator = load_calibrator(model_dir)

    @torch.no_grad()
    def score(self, query: str, docs: list[str], max_length: int = 512) -> list[float]:
        """返回校准后的概率分数列表，与 docs 一一对应。"""
        enc = self.tokenizer(
            [query] * len(docs), docs,
            max_length=max_length, truncation=True, padding="max_length", return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        raw_scores = self.model(**enc).logits.squeeze(-1).cpu().numpy()
        calibrated = self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        return calibrated.tolist()

    def rerank(self, query: str, docs: list[str], threshold: float = 0.3) -> list[dict]:
        """
        对 docs 重排序，返回 [{doc, score, is_relevant}, ...] 按分数降序。
        score < threshold 的文档标记为 is_relevant=False（可触发降级策略）。
        """
        scores = self.score(query, docs)
        results = [
            {"doc": doc, "score": score, "is_relevant": score >= threshold}
            for doc, score in zip(docs, scores)
        ]
        return sorted(results, key=lambda x: x["score"], reverse=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Platt Scaling calibration for Reranker")
    p.add_argument("--model_dir",   required=True, type=str, help="训练好的 Reranker checkpoint 目录")
    p.add_argument("--calib_file",  required=True, type=Path, help="校准集 JSONL，每行 {query, doc, label}")
    p.add_argument("--output_dir",  default=None,  type=Path, help="校准器保存目录（默认同 model_dir）")
    p.add_argument("--max_length",  type=int, default=512)
    p.add_argument("--batch_size",  type=int, default=64)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir or args.model_dir)

    logger.info("Collecting raw scores on calibration set...")
    raw_scores, labels = collect_raw_scores(
        model_dir=args.model_dir,
        calib_file=args.calib_file,
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=args.device,
    )
    logger.info("Calibration set: %d samples | Positive rate: %.2f%%",
                len(labels), labels.mean() * 100)

    logger.info("Fitting Platt Scaling...")
    calibrator = fit_platt_scaling(raw_scores, labels)

    save_calibrator(calibrator, output_dir)
    logger.info("Calibration complete.")


if __name__ == "__main__":
    main()
