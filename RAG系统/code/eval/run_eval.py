"""RAG 评测 Demo：检索层指标 + RAGAS + DeepEval 一键运行。

运行前提
--------
  cd code/
  pip install -r eval/requirements-eval.txt
  docker-compose up -d   # ES / Milvus（可选，不启动则降级为纯 BM25）

运行方式
--------
  # 只跑检索层指标（无需 LLM，< 30s）
      python eval/run_eval.py --no-llm

  # 完整评测（OpenAI API）
  OPENAI_API_KEY=sk-xxx python eval/run_eval.py

  # 完整评测（自建 vLLM，OpenAI 兼容接口）
  RAG_EVAL_LLM_BASE_URL=http://localhost:8001/v1 \\
  RAG_EVAL_LLM_MODEL=Qwen2.5-72B-Instruct \\
  OPENAI_API_KEY=no-key \\
  python eval/run_eval.py

  # 烟雾模式（只用前 4 条样本，快速验证流程）
  python eval/run_eval.py --smoke

理解各阶段的作用
----------------
  Phase 1  自研检索层指标（Recall@K / MRR / NDCG@K）
           → 定位是哪一级召回丢失了正确 Chunk
  Phase 2  RAGAS（Faithfulness / ContextPrecision / ContextRecall）
           → 评测生成质量：回答是否忠实于召回内容
  Phase 3  DeepEval（FaithfulnessMetric / ContextualRecallMetric）
           → 同 RAGAS 但框架不同，适合嵌入 pytest CI 流水线
"""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import sys
import time
from dataclasses import dataclass, field

# 保证从 code/ 目录导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.dataset import EVAL_SAMPLES
from models.schemas import DocumentChunk, RetrievedChunk

# ─────────────────────────────────────────────────────────────────────────────
# 辅助数据结构
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PipelineResult:
    """单条样本的完整运行结果。"""

    qid: str
    query: str
    query_type: str
    expected_doc_ids: list[str]
    reference_answer: str
    retrieved_chunk_ids: list[str] = field(default_factory=list)
    retrieved_texts: list[str] = field(default_factory=list)
    answer: str = ""
    gt_chunk_ids: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0：系统初始化 & 数据摄入
# ─────────────────────────────────────────────────────────────────────────────


def setup_system():
    """初始化所有组件并摄入 8 篇示例文档。"""
    from api.dependencies import init_components
    from data.sample_documents import SAMPLE_DOCUMENTS

    print("\n" + "=" * 60)
    print("Phase 0  系统初始化")
    print("=" * 60)

    comp = init_components()

    print(f"\n[Ingest] 摄入 {len(SAMPLE_DOCUMENTS)} 篇示例文档...")
    total_chunks = 0
    for doc in SAMPLE_DOCUMENTS:
        try:
            chunks = comp.ingestion_pipeline.ingest_document(
                doc_id=doc["doc_id"],
                doc_name=doc["doc_name"],
                raw_content=doc["content"],
            )
        except Exception:
            chunks = comp.ingestion_pipeline.ingest_document_direct(
                doc_id=doc["doc_id"],
                doc_name=doc["doc_name"],
                raw_content=doc["content"],
            )
        total_chunks += len(chunks)
        print(f"  {doc['doc_id']}  {doc['doc_name']} -> {len(chunks)} chunks")

    # 刷新 BM25 索引
    try:
        comp.bm25_engine.refresh()
    except Exception:
        pass
    try:
        comp.vector_engine.refresh()
    except Exception:
        pass

    print(f"\n[Ingest] 完成，共 {total_chunks} 个 chunk 入库")
    return comp


def build_gt_chunk_ids(comp, samples: list[dict]) -> dict[str, list[str]]:
    """从 chunk_store 中找出每条样本对应的 Ground Truth Chunk ID 列表。"""
    gt_map: dict[str, list[str]] = {}
    for sample in samples:
        expected = set(sample["expected_doc_ids"])
        gt_ids = [
            cid
            for cid, chunk in comp.chunk_store.items()
            if chunk.metadata.doc_id in expected
        ]
        gt_map[sample["qid"]] = gt_ids
    return gt_map


# ─────────────────────────────────────────────────────────────────────────────
# 核心：运行检索 + 生成
# ─────────────────────────────────────────────────────────────────────────────


async def _retrieve(comp, query: str, top_k: int = 10) -> list[RetrievedChunk]:
    """直接调用三级检索（不经过改写，以单独评测基础检索能力）。"""
    return await comp.retriever.retrieve(query, [query], top_k=top_k)


def _build_answer_from_chunks(query: str, chunks: list[DocumentChunk]) -> str:
    """从 chunk 直接拼装回答（跳过流式延迟，专为评测设计）。"""
    if not chunks:
        return "根据当前已知知识库，暂时无法回答该问题。"
    parts = [f"关于「{query}」，根据检索到的资料回答如下：\n"]
    for i, chunk in enumerate(chunks[:5], 1):
        snippet = chunk.text.strip()[:300]
        if len(chunk.text) > 300:
            snippet += "..."
        parts.append(f"\n**参考 {i}** [{chunk.chunk_id}]：\n{snippet}")
    return "\n".join(parts)


def run_pipeline_batch(comp, samples: list[dict], gt_map: dict) -> list[PipelineResult]:
    """对全部样本执行检索（同步包装 async），返回 PipelineResult 列表。"""

    async def _batch():
        results = []
        for sample in samples:
            retrieved = await _retrieve(comp, sample["query"], top_k=10)
            chunks = [r.chunk for r in retrieved]
            result = PipelineResult(
                qid=sample["qid"],
                query=sample["query"],
                query_type=sample["query_type"],
                expected_doc_ids=sample["expected_doc_ids"],
                reference_answer=sample["reference_answer"],
                retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved],
                retrieved_texts=[r.chunk.text for r in retrieved],
                answer=_build_answer_from_chunks(sample["query"], chunks),
                gt_chunk_ids=gt_map.get(sample["qid"], []),
            )
            results.append(result)
        return results

    return asyncio.run(_batch())


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1：自研检索层指标
# ─────────────────────────────────────────────────────────────────────────────


def _recall_at_k(retrieved_ids: list[str], gt_ids: set[str], k: int) -> float:
    return 1.0 if any(cid in gt_ids for cid in retrieved_ids[:k]) else 0.0


def _mrr(retrieved_ids: list[str], gt_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in gt_ids:
            return 1.0 / rank
    return 0.0


def _ndcg_at_k(retrieved_ids: list[str], gt_ids: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], 1)
        if cid in gt_ids
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(gt_ids), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def run_retrieval_eval(pipeline_results: list[PipelineResult]) -> dict:
    """计算 Recall@3/5/10、MRR@10、NDCG@10。"""
    print("\n" + "=" * 60)
    print("Phase 1  自研检索层指标（无需 LLM）")
    print("=" * 60)

    scores: dict[str, list[float]] = {
        "recall@3": [], "recall@5": [], "recall@10": [],
        "mrr@10": [], "ndcg@10": [],
    }

    print(f"\n{'qid':<8} {'query':<36} {'R@3':>5} {'R@10':>5} {'MRR':>6} {'失败原因'}")
    print("-" * 78)

    for r in pipeline_results:
        gt = set(r.gt_chunk_ids)
        if not gt:
            print(f"  {r.qid:<8} {'[无 GT chunk，跳过]':<36}")
            continue

        r3 = _recall_at_k(r.retrieved_chunk_ids, gt, 3)
        r5 = _recall_at_k(r.retrieved_chunk_ids, gt, 5)
        r10 = _recall_at_k(r.retrieved_chunk_ids, gt, 10)
        mrr = _mrr(r.retrieved_chunk_ids, gt)
        ndcg = _ndcg_at_k(r.retrieved_chunk_ids, gt, 10)

        scores["recall@3"].append(r3)
        scores["recall@5"].append(r5)
        scores["recall@10"].append(r10)
        scores["mrr@10"].append(mrr)
        scores["ndcg@10"].append(ndcg)

        miss_reason = ""
        if r10 == 0:
            miss_reason = "R_MISS (Top10 无 GT)"
        elif r3 == 0 and r10 > 0:
            miss_reason = "排名靠后 (>3)"

        query_short = r.query[:34] + ".." if len(r.query) > 36 else r.query
        print(
            f"  {r.qid:<8} {query_short:<36} "
            f"{'✓' if r3 else '✗':>5} {'✓' if r10 else '✗':>5} "
            f"{mrr:>6.3f}  {miss_reason}"
        )

    n = len(scores["recall@10"])
    avg = {k: sum(v) / len(v) for k, v in scores.items() if v}

    print("\n── 汇总 " + "─" * 50)
    print(f"  样本数:     {n}")
    print(f"  Recall@3:   {avg.get('recall@3', 0):.1%}")
    print(f"  Recall@5:   {avg.get('recall@5', 0):.1%}")
    print(f"  Recall@10:  {avg.get('recall@10', 0):.1%}")
    print(f"  MRR@10:     {avg.get('mrr@10', 0):.3f}")
    print(f"  NDCG@10:    {avg.get('ndcg@10', 0):.3f}")

    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2：RAGAS
# ─────────────────────────────────────────────────────────────────────────────


def _make_ragas_llm():
    """构建 RAGAS 使用的 LLM（支持自建 vLLM 端点）。"""
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=os.getenv("RAG_EVAL_LLM_MODEL", "gpt-3.5-turbo"),
        base_url=os.getenv("RAG_EVAL_LLM_BASE_URL") or None,  # None -> 使用 OpenAI 默认
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    return LangchainLLMWrapper(llm)


def run_ragas_eval(pipeline_results: list[PipelineResult]) -> dict | None:
    """运行 RAGAS 评测：Faithfulness / ContextPrecision / ContextRecall。"""
    print("\n" + "=" * 60)
    print("Phase 2  RAGAS 评测（需要 LLM API）")
    print("=" * 60)

    # ── 导入检查 ────────────────────────────────────────────────────────────
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import ContextPrecision, ContextRecall, Faithfulness
    except ImportError as e:
        print(f"\n  [跳过] RAGAS 未安装或版本不兼容: {e}")
        print("  安装方式: pip install ragas>=0.2.0 langchain-openai")
        return None
    except Exception as e:
        print(f"\n  [跳过] RAGAS 导入失败: {e}")
        return None

    # ── 构建数据集 ───────────────────────────────────────────────────────────
    # RAGAS 0.2.x 使用 SingleTurnSample + EvaluationDataset
    # 字段说明：
    #   user_input       = 用户问题
    #   response         = RAG 系统的回答（此处为 Mock 生成）
    #   retrieved_contexts = 召回的 Chunk 文本列表
    #   reference        = 标准参考答案（用于 ContextRecall 计算）
    samples = [
        SingleTurnSample(
            user_input=r.query,
            response=r.answer,
            retrieved_contexts=r.retrieved_texts[:5],  # 取 Top5 Chunk
            reference=r.reference_answer,
        )
        for r in pipeline_results
    ]
    dataset = EvaluationDataset(samples=samples)

    # ── 配置 LLM ────────────────────────────────────────────────────────────
    try:
        ragas_llm = _make_ragas_llm()
    except Exception as e:
        print(f"\n  [跳过] LLM 初始化失败: {e}")
        return None

    metrics = [
        Faithfulness(llm=ragas_llm),        # 回答中的论断是否都能在 Chunk 中找到依据
        ContextPrecision(llm=ragas_llm),     # 召回的 Chunk 有多少是真正有用的
        ContextRecall(llm=ragas_llm),        # GT 答案所需信息是否被 Chunk 覆盖
    ]

    print(f"\n  样本数: {len(samples)}")
    print("  指标:   Faithfulness / ContextPrecision / ContextRecall")
    print("  LLM:   ", os.getenv("RAG_EVAL_LLM_MODEL", "gpt-3.5-turbo"))
    print("\n  [运行中，每条样本约需 3~6 次 LLM 调用...]\n")

    t0 = time.time()
    try:
        result = ragas_evaluate(dataset=dataset, metrics=metrics)
    except Exception as e:
        print(f"\n  [错误] RAGAS 评测失败: {e}")
        print("  常见原因：API Key 无效 / 网络不通 / LLM 端点返回格式不兼容")
        return None
    elapsed = time.time() - t0

    scores = result.to_pandas().mean(numeric_only=True).to_dict()

    print("── 汇总 " + "─" * 50)
    for metric_name, score in scores.items():
        print(f"  {metric_name:<30} {score:.4f}")
    print(f"\n  耗时: {elapsed:.1f}s  ({elapsed / len(samples):.1f}s / 条)")

    # ── 关键说明：为什么 Faithfulness 在 Mock 模式下偏低 ─────────────────────
    print("\n  [注意] 当前使用 Mock LLM 生成器，回答由规则拼装而非真实推理产生。")
    print("  Faithfulness 可能偏低，这是预期现象。")
    print("  生产环境接入 Qwen2.5-72B 后指标会显著提升。")

    return scores


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3：DeepEval
# ─────────────────────────────────────────────────────────────────────────────


def _setup_deepeval_llm():
    """配置 DeepEval 使用的 LLM 端点（设置环境变量）。"""
    if base_url := os.getenv("RAG_EVAL_LLM_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = base_url
    if model := os.getenv("RAG_EVAL_LLM_MODEL"):
        # DeepEval 通过 DEEPEVAL_MODEL 或直接在 Metric 中指定 model 参数
        os.environ["DEEPEVAL_MODEL"] = model


def run_deepeval_smoke(pipeline_results: list[PipelineResult]) -> dict | None:
    """运行 DeepEval 烟雾评测：FaithfulnessMetric + ContextualRecallMetric。

    DeepEval 与 pytest 深度集成，此处演示独立运行方式。
    在 CI 中的典型用法见 tests/test_rag_quality.py。
    """
    print("\n" + "=" * 60)
    print("Phase 3  DeepEval 烟雾评测（需要 LLM API）")
    print("=" * 60)

    # ── 导入检查 ────────────────────────────────────────────────────────────
    try:
        from deepeval import evaluate as deepeval_evaluate
        from deepeval.metrics import ContextualRecallMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        print(f"\n  [跳过] DeepEval 未安装: {e}")
        print("  安装方式: pip install deepeval")
        return None

    _setup_deepeval_llm()

    # ── 构建 TestCase ────────────────────────────────────────────────────────
    # LLMTestCase 字段说明：
    #   input              = 用户问题
    #   actual_output      = RAG 系统的回答
    #   retrieval_context  = 召回的 Chunk 文本列表（用于 Faithfulness 验证）
    #   expected_output    = 标准参考答案（用于 ContextualRecall 计算）
    test_cases = [
        LLMTestCase(
            input=r.query,
            actual_output=r.answer,
            retrieval_context=r.retrieved_texts[:5],
            expected_output=r.reference_answer,
        )
        for r in pipeline_results
    ]

    metrics = [
        FaithfulnessMetric(
            threshold=0.7,
            model=os.getenv("RAG_EVAL_LLM_MODEL", "gpt-3.5-turbo"),
            include_reason=True,  # 输出失败原因，便于定界
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model=os.getenv("RAG_EVAL_LLM_MODEL", "gpt-3.5-turbo"),
            include_reason=True,
        ),
    ]

    print(f"\n  样本数: {len(test_cases)}")
    print("  指标:   FaithfulnessMetric / ContextualRecallMetric")
    print("  阈值:   0.7（低于此值视为失败）")
    print(f"  LLM:    {os.getenv('RAG_EVAL_LLM_MODEL', 'gpt-3.5-turbo')}")
    print("\n  [运行中...]\n")

    t0 = time.time()
    try:
        results = deepeval_evaluate(
            test_cases=test_cases,
            metrics=metrics,
            print_results=False,   # 我们自己格式化输出
            run_async=False,
        )
    except Exception as e:
        print(f"\n  [错误] DeepEval 评测失败: {e}")
        print("  常见原因：API Key 无效 / 网络不通 / deepeval 版本与 API 不兼容")
        return None
    elapsed = time.time() - t0

    # ── 汇总输出 ─────────────────────────────────────────────────────────────
    pass_count = sum(1 for tc in test_cases if all(m.success for m in tc.metrics_data or []))
    faith_scores = []
    recall_scores = []

    print(f"{'qid':<8} {'Faithfulness':>14} {'CtxRecall':>10} {'通过':>5}")
    print("-" * 45)

    for r, tc in zip(pipeline_results, test_cases):
        faith = next(
            (m for m in (tc.metrics_data or []) if "Faithfulness" in m.name), None
        )
        ctx_recall = next(
            (m for m in (tc.metrics_data or []) if "ContextualRecall" in m.name), None
        )
        f_score = faith.score if faith else 0.0
        c_score = ctx_recall.score if ctx_recall else 0.0
        passed = (faith.success if faith else False) and (ctx_recall.success if ctx_recall else False)
        faith_scores.append(f_score)
        recall_scores.append(c_score)
        print(
            f"  {r.qid:<8} {f_score:>14.3f} {c_score:>10.3f} {'✓' if passed else '✗':>5}"
        )

    avg_faith = sum(faith_scores) / len(faith_scores) if faith_scores else 0
    avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0

    print("\n── 汇总 " + "─" * 34)
    print(f"  Faithfulness 均值:      {avg_faith:.3f}")
    print(f"  ContextualRecall 均值:  {avg_recall:.3f}")
    print(f"  通过率:                 {pass_count}/{len(test_cases)}")
    print(f"  耗时:                   {elapsed:.1f}s  ({elapsed / len(test_cases):.1f}s / 条)")

    # ── 失败原因打印（帮助定界）────────────────────────────────────────────────
    failures = [
        (r, tc) for r, tc in zip(pipeline_results, test_cases)
        if not all(m.success for m in (tc.metrics_data or []))
    ]
    if failures:
        print("\n── 失败用例分析 " + "─" * 27)
        for r, tc in failures[:3]:  # 最多打印 3 条
            print(f"\n  qid: {r.qid}  query: {r.query}")
            for m in (tc.metrics_data or []):
                if not m.success:
                    print(f"    [{m.name}] score={m.score:.3f}  reason: {m.reason}")

    return {"faithfulness": avg_faith, "contextual_recall": avg_recall}


# ─────────────────────────────────────────────────────────────────────────────
# 最终报告
# ─────────────────────────────────────────────────────────────────────────────


def print_final_report(
    retrieval: dict,
    ragas: dict | None,
    deepeval: dict | None,
    elapsed_total: float,
):
    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)

    print("\n【Phase 1 检索层】")
    for k, v in retrieval.items():
        bar = "█" * int(v * 20)
        print(f"  {k:<12} {v:.1%}  {bar}")

    if ragas:
        print("\n【Phase 2 RAGAS】")
        for k, v in ragas.items():
            bar = "█" * int(v * 20)
            print(f"  {k:<30} {v:.4f}  {bar}")
    else:
        print("\n【Phase 2 RAGAS】  已跳过（--no-llm 或安装缺失）")

    if deepeval:
        print("\n【Phase 3 DeepEval】")
        for k, v in deepeval.items():
            bar = "█" * int(v * 20)
            print(f"  {k:<28} {v:.3f}  {bar}")
    else:
        print("\n【Phase 3 DeepEval】  已跳过（--no-llm 或安装缺失）")

    print(f"\n总耗时: {elapsed_total:.1f}s")
    print("\n提示：")
    print("  - Recall@10 低 → 排查三级召回各层（--no-llm 下已可定界）")
    print("  - Faithfulness 低 → 排查 System Prompt / 生成模型")
    print("  - ContextRecall 低 → 排查切片策略 / 知识库覆盖度")


# ─────────────────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="RAG 评测 Demo")
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="只跑检索层指标，跳过 RAGAS 和 DeepEval（无需 LLM API，< 30s）",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="烟雾模式：只使用前 4 条样本，快速验证流程",
    )
    args = parser.parse_args()

    t_start = time.time()

    # ── 选定样本 ─────────────────────────────────────────────────────────────
    samples = EVAL_SAMPLES[:4] if args.smoke else EVAL_SAMPLES
    mode = "烟雾模式" if args.smoke else "完整模式"
    llm_mode = "无 LLM（仅检索层）" if args.no_llm else "含 LLM 评测"
    print(f"\nRAG 评测 Demo  |  {mode}  |  {llm_mode}  |  {len(samples)} 条样本")

    # ── Phase 0: 初始化 ───────────────────────────────────────────────────────
    comp = setup_system()
    gt_map = build_gt_chunk_ids(comp, samples)

    # ── 检索 + 生成（批量，结果复用于 Phase 1~3）────────────────────────────
    print("\n[Pipeline] 批量执行检索...")
    pipeline_results = run_pipeline_batch(comp, samples, gt_map)
    print(f"[Pipeline] 完成，{len(pipeline_results)} 条结果")

    # ── Phase 1: 检索层指标 ───────────────────────────────────────────────────
    retrieval_metrics = run_retrieval_eval(pipeline_results)

    # ── Phase 2: RAGAS ───────────────────────────────────────────────────────
    ragas_metrics = None
    if not args.no_llm:
        if not os.getenv("OPENAI_API_KEY"):
            print(
                "\n[Phase 2] 未检测到 OPENAI_API_KEY，跳过 RAGAS。\n"
                "  设置方式：export OPENAI_API_KEY=sk-xxx\n"
                "  自建 vLLM：export RAG_EVAL_LLM_BASE_URL=http://localhost:8001/v1"
            )
        else:
            ragas_metrics = run_ragas_eval(pipeline_results)

    # ── Phase 3: DeepEval ────────────────────────────────────────────────────
    deepeval_metrics = None
    if not args.no_llm:
        if not os.getenv("OPENAI_API_KEY"):
            print("\n[Phase 3] 未检测到 OPENAI_API_KEY，跳过 DeepEval。")
        else:
            deepeval_metrics = run_deepeval_smoke(pipeline_results)

    # ── 最终报告 ─────────────────────────────────────────────────────────────
    print_final_report(
        retrieval=retrieval_metrics,
        ragas=ragas_metrics,
        deepeval=deepeval_metrics,
        elapsed_total=time.time() - t_start,
    )


if __name__ == "__main__":
    main()
