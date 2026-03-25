"""
rag.py — 三级混合检索 RAG 模块

对应全景报告第五章 5.1 节：RAG 赋能的 LLM 用例生成

架构：
  BM25 (Elasticsearch) ──┐
                          ├─► RSF 自适应融合 ──► Cross-Encoder Reranker ──► Top-K Context
  向量检索 (Milvus HNSW) ─┘

关键设计：
  1. RSF（Reciprocal Score Fusion）自适应调权：
     短查询（词数 ≤ 5）偏向 BM25，长查询偏向向量检索，不是固定 50/50
  2. 检索前置增强：
     - 指代消解：将"它的切换成功率"解析为具体实体
     - 多路问题扩展：一个查询扩展为多个角度，提升召回覆盖率
     - AutoPhrase 分词保护：防止 NR_PDCP、gNB-DU 被错误切分
  3. Cross-Encoder Reranker：
     对 BM25 + 向量的合并候选集做精排，动态阈值过滤噪声

本文件为可独立运行的演示实现，所有"外部服务"（ES、Milvus、LLM）均以
Mock 模拟，核心算法逻辑（RSF、Reranker、指代消解）完整实现。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# 第一部分：数据结构
# ═══════════════════════════════════════════════════════════════════


@dataclass
class Document:
    """知识库文档单元"""

    doc_id: str
    content: str
    source: str  # "3gpp" | "bugzilla" | "golden_case"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedDoc:
    """检索结果，携带来源和原始分数"""

    doc: Document
    score: float
    retriever: str  # "bm25" | "vector"


@dataclass
class RankedDoc:
    """Reranker 精排后的文档"""

    doc: Document
    rsf_score: float  # RSF 融合后的分数
    rerank_score: float  # Cross-Encoder 精排分数
    final_score: float  # rsf_score * rerank_weight


# ═══════════════════════════════════════════════════════════════════
# 第二部分：AutoPhrase 分词保护
# ═══════════════════════════════════════════════════════════════════

# 通信领域专有术语词典，防止被通用分词器错误切分
_5G_PHRASES: set[str] = {
    "NR_PDCP",
    "gNB-DU",
    "gNB-CU",
    "SN Status Transfer",
    "PRACH",
    "BWP",
    "N2 interface",
    "Xn interface",
    "F1-AP",
    "PDCP reorder",
    "NAS authentication",
    "RRC connection",
    "handover preparation",
    "A3 event",
    "QoS flow",
    "VoNR",
    "NGAP",
    "S1AP",
    "X2AP",
    "E-UTRA",
    "NR-ARFCN",
    "PagedAttention",
    "3GPP TS 38.331",
    "3GPP TS 38.413",
}

# 编译正则：将词典中的术语替换为带下划线的原子形式，防止切分
_PHRASE_PATTERN = re.compile(
    "|".join(re.escape(p) for p in sorted(_5G_PHRASES, key=len, reverse=True))
)


def protect_phrases(text: str) -> str:
    """将专有术语替换为 token 原子形式，BM25 分词前调用"""
    return _PHRASE_PATTERN.sub(
        lambda m: m.group(0).replace(" ", "_").replace("-", "_"), text
    )


def restore_phrases(token: str) -> str:
    """还原被保护的术语（用于展示）"""
    return token.replace("_", " ")


# ═══════════════════════════════════════════════════════════════════
# 第三部分：Mock 知识库
# ═══════════════════════════════════════════════════════════════════

KNOWLEDGE_BASE: list[Document] = [
    Document(
        doc_id="3gpp-38331-xn-handover",
        content="3GPP TS 38.331 defines Xn interface handover procedure. "
        "SN Status Transfer message must be sent within 50ms of HO command. "
        "Failure to receive SN Status Transfer causes PDCP reorder timeout.",
        source="3gpp",
        metadata={"spec": "TS 38.331", "section": "10.2.2"},
    ),
    Document(
        doc_id="3gpp-38413-nas-auth",
        content="3GPP TS 38.413 NAS authentication procedure for roaming UE. "
        "AMF sends Authentication Request, UE responds with Authentication Response. "
        "Authentication failure triggers re-authentication or rejection.",
        source="3gpp",
        metadata={"spec": "TS 38.413", "section": "8.6"},
    ),
    Document(
        doc_id="bugzilla-xn-packet-loss",
        content="BUG-2024-0891: Xn backhaul packet loss causes SN Status Transfer timeout. "
        "Handover success rate drops below 85% when Xn RTT > 20ms. "
        "Root cause: transport network congestion at aggregation switch.",
        source="bugzilla",
        metadata={"bug_id": "BUG-2024-0891", "severity": "P0", "resolved": True},
    ),
    Document(
        doc_id="bugzilla-nas-roaming-fail",
        content="BUG-2024-0456: NAS authentication failure in roaming scenario. "
        "AUSF timeout when UDM response > 2s. Affects international roaming UEs.",
        source="bugzilla",
        metadata={"bug_id": "BUG-2024-0456", "severity": "P1", "resolved": True},
    ),
    Document(
        doc_id="golden-vonr-qos",
        content="Golden Case TC-0071: VoNR bearer setup with QoS flow mapping. "
        "Expected: GBR bearer established within 200ms, QCI=1 mapping verified. "
        "Pass criteria: call success rate >= 99%, MOS >= 4.0.",
        source="golden_case",
        metadata={"tc_id": "TC-0071", "feature": "VoNR", "validated_by": "expert_team"},
    ),
    Document(
        doc_id="golden-xn-handover-baseline",
        content="Golden Case TC-0042: Xn handover baseline. "
        "100 consecutive handovers, A3 offset=3dB, speed=30km/h. "
        "Pass criteria: success_rate >= 99%, SN Status Transfer latency < 30ms.",
        source="golden_case",
        metadata={
            "tc_id": "TC-0042",
            "feature": "Xn_handover",
            "validated_by": "expert_team",
        },
    ),
    Document(
        doc_id="3gpp-38300-pdcp",
        content="3GPP TS 38.300 PDCP reordering timer. "
        "t-Reordering timer triggers when out-of-order PDCP PDUs detected. "
        "Excessive reordering indicates radio link degradation or Xn backhaul issue.",
        source="3gpp",
        metadata={"spec": "TS 38.300", "section": "6.4.2"},
    ),
]


# ═══════════════════════════════════════════════════════════════════
# 第四部分：BM25 检索（Mock Elasticsearch）
# ═══════════════════════════════════════════════════════════════════


def _tokenize(text: str) -> list[str]:
    """简单英文分词，保护专有术语"""
    protected = protect_phrases(text.lower())
    return re.findall(r"[a-z0-9_]+", protected)


def _bm25_score(
    query_tokens: list[str], doc_tokens: list[str], k1: float = 1.5, b: float = 0.75
) -> float:
    """BM25 打分（Okapi BM25 标准实现）"""
    avg_doc_len = 50  # 近似值
    doc_len = len(doc_tokens)
    doc_freq: dict[str, int] = {}
    for t in doc_tokens:
        doc_freq[t] = doc_freq.get(t, 0) + 1

    score = 0.0
    for token in query_tokens:
        tf = doc_freq.get(token, 0)
        if tf == 0:
            continue
        idf = math.log(1 + (len(KNOWLEDGE_BASE) - 1 + 0.5) / (1 + 1))
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
        score += idf * tf_norm
    return score


def bm25_retrieve(query: str, top_k: int = 5) -> list[RetrievedDoc]:
    """BM25 精确检索，适合通信专有名词、缩写命中"""
    query_tokens = _tokenize(query)
    results = []
    for doc in KNOWLEDGE_BASE:
        doc_tokens = _tokenize(doc.content)
        score = _bm25_score(query_tokens, doc_tokens)
        if score > 0:
            results.append(RetrievedDoc(doc=doc, score=score, retriever="bm25"))
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════════
# 第五部分：向量检索（Mock Milvus HNSW）
# ═══════════════════════════════════════════════════════════════════


def _mock_embed(text: str) -> list[float]:
    """
    Mock Embedding：基于词袋的确定性向量（维度 16）。
    生产中替换为 Qwen3-embedding / BGE 模型的真实输出。
    """
    tokens = _tokenize(text)
    vec = [0.0] * 16
    for token in tokens:
        for i, char in enumerate(token[:16]):
            vec[i % 16] += ord(char) / 1000.0
    # L2 归一化
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-9
    nb = math.sqrt(sum(x * x for x in b)) or 1e-9
    return dot / (na * nb)


def vector_retrieve(query: str, top_k: int = 5) -> list[RetrievedDoc]:
    """向量语义检索，适合长句语义意图理解"""
    query_vec = _mock_embed(query)
    results = []
    for doc in KNOWLEDGE_BASE:
        doc_vec = _mock_embed(doc.content)
        sim = _cosine_sim(query_vec, doc_vec)
        results.append(RetrievedDoc(doc=doc, score=sim, retriever="vector"))
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:top_k]


# ═══════════════════════════════════════════════════════════════════
# 第六部分：RSF 自适应融合算法
# ═══════════════════════════════════════════════════════════════════


def _query_word_count(query: str) -> int:
    """查询词数（用于自适应调权判断）"""
    return len(query.split())


def rsf_fuse(
    bm25_results: list[RetrievedDoc],
    vector_results: list[RetrievedDoc],
    query: str,
    k: int = 60,
) -> dict[str, float]:
    """
    RSF（Reciprocal Score Fusion）自适应融合。

    核心思路：
      短查询（≤5词）→ 精确名词匹配为主 → bm25 权重 0.7
      长查询（>5词）→ 语义意图理解为主 → vector 权重 0.7
      不是固定 50/50，而是按查询特征动态调权。

    RRF 公式: score(d) = Σ w_i / (k + rank_i(d))
    """
    word_count = _query_word_count(query)
    if word_count <= 5:
        w_bm25, w_vector = 0.7, 0.3
    else:
        w_bm25, w_vector = 0.3, 0.7

    fused: dict[str, float] = {}

    for rank, r in enumerate(bm25_results, start=1):
        doc_id = r.doc.doc_id
        fused[doc_id] = fused.get(doc_id, 0) + w_bm25 * (1.0 / (k + rank))

    for rank, r in enumerate(vector_results, start=1):
        doc_id = r.doc.doc_id
        fused[doc_id] = fused.get(doc_id, 0) + w_vector * (1.0 / (k + rank))

    return fused


# ═══════════════════════════════════════════════════════════════════
# 第七部分：Cross-Encoder Reranker
# ═══════════════════════════════════════════════════════════════════


def _cross_encoder_score(query: str, doc_content: str) -> float:
    """
    Mock Cross-Encoder 打分（生产替换为 BGE-Reranker / BCE-Reranker）。
    基于关键词重叠度的确定性近似，忽略词序差异。
    """
    query_tokens = set(_tokenize(query))
    doc_tokens = set(_tokenize(doc_content))
    if not query_tokens:
        return 0.0
    overlap = len(query_tokens & doc_tokens)
    # Jaccard 相似度的加权版本
    return overlap / (len(query_tokens) + len(doc_tokens) - overlap + 1e-9)


def rerank(
    query: str,
    candidates: list[Document],
    rsf_scores: dict[str, float],
    threshold: float = 0.05,
) -> list[RankedDoc]:
    """
    Cross-Encoder 精排 + 动态阈值噪声过滤。

    threshold：RSF 分数低于此值的候选直接丢弃，防止低质量文档污染 Context。
    """
    ranked = []
    for doc in candidates:
        rsf_score = rsf_scores.get(doc.doc_id, 0.0)
        if rsf_score < threshold:
            continue
        ce_score = _cross_encoder_score(query, doc.content)
        final_score = rsf_score * 0.4 + ce_score * 0.6
        ranked.append(
            RankedDoc(
                doc=doc,
                rsf_score=rsf_score,
                rerank_score=ce_score,
                final_score=final_score,
            )
        )
    ranked.sort(key=lambda x: x.final_score, reverse=True)
    return ranked


# ═══════════════════════════════════════════════════════════════════
# 第八部分：检索前置增强（Query Enhancement）
# ═══════════════════════════════════════════════════════════════════


def resolve_coreference(query: str) -> str:
    """
    指代消解：将模糊指代替换为具体实体。
    生产中由 Qwen3-4B 执行，此处用规则近似。

    示例：
      "它的切换成功率" → "基站A→B Xn接口切换成功率"
      "该接口的延迟"   → "Xn接口的延迟"
    """
    replacements = {
        r"它的切换": "Xn接口切换",
        r"该接口": "Xn接口",
        r"此功能": "当前测试功能",
        r"\bit\b": "the Xn interface",
        r"\bthe interface\b": "the Xn interface",
    }
    result = query
    for pattern, replacement in replacements.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def expand_query(query: str) -> list[str]:
    """
    多路问题扩展：将一个查询扩展为多个角度，提升召回覆盖率。
    生产中由 Qwen3-4B 执行，此处用模板规则近似。

    示例：
      原始查询 → [原始, 协议规范角度, 历史缺陷角度, KPI指标角度]
    """
    expansions = [query]  # 原始查询始终保留

    # 协议规范角度
    if any(kw in query.lower() for kw in ["handover", "切换", "xn"]):
        expansions.append(
            f"3GPP specification for {query} procedure and message sequence"
        )
    if any(kw in query.lower() for kw in ["auth", "鉴权", "nas"]):
        expansions.append(
            f"3GPP NAS authentication failure scenarios related to {query}"
        )

    # 历史缺陷角度
    expansions.append(f"known defects bugs related to {query}")

    # KPI 指标角度
    if any(kw in query.lower() for kw in ["verify", "test", "验证", "测试"]):
        expansions.append(
            f"KPI metrics pass criteria success rate threshold for {query}"
        )

    return expansions


# ═══════════════════════════════════════════════════════════════════
# 第九部分：完整三级检索 Pipeline
# ═══════════════════════════════════════════════════════════════════


def retrieve(query: str, top_k: int = 3) -> list[RankedDoc]:
    """
    完整 RAG 检索 Pipeline：
      1. 指代消解 + 多路扩展
      2. BM25 + 向量双路检索（对每个扩展查询）
      3. RSF 自适应融合
      4. Cross-Encoder Reranker + 动态阈值过滤
      5. 返回 Top-K 精排结果

    Args:
        query: 用户原始查询或用例生成需求
        top_k: 注入 LLM Prompt 的文档数量

    Returns:
        精排后的文档列表，按 final_score 降序
    """
    # Step 1: 查询增强
    resolved_query = resolve_coreference(query)
    expanded_queries = expand_query(resolved_query)

    # Step 2: 多路检索，合并候选集
    all_bm25: list[RetrievedDoc] = []
    all_vector: list[RetrievedDoc] = []
    for q in expanded_queries:
        all_bm25.extend(bm25_retrieve(q, top_k=5))
        all_vector.extend(vector_retrieve(q, top_k=5))

    # 去重：每个 doc_id 只保留最高分
    def dedup(results: list[RetrievedDoc]) -> list[RetrievedDoc]:
        seen: dict[str, RetrievedDoc] = {}
        for r in results:
            if r.doc.doc_id not in seen or r.score > seen[r.doc.doc_id].score:
                seen[r.doc.doc_id] = r
        return sorted(seen.values(), key=lambda x: x.score, reverse=True)

    bm25_dedup = dedup(all_bm25)
    vector_dedup = dedup(all_vector)

    # Step 3: RSF 融合（使用主查询决定权重）
    rsf_scores = rsf_fuse(bm25_dedup, vector_dedup, query=resolved_query)

    # Step 4: Reranker 精排
    all_docs = {r.doc.doc_id: r.doc for r in bm25_dedup + vector_dedup}
    ranked = rerank(resolved_query, list(all_docs.values()), rsf_scores)

    return ranked[:top_k]


def format_context(ranked_docs: list[RankedDoc]) -> str:
    """将精排文档格式化为注入 LLM Prompt 的 Context 字符串"""
    if not ranked_docs:
        return "No relevant context found."
    parts = []
    for i, rd in enumerate(ranked_docs, start=1):
        doc = rd.doc
        parts.append(
            f"[{i}] Source: {doc.source} | ID: {doc.doc_id} | Score: {rd.final_score:.3f}\n"
            f"{doc.content}"
        )
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════
# 演示入口
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_queries = [
        # 短查询 → BM25 权重 0.7
        "Xn handover SN Status Transfer",
        # 长查询 → 向量权重 0.7
        "验证基站A向基站B的Xn接口切换，近期配置变更后怀疑有问题，需要分析信令层异常",
        # 含指代的查询
        "它的切换成功率低于基线，请分析原因",
    ]

    for q in test_queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {q}")
        wc = _query_word_count(q)
        w_bm25 = 0.7 if wc <= 5 else 0.3
        print(f"  词数={wc}, BM25权重={w_bm25:.1f}, Vector权重={1 - w_bm25:.1f}")
        resolved = resolve_coreference(q)
        if resolved != q:
            print(f"  指代消解: {resolved}")
        results = retrieve(q, top_k=3)
        print(f"  Top-{len(results)} 检索结果:")
        for rd in results:
            print(f"    [{rd.doc.source}] {rd.doc.doc_id} (score={rd.final_score:.3f})")
        print("\n  Context 预览:")
        print(format_context(results)[:400])
