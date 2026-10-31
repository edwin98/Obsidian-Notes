"""检索模块在统一任务上下文中的基础行为。"""

from rag import Document, RetrievedDoc, retrieve, rrf_fuse


def test_rag_returns_context_and_empty_query_returns_nothing():
    results = retrieve("Xn handover SN Status Transfer")
    assert results
    assert len({r.doc.doc_id for r in results}) == len(results)
    assert retrieve("") == []
    assert retrieve("handover", top_k=0) == []


def test_short_query_weights_lexical_recall_more():
    a, b = Document("a", "handover", "golden_case"), Document("b", "Xn", "3gpp")
    bm25, vector = [RetrievedDoc(a, 1, "bm25")], [RetrievedDoc(b, 1, "vector")]
    short = rrf_fuse(bm25, vector, "Xn handover")
    long = rrf_fuse(bm25, vector, "verify the full Xn handover procedure now")
    assert short["a"] > short["b"] and long["a"] < long["b"]
