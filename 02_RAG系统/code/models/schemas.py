from __future__ import annotations

from pydantic import BaseModel, Field


# ---- 请求 / 响应 ----


class ChatRequest(BaseModel):
    """在线问答请求，Pydantic 强校验防止越权与非法格式注入。"""

    user_id: str = Field(..., max_length=64)
    session_id: str = Field(..., max_length=64)
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)

    model_config = {"extra": "forbid"}


class ChatResponse(BaseModel):
    answer: str
    citations: list[str] = []
    rewritten_queries: list[str] = []
    source: str = "rag"  # "exact_cache" | "semantic_cache" | "rag"


class IngestRequest(BaseModel):
    doc_id: str
    doc_name: str
    content: str  # Markdown 格式
    source_type: str = "general"


class IngestResponse(BaseModel):
    status: str = "success"
    chunks_created: int = 0


# ---- 内部数据模型 ----


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    heading_path: str = ""  # "# 5G / ## 随机接入 / ### PRACH"
    node_type: str = "leaf"  # "non_leaf" | "leaf" | "no_heading"
    is_continuation: bool = False
    parent_summary: str | None = None


class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata
    vector_384: list[float] | None = None
    vector_768: list[float] | None = None
    bm25_tokens: list[str] | None = None


class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    score: float
    source: str  # "bm25" | "vector_384" | "rsf" | "rerank"


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant" | "system"
    content: str
