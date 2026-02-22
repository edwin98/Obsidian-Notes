"""四大核心抽象基类 —— 自研微型编排骨架。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from models.schemas import DocumentChunk, RetrievedChunk


class DocumentParser(ABC):
    """将各种格式文档转换为统一 Markdown。"""

    @abstractmethod
    def parse(self, raw_content: str, file_type: str) -> str: ...


class ChunkSplitter(ABC):
    """基于标题结构的层次化切分器。"""

    @abstractmethod
    def split(
        self, markdown_text: str, doc_id: str, doc_name: str
    ) -> list[DocumentChunk]: ...


class PipelineRetriever(ABC):
    """三级渐进式混合检索器。"""

    @abstractmethod
    async def retrieve(
        self, query: str, rewritten_queries: list[str], top_k: int = 10
    ) -> list[RetrievedChunk]: ...


class LLMGenerator(ABC):
    """大模型流式生成器。"""

    @abstractmethod
    async def generate_stream(
        self,
        query: str,
        context_chunks: list[DocumentChunk],
        history: list[dict] | None = None,
    ) -> AsyncIterator[str]: ...
