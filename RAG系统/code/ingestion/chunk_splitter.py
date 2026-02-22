"""层次化切分器：基于 Markdown 标题结构的三类节点切分。

- 非叶子节点（有子标题）：>2K tokens -> 摘要; <=2K -> 直接 chunk
- 叶子节点（无子标题）：512-800 tokens，10-15% 重叠
- 无标题节点：继承最近有效标题
"""

from __future__ import annotations

import re
import uuid

from config.settings import Settings
from core.abstractions import ChunkSplitter
from models.schemas import ChunkMetadata, DocumentChunk


class HeadingNode:
    """Markdown 标题树中的节点。"""

    def __init__(self, level: int, title: str, content: str = ""):
        self.level = level
        self.title = title
        self.content = content  # 该标题下、子标题前的正文
        self.children: list[HeadingNode] = []


class HierarchicalChunkSplitter(ChunkSplitter):
    def __init__(self, settings: Settings):
        self.leaf_min = settings.chunk_leaf_min_tokens
        self.leaf_max = settings.chunk_leaf_max_tokens
        self.overlap_ratio = settings.chunk_overlap_ratio
        self.nonleaf_threshold = settings.chunk_nonleaf_threshold

    def split(
        self, markdown_text: str, doc_id: str, doc_name: str
    ) -> list[DocumentChunk]:
        tree = self._parse_heading_tree(markdown_text)
        chunks: list[DocumentChunk] = []
        self._recursive_split(tree, doc_id, doc_name, heading_path="", chunks=chunks)
        return chunks

    # ---- 解析 Markdown 为标题树 ----

    def _parse_heading_tree(self, text: str) -> HeadingNode:
        """将 Markdown 解析为嵌套的标题-内容树。根节点 level=0。"""
        root = HeadingNode(level=0, title="ROOT")
        stack: list[HeadingNode] = [root]

        heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        last_end = 0

        for match in heading_re.finditer(text):
            level = len(match.group(1))
            title = match.group(2).strip()

            # 上一个节点到当前标题之间的正文归属到栈顶节点
            between_text = text[last_end : match.start()].strip()
            if between_text:
                stack[-1].content += (
                    ("\n" + between_text) if stack[-1].content else between_text
                )

            last_end = match.end()

            new_node = HeadingNode(level=level, title=title)

            # 找到合适的父节点（level 更小的）
            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()

            stack[-1].children.append(new_node)
            stack.append(new_node)

        # 尾部剩余正文
        remaining = text[last_end:].strip()
        if remaining:
            stack[-1].content += ("\n" + remaining) if stack[-1].content else remaining

        return root

    # ---- 递归切分 ----

    def _recursive_split(
        self,
        node: HeadingNode,
        doc_id: str,
        doc_name: str,
        heading_path: str,
        chunks: list[DocumentChunk],
    ) -> None:
        current_path = f"{heading_path}/{node.title}" if heading_path else node.title

        if node.children:
            # 非叶子节点
            full_text = self._collect_text(node)
            token_count = self._estimate_tokens(full_text)

            if token_count <= self.nonleaf_threshold and full_text.strip():
                # <= 2K tokens：整段作为 chunk
                chunk = self._make_chunk(
                    text=full_text,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    heading_path=current_path,
                    node_type="non_leaf",
                )
                chunks.append(chunk)
            elif full_text.strip():
                # > 2K tokens：生成摘要作为替代 chunk
                summary = self._generate_summary(full_text, current_path)
                chunk = self._make_chunk(
                    text=summary,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    heading_path=current_path,
                    node_type="non_leaf",
                    parent_summary=summary,
                )
                chunks.append(chunk)

            # 继续递归子节点
            for child in node.children:
                self._recursive_split(child, doc_id, doc_name, current_path, chunks)
        else:
            # 叶子节点 / 无标题节点
            text = node.content.strip()
            if not text:
                return

            node_type = "leaf" if node.level > 0 else "no_heading"
            token_count = self._estimate_tokens(text)

            if token_count <= self.leaf_max:
                # 短文本直接作为 chunk
                chunk = self._make_chunk(
                    text=f"{current_path}\n\n{text}" if node.level > 0 else text,
                    doc_id=doc_id,
                    doc_name=doc_name,
                    heading_path=current_path,
                    node_type=node_type,
                )
                chunks.append(chunk)
            else:
                # 固定步长切割 + 重叠
                sub_chunks = self._split_leaf_content(text)
                for i, sub_text in enumerate(sub_chunks):
                    chunk = self._make_chunk(
                        text=f"{current_path}\n\n{sub_text}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        heading_path=current_path,
                        node_type=node_type,
                        is_continuation=(i > 0),
                    )
                    chunks.append(chunk)

    # ---- 辅助方法 ----

    def _collect_text(self, node: HeadingNode) -> str:
        """收集节点及其所有子节点的文本。"""
        parts = []
        if node.level > 0:
            parts.append("#" * node.level + " " + node.title)
        if node.content:
            parts.append(node.content)
        for child in node.children:
            parts.append(self._collect_text(child))
        return "\n\n".join(parts)

    def _split_leaf_content(self, text: str) -> list[str]:
        """叶子节点固定步长切割，10-15% 重叠。"""
        target_tokens = (self.leaf_min + self.leaf_max) // 2
        overlap_tokens = int(target_tokens * self.overlap_ratio)

        # 按句子/段落边界切割
        sentences = re.split(r"(?<=[。！？\n])", text)
        sentences = [s for s in sentences if s.strip()]

        chunks = []
        current: list[str] = []
        current_tokens = 0

        for sent in sentences:
            sent_tokens = self._estimate_tokens(sent)
            if current_tokens + sent_tokens > self.leaf_max and current:
                chunks.append("".join(current))
                # 重叠：保留尾部的一些句子
                overlap_acc = 0
                overlap_start = len(current)
                for j in range(len(current) - 1, -1, -1):
                    overlap_acc += self._estimate_tokens(current[j])
                    if overlap_acc >= overlap_tokens:
                        overlap_start = j
                        break
                current = current[overlap_start:]
                current_tokens = sum(self._estimate_tokens(s) for s in current)

            current.append(sent)
            current_tokens += sent_tokens

        if current:
            chunks.append("".join(current))

        return chunks

    def _generate_summary(self, text: str, heading_path: str) -> str:
        """Mock LLM 摘要（生产环境调用 Qwen3-4B）。"""
        lines = text.split("\n")
        # 取前几行作为摘要
        summary_lines = [line for line in lines[:10] if line.strip()]
        summary = " ".join(summary_lines)
        if len(summary) > 500:
            summary = summary[:500] + "..."
        return f"[摘要] {heading_path}: {summary}"

    def _estimate_tokens(self, text: str) -> int:
        """粗略估算 token 数。"""
        cn_chars = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        words = len(text.split())
        return int(cn_chars * 1.5 + words * 0.75) + 1

    def _make_chunk(
        self,
        text: str,
        doc_id: str,
        doc_name: str,
        heading_path: str,
        node_type: str,
        is_continuation: bool = False,
        parent_summary: str | None = None,
    ) -> DocumentChunk:
        chunk_id = f"{doc_id}_chunk_{uuid.uuid4().hex[:8]}"
        return DocumentChunk(
            chunk_id=chunk_id,
            text=text,
            metadata=ChunkMetadata(
                chunk_id=chunk_id,
                doc_id=doc_id,
                doc_name=doc_name,
                heading_path=heading_path,
                node_type=node_type,
                is_continuation=is_continuation,
                parent_summary=parent_summary,
            ),
        )
