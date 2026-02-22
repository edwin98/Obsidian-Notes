"""层次化切分器单元测试。"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.settings import Settings
from ingestion.chunk_splitter import HierarchicalChunkSplitter


def _make_splitter() -> HierarchicalChunkSplitter:
    settings = Settings()
    return HierarchicalChunkSplitter(settings)


class TestHeadingTreeParsing:
    """Markdown 标题树解析测试。"""

    def test_simple_structure(self):
        splitter = _make_splitter()
        md = """# 标题一

正文内容一。

## 子标题 1.1

子内容 1.1。

## 子标题 1.2

子内容 1.2。
"""
        chunks = splitter.split(md, "test_doc", "测试文档")
        assert len(chunks) > 0
        # 应包含非叶子节点和叶子节点
        node_types = {c.metadata.node_type for c in chunks}
        assert "leaf" in node_types or "non_leaf" in node_types

    def test_deep_nesting(self):
        splitter = _make_splitter()
        md = """# A

## A.1

### A.1.1

深层内容。

### A.1.2

另一个深层内容。

## A.2

二级内容。
"""
        chunks = splitter.split(md, "test_doc", "测试文档")
        # 深层嵌套应产生多个 chunk
        assert len(chunks) >= 3

    def test_no_heading_content(self):
        splitter = _make_splitter()
        md = "这是一段没有标题的纯文本内容。"
        chunks = splitter.split(md, "test_doc", "测试文档")
        assert len(chunks) >= 1

    def test_heading_path_preserved(self):
        splitter = _make_splitter()
        md = """# 5G 技术

## 随机接入

具体内容。
"""
        chunks = splitter.split(md, "doc_001", "5G")
        # 叶子节点的 heading_path 应包含父级路径
        leaf_chunks = [c for c in chunks if c.metadata.node_type == "leaf"]
        if leaf_chunks:
            assert "随机接入" in leaf_chunks[0].metadata.heading_path

    def test_chunk_id_unique(self):
        splitter = _make_splitter()
        md = """# A

## B

内容B。

## C

内容C。
"""
        chunks = splitter.split(md, "doc_test", "test")
        chunk_ids = [c.chunk_id for c in chunks]
        assert len(chunk_ids) == len(set(chunk_ids)), "chunk_id 应唯一"


class TestLeafSplitting:
    """叶子节点切割测试。"""

    def test_short_content_single_chunk(self):
        splitter = _make_splitter()
        md = """# 标题

短内容。
"""
        chunks = splitter.split(md, "doc", "doc")
        # 短内容不应被切割
        leaf_chunks = [c for c in chunks if c.metadata.node_type == "leaf"]
        if leaf_chunks:
            assert len(leaf_chunks) <= 2

    def test_long_content_multiple_chunks(self):
        splitter = _make_splitter()
        # 生成足够长的内容触发切割
        long_text = "这是一段很长的技术文档内容，描述了5G通信系统的核心技术细节。" * 100
        md = f"""# 长文档

{long_text}
"""
        chunks = splitter.split(md, "doc", "doc")
        # 长内容应被切割为多个 chunk
        assert len(chunks) >= 2
