"""DocumentParser：将各种格式文档转为统一 Markdown。

Demo 中仅实现 Markdown 透传；生产环境可扩展 PDF/Word/HTML 处理。
"""

from __future__ import annotations

from core.abstractions import DocumentParser


class MarkdownDocumentParser(DocumentParser):
    """Markdown 格式直接透传，其他格式做基础转换。"""

    def parse(self, raw_content: str, file_type: str = "markdown") -> str:
        if file_type in ("markdown", "md"):
            return raw_content

        if file_type == "html":
            return self._html_to_markdown(raw_content)

        if file_type == "txt":
            return raw_content

        # 默认当作 Markdown
        return raw_content

    def _html_to_markdown(self, html: str) -> str:
        """简易 HTML 转 Markdown（Demo 级别）。"""
        import re

        text = re.sub(
            r"<h([1-6])[^>]*>(.*?)</h\1>",
            lambda m: "#" * int(m.group(1)) + " " + m.group(2),
            html,
        )
        text = re.sub(r"<p[^>]*>(.*?)</p>", r"\1\n", text)
        text = re.sub(r"<br\s*/?>", "\n", text)
        text = re.sub(r"<[^>]+>", "", text)
        return text.strip()
