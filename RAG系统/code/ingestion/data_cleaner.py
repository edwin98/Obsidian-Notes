"""数据清洗：正则清洗池 + Unicode NFKC 标准化。"""

from __future__ import annotations

import re
import unicodedata


class DataCleaner:
    """工业级文本清洗，防止脏数据污染分词器和向量化效果。"""

    def clean(self, text: str) -> str:
        # Unicode NFKC 强制标准化
        text = unicodedata.normalize("NFKC", text)
        # 移除不可见十六进制控制字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # 统一换行符
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # 压缩连续空行为双换行
        text = re.sub(r"\n{3,}", "\n\n", text)
        # 压缩连续空格/制表符
        text = re.sub(r"[ \t]{2,}", " ", text)
        # 去除行首行尾多余空白（保留 Markdown 缩进中有意义的空格）
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        text = "\n".join(lines)
        return text.strip()
