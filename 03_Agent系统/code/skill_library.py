"""文件式 Skill 的最小加载器：目录元数据、按名读正文、版本指纹。

只负责把排查方法提供给模型，不执行Skill内的命令，也不增加设备权限。
为了让学习者看清格式，本例只支持简单单行 name/description 元数据。
"""

import hashlib
from pathlib import Path


class SkillLibrary:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).parent / "skills").resolve()

    def _read(self, path):
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Skill路径超出当前库")
        text = path.read_text(encoding="utf-8")
        parts = text.split("---", 2)
        if len(parts) != 3 or parts[0].strip():
            raise ValueError("Skill需要YAML风格的前置元数据")
        metadata = {}
        for line in parts[1].splitlines():
            key, separator, value = line.partition(":")
            if separator:
                metadata[key.strip()] = value.strip().strip("\"'")
        if not metadata.get("name") or not metadata.get("description"):
            raise ValueError("Skill缺少name或description")
        return metadata, parts[2].strip(), hashlib.sha256(text.encode()).hexdigest()

    def catalog(self):
        """只向上下文提供目录，不把所有正文同时塞进模型输入。"""
        return [self._read(path)[0] for path in sorted(self.root.glob("*/SKILL.md"))]

    def load(self, name):
        """只允许读取当前库内的文件，路径字符串本身不是授权。"""
        path = (self.root / name / "SKILL.md").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Skill路径超出当前库")
        metadata, instructions, digest = self._read(path)
        return {**metadata, "instructions": instructions, "content_hash": digest}
