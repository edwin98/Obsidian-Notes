"""只加载 Agent 目录，不依赖仓库里其他同名 graph/config/tools 模块。"""

import os
import sys
from pathlib import Path

import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def config():
    return {"configurable": {"thread_id": "test-thread"}, "recursion_limit": 40}
