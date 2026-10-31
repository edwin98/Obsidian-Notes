"""将当前 Agent 目录作为测试的模块来源，并关闭远程追踪。"""

import os
import sys
from pathlib import Path

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
