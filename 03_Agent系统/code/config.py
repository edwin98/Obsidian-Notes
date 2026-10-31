"""模型配置与可选追踪；导入配置不会调用模型。"""

import os

from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
os.environ.setdefault(
    "LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "5g-test-agent")
)
