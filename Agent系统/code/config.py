"""
config.py — 全局配置与环境变量

LangSmith 追踪说明
──────────────────
LangChain/LangGraph 通过读取以下三个环境变量实现自动追踪，无需修改业务代码：

  LANGCHAIN_TRACING_V2=true      启用 LangSmith 链路追踪
  LANGCHAIN_API_KEY=<key>        LangSmith API 密钥（对应 .env 中的 LANGSMITH_API_KEY）
  LANGCHAIN_PROJECT=<project>    追踪数据归属的 LangSmith 项目名

注意：这三个变量必须在任何 LangChain/LangGraph 模块 import 之前写入 os.environ，
      因此 config.py 必须是整个项目最先被 import 的模块。

追踪内容：每一次 LLM 调用、每一个节点的输入/输出、工具调用的参数与结果，
          都会自动上传至 LangSmith，可在 Web 界面中以 Trace 视图查看完整链路。
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── DeepSeek LLM ─────────────────────────────────────────────────────────────
# DeepSeek 兼容 OpenAI API 格式，只需替换 base_url 即可使用 langchain-openai
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
DEEPSEEK_MODEL: str = "deepseek-chat"

# ── LangSmith 追踪 ───────────────────────────────────────────────────────────
# setdefault：若环境变量已由 .env 设置则不覆盖，否则使用默认值
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "true"))
os.environ.setdefault("LANGCHAIN_API_KEY", os.getenv("LANGSMITH_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT", os.getenv("LANGSMITH_PROJECT", "5g-test-agent"))

# ── PostgreSQL Checkpoint ────────────────────────────────────────────────────
# LangGraph 用 Checkpointer 将每一步的 AgentState 序列化并持久化。
# PostgreSQL 版本支持跨进程恢复，是 HITL 暂停/恢复机制的存储后端。
POSTGRES_URI: str = os.getenv(
    "POSTGRES_URI",
    "postgresql://postgres:password@localhost:5432/agent_db",
)

# ── Agent 熔断阈值 ───────────────────────────────────────────────────────────
CONFIDENCE_HIGH: float = 0.85  # 自动结论，无需人工审核
CONFIDENCE_LOW: float = 0.65   # 低于此值触发 HITL（软熔断）
MAX_ERRORS: int = 2            # 连续工具错误次数上限（软熔断）
RECURSION_LIMIT: int = 15      # 图最大迭代轮次（硬熔断，由 LangGraph 强制）
