from typing import Annotated, Optional

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（LangGraph 自动追加）
    current_step: str                        # 当前执行阶段
    tool_outputs: dict                       # 工具返回缓存，供 result_judge 使用
    error_count: int                         # 连续工具错误计数（软熔断）
    confidence_score: float                  # 当前决策置信度
    hitl_required: bool                      # 是否需要人工介入
    hitl_feedback: str                       # 人工审核后的反馈
    final_result: Optional[str]              # 最终判定结果（JSON 字符串）
