"""【学习定位：旧图的工作状态】
messages使用消息reducer，其余字段描述单次图进度。新版完整业务状态使用task_models中的Task/Experiment/Run。
对应05、14；初始化必须创建新容器，不能跨任务共享可变对象。
"""

from typing import Annotated

from langchain_core.messages import HumanMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（按消息ID合并）
    current_step: str  # 当前执行阶段
    tool_outputs: dict  # 工具返回缓存，供 result_judge 使用
    error_count: int  # 连续工具错误计数（软熔断）
    confidence_score: float  # 当前决策置信度
    hitl_required: bool  # 是否需要人工介入
    hitl_feedback: str  # 人工审核后的反馈
    final_result: str | None  # 最终判定结果（JSON 字符串）
    needs_clarification: bool  # 是否需要向用户追问意图
    clarification_question: str  # 待提给用户的澄清问题
    hitl_rejected: bool  # 人工审核明确拒绝执行
    tool_history: list[dict]  # 按调用记录，保留重复工具的证据与错误
    review_history: list[dict]  # 审批历史不会随着 hitl_required 清零而消失


def initial_state(query: str) -> AgentState:
    """每次创建新容器，避免不同任务共享可变字典或消息列表。"""
    return {
        "messages": [HumanMessage(content=query)],
        "current_step": "start",
        "tool_outputs": {},
        "tool_history": [],
        "review_history": [],
        "error_count": 0,
        "confidence_score": 1.0,
        "hitl_required": False,
        "hitl_feedback": "",
        "hitl_rejected": False,
        "final_result": None,
        "needs_clarification": False,
        "clarification_question": "",
    }
