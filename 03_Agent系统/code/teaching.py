"""【学习定位：旧图的节点解释表】
把当前节点、输入状态和输出增量转换为页面上的教学提示，不参与业务判定。
对应02；展示说明与真正节点行为分开。

界面的教学说明；实际路由仍调用 graph.py 中的原始函数。
"""

from graph import (
    after_clarify,
    after_guardrail,
    after_hitl,
    after_result_judge,
    should_continue,
)

NODE_GUIDE = {
    "clarify": {
        "title": "意图澄清",
        "description": "检查测试目标是否明确；需要补充时结束本轮并追问。",
        "source": "nodes.py · clarify_node / graph.py · after_clarify",
    },
    "agent": {
        "title": "选择下一步",
        "description": "读取对话和工具观察，提出工具请求，或给出待判定的回答。",
        "source": "nodes.py · agent_node / graph.py · should_continue",
    },
    "guardrail": {
        "title": "安全检查",
        "description": "执行前检查工具名与参数，命中危险关键词则转人工审批。",
        "source": "nodes.py · guardrail_node / graph.py · after_guardrail",
    },
    "tools": {
        "title": "执行工具",
        "description": "校验依赖并执行沙盒工具，将观察写回消息、缓存和历史。",
        "source": "nodes.py · tool_node / tools.py · create_sandbox_tools",
    },
    "result_judge": {
        "title": "双轨判定",
        "description": "合并业务规则与模型归纳；硬规则失败不能被模型覆盖。",
        "source": "nodes.py · result_judge_node / graph.py · after_result_judge",
    },
    "hitl": {
        "title": "人工审批",
        "description": "interrupt 保存断点；批准允许继续，拒绝结束，不会提高置信度。",
        "source": "nodes.py · hitl_node / graph.py · after_hitl",
    },
}


def next_node(node, state):
    routers = {
        "clarify": after_clarify,
        "agent": should_continue,
        "guardrail": after_guardrail,
        "hitl": after_hitl,
        "result_judge": after_result_judge,
    }
    return "agent" if node == "tools" else routers[node](state)


def explain_step(node, state):
    """解释可观察的动作和路由，不声称展示模型内部思维。"""
    following = next_node(node, state)
    if node == "agent":
        calls = getattr(state["messages"][-1], "tool_calls", [])
        if following == "hitl":
            return "连续错误或低置信度触发人工复核，下一步进入 hitl。"
        if calls:
            names = "、".join(call["name"] for call in calls)
            return f"请求 {names}，此时工具尚未执行；下一步先进入 guardrail。"
        return "模型已停止请求工具；回答还需经过 result_judge 的证据与规则检查。"
    if node == "guardrail":
        return (
            "命中危险关键词；工具尚未执行，下一步进入 hitl 等待真实审批。"
            if following == "hitl"
            else "未命中危险关键词；下一步进入 tools 执行刚才的请求。"
        )
    if node == "tools":
        return "工具观察已写入 ToolMessage，并通过 call_id 对应模型请求；下一步交还 agent。"
    if node == "result_judge":
        return (
            "判定置信度不足，保留当前报告并进入 hitl，请人工复核。"
            if following == "hitl"
            else "业务规则与模型归纳已合并，流程结束。"
        )
    if node == "hitl":
        return (
            "审批已记录，下一步执行已审核的工具请求，不再重新规划。"
            if following == "tools"
            else "人工决定已记录；本次流程结束，原有证据和置信度保持可追溯。"
        )
    return (
        "测试目标明确，下一步交给 agent 选择动作。"
        if following == "agent"
        else "目标仍不明确，等待用户补充后继续。"
    )
