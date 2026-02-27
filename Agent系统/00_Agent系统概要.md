---
tags:
  - Agent
  - LLM
  - AI架构
status: draft
---
# Agent系统概要

## 一、什么是AI Agent (智能体)
**定义**: AI Agent（人工智能体）是指一种能够感知输入、并在特定环境中自主采取行动以实现预期目标的系统。在目前大语言模型（LLM）的背景下，Agent 演变为**以大语言模型为大脑，具备规划、记忆、工具调用能力的智能协同系统**。

根据 Andrej Karpathy 等专家的总结，现代 LLM Agent 的核心架构可以总结为公式：
**Agent = LLM (大脑) + Planning (规划) + Memory (记忆) + Tools (工具)**

### 核心组成部分：
1. **大脑 (Brain / LLM)**:
   - 担任系统的控制器。负责自然语言理解、推理、生成、以及决策。
2. **规划 (Planning)**:
   - **子目标分解 (Subgoal and decomposition)**: 将复杂的大任务拆解为更小、可管理的子任务（例如通过 Chain of Thought）。
   - **反思与纠错 (Reflection and refinement)**: Agent 能够对过去的行为进行自我批评，从错误中学习并修正未来的子步骤（例如 ReAct, Reflexion 机制）。
3. **记忆 (Memory)**:
   - **短期记忆 (Short-term memory)**: 即当前对话的上下文，受限于 LLM 的 Context Window（上下文窗口大小）。
   - **长期记忆 (Long-term memory)**: 将信息持久化存储，可以在较长时间后进行回忆。通常借助于外部向量数据库（Vector DB）和 RAG 技术实现快速检索。
4. **工具使用 (Tool Use)**:
   - 赋予模型与外部世界交互的能力。通过 API 调用、代码执行等方式，弥补 LLM 自身时效性不足和缺乏专业领域系统的缺陷。
   - 例如：计算器、网络搜索（SerpApi）、代码解释器、执行 SQL 查询。

---

## 二、核心机制与设计模式

1. **ReAct (Reason + Act)**:
   - 经典的 Agent 范式，交替进行推理（思考要做什么、为什么这么做）和行动（执行某个工具）。
   - 流程通常为：`Thought -> Action -> Observation -> Thought ...` 直到得出最终答案。
2. **CoT (Chain of Thought / 链式思考)**:
   - 引导模型“一步一步地思考”，显著提升复杂逻辑或数学问题的解决能力。
3. **ToT (Tree of Thoughts / 思维树)**:
   - CoT 的扩展，在每一步探索多种可能的选择，并在探索过程中评估选择的有效性，可以结合搜索算法（如 BFS, DFS）进行回溯。
4. **Function Calling (函数调用)**:
   - 许多模型（如 OpenAI API）原生支持的机制，允许我们向模型提供一组工具（函数的名称、描述和参数Schema），模型判断是否需要调用某个函数，并返回符合 JSON 格式的参数，是目前最稳定、标准的工具调用实现方案。

---

## 三、主流架构分类

1. **单智能体 (Single-Agent)**:
   - 系统中只有一个主要的大模型角色负责整体流程控制。适用于流程相对固定、任务类型单一的场景（如个人助理、单一功能的插件）。
2. **多智能体协同 (Multi-Agent System, MAS)**:
   - 系统中包含多个扮演不同**角色**的 Agent。它们互相合作、竞争或进行多轮对话，共同解决复杂问题。
   - 例如：赋予一个 Agent "程序员" 角色，赋予另一个 Agent "代码审查员" 角色。多 Agent 能够有效缓解单模型出现幻觉、注意力分散的问题。

---

## 四、常见挑战与工程优化手段

| 挑战 | 表现 | 解决方案 |
| :--- | :--- | :--- |
| **无限循环与死锁** | Agent 在思考或工具调用中陷入死循环 (`Action A -> Obs A -> Action A ...`)。 | 设定 `max_iterations` 或超时阈值；加入人工介入机制（Human-in-the-loop）；使用早停（Early Stopping）。 |
| **幻觉与工具误用** | 强行编造工具参数、使用了不存在的工具、不看说明书乱调。 | 强化 Prompt 工具描述（少即是多）；使用支持 Function Calling 的模型；设计健壮的工具内部错误捕捉机制，将 Error 作为 Observation 返回给 Agent。 |
| **记忆过载与迷失** | 对话轮次过多，导致超出上下文窗口，或“中间迷失”（Lost in the Middle）。 | 引入滑动窗口截断机制；对历史信息定期进行 Summary 摘要；将长期记忆分片灌入向量库（RAG 结合 Agent）。 |
| **高延迟与高成本** | Agent 为获得答案可能需要多轮（几到十几次）请求 LLM，且往往带有极长的 Prompt 预设。 | 使用并行调用；针对单一节点微调较小的模型以替代 GPT-4 等超大杯模型（知识蒸馏）；尽量流式推送用户界面中间状态以提升体验感。 |

---

## 五、需要进阶的知识体系

- [[LangChain]]：掌握如何将繁复的想法工程化落地，尤其熟悉其 AgentExecutor 机制和现成工具抽象。
- [[LangGraph]]：解决复杂控制流的关键，学习将 Agent 工作流视为一个可以控制的状态机，理解 Node、Edge 的运转逻辑。
- **Prompt Engineering 进阶**：从手写 Prompt 到框架式管理提示词。
- **Agentic Workflow（智能体工作流）**：Andrew Ng 提出的理念，强调不要迷信单一指令的零样本回答，而应建立 Reflection（反思）、Tool Use（工具）、Planning（规划）、Multi-Agent Collaboration（多智能体协作）四大设计模式的工作流。除了完全自主探索的 Agent，**工作流（Workflow）思维在当前落地应用中更具实效性。**
