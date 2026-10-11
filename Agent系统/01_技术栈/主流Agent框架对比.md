---
tags:
  - Agent
  - LLM框架
  - 核心生态
status: draft
---

# 主流 Agent 框架横向对比

框架选型需要分别看控制流、状态恢复、数据接入、工具执行、可观测性和维护状态。以下按截至 2026-10-11 的官方能力整理；历史项目采用的版本和方案仍保留当时背景。

## 1. LangChain 与 LangGraph

LangChain 1.x 的 `create_agent` 提供常见工具循环和 Middleware，底层依赖 LangGraph。LangGraph 的 StateGraph 适合明确编排节点、状态和控制流，支持 checkpoint 与 interrupt。二者可以组合，不能只按“简单链”和“复杂 Agent”机械划分。

详见 [[Agent系统/01_技术栈/LangChain]]、[[Agent系统/01_技术栈/LangGraph]]。[LangChain 迁移指南](https://docs.langchain.com/oss/python/migrate/langchain-v1)、[LangGraph 迁移指南](https://docs.langchain.com/oss/python/migrate/langgraph-v1)

## 2. LlamaIndex：数据接入、检索与 Workflows

LlamaIndex 的优势包括数据接入、索引和检索组织，例如层级检索、混合检索与图检索。它也有事件驱动 Workflows 和数据 Agent 路线，能力不限于 Retriever 封装。

Workflows 用事件组织执行步骤，适用于围绕数据处理和检索构建流程。复杂业务仍需根据状态管理、执行控制和运维需求评估。[官方 Workflows 文档](https://developers.llamaindex.ai/python/llamaagents/workflows/)

## 3. MetaGPT：SOP 与 Data Interpreter

MetaGPT 以角色分工和 SOP 组织协作，软件开发流程是经典应用。Data Interpreter 与 RAG 集成还覆盖数据分析和工具执行任务；这些是较早已有的能力，不能统称为 2026 年新增。

SOP 有助于结构化协作，但不保证任意任务都能稳定执行，仍需处理工具、环境与结果校验。[官方发布记录](https://github.com/FoundationAgents/MetaGPT/releases)

## 4. AutoGen 与 Microsoft Agent Framework

AutoGen 的早期 0.2 接口常使用 `initiate_chat` 和 GroupChat。后续 AgentChat 路线采用异步执行，并支持状态保存/加载，因此“不支持持久状态”不是跨版本结论。[AgentChat 迁移指南](https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/migration-guide.html)

截至本次核查，AutoGen 官方已进入维护模式，推荐新项目使用 Microsoft Agent Framework，并说明其已发布 1.0。维护状态与迁移成本是选型因素，旧项目无需仅因路线变化而重做。[AutoGen 官方说明](https://github.com/microsoft/autogen)

## 5. CrewAI：Crews 与 Flows

Crews 用角色、任务和协作关系组织 Agent；Flows 用事件、状态和路由组织流程，并提供持久化能力。两者可以组合，不能仅把 CrewAI 当作角色聊天或演示工具。

选型时关注流程可控性、恢复、工具执行和实际观测能力，而不是只比较初始演示所需代码量。[官方 Flows 文档](https://docs.crewai.com/en/concepts/flows)

## 6. Dify：可视化应用与工作流

Dify 提供模型接入、知识库和可视化工作流。当前发布记录还包括 MCP 协议协商、结构化工具输出和将 Workflow 暴露为 MCP，因而不局限于知识问答页面。

平台可降低集成工作，但权限、部署、版本与复杂控制流需求仍需评估。[官方发布记录](https://github.com/langgenius/dify/releases)

## 7. FastGPT：知识问答、工具与执行环境

FastGPT 从知识问答扩展到工作流、MCP 和工具调用，4.16 路线包含 Agent Sandbox、工具 Schema 及应用用户共享实例等变化。执行环境、数据模型和运行状态随版本演进，应按实际版本理解。[官方发布记录](https://github.com/labring/FastGPT/releases)

## 8. 按需求比较

| 主要需求 | 优先考察的路线 | 需要进一步确认 |
| --- | --- | --- |
| 常见工具循环与中间件 | LangChain Agent | 模型适配、状态与工具恢复 |
| 明确的业务状态机 | LangGraph StateGraph | 节点边界、checkpoint、人工介入 |
| 数据接入与复杂检索 | LlamaIndex | 检索质量、数据处理与执行成本 |
| 角色协作或 SOP | CrewAI、MetaGPT | 流程控制与工具结果校验 |
| 微软 Agent 生态 | Microsoft Agent Framework；存量 AutoGen | 维护状态、版本兼容与迁移成本 |
| 可视化应用交付 | Dify、FastGPT | 部署、权限、可扩展性与执行环境 |

同一需求可以有多种实现。应根据任务与团队条件做比较，不能把某框架的定位当成能力上限。
