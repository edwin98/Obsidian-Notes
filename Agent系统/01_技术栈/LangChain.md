---
tags:
  - Agent
  - LLM框架
  - LangChain
status: draft
---
# LangChain

## 1. 简介
**LangChain** 是当今最流行、使用者最多的大语言模型（LLM）应用程序开发框架之一，尤其在于它是搭建 AI Agent 的基础架构。它提供了一套标准接口（API）和丰富的工具抽象组件，能够帮助开发者将大型语言模型与其他计算资源或知识源结合，从而打造“感知上下文并且具有逻辑推理能力”的应用。

其核心价值在于它极大地降低了接入各种LLM、向量数据库（Vector DB）、第三方工具（Google Search, Wikipedia）的门槛，并提供了数据处理（Document Loaders, Text Splitters）的标准化管道。

## 2. 核心组件 (Core Modules)

在 LangChain 架构设计中，主要包含以下几大模块（组件）：

- **模型 I/O (Model I/O)**: 提供了统一接口调用来自 OpenAI, Anthropic, HuggingFace 等多个不同供应商的高级或开源 LLM；提供了提示词模板 (Prompt Templates) 方便构建和传递参数；以及输出解析器 (Output Parsers)，对语言模型返回的非结构化字符串抽取为 JSON、Pydantic 对象等。
- **检索增强生成 (Retrieval / RAG)**: 包含 Document loaders (加载器，如从 PDF, Notion 导入数据), Text Splitters (文档拆分工具), Text Embedding models (文本向量化), Vectorstores (向量存储，如 Chroma, FAISS), 以及 Retrievers (检索器抽象接口)。
- **链 (Chains)**: 针对简单的线性逻辑。比如先翻译问题、再回答、再总结。这是早期的用法，目前更推荐 **LCEL (LangChain Expression Language)** 这套语法将其流式串在一起：`chain = prompt | model | parser`。
- **代理 (Agents)**: 赋予 LLM 自主选择权。相比于 Chains 这种“硬编码”执行顺序，Agent 是让语言模型作为大脑，根据用户的任意输入决定**“需要用什么工具、采取哪些步骤”**。包含 Agent 核心控制逻辑、各种 `tools`（工具）定义、以及负责运行整个流程的 `AgentExecutor`。
- **记忆 (Memory)**: 为 Agent 或 Chain 提供保存历史会话记录（State / Context）的能力。如 `ConversationBufferMemory`、`ConversationSummaryMemory`。

## 3. LangChain 在 Agent 开发中的特点与瓶颈

### 3.1 特点
- **生态庞大**: 可供直接调用的第三方集成工具多得数不清，几乎满足所有开发需要（`langchain-community`, `langchain-openai` 等扩展包）。
- **封装度高**: 给初学者提供“开箱即用”的高级封装接口。只需引入 `create_openai_tools_agent` 即可快速拉起一个 Function Calling 的 Agent。
- **开箱即用工具**: 提供如 `TavilySearchResults`, `WikipediaQueryRun` 等多种工具实例对象，简化调用。

### 3.2 瓶颈与陷阱 (为何有时不被青睐？)
- **过度封装 (Magic)**: 内部高度深层的继承结构导致调试变得异常抓狂。一旦报错，堆栈信息非常长，难以看出现象。
- **控制反转严重**: 不透明的预设 Prompt。框架背后替开发者加上了大量的指令，这对于高优定制化场景极为不利。
- **复杂的循环控制**: 在传统 `AgentExecutor` 中，处理复杂的多节点循环、分支任务以及保存特定中间状态（State）是非常困难的，这就是为什么有了 **LangGraph**。
- **版本迭代激进**: 接口改版频繁，废弃功能多。

## 4. 必备代码概念与示例（伪代码层面）

### LCEL (声明式语法)
```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. 模板
prompt = ChatPromptTemplate.from_template("讲一个关于 {topic} 的笑话")
# 2. 模型
model = ChatOpenAI(model="gpt-4o")
# 3. 输出提取
output_parser = StrOutputParser()

# 4. 组成 Pipeline (LCEL语法)
chain = prompt | model | output_parser

# 5. 执行
response = chain.invoke({"topic": "狗"})
print(response)
```

### Agent 基本定义
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool

# 定义工具 
@tool
def get_weather(location: str):
    """获取指定位置的当前天气情况。"""
    return f"{location} 晴天，25度"

tools = [get_weather]

# 创建 Agent 引擎
agent = create_tool_calling_agent(model, tools, prompt)

# 执行器包含 While 循环逻辑和错误处理
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "今天北京天气如何？"})
```

## 5. 学习建议
如果你需要快速接入大模型、做工具箱集成和基础 RAG，**LangChain** 无疑是最强大的瑞士军刀。但也应逐渐摆脱其高级 API，掌握更底层模块调用以应付复杂的企业级定制系统。
