---
tags:
  - Agent
  - LLM底层特性
  - 工具调用
status: draft
---
# Function Calling (函数调用) 与工具箱

在构建现代化 AI Agent 系统时，我们常说“给大模型装上双手”，这主要依赖于模型的 **Function Calling（函数调用）/ Tool Use（工具使用）** 特性。理解它的底层原理是开发稳定 Agent 的基石。

## 1. 为什么需要 Function Calling？

早期的 Agent 比如 AutoGPT，主要是靠提示词（Prompt Engineering）强制要求大模型返回特定格式（如 JSON），并通过 Python 正则表达式去提取字段。然而这种方式极不可靠：
- 回答经常会带着解释性的文字，例如：“好的，我找到了结果，这是你的JSON：`{...}`”，直接使用 `json.loads` 会抛出解析错误。
- 必需参数可能遗漏，类型也可能不符合要求，例如应为整数却返回了字符串。
- **因此，OpenAI 等厂商在模型微调层面原生地支持了对工具 Schema 规范的理解与响应，即 Function Calling。**

它的目的**不是让大模型在云端帮我们运行这段代码**，而是让大模型稳定、精确地返回我们需要执行哪个工具以及运行这个工具的参数。

模型生成调用请求，应用执行工具，模型再利用返回结果。严格参数 Schema 控制结构，业务校验控制参数关系与权限；模型返回“调用”不等于动作已经实际完成。[结构化输出文档](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)

## 2. 运作流程（四步走）

一个完整的 Function Calling/Tool Use 流程通常由以下步骤组成：

1. **环境预设与注册工具 (Agent ➔ LLM)**: 
   - 开发者定义一组业务函数的详细描述、用途以及输入参数表（符合 JSON Schema 规范）。例如：`get_stock_price(symbol: str)`。
   - 随附着用户的提问（“苹果公司的股价是多少？”），将函数 Schema 列表发给 LLM。
2. **LLM 决定“暂停并请求工具” (LLM ➔ Agent)**: 
   - LLM 分析出自己不能直接回答该问题，必须调用外部服务。
   - LLM 返回特殊的响应模式（例如 `tool_calls` 标志位），它包含**工具名称** (`get_stock_price`) 及其生成的**结构化参数** (`{"symbol": "AAPL"}`).
3. **本地代码执行 (Agent 环境内)**: 
   - 你的工程代码拦截到模型要求调用的工具名称，在本地 Python 执行环境（或后端微服务中）去请求真实的股票接口，获取实时的价格数据：`$150.2`。
4. **将工具结果返回给模型（Agent ➔ LLM）**：
   - 将上面得到的真实数据打包成 `ToolMessage` 形式，附随在历史对话（Messages）之后，再次请求大模型。
   - 大模型终于理解了实时数据，最终产出自然语言结果呈递给最终用户（“苹果公司当前的股价为150.2美元。”）。

### Responses API 的工具循环

OpenAI 推荐新项目使用 Responses，Chat Completions 仍受支持。Responses 用不同 Items 表达消息、reasoning、function call 和工具结果，结果通过 `call_id` 与调用对应；多轮状态可使用 `previous_response_id`、手工维护 Items 或 Conversations。不能把所有输出都当成普通文本消息。[官方迁移指南](https://developers.openai.com/api/docs/guides/migrate-to-responses)

某些推理模型的工具循环还需保留接口返回的 reasoning Items。推理强度与回答长度不是同一个概念，也不要求应用伪造推理内容。[推理接口说明](https://developers.openai.com/api/docs/guides/reasoning)

上面的 ToolMessage/messages 流程描述原项目使用的消息式接口；Responses 的 Items、call_id 和状态入口按对应协议处理。

## 3. 工具箱 (Tools) 常见分类

现在的开源生态极大地丰富了我们可以调用的第三方工具包。常见的可以归为以下几类：

### 查询与搜索类
用于解决大模型的幻觉及知识陈旧问题。
- **搜索服务 (Tavily, SerpApi, Google Search, DuckDuckGo)**: 将大搜结果整合给大模型。例如 Tavily 就是专为了 LLM 和 Agent 而设计的。
- **Wiki/Arxiv 检索**: 提取维基词条内容、学术论文。

### 执行与沙箱类
赋予模型完成计算机任务的能力。但由于存在安全隐患，通常在受限环境如 Docker 容器中执行。
- **Python 代码解释器 (Code Interpreter)**: 使用如 E2B Sandbox 或 Jupyter 内核，运行模型生成的代码做复杂数学分析、画图表（类似 ChatGPT 的高级数据分析功能）。
- **Shell 命令行执行**: 直接让模型有终端读写能力（危险操作，常与 Human-in-the-loop 结合使用）。

### 数据持久层与 API 连接类
为 Agent 赋予读取特定领域或系统知识库、向其他 SaaS 服务写入等能力。
- **数据库查询 (SQLDatabaseToolkit)**: 大模型先转化为 SQL 查询数据库内容。
- **API 接口 (OpenAPI 规范 / RESTful 工具)**: 为现有公司系统增加 AI 能力的常见打法（订票、查订单、创建 Jira 任务）。

### 本地文件系统类
- **文件读写 (Read/Write File)**: 修改或生成配置项、撰写报告输出。

### Function Calling、MCP 与 Skill 的关系

Function Calling 描述模型如何表达调用意图；MCP 是能力发现与交互协议；Skill 则可以提供任务步骤与使用指导。传输、实验性 Tasks 与开放 Skills 格式详见 [[杂七杂八技术]]。[MCP 规范](https://modelcontextprotocol.io/specification/2025-11-25/changelog)

DeepSeek 的 V4 Pro/Flash 路线也支持 Responses 格式及推理强度配置，V4.1 Flash 还更新了原生视觉与别名。不同提供方和本地兼容服务实现的能力不必完全相同；协议格式相近不代表内置工具、严格 Schema 和状态功能全部等价。[V4 API 说明](https://api-docs.deepseek.com/news/news260813/)、[更新记录](https://api-docs.deepseek.com/updates/)

原项目工具循环与代码示例保留为对应时期的实现。

## 4. 最佳工程实践与防坑指南

1. **“少即是多（Less is more）”**:
   - 不要一次性传递几百个函数给 LLM。如果工具过多，模型会发生类似于“选择困难症”进而挑错工具，还浪费大量的 Token 传输 Schema 成本（现在也有模型实现了 Tool 级别的上下文缓存机制）。
   - **优化**: 可以前置挂载一个 Router 或者使用 RAG 技术查询应该传入哪些 Tools 列表。
2. **Schema 描述至关重要 (Descriptions are Prompts)**: 
   - LLM 不看你函数里的核心代码，它**仅仅**根据你对函数的长段描述 (`docstring`) 以及各个参数字段（`symbol: “股票代码或简称可以参考纳斯达克标准”`）的详情来进行挑选。为你的函数及其所有参数编写极为详尽、准确的文档描述是确保 Agent 系统稳定的重中之重。
3. **参数校验与错误拦截 (Error as Observation)**: 
   - 比如模型要求传入数字，但它给了一个字符串。此时你的代码 `try/except` 捕获了 Type Error 或者是请求 API 时抛出了 404 错误。应按错误类型返回可处理的工具结果，并设置重试次数与终止条件。
   - **处理方式**：捕获参数校验异常，将字段名、期望类型和收到的值返回给模型，让它尝试修正参数。修正后仍需重新校验，并限制重试次数；不能保证模型一次就能修复。

熟练使用和封装本地系统为 LLM 的工具 API 面板，是决定一名程序员在此浪潮中能不能把自身系统“AI Native 化”的核心区分度所在。
