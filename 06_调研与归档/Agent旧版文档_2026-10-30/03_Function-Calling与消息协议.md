---
tags:
  - Agent
  - LLM底层特性
  - 工具调用
status: draft
aliases:
  - "Function_Calling与工具箱"
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

Function Calling 描述模型如何表达调用意图；MCP 是能力发现与交互协议；Skill 则可以提供任务步骤与使用指导。传输、实验性 Tasks 与开放 Skills 格式详见 [[01_大模型/00_大模型导航]]。[MCP 规范](https://modelcontextprotocol.io/specification/2025-11-25/changelog)

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

## 消息结构与调用关联

系统基于 LangGraph + OpenAI 消息格式，State 中的 `messages` 是一个有序列表，四种消息类型：

### SystemMessage
```python
{
    "role": "system",
    "content": "你是一个5G无线网络测试助手，负责协调测试用例查询、仿真执行和结果验证..."
}
```
每次 LLM 调用都携带，内容固定（全局角色定义 + 工具使用规范 + 输出格式约束）。

### HumanMessage（用户输入）
```python
{
    "role": "user",
    "content": "帮我验证 n78 频段 X2 切换场景的切换成功率"
}
```

### AIMessage（模型输出，两种形态）
```python
# 形态 1：调用工具（finish_reason = "tool_calls"）
{
    "role": "assistant",
    "content": null,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "Test_Case_Query",
                "arguments": "{\"scenario\": \"切换\", \"feature\": \"X2切换\", \"band\": \"n78\"}"
            }
        }
    ]
}

# 形态 2：直接输出答案（finish_reason = "stop"）
{
    "role": "assistant",
    "content": "n78 频段 X2 切换测试整体判定：PASS（置信度 91%）...",
    "tool_calls": null
}
```

### ToolMessage（工具执行结果）
```python
{
    "role": "tool",
    "tool_call_id": "call_abc123",   # 与 AIMessage 中的 tool_calls[i].id 对应
    "name": "Test_Case_Query",        # 可选，部分实现会带工具名
    "content": "{\"status\": \"success\", \"cases\": [...]}"   # 必须是字符串，JSON 序列化后的结果
}
```

### 关键约束
- `tool_calls` 中的 `arguments` 是 **JSON 字符串**，不是对象，代码层需要 `json.loads()` 解析
- ToolMessage 的 `tool_call_id` 必须与对应 AIMessage 中的 `id` 完全匹配，否则 LLM 无法关联上下文
- AIMessage（含 tool_calls）和 ToolMessage 必须成对追加到历史，缺一不可

LangGraph 中通过 `add_messages` 注解自动维护这条消息链，确保每次 LLM 调用都能看到完整的推理历史。

---

## 一、工具的定义

工具定义告诉 LLM“这个功能存在、能做什么、如何调用、会返回什么”。它通常由名称、自然语言说明和输入输出 Schema 等信息组成。模型根据应用提供的定义与任务上下文提出调用请求，实际执行由工具运行时完成。

下面是消息式 Function Calling 的基本请求示例，展示名称、说明和输入参数三个部分；完整的业务工具契约还应定义输出结构、适用条件、执行影响和错误语义。业务目录中的元数据不应未经转换就加入提供方的请求字段。

```python
{
    "type": "function",
    "function": {
        # 1. 工具名称（模型调用时引用的标识符）
        "name": "Metrics_Collector",
        # 2. 功能描述（模型选工具的重要依据之一）
        #    需要精确描述：什么情况下用这个工具、不用哪个工具
        "description": (
            "通过 SSH/RPC 从网元实时拉取指定会话的 KPI 日志数据。"
            "当测试已通过 Simulation_Runner 启动（拿到 session_id 后），"
            "需要采集实际执行结果时调用此工具。"
            "不要在测试尚未启动时调用此工具。"
        ),
        # 3. 参数 Schema（告诉模型每个参数是什么、类型、是否必填）
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": {
                    "type": "string",
                    "description": "由 Simulation_Runner 返回的测试会话 ID",
                },
                "kpi_names": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "需要采集的 KPI 指标名称列表，"
                        "如 ['throughput_dl', 'ho_success_rate', 'prach_success_rate']"
                    ),
                },
                "duration_sec": {
                    "type": "integer",
                    "description": "采集时长（秒），默认 60",
                    "default": 60,
                },
            },
            "required": ["session_id", "kpi_names"],
        },
    },
}
```

**工具描述的编写原则**：
- 告诉模型"什么时候该用"，同样重要的是"什么时候不该用"（防止误调用）
- 参数描述要给出合法值的例子，不要只写 "字符串类型"
- 工具较多时，可先按能力和适用范围检索，再向模型提供当前任务相关的工具子集；效果取决于工具相似度、描述质量和模型能力，不设置统一的数量分界线

### 输出 Schema 与完整业务契约

`inputSchema` 回答“需要传什么”，`outputSchema` 回答“能获得什么”。对测试 Agent 而言，输出字段可以帮助模型判断工具能否补齐验收证据，以及如何把上游的会话、监测或快照 ID 传给下游工具。

MCP 支持可选的 `outputSchema`，用于约束 `structuredContent`。声明后，服务端必须提供符合 schema 的结构化结果，客户端应校验；这不表示所有模型 API 都接受同名的工具定义字段。接入层需按实际协议转换，并确认模型是否可见输出契约；框架没有传递时，可在工具说明中写出关键输出语义。[MCP 工具规范（2025-11-25）](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

输出 Schema 需说明字段含义、单位、必填与空值、任务状态及证据来源。例如返回 `session_id` 只表示取得任务标识，不能据此判断测试完成；未采到指标应返回空值及原因，不能填成 0。Schema 校验通过也不能证明测量数据真实或满足验收要求。

完整工具目录还应维护设备与版本适用范围、执行前置条件、权限和执行影响、异步与超时、幂等重试、并发资源约束及错误契约。其中权限、互斥和前置条件由执行器强制检查，不能仅靠模型遵守说明。局点测试的完整工具目录和字段设计见 [[03_Agent系统/04_工具设计与完整调用示例]]。

---

## 二、内部原理：模型是怎么学会调用工具的

**Function Calling 是 SFT 训练出来的能力，不是 Prompt 技巧。**

在模型预训练阶段，LLM 只会输出普通文本。Function Calling 能力来自后续的监督微调（SFT）——在训练数据中加入大量"工具调用对话"样本，让模型学会在特定条件下输出结构化的工具调用 token，而不是普通文本。

模型输出工具调用时，实际上是在生成一段特殊格式的 token 序列（各厂商实现略有差异，但结构类似）：

```
# 模型内部生成的原始 token（以 OpenAI 格式为例）
finish_reason: "tool_calls"
tool_calls: [
  {
    "id": "call_abc123",
    "type": "function",
    "function": {
      "name": "Metrics_Collector",
      "arguments": "{\"session_id\": \"sim_20240315_001\", \"kpi_names\": [\"throughput_dl\", \"ho_success_rate\"], \"duration_sec\": 60}"
    }
  }
]
```

注意 `arguments` 是 **JSON 字符串**而不是对象，需要在代码层 `json.loads()` 解析。`finish_reason` 为 `"tool_calls"` 是区分"模型要调用工具"和"模型直接输出答案"的关键标志。

**模型如何决定调用哪个工具**：模型在 Attention 机制中同时看到了对话历史 + 所有工具的 Schema，其决策过程等同于一个分类+生成任务：先判断当前 Thought 需要什么信息，再从注册的工具列表中选择最匹配描述的工具，最后生成符合该工具 Schema 约束的参数 JSON。

---

## 三、工具调用的完整四步流转

```
Step 1：注册工具 + 发送请求
┌────────────────────────────────┐
│ 开发者代码                      │
│  工具 Schema 列表               │
│  + 用户消息（对话历史）          │  ──► LLM API（携带 tools 参数）
└────────────────────────────────┘

Step 2：LLM 决定暂停并请求工具
                                    LLM 返回 ──►  finish_reason = "tool_calls"
                                                  tool_calls[0].function.name = "Metrics_Collector"
                                                  tool_calls[0].function.arguments = "{...}"

Step 3：开发者代码执行工具（本地/微服务）
┌────────────────────────────────┐
│  解析 tool_calls               │
│  路由到对应函数                 │
│  执行真实业务逻辑               │  ──► 返回真实结果
└────────────────────────────────┘

Step 4：将工具结果追加到消息历史，再次调用 LLM
[
  {"role": "user",      "content": "帮我分析这次切换测试的结果"},
  {"role": "assistant", "content": null, "tool_calls": [...]},  ← Step 2 的输出
  {"role": "tool",      "tool_call_id": "call_abc123",          ← Step 3 的结果
                        "content": "{\"throughput_dl\": {\"mean\": 2341.5, ...}}"}
]
          ──► 再次请求 LLM，模型看到 Observation 后生成最终回答
```

**关键细节**：Step 2 的模型输出（tool_calls）和 Step 3 的工具结果，必须都追加到消息历史中一起发给模型，否则模型在 Step 4 不知道自己之前做了什么。这就是 LangGraph 中 `add_messages` 注解的作用——自动维护这条完整的消息链。

---

## 四、工具的返回内容

工具返回内容统一封装为 `ToolMessage`，其 `content` 字段应为结构化 JSON 字符串（不要返回纯文本日志）：

**好的返回格式**：
```json
{
    "status": "success",
    "session_id": "sim_20240315_001",
    "duration_sec": 60,
    "metrics": {
        "throughput_dl": {
            "mean": 2341.5,
            "std": 87.3,
            "p95": 2489.2,
            "unit": "Mbps",
            "samples_count": 60
        },
        "ho_success_rate": {
            "mean": 0.987,
            "total_attempts": 152,
            "total_success": 150
        }
    }
}
```

**差的返回格式**（LLM 难以解析，且不利于置信度计算）：
```
"切换成功率为98.7%，下行吞吐量均值2341.5Mbps，采集了60秒..."
```

返回结构化 JSON 的好处：
1. LLM 解析准确率更高（不需要从自然语言中抽取数字）
2. 后续 `Baseline_Comparator` 等工具可以直接消费这个 JSON
3. 结果判定节点可以直接对 `mean/std` 做统计检验，不依赖 LLM 转述

---

## 项目完整实例

工具定义、用例存储及从模型请求到 Observation 的完整示例见 [[03_Agent系统/04_工具设计与完整调用示例]]。消息式接口与 Responses Items 应分别理解，`tool_call_id` / `call_id` 不得混用。
