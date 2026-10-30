---
title: "MCP 协议与服务封装"
updated: 2026-10-29
---

# MCP 协议与服务封装

本页维护协议能力、传输与 API 封装。Function Calling 是模型表达调用意图的机制，Skill 提供过程指导，三者的职责分别见 [[03_Agent系统/03_Function-Calling与消息协议]]、[[03_Agent系统/10_Agent-Skills与渐进披露]]。

## 一句话理解

- **API**：软件接口合约，可由 HTTP、gRPC 等承载；是否有状态、是否携带上下文取决于接口设计。
- **MCP**：规范工具、资源和提示词等能力的发现与交互。MCP 客户端发现能力并向模型提供工具描述，模型提出调用请求，由宿主/客户端执行；协议也支持无状态的 HTTP 实现。

## 核心区别对比

| 维度 | 传统 API | MCP |
|---|---|---|
| 调用方 | 通常由代码预先指定接口 | 客户端发现工具，模型选择工具，客户端执行调用 |
| 能力发现 | 依赖文档/OpenAPI Spec（离线） | 运行时 `tools/list` 动态枚举 |
| 协议 | HTTP REST / gRPC 等 | JSON-RPC 2.0；标准传输为 stdio 和 Streamable HTTP，旧 HTTP+SSE 已被替代 |
| 上下文 | 可携带会话、资源等，由接口设计决定 | 规范资源和会话交互，但不自动理解业务上下文 |
| 类型系统 | 可用 OpenAPI、Protobuf 等 | 工具参数通常采用 JSON Schema |
| 错误反馈 | HTTP 状态及业务错误等 | 协议错误与工具执行错误，重试由客户端策略决定 |
| 主要场景 | 服务间调用 | LLM ↔ 外部能力 |

## Agent 如何连接、发现和调用 MCP 工具

这里的 Agent 包含大模型和运行程序，两者职责需要分开理解：

| 角色 | 职责 |
|---|---|
| 大模型 | 理解需求、选择工具、生成参数、分析结果 |
| Agent 运行程序（宿主及 MCP 客户端） | 读取配置、连接服务器、获取工具定义、组装模型上下文、执行调用 |
| MCP Server | 注册工具及其说明，执行对应函数并返回结果 |

### 服务器地址如何配置

地址写在应用配置中，由运行程序读取。以 Codex 的 HTTP MCP 配置为例：

```toml
[mcp_servers.notes]
url = "http://127.0.0.1:8000/mcp"
```

上述内容可放入用户级 `~/.codex/config.toml`；`notes` 是配置名称，`url` 指向已经运行的 MCP 服务。本地 stdio 模式则配置启动命令和参数，由客户端启动服务器进程，通过标准输入、标准输出通信，无需 HTTP 地址。[Codex 配置说明](https://learn.chatgpt.com/docs/extend/mcp?surface=cli)

### 工具清单如何进入模型上下文

1. **发现工具**：客户端向服务器发送 `tools/list`，服务器返回工具清单，包含 `name`、`description`、`inputSchema` 等定义；实现代码保留在服务器端。
2. **提供给模型**：运行程序将可用工具定义与工作指令、对话历史、当前问题等一起提供给模型。工具定义可通过模型 API 的 `tools` 字段传入，不一定拼接成普通聊天文本。
3. **选择工具**：模型根据需求和工具说明，生成工具名与调用参数。
4. **执行调用**：客户端向服务器发送 `tools/call`，携带工具名与参数；服务器找到对应函数并执行。
5. **回传结果**：运行程序把调用记录和结果加入模型上下文，模型生成回答或继续请求工具。

```text
大模型 ←→ Agent 运行程序 ←→ MCP Server
       模型接口             MCP 协议
```

**连接、请求、解析和执行都是程序化完成的；模型负责决定使用什么工具及如何使用结果。** MCP 规定客户端与服务器如何交换信息，具体如何组装模型上下文由 Agent 应用决定。[MCP 架构](https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture)、[工具规范](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)、[Function Calling](https://developers.openai.com/api/docs/guides/function-calling)

### 提前加载与按需加载

工具较少时，程序可以提前提供全部定义；工具较多时，可以先提供目录概述和工具搜索能力，模型提出发现请求后，再由程序加载具体定义。因此，**程序掌握的工具目录可以大于模型当前看到的工具集合**。[按需加载说明](https://developers.openai.com/api/docs/guides/tools-tool-search)

`tools/list` 返回的是能力清单；“先搜索、再读取、最后修改”这样的执行顺序，通常由 Agent 根据任务和中间结果决定，也可以由程序预先编排。

## MCP 的能力与传输版本

2025-11-25 规范增加或完善授权发现、增量权限申请、URL elicitation、sampling 工具调用及实验性 Tasks。Tasks 要明确标注实验性，不能仅凭规范有定义就推断所有客户端/服务器已经支持。[规范变更](https://modelcontextprotocol.io/specification/2025-11-25/changelog)

当前传输包含 stdio 和 Streamable HTTP。旧 HTTP+SSE 使用双端点，属于历史兼容路线；Streamable HTTP 可以用同一 MCP endpoint 处理 POST/GET，并可使用 SSE，因此“新传输不用 SSE”和“MCP 必须两个端点”都不准确。还需区分协议版本协商、授权与 HTTP Origin 校验。[传输规范](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)

使用旧 `sse` 接口的实现应按对应历史版本理解。

## 把一个已有 API 封装成 MCP 的步骤

### 1. 选择 MCP SDK

```bash
# Python
pip install mcp

# TypeScript/Node
npm install @modelcontextprotocol/sdk
```

### 2. 定义 Tool Schema（对应原 API 的参数）

原 API：`GET /weather?city=Beijing&unit=celsius`

MCP Tool 定义：
```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import httpx

app = Server("weather-mcp")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="get_weather",
            description="获取指定城市的实时天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                },
                "required": ["city"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "get_weather":
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                "https://api.weather.com/weather",
                params=arguments
            )
        return [TextContent(type="text", text=resp.text)]
```

### 3. 启动 MCP Server

以下基于 Python SDK 的低层 Server API，须与前面的 app 定义组合运行。[SDK](https://github.com/modelcontextprotocol/python-sdk)、[传输规范](https://modelcontextprotocol.io/specification/2025-11-25/basic/transports)。

```python
import asyncio
from mcp.server.stdio import stdio_server


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
```

```json
// claude_desktop_config.json 注册
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather_mcp.py"]
    }
  }
}
```

### 4. 封装要点总结

- **description 写清楚**：模型会结合描述、参数 Schema、对话和系统指令决定工具调用，描述不是唯一依据
- **inputSchema 用 JSON Schema 严格描述**：模型参数仍需校验；是否严格约束取决于提供方与配置，枚举值用 `enum`，必填用 `required`
- **错误要结构化返回**：不要抛异常，返回带 `isError: true` 的 Content，LLM 可据此重试
- **一个 API ≈ 一个 Tool**：粒度对齐，避免一个 Tool 做太多事

---

## 5G 项目是否引入 MCP 与 Skill

当前文档记录的工具主要面向内部 5G 测试，若无跨团队、跨框架复用需求，可以继续使用本地工具抽象；是否已接入生产以 [[03_Agent系统/01_项目事实与实现状态]] 为准。

### MCP 的引入条件与边界

跨框架共享工具、接入 MCP 客户端或由独立团队维护工具服务时，标准协议具有价值。引入 MCP 本身不必然破坏安全架构；关键是服务端也验证身份、授权、资源范围及参数，避免绕过 Agent 侧 Guardrail 后直接执行高危动作。Agent 侧审批和执行侧校验可同时保留。

### Skill 与复合工具的分工

LangChain Tool 提供可执行函数与参数 Schema；文件式 Agent Skills 提供适用条件、过程指导、脚本和资源，二者不等价。固定高频动作可以封装为工作流或复合工具，再由 Skill 说明何时使用。

```python
# 复合工具/工作流示意，不把 Python 类本身称为文件式 Agent Skill。
class HandoverValidationWorkflow:
    def run(self, scenario: str, baseline_version: str) -> dict:
        cases = test_case_query(scenario)
        session = simulation_runner(cases)
        logs = log_analyzer(session)
        return baseline_comparator(logs, baseline_version)
```

完整封装仍需处理异步等待、参数校验和失败恢复。是否固化流程，由规划错误、复用频率与维护成本决定。Skill 的开放格式与加载方式见 [[03_Agent系统/10_Agent-Skills与渐进披露]]。
