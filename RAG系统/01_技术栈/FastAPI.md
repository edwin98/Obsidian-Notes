# FastAPI 微服务架构与实时交互

FastAPI 是本项目在线问答链路的“大脑”，负责连接前端请求、中台检索逻辑与底层大模型生成。

## 0. 技术原理与背景信息
*   **技术原理**：FastAPI 基于 Starlette（负责 Web 路由与 ASGI 异步服务器规范）和 Pydantic（负责数据验证）。它深度依赖 Python 3.6+ 引入的协程（`async/await`）机制。当处理 I/O 密集型任务（如等待数据库返回、等待 LLM 推理响应）时，FastAPI 会主动挂起当前请求，交出 CPU 控制权给事件循环，从而用少数几个线程就能并发处理成千上万个网络连接，彻底打破了传统同步框架（如 Flask/Django）每一个请求独占一个线程导致的性能瓶颈。
*   **背景信息（为什么需要它）**：在 RAG 在线问答中，最核心的刚需是**“流式输出（Streaming/打字机效果）”**与**“应对大模型漫长的推理等待时间”**。传统框架在等待 LLM 输出时会把宝贵的服务器线程阻塞卡死，并发量一高整个服务就会瘫痪。而且 RAG 后端需要和前段随时保持协议的同步对齐。FastAPI 天生异步的设计完美契合了 LLM 长连接推流场景，同时它能根据代码自动生成 Swagger 接口文档，大大降低了团队协作成本。

## 1. 核心作用
*   **流量分发与编排**：接收 HTTP 请求，并发调度 Redis、ES、Milvus 及大模型 API 资源。
*   **接口规范化**：基于 OpenAPI/Swagger 标准，提供类型安全的外部交互协议。
*   **流式内容吐出**：通过 SSE 协议实现 LLM 的逐字推流功能，极速响应用户。

## 2. 核心能力
- **原生异步 (Asyncio)**：利用非阻塞 I/O，单台服务器可抗住数千个并发长连接（由于 RPC 等待时间长，异步是提升 QPS 的唯一出路）。
- **极速性能**：基于 Starlette 和 Pydantic，它是 Python 系中性能最强的框架之一。
- **自动文档化**：实时生成的 `/docs` 界面，极大降低了前后端联调成本。
- **依赖注入系统**：方便地管理数据库连接池、日志器以及权限校验逻辑。

## 3. 常见用法
- **StreamingResponse**：打字机效果。配合异步生成器（Generator）将大模型生成的 Token 实时回传。
- **BackgroundTasks**：轻量异步动作。例如：用户发起提问的同时，在后台异步记录埋点日志，不阻塞 API 返回。
- **Middlewares**：全局拦截。用于实现跨域 (CORS) 处理、请求耗时监控及异常统一捕获。

## 4. 技术实现示例
```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio

app = FastAPI(title="RAG-Brain-Service")

async def llm_stream_generator(query: str):
    """
    模拟调用底层 VLLM 框架的流式协程迭代器
    """
    # 实际场景这里会 yield 外部 LLM 的输出
    for chunk in ["这", "是", "RAG", "生", "成", "的答", "案"]:
        # 遵循 SSE 协议格式：data: 内容 \n\n
        yield f"data: {chunk}\n\n"
        await asyncio.sleep(0.05)

@app.get("/api/chat/stream")
async def chat_stream_endpoint(query: str):
    """
    流式问答端点：
    极速返回 200 并挂载 StreamingResponse，严防网关 HTTP 阻塞超时
    """
    return StreamingResponse(
        llm_stream_generator(query), 
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
```

## 5. 注意事项与坑点
- **长连接超时**：网关（如 Nginx 或企业网关）默认可能有 60s 的超时阈值。由于 Reranker + LLM 过程可能很长，需同步调整网关的 `proxy_read_timeout`。
- **阻塞同步调用**：如果代码中调用了不支持异步的 `requests` 库或密集型计算函数，必须使用 `run_in_executor` 扔进进程池，否则会锁死 FastAPI 的事件循环，导致其他请求全部卡死。
- **内存泄漏**：在流式生成中，由于 generator 保持了较长时间的变量引用，需注意对象的周期释放。
- **Uvicorn 配置**：生产环境务必开启 `--workers` 多进程模式，并配合 `prometheus-fastapi-instrumentator` 监控服务健康。
