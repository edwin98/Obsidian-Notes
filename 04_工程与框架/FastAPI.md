# FastAPI 微服务架构与实时交互

FastAPI 是本项目在线问答服务的接口层，负责接收前端请求、调用检索与生成组件，并返回结果。

## 0. 技术原理与背景信息
*   **技术原理**：FastAPI 基于 Starlette（负责 Web 路由与 ASGI 异步服务器规范）和 Pydantic（负责数据验证）。异步接口使用 Python 的 `async/await` 协程机制，运行时版本要求以所用 FastAPI 版本为准。当处理 I/O 密集型任务（如等待数据库返回、等待 LLM 推理响应）时，FastAPI 会主动挂起当前请求，交出 CPU 控制权给事件循环，从而用少数几个线程就能并发处理成千上万个网络连接，提高 I/O 等待场景下的并发利用率；不能将所有 Flask/Django 部署概括为同一种同步执行模型。
*   **背景信息（为什么需要它）**：RAG 问答需要等待检索和模型生成，并向前端持续返回内容。FastAPI 支持异步请求处理，可在等待网络 I/O 时处理其他请求；也可以通过 SSE 实现流式输出，并自动生成 OpenAPI 接口文档。并发能力仍取决于调用方式和下游服务容量。

## 1. 核心作用
*   **流量分发与编排**：接收 HTTP 请求，并发调度 Redis、ES、Milvus 及大模型 API 资源。
*   **接口规范化**：基于 OpenAPI/Swagger 标准，提供类型安全的外部交互协议。
*   **流式输出**：通过 SSE 持续发送生成内容，减少用户等待完整答案的时间。

## 2. 核心能力
- **原生异步 (Asyncio)**：利用非阻塞 I/O，单台服务器可抗住数千个并发长连接（实际容量仍需按负载、连接池与部署资源测试）。
- **异步请求处理**：结合 Starlette 与 Pydantic，提供异步路由和数据校验；实际性能取决于业务逻辑和部署配置。
- **自动文档化**：实时生成的 `/docs` 界面，极大降低了前后端联调成本。
- **依赖注入系统**：方便地管理数据库连接池、日志器以及权限校验逻辑。

## 3. 常见用法
- **StreamingResponse**：打字机效果。配合异步生成器（Generator）将大模型生成的 Token 实时回传。
- **BackgroundTasks**：轻量异步动作。例如：用户发起提问的同时，在后台异步记录埋点日志，不阻塞 API 返回。
- **Middlewares**：全局拦截。用于实现跨域 (CORS) 处理、请求耗时监控及异常统一捕获。

### 指标与分布式追踪

FastAPI 0.142.0 引入原生 OpenTelemetry 支持；2026-10-08 的 0.143.0 将自动 exporter 配置改为显式启用。原生接入能力不代表所有服务会自动完整采集和导出遥测，仍有配置边界。[官方发布记录](https://fastapi.tiangolo.com/release-notes/)

Prometheus 指标主要用于观察请求量、错误率和延迟分布；tracing 用于关联一次请求经过检索、重排、模型调用等阶段的耗时。两者解决不同的定位问题，可以互补。

Pydantic 兼容路线也随 FastAPI 版本演进。原项目配置仍按对应版本解释。关联：[[04_工程与框架/Pydantic]]。

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
    使用 StreamingResponse 逐步返回内容，减少等待完整回答造成的超时风险
    """
    return StreamingResponse(
        llm_stream_generator(query),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
```

## 5. 注意事项与坑点
- **长连接超时**：网关（如 Nginx 或企业网关）默认可能有 60s 的超时阈值。由于 Reranker + LLM 过程可能很长，需同步调整网关的 `proxy_read_timeout`。
- **阻塞调用**：`loop.run_in_executor(None, func)` 默认使用线程池，`asyncio.to_thread` 也使用线程。纯 Python CPU 密集计算可显式使用 `ProcessPoolExecutor` 或独立 worker；在 `async def` 中直接运行阻塞函数会阻塞事件循环。
- **内存泄漏**：在流式生成中，由于 generator 保持了较长时间的变量引用，需注意对象的周期释放。
- **Uvicorn 配置**：生产环境务必开启 `--workers` 多进程模式，并配合 `prometheus-fastapi-instrumentator` 监控服务健康。

## 6. FastAPI 常见面试问题及回答

### Q1: 为什么在 RAG 或 LLM 场景下首选 FastAPI 而不是 Flask 或 Django？
**回答**：
主要原因是**支持 Python 异步调用（Asyncio）**。模型服务调用通常需要等待数秒甚至数十秒，应用可在等待网络响应时处理其他请求。
- **传统框架（Flask/Django同步模式）**：一个请求就会阻塞挂起一个服务器线程（或进程）。当并发稍微一多，比如 100 个人同时问问题，线程池立刻被打满，新用户连界面都打不开（直接 502）。
- **FastAPI**：采用 ASGI 接口规范。异步调用模型 API 时，`await` 可以让出执行权，使事件循环继续处理其他请求。SSE 可用于逐步返回模型输出；实际并发容量需结合负载测试确认。

### Q2: FastAPI 里的同步函数 (def) 和异步函数 (async def) 在执行时有什么区别？
**回答**：
- **`async def`**：异步路由由事件循环调度。如果直接执行 `time.sleep`、同步 `requests.get` 或大量计算，会阻塞当前事件循环，影响同一 Worker 处理的其他请求。应使用异步接口，或将阻塞操作移到线程池、进程或后台任务中。
- **普通的 `def`**：FastAPI 很聪明地检测到它不是异步函数，会自动把它放进一个**外部的后台线程池**中运行，保证它哪怕阻塞了也不会卡死主线程的事件循环。因此，如果没有对应的 `async` 三方库，老老实实写普通的 `def` 反而是最安全的防线。

### Q3: 什么是依赖注入 (Dependency Injection)？FastAPI 的依赖注入有什么用？
**回答**：
依赖注入是一种解耦模式。在 FastAPI 中，通过 `Depends()` 关键字实现。
- **作用**：将通用的预处理逻辑（如数据库连接的获取与释放、JWT 头部的鉴权验证、配置文件的读取）从具体的路由（Router）核心流程中抽取出来。
- **好处**：提高代码复用率；在测试时，可以通过 `app.dependency_overrides` 方便地替换掉线上数据库用于 Mock 单测。
