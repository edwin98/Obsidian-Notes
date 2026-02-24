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

## 6. FastAPI 常见面试问题及回答

### Q1: 为什么在 RAG 或 LLM 场景下首选 FastAPI 而不是 Flask 或 Django？
**回答**：
最核心的原因是**对原生异步（Asyncio）的完美支持**。LLM 对话是一个典型的 I/O 密集型/长耗时场景，一次推理可能长达好几秒甚至几十秒。
- **传统框架（Flask/Django同步模式）**：一个请求就会阻塞挂起一个服务器线程（或进程）。当并发稍微一多，比如 100 个人同时问问题，线程池立刻被打满，新用户连界面都打不开（直接 502）。
- **FastAPI**：底层的 ASGI 规范基于事件循环，当遇到 `await` 调用大模型 API 时，它会自动释放当前执行权去处理别的 HTTP 请求。凭借单进程事件循环，极小的资源消耗即可支撑数千并发，且天生支持 SSE（Server-Sent Events）用于打字机流式输出。

### Q2: FastAPI 里的同步函数 (def) 和异步函数 (async def) 在执行时有什么区别？
**回答**：
- **`async def`**：FastAPI 会把这个函数直接放入主线程的事件循环（Event Loop）中调度。**坑点**：如果在这个函数中执行了不支持异步操作的阻塞代码（如 `time.sleep`、同步的 `requests.get`、大量的计算如矩阵乘法），不管你是哪种请求，**整个服务器的事件循环会被彻底卡死**，所有其他用户的连接全部假死。
- **普通的 `def`**：FastAPI 很聪明地检测到它不是异步函数，会自动把它放进一个**外部的后台线程池**中运行，保证它哪怕阻塞了也不会卡死主线程的事件循环。因此，如果没有对应的 `async` 三方库，老老实实写普通的 `def` 反而是最安全的防线。

### Q3: 什么是依赖注入 (Dependency Injection)？FastAPI 的依赖注入有什么用？
**回答**：
依赖注入是一种解耦模式。在 FastAPI 中，通过 `Depends()` 关键字实现。
- **作用**：将通用的预处理逻辑（如数据库连接的获取与释放、JWT 头部的鉴权验证、配置文件的读取）从具体的路由（Router）核心流程中抽取出来。
- **好处**：提高代码复用率；在测试时，可以通过 `app.dependency_overrides` 极其方便地替换掉线上数据库用于 Mock 单测。
