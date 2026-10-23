# Celery 任务队列机制

在 RAG 项目中，Celery 负责将那些“不需要实时反馈结果”的沉重计算任务带离主线程，典型的业务场景是**超长会话的摘要生成**和**检索召回率的夜间巡检**。

## 0. 技术原理与背景信息
*   **技术原理**：Celery 采用经典的“生产者-消费者”模型。当主程序（生产者）遇到耗时任务时，不会自己执行，而是将任务的函数名和参数打包序列化生成一条消息，投递到消息中间件（Broker，本项目通常用 Redis 或 RabbitMQ）。运行在后台或其它服务器上的 Worker（消费者）进程持续监听 Broker，一旦发现新任务，便拉取下来进行处理；如果处理失败，则根据设定的退避策略进行重试，完成后将状态写回 Backend。
*   **背景信息（为什么需要它）**：Kafka 主要用于消息与数据流，Celery 更侧重于**函数任务的异步执行、状态追踪、重试和定时调度**。RAG 中的外部 API 重试、定时索引检查等任务，可交给 Celery Worker 执行，减少在 Web 服务内自行维护线程和调度逻辑的工作。

## 1. 核心作用
*   **异步后台处理**：当对话历史触及 Token 阈值，Celery 被唤起去调轻量模型生成摘要。
*   **任务失败重试**：针对大模型 API 偶发的 503 超时，Celery 提供自动的指数退避重试能力。
*   **定时任务调度 (Beat)**：安排每日凌晨 2 点执行知识库的自动优化、合并与向量库同步。

## 2. 核心能力
- **松耦合架构**：生产者（FastAPI）发送任务到 Broker（Redis），消费者（Celery Worker）在另一台甚至多台机器上执行。
- **并发控制**：可以设置 `concurrency` 参数，精确控制由于 LLM 推理带来的资源占用上限。
- **状态追踪**：能够查询任务是 `PENDING`、`STARTED` 还是 `SUCCESS`，并储存执行结果。

## 3. 常见用法
- **离线摘要压缩**：大模型 RAG 中，为了不阻塞当前问答，摘要压缩任务被打成后台 Task。
- **文档重采样评估**：从知识库随机抽取 Chunk，由 Celery 调用模型生成对应 Question，以此来评价召回率。
- **报警与监控**：当 RAG 系统检测到由于某种原因召回结果为空达到一定频次，由异步任务发送企业微信报警。

### ETA/countdown 与队列配置

5.6 增加 `worker_eta_task_limit`，限制 worker 预取并放在内存中的 ETA/countdown 任务数量；默认不自动设限。大量远期任务可能占用 worker 内存，因此“任务尚未执行”不等于“任务一定只在 broker 中等待”。它还扩展了自动创建 quorum queue 的配置并修复若干内存、关闭行为问题。[官方 5.6 更新](https://docs.celeryq.dev/en/stable/history/whatsnew-5.6.html)

这些机制改善资源与队列管理，不会自动带来 exactly-once。任务重投与业务进度恢复不同，重试和幂等依然是原有知识的关键部分。

原项目队列和任务实现仍按当时版本解释。

## 4. 技术实现示例
```python
from celery import Celery
import time

# 定义 Celery App 实例
# 选用 Redis 为消息代理 (Broker) 与结果状态载体 (Backend)
celery_app = Celery(
    "rag_tasks",
    broker="redis://redis-cluster:6379/1",
    backend="redis://redis-cluster:6379/2",
)

# 配置任务路由，将耗时任务专线处理
celery_app.conf.task_routes = {"tasks.summarize": {"queue": "gpu_intensive"}}


@celery_app.task(name="tasks.summarize", bind=True, max_retries=5, rate_limit="10/m")
def async_summarize_session(self, session_id):
    """
    异步对话压缩任务：
    支持指数退避重试 (exponential backoff)
    """
    try:
        # 获取业务上下文
        history = db.get_session_session(session_id)

        # 核心 AI 压缩操作 (耗时可能 > 5s)
        summary = llm.summarize_context(history)

        # 更新数据库
        db.update_session_summary(session_id, summary)
        return {"status": "success", "session": session_id}
    except Exception as exc:
        # 异常重试策略：1s, 2s, 4s, 8s...
        raise self.retry(exc=exc, countdown=2**self.request.retries)
```

## 5. 注意事项与坑点
- **任务结果持续占用内存**：启用 `result_backend` 后，如果不清理历史结果，Redis 内存占用会持续增长。应设置 `result_expires`，并监控结果存储的容量。
- **数据库连接泄漏**：Celery Worker 是多进程执行。在任务代码中如果直接实例化 DB 连接而不在任务结束时关闭，会迅速耗尽数据库连接池。
- **可视化利器 Flower**：强烈建议开启 Flower 界面，实时观察任务成功率、并发数以及哪些节点处于负载状态。
- **序列化选择**：优先 JSON；大数组等数据存入对象存储/缓存，任务只传 ID。pickle 可执行反序列化代码，不应用于不可信消息。

## 6. Celery 常见面试问题及回答

### Q1: Celery 的系统架构由哪些部分组成？Broker 和 Backend 的区别是什么？
**回答**：
- **架构组成**：生产者（投递任务的客户端，如 FastAPI）、Broker（消息中间件/任务队列，如 Redis/RabbitMQ）、Worker（执行任务的消费者进程）、Backend（存储任务执行结果和状态的后端）。
- **区别**：Broker 负责暂存待处理的任务消息，相当于“收发室”；Backend 负责存储任务执行完成后的状态（成功、失败）和返回值，相当于“档案室”。在 RAG 中为了简化部署，通常都使用 Redis。

### Q2: 如果 Celery Worker 处理任务失败了怎么办？如何保证任务不丢失？
**回答**：
- **重试**：`max_retries` 只设上限，不会自动重试。显式调用 `self.retry`，或配置 `autoretry_for` 才会触发相应重试策略。
- **ACK**：默认在执行前确认。`acks_late=True` 推迟到执行后确认；worker 子进程异常退出等情况还需考虑 `task_reject_on_worker_lost=True`、broker 的恢复机制和消息持久化。重投可能导致重复执行，应设计幂等，不能承诺仅靠 ACK 配置绝不丢失。

### Q3: 为什么在 RAG 场景中，把长任务交给 Celery 而不是直接用 Python 的多线程或协程去跑？
**回答**：
1. **可靠性**：broker 持久化及确认配置可支持故障后重投任务，但通常从头执行；恢复业务处理进度需要应用自己持久化 checkpoint。FastAPI BackgroundTasks 则依赖 Web 进程生命周期。
2. **算力隔离与扩容**：大模型的向量计算或大文档解析极耗 CPU/GPU。Celery 能够将 Web 服务器（I/O 密集型）与 Worker 节点（CPU密集型）物理隔离。如果遇到离线入库高峰，我们可以单独为 Worker 增加机器（横向扩容），而不影响在线 Web 服务的稳定。
3. **复杂编排**：Celery 原生支持延时任务、定时任务（Celery Beat）、失败重试序列与任务链（Chains）等复杂的编排需求，这些如果是用协程手写，代码会非常臃肿且难以维护。
