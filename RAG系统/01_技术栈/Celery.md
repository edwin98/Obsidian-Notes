# Celery 任务队列机制

在 RAG 项目中，Celery 负责将那些“不需要实时反馈结果”的沉重计算任务带离主线程，典型的业务场景是**超长会话的摘要生成**和**检索召回率的夜间巡检**。

## 0. 技术原理与背景信息
*   **技术原理**：Celery 采用经典的“生产者-消费者”模型。当主程序（生产者）遇到耗时任务时，不会自己执行，而是将任务的函数名和参数打包序列化生成一条消息，投递到消息中间件（Broker，本项目通常用 Redis 或 RabbitMQ）。运行在后台或其它服务器上的 Worker（消费者）进程持续监听 Broker，一旦发现新任务，便拉取下来进行处理；如果处理失败，则根据设定的退避策略进行重试，完成后将状态写回 Backend。
*   **背景信息（为什么需要它）**：Kafka 虽然也是消息队列，但它偏重于海量数据流的吞吐；而 Celery 更偏重于**具体函数任务的异步执行、状态追踪和定时调度**。在 RAG 系统中，有很多业务逻辑如“失败重试去调外部 API”、“每天凌晨自动执行向量数据库的对齐检查”，这些属于复杂的“任务编排”。让 FastAPI 去自己写多线程跑各种定时和重试不仅极易导致内存泄漏，还难以做分布式扩展。Celery 为 Python 提供了开箱即用的异步任务执行框架，完美解决了耗时业务的编排问题。

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

## 4. 技术实现示例
```python
from celery import Celery
import time

# 定义 Celery App 实例
# 选用 Redis 为消息代理 (Broker) 与结果状态载体 (Backend)
celery_app = Celery(
    'rag_tasks', 
    broker='redis://redis-cluster:6379/1',
    backend='redis://redis-cluster:6379/2'
)

# 配置任务路由，将耗时任务专线处理
celery_app.conf.task_routes = {
    'tasks.summarize': {'queue': 'gpu_intensive'}
}

@celery_app.task(bind=True, max_retries=5, rate_limit='10/m')
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
        # 异常重试策略：2s, 4s, 8s, 16s...
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
```

## 5. 注意事项与坑点
- **Result Backend 爆炸**：由于 RAG 任务多，如果开启了 `result_backend` 且不设置存活时间，Redis 内存会被任务结果塞满。务必设置 `result_expires`。
- **数据库连接泄漏**：Celery Worker 是多进程执行。在任务代码中如果直接实例化 DB 连接而不在任务结束时关闭，会迅速耗尽数据库连接池。
- **可视化利器 Flower**：强烈建议开启 Flower 界面，实时观察任务成功率、并发数以及哪些节点处于负载状态。
- **序列化选择**：默认使用 `json`。如果任务中需要传递复杂的 Numpy 数组（如向量），建议配置使用 `pickle` 或先将数据入缓存，只传递 ID。

## 6. Celery 常见面试问题及回答

### Q1: Celery 的系统架构由哪些部分组成？Broker 和 Backend 的区别是什么？
**回答**：
- **架构组成**：生产者（投递任务的客户端，如 FastAPI）、Broker（消息中间件/任务队列，如 Redis/RabbitMQ）、Worker（执行任务的消费者进程）、Backend（存储任务执行结果和状态的后端）。
- **区别**：Broker 负责暂存待处理的任务消息，相当于“收发室”；Backend 负责存储任务执行完成后的状态（成功、失败）和返回值，相当于“档案室”。在 RAG 中为了简化部署，通常都使用 Redis。

### Q2: 如果 Celery Worker 处理任务失败了怎么办？如何保证任务不丢失？
**回答**：
- **重试机制**：可以在 `@celery_app.task(bind=True, max_retries=3)` 中配置自动重试，并且可以在异常捕获中通过 `raise self.retry(exc=exc, countdown=2 ** self.request.retries)` 实现**指数退避重试**（避免密集重试压垮第三方 API）。
- **防丢失（ACK 机制）**：Worker 默认在任务执行完成后（不管成功还是抛出异常）才会向 Broker 发送 ACK（确认信号）。如果在执行中途 Worker 进程崩溃（如 OOM），Broker 没有收到 ACK，等超时后会将该任务重新分配给其他 Worker 再次执行。

### Q3: 为什么在 RAG 场景中，把长任务交给 Celery 而不是直接用 Python 的多线程或协程去跑？
**回答**：
1. **可靠性**：FastAPI 的协程/后台任务（BackgroundTasks）受限于 Web 进程的生命周期。如果 Web 节点重启或崩溃，内存中还在跑的任务就会全部丢失。Celery 将任务持久化在 Broker 中，即使整个集群重启也能恢复断点。
2. **算力隔离与扩容**：大模型的向量计算或大文档解析极耗 CPU/GPU。Celery 能够将 Web 服务器（I/O 密集型）与 Worker 节点（CPU密集型）物理隔离。如果遇到离线入库高峰，我们可以单独为 Worker 增加机器（横向扩容），而不影响在线 Web 服务的稳定。
3. **复杂编排**：Celery 原生支持延时任务、定时任务（Celery Beat）、失败重试序列与任务链（Chains）等复杂的编排需求，这些如果是用协程手写，代码会非常臃肿且难以维护。
