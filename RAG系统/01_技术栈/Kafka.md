# Kafka 消息中间件与离线计算解耦

Kafka 是本项目中处理海量文档摄入（Ingestion Pipeline）的核心动力，负责将繁重的预处理任务从在线问答链路中完全剥离。

## 0. 技术原理与背景信息
*   **技术原理**：Kafka 是一个分布式的发布-订阅消息系统。它的核心原理是基于磁盘的“顺序写”加“零拷贝（Zero-Copy）”技术。生产者（Producer）只管把消息按顺序追加写入到文件系统的日志中，消费者（Consumer）通过记录并移动偏移量（Offset）来顺序读取消息。这种设计使得 Kafka 能在极低内存占用下，承受每秒百万级别的极致数据吞吐量。同时通过多副本机制保证了数据不丢失。
*   **背景信息（为什么需要它）**：在构建 RAG 知识库时，如果用户一次性上传了数万份 PDF 文档，系统需要进行极其耗时的 OCR 识别、切分、向量化（Embedding）操作。如果让 Web API 直接去等结果，不仅请求会超时，后端算力也会在一瞬间被洪水般的流量彻底压垮导致 OOM。通过引入 Kafka 这道“缓冲大坝”，我们将所有文档处理任务化作一条条消息存放在 Kafka 里，后端的离线消费者根据自身的处理能力，以平滑的节奏从 Kafka 中抽取任务慢慢消化，保护了整个系统的架构稳定性。

## 1. 核心作用
*   **削峰填谷**：应对批量上传文档时的瞬间流量洪峰，保护后端切片与向量化模型集群不被 OOM 压垮。
*   **异步解耦**：FastAPI 服务端只需将任务抛入 Kafka 即可返回，无需等待耗时的 PDF 解析和摘要生成。
*   **任务编排**：作为流水线的中转站，将一个文档处理流程拆解为多个阶段（OCR -> Clean -> Slice -> Embedding）。

## 2. 核心能力
- **高吞吐持久化**：支持 PB 级数据的顺序写入，保障数据即使在服务宕机时也不会丢失。
- **消费者组隔离**：允许不同的服务（如：监控服务、解析服务、入库服务）同时消费同一份数据，互不干扰。
- **顺序保证**：基于 `Key`（如 `doc_id`）的哈希分区，确保同一文档的所有切片操作按顺序由同一个消费者处理。

## 3. 常见用法
- **多 Topic 流水线**：
  - `doc_upload`: 原始文件数据负载。
  - `doc_parsed`: PDF 转 Markdown 后的结构化文本。
  - `vector_ingested`: 成功入库向量数据库的确认消息。
- **ACKS 配置**：设置 `acks='all'` 确保数据在所有备份节点落盘，追求研发资产存储的绝对安全。

## 4. 技术实现示例
```python
from aiokafka import AIOKafkaProducer
import json
import asyncio

async def push_to_ingestion_pipeline(doc_metadata: dict):
    # 初始化原生异步高吞吐 Producer
    producer = AIOKafkaProducer(
        bootstrap_servers='kafka-cluster.internal:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        acks='all',               # 强一致性落盘
        compression_type='lz4',   # 开启数据压缩，带宽友好
        batch_size=16384          # 攒批发送
    )
    await producer.start()
    try:
        # 发送处理任务
        await producer.send_and_wait(
            topic="doc_ingestion_tasks", 
            value=doc_metadata,
            # 利用 doc_id 实现分区锁定，保证同文件语义时序相关性
            key=doc_metadata['doc_id'].encode('utf-8') 
        )
    finally:
        await producer.stop()
```

## 5. 注意事项与坑点
- **消息大小限制 (Message Max Bytes)**：默认 1MB。PDF 解析出的原始文本若超过此限制会导致发送失败。需调大 `message.max.bytes` 或将大文本存入 S3，Kafka 只传 URL。
- **消费积压监控**：必须严格监控 `Lag` 指标。若文档入库有数小时延迟，需及时横向扩展消费者 Pod 数量。
- **再平衡 (Rebalance) 震荡**：消费者频繁上下线会导致长时间不消费。需合理配置 `session.timeout.ms`。
- **幂等性保障**：由于 Kafka 的 `At-Least-Once` 特性，写入向量数据库的代码必须根据 `chunk_id` 实现幂等操作（即：重复消费同一消息不应导致数据库出现重复记录）。

## 6. Kafka 常见面试问题及回答

### Q1: 为什么 Kafka 能拥有极高的吞吐量和极低的延迟？
**回答**：
主要得益于两项核心底层的 OS 技术，将磁盘读写的速度压榨到了极限：
1. **顺序追加写 (Sequential I/O)**：磁盘如果是随机读写寻道时间极慢，但顺序写盘的速度甚至堪比写内存。Kafka 的日志文件一经写入便不可修改，所有的写入都只是追加到文件末尾。
2. **零拷贝 (Zero-Copy) 与 Page Cache**：现代 OS 在内存中划分了页缓存（Page Cache）存放磁盘文件的副本。Kafka 不会在 JVM 堆内存里做过多的缓存。在传输时，利用 Linux 的 `sendfile` 系统调用，数据直接从内核态的 Page Cache 被 DMA 拷入网卡发送，越过了用户态和内核态之间的高昂数据拷贝损耗。

### Q2: Kafka 的消费者组 (Consumer Group) 是什么概念？一个 Partition 可以被同一个组内的多个消费者消费吗？
**回答**：
- **消费者组**：是一组为了“共同分担”同一个 Topic 下消息消费压力的逻辑上组合在一起的 Consumer 实例（比如起了 5 台服务器一起处理文档解析）。
- **原则**：为了保证消息消费顺序性，Kafka 严格规定，**一个 Partition 只能被同一个消费者组内的一个消费者消费**。这也意味着，如果你的 Topic 只有 3 个 Partition，即使你在组里起了 10 个消费者实例，也最多只有 3 个实例在干活，剩下 7 个处于闲置饥饿状态。因此，**扩容消费者的前提是提前扩容好足够的 Partition**。

### Q3: 消息怎么保证不丢失（如何实现 At-Least-Once）？
**回答**：
需要生产者、Broker、消费者三端配合：
1. **生产者 (Producer)**：配置 `acks=all`（或者 `-1`），意味着必须等所有副本节点（Followers）都把数据落盘才认为发送成功；配置 `retries` 进行失败重试机制。
2. **消息中间件端 (Broker)**：配置多副本（Replication Factor），并将 `min.insync.replicas`（最小同步副本数）设为至少大于 1，以防单点宕机丢失数据。
3. **消费者 (Consumer)**：必须**关闭自动提交 Offset (`enable.auto.commit=false`)**。只有当文档真正在后端入库（比如写入 Milvus / MySQL）确认无误后，才在代码尾部手动提交 Offset (`commitSync()`)，防止执行中途崩溃导致位点丢失从而漏消费。
