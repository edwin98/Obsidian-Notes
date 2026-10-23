"""评测数据集：12 条带 Ground Truth 的评测样本。

覆盖全部 8 篇示例文档，每条样本包含：
- qid: 唯一标识
- query: 测试问题
- query_type: 提问方式（原文 / 总结 / 容错）
- expected_doc_ids: Ground Truth 文档 ID 列表
- reference_answer: 标准参考答案（供 RAGAS Context Recall 使用）
"""

EVAL_SAMPLES: list[dict] = [
    # ── doc_001: 5G NR 随机接入流程 ──────────────────────────────────────────
    {
        "qid": "q001",
        "query": "5G随机接入的四步流程是什么？",
        "query_type": "总结",
        "expected_doc_ids": ["doc_001"],
        "reference_answer": (
            "四步随机接入（4-step RACH）包含：MSG1（前导码发送，UE 在 PRACH 上随机选择前导码）、"
            "MSG2（RAR 响应，基站返回时间提前量、临时 C-RNTI 和上行授权）、"
            "MSG3（RRC 连接请求，UE 用分配的上行资源发送 RRCSetupRequest）、"
            "MSG4（竞争解决，基站通过 PDCCH 解决多 UE 冲突）。"
        ),
    },
    {
        "qid": "q002",
        "query": "MSG2 RAR消息包含哪些内容",
        "query_type": "原文",
        "expected_doc_ids": ["doc_001"],
        "reference_answer": (
            "MSG2 RAR 响应包含：时间提前量（Timing Advance，用于上行同步校准）、"
            "临时 C-RNTI（用于后续消息的标识）、上行授权（UL Grant，用于 MSG3 的资源分配）、"
            "RAR 窗口（ra-ResponseWindow，典型值 10ms）。"
        ),
    },
    {
        "qid": "q003",
        "query": "2步rach比4步rach快多少毫秒",
        "query_type": "容错",
        "expected_doc_ids": ["doc_001"],
        "reference_answer": (
            "两步随机接入将接入时延从四步的约 13ms 缩短到约 8ms，减少约 5ms，"
            "适用于低时延 URLLC 场景。"
        ),
    },
    # ── doc_002: 载波聚合 ────────────────────────────────────────────────────
    {
        "qid": "q004",
        "query": "CA是什么技术",
        "query_type": "原文",
        "expected_doc_ids": ["doc_002"],
        "reference_answer": (
            "CA（载波聚合，Carrier Aggregation）是将多个分量载波（Component Carrier，CC）"
            "的带宽聚合使用的技术，通过同时利用多个载波来增加系统传输带宽，"
            "从而提升峰值数据速率和系统吞吐量。"
        ),
    },
    {
        "qid": "q005",
        "query": "NR Rel-17下行最多可以聚合多少个载波",
        "query_type": "原文",
        "expected_doc_ids": ["doc_002"],
        "reference_answer": "NR Rel-17 下行最多支持 32 个分量载波（CC）聚合，上行最多 4 个。",
    },
    # ── doc_003: Massive MIMO 波束管理 ───────────────────────────────────────
    {
        "qid": "q006",
        "query": "波束恢复时延目标是多少毫秒",
        "query_type": "原文",
        "expected_doc_ids": ["doc_003"],
        "reference_answer": "波束恢复时延目标为小于 50ms（从波束失败检测到恢复完成）。",
    },
    # ── doc_004: 5G 网络切片 ─────────────────────────────────────────────────
    {
        "qid": "q007",
        "query": "URLLC切片的时延和可靠性指标要求",
        "query_type": "总结",
        "expected_doc_ids": ["doc_004"],
        "reference_answer": (
            "URLLC（超可靠低时延通信，SST=2）切片的关键指标：时延小于 1ms，可靠性 99.999%。"
        ),
    },
    # ── doc_005: gNodeB 参数配置 ─────────────────────────────────────────────
    {
        "qid": "q008",
        "query": "AAU5613最大功率多少瓦，适合什么场景",
        "query_type": "原文",
        "expected_doc_ids": ["doc_005"],
        "reference_answer": (
            "AAU5613 工作在 n78（3.5GHz）频段，最大功率 200W，天线阵列 64T64R，"
            "适用于城区宏覆盖场景。"
        ),
    },
    # ── doc_006: RRC 连接建立 ────────────────────────────────────────────────
    {
        "qid": "q009",
        "query": "T300定时器超时后UE如何处理",
        "query_type": "总结",
        "expected_doc_ids": ["doc_006"],
        "reference_answer": (
            "T300 超时且 RRC 连接建立未完成时：UE 通知高层连接建立失败、释放 MAC 配置、"
            "重新进入 RRC_IDLE 态，并在 T302 定时器超时前不允许再次发起连接建立（接入控制）。"
        ),
    },
    # ── doc_007: HARQ 重传 ──────────────────────────────────────────────────
    {
        "qid": "q010",
        "query": "HARQ最大重传次数是多少",
        "query_type": "原文",
        "expected_doc_ids": ["doc_007"],
        "reference_answer": (
            "HARQ 最大重传次数由 maxNrofHARQ-Retransmissions 配置，典型值为 4。"
            "超过最大重传次数后由高层 RLC ARQ 机制负责恢复。"
        ),
    },
    {
        "qid": "q011",
        "query": "增量冗余和追踪合并的区别",
        "query_type": "总结",
        "expected_doc_ids": ["doc_007"],
        "reference_answer": (
            "Chase Combining（CC，追踪合并）：重传数据与初传完全相同，接收端做最大比合并（MRC）。"
            "Incremental Redundancy（IR，增量冗余）：每次重传发送不同的冗余版本（RV），"
            "提供额外的编码增益，频谱效率更高。"
        ),
    },
    # ── doc_008: 上行功率控制 ────────────────────────────────────────────────
    {
        "qid": "q012",
        "query": "城区密集部署场景上行功控alpha推荐值是多少",
        "query_type": "原文",
        "expected_doc_ids": ["doc_008"],
        "reference_answer": (
            "城区密集部署场景建议 alpha 取 0.8，P0 调整方向为降低，目的是减少对邻小区的干扰。"
        ),
    },
]
