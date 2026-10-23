"""内置示例文档：8 篇无线通信领域模拟文档（Markdown 格式）。

覆盖场景：
- 多级标题层次结构（测试层次化切分）
- 术语缩写（测试 BM25 关键词匹配）
- 长文档（测试非叶子节点摘要）
- 具体参数数据（测试精确匹配）
- 步骤序列（测试多跳检索）
"""

SAMPLE_DOCUMENTS = [
    {
        "doc_id": "doc_001",
        "doc_name": "5G NR 随机接入流程",
        "content": """# 5G NR 随机接入流程

## 概述

5G NR（New Radio）随机接入是终端设备（UE）接入基站（gNodeB）的关键过程。随机接入流程用于建立上行同步、获取上行资源分配，是 UE 从空闲态转为连接态的必经之路。

随机接入的主要触发场景包括：
- 初始接入（开机入网）
- RRC 连接重建
- 切换（Handover）
- 上行数据到达但无上行同步
- 下行数据到达需建立上行链路

## PRACH 信道

PRACH（Physical Random Access Channel）是承载随机接入前导码的物理信道。

### 前导码格式

5G NR 定义了多种前导码格式（Preamble Format），分为长格式和短格式两大类：

| 格式 | 序列长度 | 子载波间隔 | 适用场景 |
|------|---------|-----------|---------|
| 0 | 839 | 1.25 kHz | 大覆盖小区 |
| 1 | 839 | 1.25 kHz | 大覆盖小区（扩展CP） |
| 2 | 839 | 1.25 kHz | 超大覆盖 |
| 3 | 839 | 5 kHz | 大覆盖高速场景 |
| A1 | 139 | 15/30 kHz | 中小覆盖 |
| A2 | 139 | 15/30 kHz | 中小覆盖（受限） |
| A3 | 139 | 15/30 kHz | 中覆盖 |
| B1 | 139 | 15/30 kHz | 小覆盖 |
| B4 | 139 | 15/30 kHz | 室内小站 |
| C0 | 139 | 15/30 kHz | 超密集部署 |
| C2 | 139 | 15/30 kHz | 毫米波 |

### 时频资源配置

PRACH 的时频资源由 RRC 参数 RACH-ConfigCommon 配置，主要参数包括：
- prach-ConfigIndex：确定时域资源位置（子帧/时隙编号）
- msg1-FrequencyStart：确定频域起始位置
- msg1-FDM：频分复用的 PRACH 资源数量（1/2/4/8）
- ssb-perRACH-OccasionAndCB-PreamblesPerSSB：SSB 与 RACH Occasion 的映射关系

## 四步随机接入（4-step RACH）

### MSG1：前导码发送

终端在选定的 PRACH 资源上发送随机接入前导码。前导码选择遵循以下规则：
1. 根据 SSB 测量结果选择最佳波束方向
2. 在对应的 RACH Occasion 上随机选择一个前导码序列
3. 按照功率控制公式确定发射功率：P_PRACH = min(P_CMAX, P_target + PL)

### MSG2：RAR 响应

基站在检测到前导码后，通过 PDCCH（RA-RNTI 加扰）调度 RAR（Random Access Response）消息，包含：
- 时间提前量（Timing Advance）：用于上行同步校准
- 临时 C-RNTI：用于后续消息的标识
- 上行授权（UL Grant）：用于 MSG3 的资源分配
- RAR 窗口：ra-ResponseWindow 配置，典型值 10ms

### MSG3：RRC 连接请求

终端使用 RAR 中分配的上行资源发送 RRC 消息（如 RRCSetupRequest），携带：
- UE 标识信息（随机值或 S-TMSI）
- 建立原因（Establishment Cause）
- 需要传输的 NAS 消息

### MSG4：竞争解决

基站通过 PDCCH 发送竞争解决消息，解决多个 UE 选择相同前导码的冲突：
- 如果使用 S-TMSI 标识，在 PDCCH 中用 C-RNTI 加扰
- 如果使用随机值标识，在 MAC PDU 中携带 UE Contention Resolution Identity
- UE 比对标识后确认接入成功或失败

## 两步随机接入（2-step RACH）

为降低接入时延，5G NR 引入了两步随机接入（2-step RACH）机制：

### MSGA

MSGA 合并了四步流程中的 MSG1 和 MSG3：
- 在 PRACH 上发送前导码
- 紧随其后在 PUSCH 上发送 RRC 消息
- 两者时域位置紧密关联

### MSGB

MSGB 合并了 MSG2 和 MSG4：
- 包含 RAR 信息和竞争解决结果
- 如果 fallback 到四步流程，则转为发送普通 RAR

两步随机接入的优势：将接入时延从 4 步的约 13ms 缩短到 2 步的约 8ms，适用于低时延 URLLC 场景。
""",
    },
    {
        "doc_id": "doc_002",
        "doc_name": "载波聚合技术详解",
        "content": """# 载波聚合（CA）技术详解

## 定义

载波聚合（Carrier Aggregation，CA）是一种将多个分量载波（Component Carrier，CC）的带宽聚合使用的技术，通过同时利用多个载波来增加系统传输带宽，从而提升峰值数据速率和系统吞吐量。

CA 是 LTE-A 引入的核心技术之一，在 5G NR 中得到了进一步增强。

## 基本概念

### 分量载波（CC）

每个分量载波独立配置：
- 带宽：支持 5/10/15/20/25/30/40/50/60/80/100 MHz
- 子载波间隔：15/30/60/120/240 kHz
- 帧结构：可独立配置 TDD/FDD

### 聚合方式

CA 支持三种聚合方式：
1. **带内连续聚合（Intra-band Contiguous）**：分量载波在同一频段且频域相邻
2. **带内非连续聚合（Intra-band Non-contiguous）**：分量载波在同一频段但频域不相邻
3. **带间聚合（Inter-band）**：分量载波分布在不同频段

### 主辅载波

- **PCell（Primary Cell）**：主小区，提供 NAS 信令连接，始终激活
- **SCell（Secondary Cell）**：辅小区，用于扩展带宽，可动态激活/去激活
- **PSCell（Primary Secondary Cell）**：双连接场景下辅基站组的主小区

## 关键参数配置

### 最大聚合载波数

| 标准版本 | 下行最大 CC | 上行最大 CC |
|---------|-----------|-----------|
| LTE Rel-10 | 5 | 5 |
| NR Rel-15 | 16 | 4 |
| NR Rel-16 | 16 | 4 |
| NR Rel-17 | 32 | 4 |

### gNodeB 参数配置

CA 激活需要在 gNodeB 侧配置以下关键参数：
- SCC 频点配置（ServingCellConfigCommon）
- 测量配置（MeasConfig / MeasObjectNR）
- 添加/释放辅小区事件（Event A4/A6）
- SCell 激活/去激活 MAC CE

## 性能优势

1. **峰值速率提升**：聚合 N 个 CC，理论峰值速率可线性提升 N 倍
2. **频谱利用率优化**：将碎片化的频谱资源整合利用
3. **负载均衡**：跨载波调度实现多 CC 间的负载均衡
4. **覆盖增强**：低频 + 高频协同，兼顾覆盖与容量

## 典型部署场景

- 城区宏站：n78 (3.5GHz) + n28 (700MHz) 带间 CA
- 室内覆盖：n78 + n41 (2.6GHz) 带间 CA
- 热点区域：n78 多载波带内 CA
""",
    },
    {
        "doc_id": "doc_003",
        "doc_name": "Massive MIMO 波束管理",
        "content": """# Massive MIMO 波束管理

## 概述

Massive MIMO（大规模多入多出）是 5G NR 的核心技术之一。通过在基站侧部署大量天线阵元（通常 32/64/128/256 个），实现精确的波束赋形，显著提升频谱效率和覆盖能力。

波束管理（Beam Management）是 Massive MIMO 系统正常运行的基础，涵盖波束扫描、测量、上报和切换的全过程。

## 波束管理框架

### P1：初始波束扫描

基站周期性发送 SSB（Synchronization Signal Block），每个 SSB 对应不同的波束方向：
- FR1（Sub-6GHz）：最多 4/8 个 SSB 波束
- FR2（毫米波）：最多 64 个 SSB 波束
- SSB 突发集周期：5/10/20/40/80/160 ms

UE 通过测量不同 SSB 的 SS-RSRP/SS-RSRQ/SS-SINR，选择最佳波束。

### P2：基站侧波束细化

在 P1 确定粗波束后，基站通过 CSI-RS（Channel State Information Reference Signal）进行细粒度波束扫描：
- 在粗波束覆盖范围内进一步划分细波束
- CSI-RS 资源配置通过 NZP-CSI-RS-ResourceSet 下发
- 波束细化可将波束宽度从 15-30 度收窄到 5-10 度

### P3：UE 侧波束细化

UE 在确定最佳服务波束后，调整自身的接收波束：
- 通过接收不同方向的 SSB/CSI-RS 信号，选择最佳接收波束
- 适用于 UE 具有多个接收面板的场景（如 FR2 终端）

## 波束测量与上报

### 测量资源

| 参考信号 | 用途 | 配置参数 |
|---------|------|---------|
| SSB | 初始接入、波束管理 L1 测量 | ssb-ConfigMobility |
| CSI-RS | 精细波束管理、CSI 获取 | CSI-MeasConfig |
| SRS | 上行波束管理 | SRS-Config |

### L1-RSRP 上报

UE 通过 PUCCH/PUSCH 上报 L1-RSRP 测量结果：
- 上报触发：周期性 / 半持续 / 非周期
- 上报内容：最佳波束 ID + RSRP 值（可上报多个波束对）
- 报告配置：CSI-ReportConfig 中 reportQuantity 设置

## 波束恢复

当服务波束质量急剧下降（如 UE 移动到遮挡区域）时，触发波束恢复流程：

1. **波束失败检测**：服务波束的 L1-RSRP 低于门限 beamFailureDetectionTimer 次
2. **候选波束识别**：从监测的 SSB/CSI-RS 中找到质量满足 candidateBeamRSList 的波束
3. **波束恢复请求（BFR）**：在专用 PRACH 资源上发送恢复请求
4. **基站响应**：通过 PDCCH 指示新的服务波束

波束恢复时延目标：< 50ms（从波束失败到恢复完成）。
""",
    },
    {
        "doc_id": "doc_004",
        "doc_name": "5G 网络切片架构",
        "content": """# 5G 网络切片架构

## 概述

网络切片（Network Slicing）是 5G 网络架构的革命性特征，允许在同一物理网络基础设施上创建多个逻辑隔离的端到端虚拟网络，每个网络切片可针对特定业务场景进行定制优化。

## 切片类型

### 标准切片类型（SST）

3GPP 定义了以下标准切片/服务类型：

| SST 值 | 名称 | 典型场景 | 关键指标 |
|--------|------|---------|---------|
| 1 | eMBB | 增强移动宽带 | 峰值速率 > 10 Gbps |
| 2 | URLLC | 超可靠低时延通信 | 时延 < 1ms, 可靠性 99.999% |
| 3 | MIoT | 大规模物联网 | 百万级连接密度 |
| 4 | V2X | 车联网 | 时延 < 10ms |

### S-NSSAI

S-NSSAI（Single Network Slice Selection Assistance Information）由 SST + SD 组成：
- SST（Slice/Service Type）：1 字节，标识切片类型
- SD（Slice Differentiator）：3 字节（可选），区分同类型的不同切片实例

## 切片架构

### RAN 切片

无线接入网侧的切片实现：
- **资源隔离**：通过 RRM（无线资源管理）策略实现 PRB 级别的资源隔离
- **QoS 保障**：基于 5QI（5G QoS Identifier）进行流级别的 QoS 管理
- **调度策略**：不同切片可配置不同的调度算法（如 URLLC 优先抢占）

### 核心网切片

核心网侧的切片实现基于 SBA（服务化架构）：
- **NF 共享/专用**：AMF、SMF、UPF 等网元可共享或切片专用
- **选择机制**：NSSF（Network Slice Selection Function）负责切片选择
- **隔离等级**：L1（共享 NF）、L2（专用 NF 实例）、L3（专用硬件）

## 切片管理

### 生命周期管理

切片的生命周期包括：
1. **准备阶段**：设计切片模板、定义 SLA 需求
2. **部署阶段**：实例化网络功能、配置资源编排
3. **运行阶段**：监控性能指标、动态资源调整
4. **退役阶段**：释放资源、数据归档

### NSSP（网络切片选择策略）

UE 侧通过 URSP 规则将业务流映射到对应切片：
- 基于 APP ID / IP 描述符 / 域名等条件匹配
- 优先级排序，首个匹配规则生效
""",
    },
    {
        "doc_id": "doc_005",
        "doc_name": "gNodeB 基站参数配置指南",
        "content": """# gNodeB 基站参数配置指南

## 基站型号概览

| 型号 | 频段 | 最大功率 | 天线阵列 | 适用场景 |
|------|------|---------|---------|---------|
| AAU5613 | n78 (3.5GHz) | 200W | 64T64R | 城区宏覆盖 |
| AAU5639 | n41 (2.6GHz) | 240W | 64T64R | 城区/郊区宏覆盖 |
| AAU5611 | n28 (700MHz) | 4x60W | 4T4R | 广覆盖补盲 |
| pRRU | n78 | 2x250mW | 2T2R | 室内分布 |
| LampSite | n78/n41 | 500mW | 4T4R | 室内热点 |

## 关键参数配置

### 小区级参数

```
# 小区标识
cellId = 0
pci = 128
trackingAreaCode = 12345

# 频率配置
absoluteFrequencySSB = 627264    # SSB 频点 (n78)
carrierBandwidth = 273           # 100MHz 对应 273 个 RB

# 功率配置
ss-PBCH-BlockPower = -10         # SSB 发射功率 (dBm/RE)
p-Max = 23                       # UE 最大发射功率

# 定时器配置
t300 = 1000                      # RRC 连接建立定时器 (ms)
t301 = 1000                      # RRC 连接重建定时器 (ms)
t310 = 1000                      # 无线链路失败检测定时器 (ms)
n310 = 1                         # 连续失步指示计数
n311 = 1                         # 连续同步指示计数
```

### RACH 参数配置

```
# 随机接入参数
prach-ConfigIndex = 160          # PRACH 配置索引
msg1-FrequencyStart = 0          # PRACH 频域起始 RB
msg1-FDM = 1                     # 频分复用数
preambleReceivedTargetPower = -100  # 前导码目标接收功率 (dBm)
preambleTransMax = 10            # 最大前导码发送次数
powerRampingStep = 2             # 功率爬升步长 (dB)
ra-ResponseWindow = 10           # RAR 窗口 (slots)
```

### 波束管理参数

```
# SSB 配置
ssb-PositionsInBurst = 0xFF      # 8 个 SSB 波束全部使能
ssb-periodicityServingCell = 20  # SSB 周期 20ms
beamFailureDetectionTimer = 10   # 波束失败检测定时器 (ms)
beamFailureRecoveryTimer = 50    # 波束恢复定时器 (ms)
```

## 典型告警处理

### ALM-26214：VSWR 过高告警

**原因分析**：天馈系统驻波比异常
**排查步骤**：
1. 检查天馈接头是否松动或进水
2. 使用天馈测试仪测量 VSWR 值
3. 正常值应 < 1.5，告警门限 2.0
4. 检查馈线弯曲半径是否符合规范

### ALM-26001：小区退服告警

**原因分析**：小区不可用
**排查步骤**：
1. 检查基带板和射频板工作状态
2. 检查 CPRI/eCPRI 光模块连接
3. 核实 License 是否过期
4. 查看操作日志是否有人为闭站
""",
    },
    {
        "doc_id": "doc_006",
        "doc_name": "RRC 连接建立信令流程",
        "content": """# RRC 连接建立信令流程

## 概述

RRC（Radio Resource Control）连接建立是 UE 从 RRC_IDLE 态转为 RRC_CONNECTED 态的过程。该流程是所有用户面数据传输和大部分控制面操作的前提。

## 详细信令流程

### 第一步：系统消息获取

UE 开机后首先进行小区搜索和选择：
1. PSS/SSS 检测：获取 PCI、帧定时
2. 解码 MIB（PBCH）：获取系统帧号、SSB 子载波偏移、PDCCH 配置
3. 解码 SIB1（PDSCH）：获取小区接入限制信息、公共配置参数
4. 根据需要获取其他 SIB（SIB2-SIB9）

### 第二步：随机接入

执行四步随机接入流程（参见随机接入流程文档）。

### 第三步：RRC Setup

gNodeB 向 UE 发送 RRCSetup 消息，包含：
- SRB1 配置（Signaling Radio Bearer 1）
- 初始 BWP 配置
- MAC 配置（BSR/PHR/SR 参数）
- 物理层配置

### 第四步：RRC Setup Complete

UE 向 gNodeB 回复 RRCSetupComplete，携带：
- 选择的 PLMN ID
- NAS 消息（如 Registration Request）
- 注册的 S-NSSAI（切片信息）

### 第五步：安全激活

1. gNodeB 向 UE 发送 SecurityModeCommand
2. UE 激活 AS 安全（加密 + 完整性保护）
3. UE 回复 SecurityModeComplete

### 第六步：RRC 重配置

gNodeB 通过 RRCReconfiguration 配置完整的用户面参数：
- DRB 配置（Data Radio Bearer）
- 测量配置（MeasConfig）
- SCell 配置（如需 CA）
- 波束管理参数

## 关键定时器

| 定时器 | 功能 | 典型值 |
|--------|------|--------|
| T300 | RRC 连接建立超时 | 1000 ms |
| T301 | RRC 连接重建超时 | 1000 ms |
| T310 | 无线链路失败检测 | 1000 ms |
| T311 | RRC 重建过程中小区选择 | 1000 ms |
| T304 | 切换执行超时 | 150 ms |

## 异常处理

### RRC 连接建立失败

如果 T300 超时且 RRC 连接建立未完成：
1. UE 通知高层连接建立失败
2. 释放 MAC 配置
3. 重新进入 RRC_IDLE 态
4. 在 T302 定时器超时前不允许再次发起连接建立（接入控制）
""",
    },
    {
        "doc_id": "doc_007",
        "doc_name": "HARQ 重传机制",
        "content": """# HARQ 重传机制

## 基本概念

HARQ（Hybrid Automatic Repeat Request，混合自动重传请求）结合了前向纠错编码（FEC）和自动重传请求（ARQ），是 5G NR 保障数据传输可靠性的核心机制。

## 工作原理

### 软合并

HARQ 的核心优势在于软合并（Soft Combining）：接收端将每次传输的数据保存在 HARQ 缓存中，将多次传输的信号在软比特级别进行合并，从而提升解码成功概率。

主要合并方式：
- **Chase Combining (CC)**：重传数据与初传完全相同，接收端做最大比合并（MRC）
- **Incremental Redundancy (IR)**：每次重传发送不同的冗余版本（RV），提供额外的编码增益

### HARQ 进程

NR 下行支持最多 16 个 HARQ 进程，上行最多 16 个：
- 每个 HARQ 进程独立运行
- 采用异步 HARQ（基站灵活调度重传时机）
- DCI 中携带 HARQ Process ID 和 NDI（New Data Indicator）

## 反馈机制

### ACK/NACK

UE 通过 PUCCH 或 PUSCH 反馈 HARQ-ACK 信息：
- ACK：解码成功
- NACK：解码失败，请求重传
- DTX：未检测到 PDCCH（不发送反馈）

### 定时关系

PDSCH 到 HARQ-ACK 的反馈定时由 K1 参数控制：
- K1 取值范围：{0, 1, 2, 3, 4, 5, 6, 7, 8} slots
- 通过 DCI 中的 PDSCH-to-HARQ_feedback_timing_indicator 指示

### CBG 级 HARQ

NR 引入了码块组（Code Block Group，CBG）级别的 HARQ 反馈：
- 一个传输块可分为多个 CBG
- 仅重传解码失败的 CBG，而非整个传输块
- 显著减少重传数据量，提升频谱效率

## 最大重传次数

最大重传次数由 maxNrofHARQ-Retransmissions 配置，典型值为 4。超过最大重传次数后：
1. HARQ 进程丢弃该数据
2. 由高层 RLC ARQ 机制负责恢复
3. 如果 RLC 也无法恢复，由 PDCP 层处理
""",
    },
    {
        "doc_id": "doc_008",
        "doc_name": "上行功率控制算法",
        "content": """# 上行功率控制算法

## 概述

上行功率控制（Uplink Power Control）是 5G NR 无线资源管理的关键机制，目标是在保证上行链路质量的同时，尽量降低 UE 发射功率，减少对邻小区的干扰。

## PUSCH 功率控制

### 开环功控公式

PUSCH 发射功率计算公式：

P_PUSCH = min(P_CMAX, P_0 + 10*log10(2^mu * M_RB) + alpha * PL + Delta_TF + f)

其中：
- P_CMAX：UE 最大允许发射功率（dBm）
- P_0：目标接收功率参数（dBm），由高层参数 p0-NominalWithoutGrant 配置
- mu：子载波间隔指数（0/1/2/3 对应 15/30/60/120 kHz）
- M_RB：分配的 RB 数
- alpha：路径损耗补偿因子（0/0.4/0.5/0.6/0.7/0.8/0.9/1.0）
- PL：下行路径损耗估计值（dB）
- Delta_TF：传输格式补偿
- f：闭环功控调整量

### 闭环功控

基站通过 DCI 中的 TPC（Transmit Power Control）命令动态调整 UE 发射功率：
- TPC 步长：{-1, 0, 1, 3} dB（累积模式）
- TPC 步长：{-4, -1, 1, 4} dB（绝对模式）
- 累积量 f 有上下限约束

## PUCCH 功率控制

PUCCH 发射功率公式：

P_PUCCH = min(P_CMAX, P_0_PUCCH + 10*log10(2^mu * M_RB) + PL + Delta_F + Delta_TF + g)

## SRS 功率控制

SRS（Sounding Reference Signal）的功率控制类似 PUSCH，但使用独立的参数集：
- P_0_SRS：SRS 目标功率
- alpha_SRS：SRS 路径损耗补偿因子

## 功控参数调优建议

| 场景 | alpha 建议值 | P_0 调整方向 |
|------|-------------|-------------|
| 城区密集部署 | 0.8 | 降低（减少干扰） |
| 郊区广覆盖 | 1.0 | 提高（保证覆盖） |
| 室内覆盖 | 0.6 | 降低（小区间距小） |
| 高速场景 | 0.9 | 适度提高 |
""",
    },
]


# ---- 预设测试查询 ----

TEST_QUERIES = [
    {
        "query": "5G随机接入的四步流程是什么？",
        "expected_doc": "doc_001",
        "description": "语义理解测试：应命中随机接入文档",
    },
    {
        "query": "CA是什么",
        "expected_doc": "doc_002",
        "description": "缩写扩展测试：CA -> 载波聚合",
    },
    {
        "query": "gNodeB AAU5613 的最大功率是多少",
        "expected_doc": "doc_005",
        "description": "精确参数匹配测试：BM25 优势场景",
    },
    {
        "query": "波束失败后怎么恢复",
        "expected_doc": "doc_003",
        "description": "语义理解测试：波束恢复流程",
    },
    {
        "query": "URLLC 切片的时延要求",
        "expected_doc": "doc_004",
        "description": "多维度匹配：术语 + 参数",
    },
    {
        "query": "HARQ 最多能重传几次",
        "expected_doc": "doc_007",
        "description": "精确事实查询",
    },
    {
        "query": "上行功率控制的计算公式",
        "expected_doc": "doc_008",
        "description": "技术细节查询",
    },
    {
        "query": "RRC连接建立需要几步",
        "expected_doc": "doc_006",
        "description": "流程类查询",
    },
]
