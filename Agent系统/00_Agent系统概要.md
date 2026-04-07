---
tags:
  - Agent
  - LLM
  - AI架构
status: active
---
# Agent 系统概要

## 系统骨架：三个嵌套闭环

理解这个 Agent 系统，最重要的是把它看成**三个运行在不同时间尺度上的嵌套闭环**，而不是一堆平铺的技术组件。

```
┌─────────────────────────────────────────────────────────┐
│  闭环 3：学习飞轮（离线，天/周级）                         │
│                                                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  闭环 2：安全围栏（请求级，秒-小时）               │    │
│  │                                                  │    │
│  │  ┌─────────────────────────────────────────┐    │    │
│  │  │  闭环 1：ReAct 推理循环（毫秒级）          │    │    │
│  │  │                                          │    │    │
│  │  │  Agent → Guardrail → Tool → Observation  │    │    │
│  │  │        ↑__________________________|     │    │    │
│  │  └─────────────────────────────────────────┘    │    │
│  │                                                  │    │
│  │  置信度熔断 / 高危拦截 → HITL → 人工修正后恢复    │    │
│  └─────────────────────────────────────────────────┘    │
│                                                          │
│  HITL 数据 → DPO/SFT → 模型更安全                       │
│  专家确认用例 → Embedding 入库 → RAG 更准确              │
└─────────────────────────────────────────────────────────┘
```

三个闭环各自负责不同的问题：
- **闭环 1** 解决"怎么完成一次任务"
- **闭环 2** 解决"怎么保证任务安全可控"
- **闭环 3** 解决"怎么让系统越用越聪明"

---

## 一、什么是 AI Agent

**定义**：AI Agent 是能够感知输入、在特定环境中自主采取行动以实现预期目标的系统。在大语言模型背景下，Agent 演变为**以 LLM 为大脑，具备规划、记忆、工具调用能力的自主决策系统**。

> **Agent = LLM（大脑）+ Planning（规划）+ Memory（记忆）+ Tools（工具）**

**Agent 与普通 LLM 调用的本质区别**：普通调用是单次 input → output 的无状态变换；Agent 是一个能自主决策"下一步做什么"的持续性控制循环（Control Loop），它能在目标达成前反复调用工具、感知结果、调整策略。

---

## 二、核心组成部分

### 1. 大脑（LLM）

LLM 担任系统的控制器，负责：
- 自然语言理解与意图解析
- 推理与逻辑规划
- 工具调用决策（什么时候、调用哪个工具、传什么参数）
- 最终答案生成

**模型选型思考**：在工业落地中，不同任务节点对模型能力要求差异悬殊。主干规划推理节点需要强大的 32B/72B 级模型（如 Qwen3-32B）；意图分类、Guardrail 检测、格式校验等单一子任务可以用经过 SFT 的 4B/7B 小模型替代，大幅降低延迟和成本。

### 2. 规划（Planning）

| 机制 | 描述 | 适用场景 |
|:---|:---|:---|
| **CoT（链式思考）** | 引导模型逐步推理，减少跳跃性错误 | 数学、逻辑推理 |
| **ReAct（推理+行动）** | `Thought → Action → Observation` 循环，边思考边执行 | 工具调用、多步骤任务 |
| **ToT（思维树）** | 在每步探索多个分支并评估优劣，支持回溯 | 复杂规划、游戏决策 |
| **Reflexion（反思）** | Agent 对自身过去行为进行批判性反思，生成改进计划 | 需要自我纠错的长任务 |

**5G 测试 Agent 采用的 ReAct 循环**：
```
用户输入（测试需求）
   ↓
[Thought] 需要先查询切换场景的测试用例
   ↓
[Action]  调用 Test_Case_Query 工具
   ↓
[Observation] 返回 42 条匹配用例
   ↓
[Thought] 用例已获取，触发仿真执行
   ↓
[Action]  调用 Simulation_Runner 工具
   ↓
...直到收集到足够证据，输出判定结论
```

### 3. 记忆（Memory）

```
记忆分类
├── 短期记忆（Short-term Memory）
│   └── 当前对话的上下文窗口（Context Window）
│       受限于 LLM 最大 Token 数（如 128K/1M）
│
└── 长期记忆（Long-term Memory）
    ├── 向量数据库（Milvus）
    │   └── 语义相似度检索，适合模糊知识回忆
    │   └── 存储：3GPP 协议文档、历史缺陷库、Golden Cases
    └── 结构化数据库（Postgres）
        └── 精确键值检索，适合状态存档（Checkpoint）
```

**记忆过载问题（Lost in the Middle）**：当上下文过长时，LLM 倾向于忽略中间段内容。工程解法：滑动窗口截断 + 定期 Summary 摘要压缩 + 长期记忆分片存入向量库。

### 4. 工具（Tools）

工具赋予 Agent 与真实世界交互的能力，弥补 LLM 的两大天然缺陷：**时效性不足**（训练数据截止）和**无法执行副作用**（数据库写入、API 调用）。

5G 测试 Agent 的核心工具链：

| 工具 | 职责 |
|:---|:---|
| `Test_Case_Query` | 按场景/频段/特性从知识库检索测试用例 |
| `Simulation_Runner` | 触发仿真平台执行测试，返回 session_id |
| `Metrics_Collector` | 通过 SSH/RPC 从网元实时拉取 KPI 日志 |
| `Baseline_Comparator` | 与历史安全版本做统计学对比（T-Test / KS-Test）|
| `Log_Analyzer` | 解析 RRC/NAS/PDCP 信令日志，识别协议级异常 |
| `Fleet_Manager` | 多局点并发探针集群调度（Multi-Agent 场景）|

---

## 三、核心设计模式与架构图

### 3.1 ReAct 单体 Agent 架构（基础模式）

```
┌─────────────────────────────────────────────────┐
│                   ReAct Agent                    │
│                                                  │
│  用户请求  ──►  ┌──────────┐                    │
│                │  LLM 推理  │ ◄── System Prompt  │
│                │  (Thought) │     + 工具描述      │
│                └─────┬─────┘                    │
│                      │                          │
│              ┌───────▼───────┐                  │
│              │ 是否需要工具?  │                  │
│              └───────────────┘                  │
│                 /           \                   │
│               是              否                │
│               ↓              ↓                  │
│        ┌──────────┐    最终输出给用户            │
│        │ 工具执行  │                             │
│        │(Action)  │                             │
│        └────┬─────┘                             │
│             │ Observation                       │
│             └──── 反馈给 LLM 继续推理 ──────►   │
└─────────────────────────────────────────────────┘
```

### 3.2 LangGraph 状态机架构（企业级实现）

LangGraph 将 Agent 流程抽象为**有向图 + 状态机**，是目前工业落地最可控的方案。本系统的图结构如下：

```mermaid
graph TD
    START((START)) --> agent_node["Agent 节点\n(LLM 推理决策)"]

    agent_node --> router{"条件路由\nshould_continue()"}

    router -- "有工具调用 & 连续错误未超限" --> guardrail_node["Guardrail 节点\n(安全检测)"]
    router -- "无工具调用 & 置信度正常" --> result_judge_node["结果判定节点\n(双轨判定)"]
    router -- "连续错误超限 / 置信度过低" --> hitl_node["HITL 节点\n(人工介入)"]

    guardrail_node --> guard_router{"after_guardrail()"}
    guard_router -- "无高危操作" --> tool_node["工具执行节点\n(Tool Node)"]
    guard_router -- "检测到高危操作" --> hitl_node

    tool_node -- "Observation 追加至 State" --> agent_node
    hitl_node -- "人工修正 + Approve" --> agent_node

    result_judge_node --> judge_router{"after_result_judge()"}
    judge_router -- "置信度 >= 0.65" --> END((END))
    judge_router -- "置信度 < 0.65" --> hitl_node

    style agent_node fill:#4a90d9,color:#fff
    style guardrail_node fill:#e67e22,color:#fff
    style tool_node fill:#5cb85c,color:#fff
    style hitl_node fill:#f0ad4e,color:#fff
    style result_judge_node fill:#9b59b6,color:#fff
```

**Guardrail 节点**是这个图与教科书 ReAct 的最大区别：工具调用在执行前必须经过安全检测，而不是直接执行。这一节点拦截包含高危关键词（`format_c`、`reset_all`、`force_reboot` 等）的工具参数，防止幻觉生成的危险指令直接下发。

**核心 State 设计**：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（自动追加）
    current_step: str                        # 当前执行阶段
    tool_outputs: dict                       # 工具返回缓存，供 result_judge 使用
    error_count: int                         # 连续工具错误计数（软熔断）
    confidence_score: float                  # 当前决策置信度
    hitl_required: bool                      # 是否需要人工介入
    hitl_feedback: str                       # 人工审核后的反馈文本
    final_result: Optional[str]              # 最终判定结果（JSON 字符串）
```

### 3.3 5G 测试验证 Agent 完整系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    5G 智能测试验证 Agent 系统                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   用户 / CI 系统                                                     │
│       │ 测试需求（自然语言 / YAML）                                   │
│       ▼                                                             │
│   FastAPI 接口层  ──SSE推流──► 前端实时状态展示                       │
│       │                                                             │
│       ▼                                                             │
│   ┌─────────────────────────────────────────┐                      │
│   │           LangGraph 状态机引擎           │                      │
│   │                                         │                      │
│   │  [Agent 节点] ──► [Guardrail 节点]       │                      │
│   │      ↑                  │               │                      │
│   │      │           安全/高危              │                      │
│   │      │                  ↓    ↓          │                      │
│   │      │          [Tool 节点] [HITL 节点] │                      │
│   │      │______________↑       │           │                      │
│   │                             ↓           │                      │
│   │                      [结果判定节点]      │                      │
│   │                      统计轨 + 语义轨     │                      │
│   └─────────────────────────────────────────┘                      │
│                                                                     │
├──────────────────── 知识层（RAG）─────────────────────────────────-─┤
│                                                                     │
│  Milvus（语义向量）  Elasticsearch（BM25）  Cross-Encoder Reranker   │
│  3GPP 协议文档  |  历史 Bugzilla 缺陷库  |  专家确认 Golden Cases    │
│                                                                     │
├──────────────────── 工具层（Tool Layer）───────────────────────────-┤
│                                                                     │
│  Test_Case_Query  Simulation_Runner  Metrics_Collector              │
│  Baseline_Comparator  Log_Analyzer  Fleet_Manager                   │
│                                                                     │
├──────────────────── 基础设施层 ──────────────────────────────────-──┤
│                                                                     │
│  Postgres（Checkpoint）  Redis（缓存）  Kafka（异步消息）             │
│  Celery（任务队列）  vLLM（推理服务）  Qwen3-32B（主干模型）          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**RAG 在 Agent 中的定位**：知识层不是独立系统，而是 Agent 的"长期记忆"输入。每次用例生成时，Agent 主动从 Milvus/ES 检索相关协议规范和历史缺陷，作为 RAG Context 注入 Prompt，保证生成内容有协议依据、避免重蹈历史漏测覆辙。

---

## 四、关键工程问题与解决方案

### 4.1 双重熔断机制（防死循环 + 防幻觉）

Agent 在复杂任务中极易陷入无效循环或产生幻觉递归，需要两层防护：

| 熔断层级 | 触发条件 | 机制 | 处理方式 |
|:---|:---|:---|:---|
| **硬熔断（第一层）** | 图迭代深度超过阈值（如 15 跳） | LangGraph `recursion_limit` | 强制挂起，上报异常 |
| **软熔断（第二层）** | 连续工具错误 ≥ 3 次，或置信度 < 0.65 | State 中 `error_count` / `confidence_score` | 路由至 HITL 人工介入 |

**置信度阈值的确定**：不是拍脑袋，而是通过大量压测找到相变点。在 5G 测试项目中，0.65 是关键阈值——低于此值时 Agent 自主执行的错误率急剧上升。

### 4.2 Human-in-the-Loop（HITL）工程实现

HITL 是 Agent 从"玩具"走向"工业生产力"的核心桥梁，工程难点在于：**如何在不阻塞服务器线程的前提下等待人工审批**？

```
传统方式（有问题）：
While 循环等待 ──► 线程持续阻塞 ──► 内存泄漏 / HTTP 504 超时

LangGraph 方式（正确）：
触发高危预警 / 低置信度
   │
   ▼
LangGraph 将 State 快照序列化至 Postgres（interrupt_before 钩子）
   │
   ▼
释放计算资源（进程终止，显存释放）
   │
   ▼
Celery 异步触发钉钉告警 Webhook
   │
   ▼
专家在 Web 前端 Review，修改参数，点击 Approve
   │
   ▼
API 唤醒 → 从 Postgres 拉取 Thread_ID 对应的 State
   │
   ▼
将修改后的参数注入 State，调用 .resume() 恢复执行
```

**HITL 解决的三类业务问题**：
1. **灾难性环境摧毁防护**：拦截包含高危写权限信令的用例（如 `format_c`、`reset_all`）
2. **模糊边界判定**：置信度处于灰度区间（0.60~0.65）时交由专家会诊
3. **新奇用例确权**：LLM 生成的 Novel Case（与知识库相似度极低），需专家鉴别价值后决定是否执行

### 4.3 RAG 与 Agent 的结合（知识增强）

Agent 中的 RAG 不仅是检索问答，更是**用例生成的背景知识供给**和**历史缺陷的防覆辙机制**。

```
知识库构成：
├── 3GPP 协议文档（通信规范，BM25 精确匹配专有名词）
├── 历史 Bugzilla 缺陷库（向量语义检索，补全边界场景）
└── 人工确认的 Golden Cases（经过专家背书的高质量用例）

混合检索策略：
ES(BM25) 精确匹配  ──┐
                      ├──► 自研融合算法 ──► Cross-Encoder Rerank ──► Top-K Context
Milvus 语义检索    ──┘

数据飞轮（Data Flywheel）：
新用例生成 → HITL 专家审核 → 通过后自动 Embedding 入库
         → 系统越用越聪明，知识边界自我拓展
```

### 4.4 结果判定双轨机制

单纯依赖 LLM 判定存在幻觉风险，单纯依赖统计规则又无法处理复杂信令异常：

```
测试执行完成
      │
      ▼
┌─────────────────────────────────┐
│           双轨并行判定           │
├─────────────────┬───────────────┤
│   统计轨（硬规则）│   语义轨（LLM）│
│                 │               │
│ KPI 均值/方差   │ PCAP 信令解析  │
│ T-Test / KS-Test│ 日志语义理解   │
│ 阈值包络策略    │ 根因关联推理   │
└────────┬────────┴───────┬───────┘
         │                │
         ▼                ▼
    明显违规: FAIL    隐性异常: 详细诊断
         │                │
         └───── 综合判定 ──┘
                  │
         ┌────────▼────────┐
         │  置信度评分输出  │
         │ ≥ 0.65: 自动结论 │
         │ < 0.65: HITL    │
         └─────────────────┘
```

---

## 五、常见挑战与系统性对策

| 挑战 | 具体表现 | 工程对策 |
|:---|:---|:---|
| **无限循环与死锁** | `Action A → Obs A → Action A` 死循环 | 双重熔断（硬迭代上限 + 软熔断路由）|
| **幻觉与工具误用** | 捏造工具参数，生成高危指令 | Guardrail 节点（关键词 + 轻量分类器）+ Function Calling 结构化约束 |
| **记忆过载** | 超出 Context Window，中间信息丢失 | 滑动窗口 + Summary 摘要压缩 + 长期记忆向量化 |
| **高延迟高成本** | 多轮 LLM 调用 + 超长 Prompt | 并行节点执行 + 小模型替代非核心节点 + 流式推送 |
| **危险指令生成** | 幻觉生成带高危参数的测试用例 | Guardrail 双缝检测 + HITL 拦截 + DPO 安全对齐 |
| **评估难** | 长链路无单一准确率指标 | 四维评估矩阵（格式/工具/轨迹/安全）+ LLM-as-Judge + 沙盒执行 |

---

## 六、主流架构分类

### 6.1 单智能体（Single-Agent）

适用于流程相对固定、任务类型单一的场景。5G 测试 Agent 的核心链路（用例生成→执行→判定）即采用单体架构。

```
用户 → [单一 LLM Agent] → 工具调用 → 结果输出
```

### 6.2 多智能体协同（Multi-Agent）

当任务包含多个独立可并行的子任务时，引入 Multi-Agent。

```
                    ┌────────────────────────┐
                    │   Supervisor Agent      │
                    │   （任务分配与协调）     │
                    └────────────────────────┘
                        /        |        \
                       ↓         ↓         ↓
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │ Worker 1 │ │ Worker 2│ │ Worker 3│
              │ RRC 分析 │ │ NAS 分析│ │ KPI 分析│
              └─────────┘ └─────────┘ └─────────┘
                 │              │           │
                 └──────────────┴───────────┘
                       汇总节点（Reduce）
                       → 综合诊断报告
```

**5G 测试中的 Map-Reduce 模式**：主控 Agent 将 GB 级日志分发给多个 Worker Agent 并行解析（RRC / NAS / KPI 各一路），最后汇总输出综合报告。判断是否引入 Multi-Agent 的标准：**当单任务有多个独立可并行的子任务时才引入，不为架构而架构**。

### 6.3 Workflow 思维（实用主义）

Andrew Ng 提出的四大工作流设计模式，在工业落地中往往比完全自主的 Agent 更具实效：

1. **Reflection（反思）**：让 Agent 对自身输出进行批评和改进
2. **Tool Use（工具调用）**：赋予模型与外部系统交互的能力
3. **Planning（规划）**：将复杂任务分解为可执行的子步骤序列
4. **Multi-Agent Collaboration（多智能体协作）**：不同角色并行协同

> 明确的状态流转比让模型自由发挥更可控、更稳定。Workflow 优先，自主 Agent 其次。

---

## 七、模型后训练闭环（SFT + DPO）

通用基座模型在垂直领域面临三个核心问题：**领域术语理解不足**、**结构化格式输出不稳定**、**危险逻辑幻觉**。后训练是从"可用"走向"高可靠"的必经之路。

### 7.1 SFT 监督微调（解决格式与术语）

```
数据来源：
├── Self-Instruct / Magpie 合成数据流水线（含失败案例增强容错性）
├── HITL 沉淀的 Golden Dataset（专家确认的高质量样本）
└── 3GPP 协议注入的领域知识对话

技术方案（32B 量级）：
DeepSpeed ZeRO-3 + LoRA（BF16 全精度，无量化）
LoRA 插入位置：q_proj, v_proj, o_proj, gate_proj, up_proj
超参数：r=64, α=128, dropout=0.05, lr=5e-5 (Cosine 退火)
硬件：8×A100 80G，FlashAttention-2 + Gradient Checkpointing
说明：算力充足时选 LoRA 而非 QLoRA，避免 nf4 量化误差影响工具调用格式精确性
```

### 7.2 DPO 偏好对齐（抑制危险幻觉）

HITL 组件是天然的 DPO 数据采集器：

```
HITL 拦截流程          →   DPO 数据生成
─────────────────────────────────────────
高危用例被专家打回      →   Y_rejected（负样本）
专家修改后通过         →   Y_chosen（正样本）

三元组: (Prompt, Y_chosen, Y_rejected)
           ↓
        DPO 训练
           ↓
      合法信令↑概率，危险词汇↓权重
```

**DPO 相比 RLHF 的优势**：省去独立 Reward Model 的训练复杂度，直接通过偏好对数据进行梯度对齐，工程实现更轻量。

---

## 八、Agent 系统评测体系

Agent 评估不能简单用准确率衡量，需要**四维矩阵式评估**：

| 维度 | 指标 | 评估方法 |
|:---|:---|:---|
| **格式合规** | 指令遵从率（JSON Schema 合法性）| pytest + Pydantic Validators |
| **工具调用** | 工具选择准确率 + 参数 F1-Score | 与 Ground Truth 轨迹对比 |
| **轨迹效率** | 成功率 + 步数效率比（Agent步数/专家步数）| Golden Trajectories 对比 |
| **安全合规** | 高危场景阻断率（生产网必须 100%）| 毒药测试集压测 |

**三层评估技术栈**：
1. **静态断言层**：pytest + Ragas/DeepEval 量化检索精度（Context Precision、Answer Relevance）
2. **LLM-as-Judge 层**：Prometheus-Eval 或 GPT-4o 对长文本报告进行自动语义评分
3. **沙盒执行层**（最关键）：Agent 生成的用例直接在 Docker 隔离环境中运行，以执行结果而非文本描述为最终判定依据

**关键指标基准**（基于 1258 题标准评测集）：
- 任务完成率：88%
- 判定准确率：94%
- 高危场景阻断率：100%（生产网硬性要求）

---

## 九、工具输入输出完整定义

所有工具均基于 LangChain `StructuredTool` + Pydantic 模型定义，LLM 通过 Function Calling 接口获取工具的 JSON Schema，推理后生成结构化参数。

### 9.1 Test_Case_Query（用例检索）

```python
# ---- 输入 Schema ----
class TestCaseQueryInput(BaseModel):
    scenario: str          # 测试场景名称，如 "handover", "idle_mode_reselection"
    band: str              # 频段，如 "n78_100MHz", "n41_60MHz"
    feature_tags: list[str] # 特性标签，如 ["mobility", "beamforming"]
    top_k: int = 10        # 最大返回用例数，默认 10

# ---- 输出 Schema ----
class TestCaseQueryOutput(BaseModel):
    cases: list[TestCase]  # 匹配用例列表
    total_found: int        # 库中总命中数（用于判断检索质量）
    retrieval_scores: list[float]  # 每条用例的相关性得分（0~1）

class TestCase(BaseModel):
    case_id: str           # 唯一标识，如 "TC_HO_NR_001"
    title: str             # 用例标题
    steps: list[str]       # 执行步骤
    expected_kpi: dict     # 期望 KPI，如 {"ho_success_rate": ">= 0.98"}
    source: str            # 来源：3gpp / bugzilla / golden
```

**工具描述（注入 System Prompt 的 Schema 片段）**：
```
Tool: Test_Case_Query
Description: 从知识库检索匹配指定场景和频段的测试用例。
             当需要获取测试用例时调用，不可凭空生成用例。
Parameters:
  - scenario (str, required): 测试场景关键字
  - band (str, required): 频段规格
  - feature_tags (list[str], optional): 细化特性筛选
  - top_k (int, optional): 返回条数上限，默认 10
```

---

### 9.2 Simulation_Runner（仿真执行）

```python
# ---- 输入 Schema ----
class SimulationRunnerInput(BaseModel):
    case_ids: list[str]     # 待执行的用例 ID 列表
    env_config: dict        # 仿真环境参数，如 {"ue_count": 32, "speed_kmh": 120}
    timeout_seconds: int = 300  # 单用例超时时间

# ---- 输出 Schema ----
class SimulationRunnerOutput(BaseModel):
    session_id: str         # 异步任务 ID，后续 Metrics_Collector 凭此轮询
    status: Literal["queued", "running", "failed"]
    estimated_duration_s: int   # 预估完成时间（秒）
    error_msg: str | None   # 若 failed 则携带错误原因
```

> 注意：Simulation_Runner 是**异步工具**，只返回 session_id，不等待执行完毕。  
> Agent 需要在下一轮 Action 中调用 Metrics_Collector 轮询结果。

---

### 9.3 Metrics_Collector（指标采集）

```python
# ---- 输入 Schema ----
class MetricsCollectorInput(BaseModel):
    session_id: str         # Simulation_Runner 返回的任务 ID
    metric_names: list[str] # 需要采集的指标，如 ["ho_success_rate", "rlf_count"]
    poll_interval_s: int = 5    # 轮询间隔（工具内部重试逻辑）
    max_wait_s: int = 600   # 最大等待时间

# ---- 输出 Schema ----
class MetricsCollectorOutput(BaseModel):
    session_id: str
    status: Literal["running", "completed", "timeout"]
    metrics: dict[str, float]   # 指标名 → 实测值，如 {"ho_success_rate": 0.976}
    raw_log_path: str           # 原始日志在共享存储的路径
    collection_time: str        # ISO8601 时间戳
```

---

### 9.4 Baseline_Comparator（基线对比）

```python
# ---- 输入 Schema ----
class BaselineComparatorInput(BaseModel):
    session_id: str             # 当前测试会话
    baseline_version: str       # 对比基准版本，如 "v2.3.1_golden"
    metrics: dict[str, float]   # 当前实测值（来自 Metrics_Collector 的输出）
    test_method: Literal["t_test", "ks_test", "envelope"] = "t_test"

# ---- 输出 Schema ----
class BaselineComparatorOutput(BaseModel):
    verdict: Literal["PASS", "FAIL", "INCONCLUSIVE"]
    p_value: float | None       # T-Test/KS-Test 显著性检验结果
    delta_summary: dict         # 每项指标与基线的差值和百分比
    regression_flags: list[str] # 检测到的回归项，如 ["ho_success_rate 下降 3.2%"]
    confidence: float           # 本判定的置信度（0~1）
```

---

### 9.5 Log_Analyzer（日志解析）

```python
# ---- 输入 Schema ----
class LogAnalyzerInput(BaseModel):
    log_path: str               # 原始日志文件路径（来自 Metrics_Collector.raw_log_path）
    analysis_type: Literal["rrc", "nas", "pdcp", "full"]
    anomaly_keywords: list[str] = []  # 额外关注的异常关键词

# ---- 输出 Schema ----
class LogAnalyzerOutput(BaseModel):
    anomalies: list[Anomaly]    # 发现的异常事件列表
    root_cause_hypothesis: str  # LLM 辅助生成的根因假设（自然语言）
    severity: Literal["INFO", "WARNING", "CRITICAL"]
    relevant_3gpp_clause: str | None  # 关联的 3GPP 条款，如 "TS 38.300 §9.2.3"

class Anomaly(BaseModel):
    timestamp: str
    event_type: str             # 如 "RLF", "HO_FAILURE", "T310_EXPIRY"
    ue_id: str
    description: str
```

---

### 9.6 Fleet_Manager（多局点调度）

```python
# ---- 输入 Schema ----
class FleetManagerInput(BaseModel):
    task_type: Literal["dispatch", "status_query", "abort"]
    worker_ids: list[str] | None  # dispatch 时指定 Worker，None 则自动选择
    subtasks: list[dict] | None   # dispatch 时的子任务列表
    session_id: str | None        # status_query / abort 时使用

# ---- 输出 Schema ----
class FleetManagerOutput(BaseModel):
    dispatched_sessions: list[str]  # 每个 Worker 的 session_id
    worker_allocation: dict[str, str]  # worker_id → session_id 映射
    status_summary: dict | None     # status_query 时返回各 Worker 状态
    aborted_count: int | None       # abort 时返回终止数量
```

---

## 十、完整 Agent 调用模拟

**测试场景**：验证 5G NR n78 频段 100MHz 切换（Handover）场景的掉线率是否满足 3GPP TS 38.300 规范。

**初始用户输入**：
```
"验证 n78_100MHz 频段切换场景的掉线率，与 v2.3.1_golden 基线对比，生成判定报告。"
```

---

### Step 0：初始状态

```python
# ===== AgentState 初始值 =====
state = AgentState(
    messages=[
        SystemMessage(content="你是一个 5G 测试验证 Agent，负责自动执行测试、分析结果并给出判定。\n可用工具：[Test_Case_Query, Simulation_Runner, Metrics_Collector, Baseline_Comparator, Log_Analyzer]"),
        HumanMessage(content="验证 n78_100MHz 频段切换场景的掉线率，与 v2.3.1_golden 基线对比，生成判定报告。")
    ],
    current_step="init",
    tool_outputs={},
    error_count=0,
    confidence_score=1.0,
    hitl_required=False,
    hitl_feedback="",
    final_result=None
)

# messages 长度：2（System + Human）
```

---

### Step 1：Agent 节点第一次推理

**LLM 输入（送入推理的 messages）**：
```
[SystemMessage] 你是一个 5G 测试验证 Agent...（含工具 Schema）
[HumanMessage]  验证 n78_100MHz 频段切换场景的掉线率...
```

**LLM 输出（AIMessage with tool_calls）**：
```python
AIMessage(
    content="",   # 有工具调用时 content 为空
    tool_calls=[{
        "id": "call_001",
        "name": "Test_Case_Query",
        "args": {
            "scenario": "handover",
            "band": "n78_100MHz",
            "feature_tags": ["mobility"],
            "top_k": 10
        }
    }]
)
```

**State 变化**：
```python
state["messages"].append(AIMessage(...))   # +1 条 AIMessage
state["current_step"] = "tool_call"
# messages 长度：3
```

---

### Step 2：Guardrail 节点检查

**输入**：读取 `state["messages"][-1].tool_calls`，提取工具名和参数

**检查逻辑**：
```python
dangerous_keywords = ["format_c", "reset_all", "force_reboot", "flash_fw"]
args_str = json.dumps({"scenario": "handover", "band": "n78_100MHz", ...})
# 无命中 → 安全
```

**输出**：路由至 Tool 节点，**State 无写入变化**

---

### Step 3：Tool 节点执行 Test_Case_Query

**工具调用**：
```python
# 输入（来自 LLM tool_calls.args）
input = TestCaseQueryInput(
    scenario="handover",
    band="n78_100MHz",
    feature_tags=["mobility"],
    top_k=10
)

# 输出（工具实际返回）
output = TestCaseQueryOutput(
    cases=[
        TestCase(case_id="TC_HO_NR_001", title="NR 同频切换基础成功率",
                 steps=["配置源/目标小区...", "触发 A3 事件...", "统计 HO 成功率"],
                 expected_kpi={"ho_success_rate": ">= 0.98"}, source="3gpp"),
        TestCase(case_id="TC_HO_NR_007", title="高速移动切换掉线率",
                 steps=["设置 UE 速度 120km/h...", "执行切换序列..."],
                 expected_kpi={"rlf_rate": "<= 0.02"}, source="golden"),
        # ...共 8 条
    ],
    total_found=8,
    retrieval_scores=[0.94, 0.91, 0.88, ...]
)
```

**State 变化**：
```python
# ToolMessage 追加至 messages
state["messages"].append(
    ToolMessage(
        tool_call_id="call_001",
        content=json.dumps(output.dict()),  # 序列化工具结果
        name="Test_Case_Query"
    )
)
state["tool_outputs"]["Test_Case_Query"] = output.dict()
# messages 长度：4（System + Human + AI + Tool）
```

---

### Step 4：Agent 节点第二次推理

**LLM 输入（完整 messages 历史）**：
```
[SystemMessage]  工具描述 + System Prompt
[HumanMessage]   验证 n78_100MHz 切换场景...
[AIMessage]      tool_calls: [Test_Case_Query(handover, n78_100MHz)]
[ToolMessage]    {"cases": [...8条用例...], "total_found": 8}
```

**LLM 推理（Thought）**：
> 已获取 8 条用例，其中 TC_HO_NR_001 和 TC_HO_NR_007 最相关。
> 下一步需要触发仿真执行这两条用例。

**LLM 输出**：
```python
AIMessage(
    content="",
    tool_calls=[{
        "id": "call_002",
        "name": "Simulation_Runner",
        "args": {
            "case_ids": ["TC_HO_NR_001", "TC_HO_NR_007"],
            "env_config": {"ue_count": 32, "speed_kmh": 120},
            "timeout_seconds": 300
        }
    }]
)
```

**State 变化**：
```python
state["messages"].append(AIMessage(...))   # +1 条
# messages 长度：5
```

---

### Step 5：Guardrail + Simulation_Runner 执行

**Guardrail 检查**：参数无危险关键词，放行

**工具调用**：
```python
# 输入
input = SimulationRunnerInput(
    case_ids=["TC_HO_NR_001", "TC_HO_NR_007"],
    env_config={"ue_count": 32, "speed_kmh": 120},
    timeout_seconds=300
)

# 输出（异步，只返回 session_id）
output = SimulationRunnerOutput(
    session_id="sim_20240406_a3f2c1",
    status="queued",
    estimated_duration_s=180,
    error_msg=None
)
```

**State 变化**：
```python
state["messages"].append(
    ToolMessage(tool_call_id="call_002",
                content=json.dumps(output.dict()),
                name="Simulation_Runner")
)
state["tool_outputs"]["Simulation_Runner"] = output.dict()
# messages 长度：6
```

---

### Step 6：Agent 节点第三次推理

**LLM 输入（6 条 messages）**：完整历史，包含 session_id

**LLM 输出**：
```python
AIMessage(
    content="",
    tool_calls=[{
        "id": "call_003",
        "name": "Metrics_Collector",
        "args": {
            "session_id": "sim_20240406_a3f2c1",
            "metric_names": ["ho_success_rate", "rlf_count", "rlf_rate"],
            "max_wait_s": 600
        }
    }]
)
```

**State 变化**：messages 长度 → 7

---

### Step 7：Metrics_Collector 执行

**工具调用**（工具内部轮询约 3 分钟后仿真完成）：
```python
# 输入
input = MetricsCollectorInput(
    session_id="sim_20240406_a3f2c1",
    metric_names=["ho_success_rate", "rlf_count", "rlf_rate"],
    max_wait_s=600
)

# 输出（仿真已完成）
output = MetricsCollectorOutput(
    session_id="sim_20240406_a3f2c1",
    status="completed",
    metrics={
        "ho_success_rate": 0.963,   # 低于期望值 0.98，需要关注
        "rlf_count": 7,
        "rlf_rate": 0.021           # 略超 0.02 阈值
    },
    raw_log_path="/logs/sim_20240406_a3f2c1/full_trace.pcap",
    collection_time="2024-04-06T14:32:11Z"
)
```

**State 变化**：
```python
state["messages"].append(ToolMessage(...))
state["tool_outputs"]["Metrics_Collector"] = output.dict()
# messages 长度：8
```

---

### Step 8：Agent 节点第四次推理

**LLM 推理（Thought）**：
> ho_success_rate=0.963 低于期望 0.98；rlf_rate=0.021 超过 0.02 阈值。
> 需要与基线对比确认是否为回归，并解析日志找根因。
> 并行调用 Baseline_Comparator 和 Log_Analyzer。

**LLM 输出（并行工具调用）**：
```python
AIMessage(
    content="",
    tool_calls=[
        {
            "id": "call_004",
            "name": "Baseline_Comparator",
            "args": {
                "session_id": "sim_20240406_a3f2c1",
                "baseline_version": "v2.3.1_golden",
                "metrics": {"ho_success_rate": 0.963, "rlf_rate": 0.021},
                "test_method": "t_test"
            }
        },
        {
            "id": "call_005",
            "name": "Log_Analyzer",
            "args": {
                "log_path": "/logs/sim_20240406_a3f2c1/full_trace.pcap",
                "analysis_type": "rrc",
                "anomaly_keywords": ["T310", "RLF", "HO_FAILURE"]
            }
        }
    ]
)
```

**State 变化**：messages 长度 → 9

---

### Step 9：并行执行 Baseline_Comparator + Log_Analyzer

**Baseline_Comparator 输入/输出**：
```python
# 输出
output_bc = BaselineComparatorOutput(
    verdict="FAIL",
    p_value=0.031,          # p < 0.05，统计显著回归
    delta_summary={
        "ho_success_rate": {"delta": -0.017, "percent": "-1.7%"},
        "rlf_rate":         {"delta": +0.003, "percent": "+16.7%"}
    },
    regression_flags=["ho_success_rate 显著下降 1.7%", "rlf_rate 超基线 16.7%"],
    confidence=0.82
)
```

**Log_Analyzer 输入/输出**：
```python
# 输出
output_la = LogAnalyzerOutput(
    anomalies=[
        Anomaly(timestamp="14:28:03.441", event_type="T310_EXPIRY",
                ue_id="UE_015", description="T310 计时器超时触发 RLF，A3 事件上报延迟 47ms"),
        Anomaly(timestamp="14:29:11.882", event_type="HO_FAILURE",
                ue_id="UE_023", description="切换准备阶段目标小区资源分配失败"),
    ],
    root_cause_hypothesis="T310 定时器参数（当前 1000ms）过短，高速移动场景下 A3 测量上报延迟导致虚假 RLF 触发",
    severity="WARNING",
    relevant_3gpp_clause="TS 38.331 §5.3.10.3"
)
```

**State 变化**：
```python
# 两个 ToolMessage 追加
state["messages"].append(ToolMessage(tool_call_id="call_004", ...))  # BC 结果
state["messages"].append(ToolMessage(tool_call_id="call_005", ...))  # LA 结果
state["tool_outputs"]["Baseline_Comparator"] = output_bc.dict()
state["tool_outputs"]["Log_Analyzer"] = output_la.dict()
# messages 长度：11
```

---

### Step 10：Agent 节点第五次推理（生成最终结论）

**LLM 输入（11 条 messages）**：完整推理链路

**LLM 推理（Thought）**：
> 统计轨：p=0.031 确认显著回归，verdict=FAIL
> 语义轨：根因是 T310 参数过短（1000ms），参考 TS 38.331 §5.3.10.3
> 综合置信度 0.82 ≥ 0.65，可以自动输出结论，无需 HITL

**LLM 输出（无工具调用，生成最终答案）**：
```python
AIMessage(
    content="""
## 测试判定报告

**判定结论：FAIL（回归确认）**
**置信度：0.82**

### 问题指标
| 指标 | 实测值 | 基线值 | 变化 | 状态 |
|---|---|---|---|---|
| ho_success_rate | 0.963 | 0.980 | -1.7% | 回归 |
| rlf_rate | 0.021 | 0.018 | +16.7% | 回归 |

### 根因分析
T310 计时器参数（当前 1000ms）过短。
在 120km/h 高速移动场景下，A3 测量上报出现 47ms 以上延迟，
导致 T310 超时误判为 RLF，引发不必要的无线链路失败流程。
参考标准：3GPP TS 38.331 §5.3.10.3

### 建议措施
1. 将 T310 定时器调整为 1500~2000ms（高速场景优化值）
2. 复核 A3 事件 TTT（Time-To-Trigger）配置是否合理
""",
    tool_calls=[]
)
```

**State 变化**：
```python
state["messages"].append(AIMessage(content="## 测试判定报告..."))
state["current_step"] = "result_judge"
state["confidence_score"] = 0.82
# messages 长度：12
```

---

### Step 11：结果判定节点

**检查逻辑**：
```python
confidence = state["confidence_score"]  # 0.82
if confidence >= 0.65:
    state["final_result"] = json.dumps({
        "verdict": "FAIL",
        "confidence": 0.82,
        "report": state["messages"][-1].content
    })
    # 路由 → END
else:
    state["hitl_required"] = True
    # 路由 → HITL 节点
```

---

### 完整 State 演变汇总

```
时刻        current_step      messages数  tool_outputs keys        confidence  hitl
─────────────────────────────────────────────────────────────────────────────────
Step 0      init              2           {}                        1.0         False
Step 1      tool_call         3           {}                        1.0         False
Step 3      tool_call         4           {TCQ}                     1.0         False
Step 4      tool_call         5           {TCQ}                     1.0         False
Step 5      tool_call         6           {TCQ, SR}                 1.0         False
Step 6      tool_call         7           {TCQ, SR}                 1.0         False
Step 7      tool_call         8           {TCQ, SR, MC}             1.0         False
Step 8      tool_call         9           {TCQ, SR, MC}             1.0         False
Step 9      tool_call         11          {TCQ, SR, MC, BC, LA}     1.0         False
Step 10     result_judge      12          {TCQ, SR, MC, BC, LA}     0.82        False
Step 11     END               12          {TCQ, SR, MC, BC, LA}     0.82        False
```

TCQ=Test_Case_Query, SR=Simulation_Runner, MC=Metrics_Collector, BC=Baseline_Comparator, LA=Log_Analyzer

### messages 完整序列（最终状态）

```
[0]  SystemMessage   — 系统角色定义 + 工具 Schema
[1]  HumanMessage    — 用户原始请求
[2]  AIMessage       — tool_calls: [Test_Case_Query]
[3]  ToolMessage     — Test_Case_Query 返回 8 条用例
[4]  AIMessage       — tool_calls: [Simulation_Runner]
[5]  ToolMessage     — Simulation_Runner 返回 session_id
[6]  AIMessage       — tool_calls: [Metrics_Collector]
[7]  ToolMessage     — Metrics_Collector 返回实测指标
[8]  AIMessage       — tool_calls: [Baseline_Comparator, Log_Analyzer] 并行
[9]  ToolMessage     — Baseline_Comparator 返回 FAIL + 统计结果
[10] ToolMessage     — Log_Analyzer 返回根因假设
[11] AIMessage       — 最终判定报告（无 tool_calls，直接输出文本）
```

---

## 十一、进阶知识体系

- [[LangChain]]：掌握工具抽象、AgentExecutor 机制、LCEL 链式组合
- [[LangGraph]]：复杂控制流的核心，State Machine + Checkpoint + HITL 完整实现
- [[主流Agent框架对比]]：LlamaIndex / MetaGPT / AutoGen / CrewAI / Dify 选型思路
- [[Function_Calling与工具箱]]：结构化工具调用的底层实现细节
- [[Memory机制与向量库接入]]：短期/长期记忆的工程化接入方案
- [[02_Agent技术报告]]：5G 测试验证 Agent 完整技术深度报告（面试专用）
