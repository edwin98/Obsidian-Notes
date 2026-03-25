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
DeepSpeed ZeRO-3 + QLoRA (nf4 量化)
LoRA 插入位置：q_proj, v_proj, o_proj, gate_proj, up_proj
超参数：r=64, α=128, dropout=0.05, lr=5e-5 (Cosine 退火)
硬件：8×A100 80G，FlashAttention-2 + Gradient Checkpointing
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

## 九、进阶知识体系

- [[LangChain]]：掌握工具抽象、AgentExecutor 机制、LCEL 链式组合
- [[LangGraph]]：复杂控制流的核心，State Machine + Checkpoint + HITL 完整实现
- [[主流Agent框架对比]]：LlamaIndex / MetaGPT / AutoGen / CrewAI / Dify 选型思路
- [[Function_Calling与工具箱]]：结构化工具调用的底层实现细节
- [[Memory机制与向量库接入]]：短期/长期记忆的工程化接入方案
- [[02_Agent技术报告]]：5G 测试验证 Agent 完整技术深度报告（面试专用）
