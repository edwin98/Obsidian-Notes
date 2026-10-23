# LLM Agent 技术与产业调研报告

> [!info] 历史资料
> 本文保留原审阅或调研日期的内容与结论，不代表当前代码和生态状态。当前学习入口见 [[00_导航与计划/知识库导航]]，未关闭事项见 [[00_导航与计划/待核实事项]]。

**版本日期：** 2026 年 9 月 29 日

**面向对象：** Agent 应用开发求职者、技术负责人、产品负责人、创新与数字化管理者

**研究范围：** 以软件型 LLM Agent 为主，兼顾多智能体、计算机操作、语音与物理世界智能体；重点讨论 2022—2026 年的演进、现状、趋势与企业落地。

**说明：** 本报告优先采用论文、官方文档、标准组织与研究机构的一手资料。框架和产品更新很快，涉及具体版本时应以厂商最新文档为准。

---

## 一、执行摘要

### 1. 核心结论

1. **LLM Agent 不是一个单独模型，而是一套运行系统。** 它通常由模型、指令与上下文、工具、记忆、规划/编排、执行环境、状态管理、评估监控和权限治理共同组成。模型负责判断，工具负责行动，运行时负责把一个不稳定的概率模型约束成可持续工作的系统。

2. **发展主线已经从“会思考”走到“能可靠执行”。** 2022—2023 年的重点是推理与行动循环、工具调用和自主规划；2024 年开始转向工作流、可观测性与计算机操作；2025 年形成 MCP、A2A、Agent Skills 等开放协议；到 2026 年，竞争重点进一步转向长任务运行时、上下文压缩、沙箱、断点恢复、子智能体、权限和生产评估。

3. **Agent 能力进步很快，但仍存在明显的“锯齿状能力边界”。** Stanford 2026 AI Index 指出，OSWorld 上的智能体任务成功率已从约 12% 提升到约 66%，但仍约有三分之一任务失败；同一模型可能完成复杂编程，却在简单界面状态、权限边界或异常恢复上失误。[Stanford AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)

4. **企业应用处于“高试验、低规模化”阶段。** McKinsey 2025 调研中，62% 的受访组织已至少开始试验 Agent，但只有 23% 表示正在某处规模化，且单个职能中规模化比例均不超过 10%。Stanford 的汇总也显示，多数业务职能的 Agent 规模化应用仍为个位数。[McKinsey 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-2025)、[Stanford AI Index：Economy](https://hai.stanford.edu/ai-index/2026-ai-index-report/economy)

5. **近期最确定的高价值场景不是“全能数字员工”，而是边界清晰、反馈可验证的知识工作。** 当前成熟度相对较高的方向包括软件研发、深度研究、IT 服务台、知识管理、文档处理、数据分析、客服辅助和合规初审。共同特征是：输入输出可数字化、工具接口存在、结果可通过测试或规则验证、错误可回滚。

6. **多智能体不是默认答案。** 它适合可并行、需要不同上下文或专业角色的任务；对于强顺序依赖、共享状态密集或简单任务，多智能体往往增加成本、延迟、协调错误和调试难度。生产系统应先从单 Agent 或确定性工作流开始，再以评估结果决定是否拆分。

7. **开放标准正在形成 Agent 技术栈的“接口层”。** MCP 主要解决 Agent 与工具/数据的连接，A2A 主要解决独立 Agent 之间的发现、通信和委派，Agent Skills/`SKILL.md` 用于封装可复用的过程知识，`AGENTS.md` 用于给代码 Agent 提供仓库级约束。这些标准降低了连接成本，但并没有自动解决身份、授权、信任、质量与责任归属问题。

8. **安全问题的本质从“输出是否有害”升级为“系统是否会执行有害动作”。** 间接提示注入、工具供应链、过度授权、持久记忆污染、数据外泄、重复执行和责任不可追溯，已成为 Agent 特有的主要风险。NIST 2026 的大规模红队研究仍将 Agent 劫持视为现实问题。[NIST Agent Security](https://www.nist.gov/blogs/caisi-research-blog/insights-ai-agent-security-large-scale-red-teaming-competition)

9. **未来 12—24 个月的竞争壁垒不会只在模型。** 更持久的壁垒将来自：高质量私有流程与数据、可验证的工具、领域评估集、权限体系、执行基础设施、用户信任和业务分发渠道。

### 2. 给决策者的一句话建议

把 Agent 当成“具有不确定性的业务执行系统”，而不是“更聪明的聊天机器人”：先选择可验证、可回滚的流程，建立评估和权限边界，再逐步增加自主性。

---

## 二、概念与边界

### 1. 什么是 LLM Agent

本报告采用如下工作定义：

> LLM Agent 是以大语言模型为决策核心，能够感知上下文、形成或调整计划、选择并调用工具、观察环境反馈，并在一定自主程度下持续完成目标的软件系统。

它与普通聊天机器人的差别不在于“是否使用大模型”，而在于是否形成了持续的闭环：

**目标 → 判断/计划 → 行动 → 获取真实反馈 → 更新状态 → 继续或停止**

Anthropic 将“工作流”和“Agent”作了有用区分：工作流由代码预先规定路径，Agent 则由模型动态决定过程和工具使用。现实中的系统常是两者混合，而非二选一。[Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)

### 2. 与 RAG、工作流和自动化的区别

| 形态 | 控制中心 | 路径是否预先确定 | 是否执行外部动作 | 适合场景 |
|---|---|---:|---:|---|
| 单次 LLM 调用 | 提示词 | 是 | 否或一次 | 摘要、分类、改写 |
| RAG 应用 | 检索逻辑 + LLM | 基本确定 | 通常只读 | 企业知识问答 |
| LLM 工作流 | 程序/图编排 | 大部分确定 | 可以 | 稳定、规则清晰的流程 |
| LLM Agent | 模型 + 运行时 | 动态 | 是 | 开放式、多步骤、路径未知的任务 |
| 多智能体系统 | 协调者 + 多 Agent | 动态或混合 | 是 | 可并行、跨专业、上下文隔离任务 |
| RPA | 规则/脚本 | 确定 | 是 | 高重复、界面和规则稳定的操作 |

Agent 不是 RPA 的简单替代。更现实的组合是：Agent 负责理解、决策和异常处理，确定性代码/RPA 负责高频、强约束执行。

### 3. 典型技术架构

一个生产级 Agent 通常包含九层：

1. **模型层：** 推理、多模态理解、工具选择、结构化输出。
2. **上下文层：** 系统指令、用户目标、领域知识、会话状态、检索结果。
3. **规划与编排层：** ReAct 循环、状态机、图工作流、任务分解、路由和终止条件。
4. **工具层：** API、数据库、搜索、浏览器、代码执行、业务系统和其他 Agent。
5. **记忆层：** 短期会话、任务状态、长期偏好、经验摘要；长期记忆不等于简单向量库。
6. **执行层：** 沙箱、文件系统、网络策略、凭据代理、队列、重试、幂等和断点恢复。
7. **可观测层：** Trace、工具调用记录、状态迁移、成本、延迟、失败原因和人工干预。
8. **评估层：** 离线回归、在线监控、仿真、红队、安全评估和业务 KPI。
9. **治理层：** 身份、最小权限、审批、数据边界、审计、责任和合规。

生产差距通常不在第 1 层，而在第 6—9 层。

---

## 三、发展脉络

### 阶段一：基础积累——语言模型获得“推理接口”（2020—2022）

大模型、指令微调和思维链提示使模型能够把复杂问题拆成中间步骤。但此时模型主要“生成答案”，无法验证事实或改变外部世界。核心局限是知识静态、没有环境反馈、行动能力弱。

### 阶段二：推理与行动闭环——Agent 原型形成（2022—2023）

- **ReAct（2022）** 将推理轨迹与外部行动交替执行，确立了今天多数工具型 Agent 的基本循环。[ReAct 论文](https://arxiv.org/abs/2210.03629)
- **Toolformer（2023）** 研究模型如何学习何时调用 API、传递什么参数及如何使用结果，推动“工具使用”从提示技巧走向模型能力。[Toolformer 论文](https://arxiv.org/abs/2302.04761)
- 2023 年出现 AutoGPT、BabyAGI 等自主循环实验，同时 LangChain、AutoGen 等框架降低了工具编排和多智能体实验门槛。AutoGen 将多 Agent 对话和协作模式系统化。[AutoGen 论文](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/08/LLM_agent.pdf)

这一阶段证明了 Agent 的潜力，也暴露出循环失控、成本高、错误累积、上下文污染和“看起来忙但没有完成任务”等问题。

### 阶段三：从 Demo 到工程系统（2024）

行业开始从“自主性最大化”转向“可靠性工程”：

- 确定性工作流与 Agent 循环混合；
- 图编排、检查点、人工审批、Trace 和离线评估成为重要能力；
- 编码 Agent、浏览器 Agent、研究 Agent 成为主要验证场；
- Anthropic 在 2024 年 11 月发布 MCP，将模型与外部数据/工具连接抽象成开放协议。[MCP 发布](https://www.anthropic.com/news/model-context-protocol)

### 阶段四：协议化与产品化（2025）

2025 年是 Agent 从框架概念转向平台能力的一年：

- OpenAI 发布 Responses API、内置网页搜索、文件搜索、计算机操作和 Agents SDK；[OpenAI Agent 构建工具](https://openai.com/index/new-tools-for-building-agents/)
- Google 发布 A2A 协议，解决不同厂商和框架的 Agent 发现、通信、任务委派与结果返回；[A2A 发布](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- LangChain/LangGraph 1.0 强调长期运行、持久化、人工介入和稳定 API；[LangChain/LangGraph 1.0](https://www.langchain.com/blog/langchain-langgraph-1dot0)
- MCP 被捐赠给 Linux Foundation 下的 Agentic AI Foundation；Anthropic 称当时已有超过 10,000 个活跃公共 MCP Server。[MCP 与 AAIF](https://www.anthropic.com/news/donating-the-model-context-protocol-and-establishing-of-the-agentic-ai-foundation)
- Agent Skills 和 `AGENTS.md` 把领域流程与项目约束变成可移植的文件化资产。[Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)、[AGENTS.md 与 AAIF](https://openai.com/index/agentic-ai-foundation/)

### 阶段五：长任务运行时与“Agent 工程”形成（2026）

到 2026 年，前沿平台开始提供托管沙箱、持久会话、上下文压缩、工具动态加载、子智能体和长时间运行能力。OpenAI 于 2026 年 9 月发布 Agents API 公测，把 Codex 类 Agent harness 作为托管基础设施开放，体现出平台竞争已从“模型 API”扩展到“Agent 运行时”。[OpenAI Agents API](https://openai.com/index/introducing-the-agents-api/)

与此同时：

- Microsoft 将 AutoGen 与 Semantic Kernel 的经验收敛到 Microsoft Agent Framework，提供图工作流、检查点、人工介入及 MCP/A2A 支持；[Microsoft Agent Framework RC](https://devblogs.microsoft.com/agent-framework/migrate-your-semantic-kernel-and-autogen-projects-to-microsoft-agent-framework-release-candidate/)
- Google ADK 扩展至多语言、评估、部署、Trace 和沙箱执行；[Google ADK](https://google.github.io/adk-docs/)
- 国内平台逐渐从“低代码 Bot 搭建”升级到多 Agent、MCP、长期记忆、本地执行与团队协作。例如扣子 3.0 强调多人多 Agent、本地 Agent 接入和跨端持续任务；腾讯元器区分标准、单工作流和 Multi-Agent 模式；AgentScope 2.0 强调生产部署、沙箱、MCP/A2A 和可观测性。[扣子 3.0](https://docs.coze.cn/cozespace_coze_app_faq)、[腾讯元器模式](https://yuanqi.tencent.com/guide/agent-build-choose-mode)、[AgentScope](https://github.com/agentscope-ai/agentscope)

---

## 四、2026 年的技术与产业现状

### 1. 能力进展

**工具调用更可靠。** 新一代模型对函数模式、结构化输出和工具选择进行了专门训练；平台还开始允许模型生成受限程序，批量、并行或条件调用多个工具，减少中间数据进入上下文的成本。[OpenAI Programmatic Tool Calling](https://developers.openai.com/api/docs/guides/tools-programmatic-tool-calling)

**计算环境成为标准组件。** Agent 不再只调用小型函数，而是获得隔离文件系统、Shell、代码执行、浏览器和持久工作区。这样可以把大数据和中间产物留在环境中，而不是反复塞入提示词。[从模型到带计算环境的 Agent](https://openai.com/index/equip-responses-api-computer-environment/)

**长任务能力提升。** METR 提出用“能够独立完成多长的人类任务”衡量智能体能力，而不只看静态问答准确率。[METR 长任务研究](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) Anthropic 对真实使用的研究显示，Claude Code 最长一档会话中，单次自主工作时长在三个月内从不足 25 分钟增至超过 45 分钟；熟练用户更常自动批准，也更常在必要时主动打断。[Measuring AI Agent Autonomy](https://www.anthropic.com/research/measuring-agent-autonomy)

**计算机操作进步但仍不稳定。** 图形界面是连接无 API 系统的重要通道，但页面变化、弹窗、状态感知和误点击都会导致失败。它适合填补遗留系统接口空白，不应替代稳定 API。

### 2. 商业采用

企业对 Agent 的兴趣很高，但“使用 AI”与“规模化 Agent”之间存在显著落差：

- McKinsey 2025：88% 的受访组织在至少一个职能中经常使用 AI；62% 至少试验 Agent；23% 在企业某处规模化 Agent。
- Stanford 2026：Agent 在几乎所有业务职能中的规模化部署仍是个位数；科技行业相对领先，软件工程、IT 和服务运营的规模化比例分别约为 24%、22% 和 21%。
- Deloitte 2026：在 3,235 名业务和 IT 领导者调查中，仅 21% 表示已有成熟的 Agent 治理模型。[Deloitte 2026](https://www.deloitte.com/us/en/insights/topics/emerging-technologies/ai-agents-scaling-faster.html)

这说明 2026 年仍不是“无人值守 Agent 普遍接管企业流程”的阶段，而是“从局部生产验证走向平台化治理”的阶段。

### 3. 当前最成熟的应用方向

| 方向         | 典型任务               | 成熟度判断 | 为什么相对成熟               |
| ---------- | ------------------ | ----- | --------------------- |
| 软件研发       | 理解代码、改动、测试、修复、迁移   | 高     | 有编译器、测试和版本控制提供反馈      |
| 深度研究       | 多轮搜索、阅读、归纳、引用      | 中高    | 任务价值高，结果可由引用和人工复核     |
| IT 运维/SRE  | 告警调查、日志分析、处置建议     | 中高    | 数据和工具数字化，流程可设审批       |
| 知识管理       | 跨库检索、报告、问答         | 中高    | 主要为只读操作，风险相对可控        |
| 客服与服务台     | 诊断、回复、工单和有限操作      | 中     | 高频、价值明确，但需防越权和误操作     |
| 数据分析       | 取数、清洗、代码分析、图表和报告   | 中     | 可通过 SQL、代码和数据校验，但口径易错 |
| 销售/营销运营    | 调研、线索丰富、内容与 CRM 更新 | 中     | 流程数字化，但隐私、品牌和误发风险较高   |
| 财务、法务、医疗   | 初审、资料整理、辅助决策       | 中低到中  | 价值高，但容错率低、监管和责任要求高    |
| 全自动跨系统业务办理 | 端到端交易、审批、支付        | 低到中   | 状态复杂、外部副作用大、异常长尾多     |

---

## 五、当前主要趋势

### 趋势 1：从“模型能力”转向“Agent harness/运行时能力”

模型仍重要，但生产效果越来越取决于上下文管理、工具发现、状态持久化、执行环境、重试恢复、权限和评估。2026 年的托管 Agents API、Managed Agents 和各类 Agent Runtime，说明行业正在复制云计算的发展路径：把复杂的执行基础设施平台化。

**影响：** 企业选型不应只比较模型榜单，也要比较运行时可控性、数据边界、可移植性、故障恢复和运营工具。

### 趋势 2：开放协议把 Agent 生态拆成可组合层

- **MCP：** Agent/AI 应用连接工具与数据。
- **A2A：** 独立 Agent 之间发现、通信、委派和交换任务状态。
- **Agent Skills：** 封装说明、脚本、模板和资源，提供按需加载的过程知识。
- **`AGENTS.md`：** 在代码仓库中声明构建、测试、风格与安全约束。

A2A 在 2026 年一周年时已获得超过 150 个组织支持，并进入主要云平台；MCP 已形成广泛的客户端、服务端和基础设施生态。[A2A 一周年](https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year)

**判断：** 协议层会趋于开放，但企业级“连接器”仍需要身份、授权、审计、服务等级和语义契约。兼容协议不等于可以互相信任。

### 趋势 3：工具从“几十个静态函数”走向动态发现和程序化编排

当工具增长到数百或数千个，把全部工具描述放进上下文既昂贵又容易误选。平台开始通过工具搜索按需加载，再让模型用受限代码并行、过滤、聚合结果。

**影响：** 工具目录、命名、输入输出模式、错误语义、幂等性和权限元数据，将成为新的“Agent API 设计”。

### 趋势 4：长任务、后台运行和可恢复执行成为分水岭

真实工作往往持续数十分钟到数天，会遭遇网络失败、限流、人工等待和上下文上限。生产运行时因此需要：检查点、持久状态、事件队列、后台模式、暂停/恢复、任务取消、幂等重放和上下文压缩。

**判断：** “一次请求能否回答”将让位于“任务中断后能否正确继续，以及能否证明完成了什么”。

### 趋势 5：多 Agent 从概念热潮进入理性分工

多 Agent 的有效模式正在收敛：

- 主管—执行者：主管分解、委派、汇总；
- 路由：把输入交给专业 Agent；
- 并行研究：不同 Agent 独立搜索或分析，再汇总；
- 评审者—执行者：一个产出、另一个验证；
- 跨组织 A2A：不同系统中的 Agent 通过协议协作。

**反趋势：** 为了“看起来先进”而把一个简单流程拆成多个角色，通常得不偿失。最重要的判断标准是：是否存在上下文隔离、专业工具隔离或真实并行收益。

### 趋势 6：编码任务成为 Agent 的重要验证场景，并向其他知识工作扩展

软件工程具备机器可读环境、精确工具、自动测试和可逆版本控制，因此最适合训练与验证 Agent。其工程模式正在外溢到研究、数据分析和运营：工作区、文件产物、测试、评审、差异比较和回滚。

需要注意，公共基准正在趋于饱和并出现数据污染、测试缺陷和统计区分度不足的问题。2026 年的研究已指出 SWE-bench Verified 顶部系统之间的微小差异未必具有统计显著性，推动评估转向更新鲜、多语言和企业私有任务集。[SWE-bench 排名区分研究](https://arxiv.org/abs/2609.17394)、[SWE-bench-Live](https://swe-bench-live.github.io/)

### 趋势 7：评估从“最终答案”转向全过程与业务结果

Agent 的失败可能发生在计划、工具选择、参数、权限、环境状态、事实引用、停止条件或人工交接中的任何一步。单一准确率不足以描述系统。

新的评估体系会同时衡量：

- 任务成功率与结果正确性；
- 关键证据覆盖率；
- 工具选择与参数正确率；
- 安全违规和越权率；
- 人工介入率、返工率和升级率；
- 延迟、Token、工具和基础设施成本；
- 重试后恢复率、重复副作用率；
- 用户接受率及实际业务 KPI。

### 趋势 8：安全重心转向身份、授权与动作边界

Agent 会阅读不可信内容，也会持有执行权限，因此“数据”和“指令”的边界容易被混淆。NIST 将间接提示注入/Agent 劫持列为重要威胁，并在 2026 年推动 Agent 身份、授权、审计和不可抵赖方面的标准实践。[NIST Agent 身份与授权](https://www.nist.gov/news-events/news/2026/02/new-concept-paper-identity-and-authority-software-agents)

**行业方向：** 每个 Agent 使用独立身份；短期、最小化凭据；读写权限分离；高风险动作审批；沙箱与网络白名单；工具调用完整审计；把外部内容始终视为不可信数据。

### 趋势 9：自主性将是动态配置，而不是固定等级

用户并非简单选择“自动”或“手动”。Anthropic 的实证研究显示，熟练用户一方面更常自动批准，另一方面也更会在关键时点打断；Agent 自己请求澄清也是监督机制的一部分。

**判断：** 更好的产品会根据动作风险、可逆性、置信度、金额、数据敏感度和用户经验动态调整审批，而不是为整个 Agent 设置一个永久的“全自动”开关。

### 趋势 10：成本优化从 Token 单价走向“每个成功任务成本”

Agent 可能多轮推理、重复检索、运行沙箱并调用付费工具。仅比较模型每百万 Token 价格会误导决策。更合理的指标是：

> 每个成功任务成本 = 全部模型、工具、计算、人工复核和失败重试成本 ÷ 成功完成且被接受的任务数

模型路由、缓存、上下文压缩、小模型执行简单步骤、代码聚合工具结果和并行度控制，都会成为标准工程手段。

### 趋势 11：多模态、实时语音和具身 Agent 逐步汇流

文本、视觉、语音和界面操作正在进入同一个 Agent 循环。短期内，语音客服、现场辅助、屏幕理解和桌面协作比通用机器人更容易规模化；物理世界的安全性、实时性和硬件成本使具身 Agent 的落地节奏更慢。

### 趋势 12：监管从模型内容扩展到交互透明与可追溯性

中国《人工智能生成合成内容标识办法》已于 2025 年 9 月 1 日施行，要求显式和隐式标识；欧盟 AI Act 的部分透明度义务自 2026 年 8 月 2 日开始适用，包括告知用户正在与 AI 交互及对生成/修改内容作标识。[中国网信办](https://www.cac.gov.cn/2025-03/14/c_1743654685899683.htm)、[欧盟委员会](https://digital-strategy.ec.europa.eu/en/news/commission-starts-enforcing-ai-act-rules-and-new-transparency-requirements-2-august)

**影响：** 面向用户的 Agent 应默认保留身份披露、内容来源、行动日志和人工责任链，而不是上线后再补。

---

## 六、主流技术路线与选型建议

### 1. 技术路线分类

| 路线 | 代表 | 优势 | 主要约束 | 适合 |
|---|---|---|---|---|
| 模型厂商托管运行时 | OpenAI Agents API/SDK、Claude Managed Agents、Google Agent Runtime | 集成快，内置模型与工具优化，减少基础设施工作 | 厂商依赖、数据和执行边界需核实 | 快速交付、模型能力优先 |
| 通用开源编排框架 | LangGraph/LangChain、Microsoft Agent Framework | 多模型、可自托管、流程可控、生态广 | 需要自行集成运维与安全体系 | 复杂企业流程、跨模型策略 |
| 云平台 Agent 套件 | Google ADK/Vertex、Azure/Foundry、AWS Agent 相关服务 | 与云 IAM、监控、数据和部署结合 | 云绑定和成本结构复杂 | 已深度使用对应云的企业 |
| 国内低代码/平台 | 阿里云百炼、扣子、腾讯元器、百度千帆等 | 中文生态、渠道和国内合规集成，搭建快 | 可移植性与深度定制差异大 | 运营、客服、知识和快速试点 |
| 国内开源框架 | AgentScope、Dify、Coze Studio 等 | 自主可控、本地化、二次开发 | 生产运维和版本治理由团队承担 | 私有部署、平台型建设 |
| 自研轻量运行时 | 直接使用模型 API + 状态机/队列 | 最透明，依赖少，针对性强 | 需要自行补齐评估、恢复、观测 | 单一关键流程、强控制要求 |

### 2. 选型不应只看功能清单

建议按以下优先级做实测：

1. **任务效果：** 在企业自己的代表性任务集上比较，不以公开榜单代替。
2. **控制与安全：** 身份、最小权限、审批、网络、沙箱、密钥和数据驻留。
3. **可靠性：** 持久状态、暂停恢复、幂等、重试、超时、并发和故障隔离。
4. **可观测与评估：** 是否能看到完整轨迹，是否支持版本对比和在线告警。
5. **可移植性：** 模型、工具协议、状态数据和评估集能否迁移。
6. **总成本：** 以每个成功任务成本衡量，而非只看 Token 单价。
7. **生态与人才：** 团队语言栈、现有云、社区稳定性和长期维护能力。

### 3. 简化的选择原则

- 流程完全可预先描述：优先传统代码或 LLM 工作流。
- 路径难预测但动作低风险：单 Agent + 工具 + 明确停止条件。
- 任务可拆成独立并行部分：在评估证明收益后使用多 Agent。
- 需要操作遗留 GUI：先找 API；确实没有 API 时，再采用计算机操作并设置审批。
- 高风险领域：Agent 负责准备材料和建议，最终决策与不可逆动作保留给授权人员。

---

## 七、主要瓶颈与风险

### 1. 可靠性与错误累积

单步 95% 正确并不意味着 20 步任务可靠。若错误近似独立，连续 20 步全对的概率只有约 36%。现实错误并不独立，还可能因错误状态继续放大。因此必须减少不必要步骤、在关键节点验证，并尽量从环境获得可机器验证的事实。

### 2. 目标与完成条件模糊

Agent 经常“做了很多事”但没有真正完成目标。任务应有明确交付物、验收条件、预算、截止条件和不可做事项。最好让完成状态能通过测试、查询或规则确认。

### 3. 上下文与记忆污染

长上下文不是无限记忆。历史噪声、过期事实和错误摘要会持续影响后续决策。长期记忆还可能保存敏感数据或被恶意内容污染。记忆需要来源、时间、作用域、生命周期和删除机制。

### 4. 间接提示注入

Agent 读取网页、邮件、文档或代码仓库时，攻击者可以把恶意指令伪装成内容，诱导 Agent 泄露数据或执行工具。只靠提示词声明“忽略恶意指令”不足以构成安全边界。

### 5. 过度授权与副作用

一个能读取全部文档、发送邮件、修改数据库和付款的 Agent，风险远高于能力之和。权限应按工具、资源、动作、金额和时间分割，关键写操作应具备预览、审批、幂等键和撤销方案。

### 6. 评估失真

公开基准可能被训练数据污染或与企业任务不匹配；LLM-as-judge 可能受偏好、提示和答案风格影响。需要公共基准、私有任务集、规则验证、人工抽检和线上业务指标组合使用。

### 7. 经济性不确定

Agent 可能节省人工时间，但引入模型、工具、平台、复核和失败处置成本。如果流程本身低频、输入混乱或组织无法采用输出，技术成功也不会产生 ROI。

### 8. 责任与组织变革

Agent 会跨越原有岗位和系统边界。若没有流程所有者，出现错误时很难判断由模型、工具、数据、平台还是审批人负责。规模化往往需要重新设计流程和岗位，而不仅是部署软件。

---

## 八、企业落地方法

### 1. 场景优先级评分

建议从以下八个维度按 1—5 分评分：

| 维度 | 高分含义 |
|---|---|
| 业务价值 | 高频、高人工成本、直接影响收入或服务质量 |
| 可数字化 | 输入、状态和动作都可由系统访问 |
| 可验证性 | 结果可测试、对账或规则校验 |
| 可回滚性 | 错误可撤销，不造成永久损失 |
| 流程复杂度 | 需要判断，但不是无限开放问题 |
| 数据准备度 | 数据可用、权限清晰、质量可接受 |
| 集成准备度 | 有稳定 API、MCP 或可控工具 |
| 风险可控度 | 隐私、合规、金额和声誉风险较低 |

首批项目应优先选择“高价值 + 高可验证 + 高可回滚 + 数据/工具已就绪”的场景。

### 2. 分阶段路线图

#### 第 0 阶段：定义基线（1—2 周）

- 描述当前人工流程、耗时、错误率和成本；
- 收集 50—200 个真实、去敏的代表性任务；
- 明确成功标准、禁止动作和人工责任人；
- 判断是否真的需要 Agent，还是规则、搜索或普通工作流即可。

#### 第 1 阶段：只读副驾驶（2—6 周）

- Agent 只读取数据、生成建议或草稿；
- 所有输出人工确认；
- 建立 Trace、版本、成本和失败分类；
- 以真实任务集进行回归测试。

#### 第 2 阶段：受控执行（4—8 周）

- 开放低风险、可撤销的写工具；
- 为关键动作增加预览、审批、幂等和回滚；
- 建立沙箱、网络白名单、短期凭据和速率/预算限制；
- 做间接提示注入与数据外泄红队测试。

#### 第 3 阶段：局部自治与规模化（持续）

- 只对评估稳定的任务提高自动批准范围；
- 根据风险动态设置人机交接；
- 建立生产抽检、漂移监测、事件响应和模型/提示回滚；
- 将优秀流程沉淀为可版本化 Skill、工具和评估集。

### 3. 最小生产架构

一个足够简单、但可生产的起点应包含：

- 单一主 Agent；
- 少量、边界清楚的工具；
- 显式状态机或任务状态；
- 持久化检查点；
- 工具级身份和最小权限；
- 高风险操作审批；
- 完整 Trace 和成本记录；
- 代表性回归测试集；
- 最大步骤、最大成本和超时限制；
- 可取消、可恢复、可回滚。

不要一开始就建设复杂的“Agent 中台”或庞大多 Agent 组织。让第一个真实场景暴露公共能力需求，再抽象平台。

---

## 九、评估指标体系

### 1. 离线评估

- 端到端任务成功率；
- 结果正确性、完整性和证据覆盖；
- 工具选择、参数和调用顺序正确率；
- 关键约束遵守率；
- 安全攻击成功率与越权率；
- 重试恢复率；
- 平均和 P95 延迟；
- 每任务与每成功任务成本。

### 2. 在线指标

- 用户接受/采用率；
- 人工修改率与修改幅度；
- 升级人工率、用户中断率、Agent 主动澄清率；
- 自动完成率和首次完成率；
- 业务周期缩短、成本下降、收入或质量变化；
- 事故、越权、数据泄露和重复执行次数；
- 不同模型、提示、工具版本的回归情况。

### 3. 评估原则

1. 先有测试集，再调提示和编排。
2. 测最终结果，也测过程中的关键不变量。
3. 把失败案例持续加入回归集。
4. 公共榜单只作参考，私有真实任务决定选型。
5. 对不可逆操作，安全指标优先于平均成功率。
6. 每次模型、提示、工具或知识库变更都应可追溯并触发回归。

---

## 十、安全与治理清单

### 设计阶段

- 为 Agent 建立独立身份，不借用最终用户或管理员的长期凭据；
- 对数据、工具和动作实施最小权限；
- 区分可信指令与不可信内容；
- 明确哪些动作可自动执行，哪些必须审批；
- 为写操作设计预览、幂等、撤销和补偿事务；
- 对网络、文件系统和代码执行使用隔离环境；
- 为记忆设置来源、作用域、保留期和删除机制。

### 上线前

- 进行提示注入、工具欺骗、权限绕过和数据外泄测试；
- 验证故障重试不会重复发信、下单或付款；
- 模拟工具超时、部分失败、返回恶意内容和状态变化；
- 设置步骤、时间、Token、金额和调用次数预算；
- 准备停用开关、降级路径和人工接管流程。

### 运行中

- 保存可检索的任务、工具、审批和版本审计记录；
- 对异常工具组合、敏感数据访问和成本激增告警；
- 持续抽检成功任务，避免只分析显性失败；
- 定期撤销闲置凭据和工具；
- 对模型、工具、协议和依赖升级做回归评估。

---

## 十一、未来 12—24 个月展望

### 高置信度判断

1. **Agent 运行时将成为云和模型平台的标准产品层。** 长任务、状态、沙箱、工具、评估和权限会进一步托管化。
2. **MCP、A2A、Skills 等开放接口继续普及。** 企业会要求工具和 Agent 具备跨平台可发现、可授权和可审计能力。
3. **编码、研究、IT 运维和客服仍是主要落地前沿。** 这些领域的反馈和价值链最清晰。
4. **评估与安全预算上升。** 企业从试点进入生产后，会发现 Trace、私有评估集、权限和事件响应不是可选项。
5. **自主性以风险分层方式扩大。** 低风险、可逆任务自动化率提升，高风险决策仍保留明确的人类责任。

### 中等置信度判断

1. **Agent 市场将从独立应用转向现有软件的内嵌能力。** CRM、ERP、办公、开发和数据平台会把 Agent 变成默认交互和执行层。
2. **“数字员工”会先以角色包而非完全独立主体出现。** 一个角色包包含 Skill、工具、权限、记忆、评估和审计策略。
3. **模型路由与多模型组合成为常态。** 强模型负责规划和疑难步骤，小模型或确定性代码负责分类、抽取和批处理。
4. **Agent 身份和授权标准快速发展。** 工作负载身份、委派凭证、动作签名和责任追踪将与 MCP/A2A 生态结合。
5. **GUI 操作仍是过渡接口。** 能稳定提供 API 的系统最终会优先 API；计算机操作主要服务于遗留系统、临时任务和无法集成的长尾界面。

### 低置信度或需要谨慎对待的叙事

1. 未来一两年内通用 Agent 大规模、完全无人监督地替代完整岗位；
2. 只靠多 Agent 讨论就能稳定消除幻觉和错误；
3. 一个通用公开基准可以准确预测企业真实 ROI；
4. 协议互通自然带来安全互信；
5. 更长上下文可以替代记忆设计、检索和状态管理。

---

## 十二、对企业的建议

### 对业务负责人

- 从流程结果和责任边界定义项目，不从“要不要做 Agent”出发；
- 先挑选可验证、可回滚的高频流程；
- 把采用率、返工率和业务周期纳入 ROI，不只计算节省的人时；
- 让一线人员参与任务集、失败分类和审批设计。

### 对技术负责人

- 保持架构简单：优先单 Agent、少量工具、明确状态；
- 把工具做成窄接口、结构化输出、可幂等和可审计；
- 使用开放协议降低耦合，但在协议之上实现企业权限和信任层；
- 建立私有评估集、持续回归和完整 Trace；
- 将模型视为可替换组件，将流程、工具、数据与评估集视为核心资产。

### 对安全与合规负责人

- 按“会执行动作的软件主体”而非“内容生成器”评估 Agent；
- 推动 Agent 独立身份、最小权限、短期凭据和审批策略；
- 把间接提示注入纳入常规应用安全测试；
- 要求交互披露、生成内容标识、审计和数据生命周期管理；
- 对金融、医疗、法务、人事等高风险场景保留明确的人类决策责任。

---

## 十三、最终判断

LLM Agent 已越过纯概念验证阶段，但尚未进入普遍、完全自治的成熟期。2026 年最重要的变化不是 Agent 突然“像人一样工作”，而是支撑它工作的工程基础正在快速成形：开放协议、长任务运行时、沙箱、状态恢复、工具动态加载、评估、权限和审计。

未来真正有竞争力的 Agent 系统，不会是自主步骤最多的系统，而是能够在明确边界内，以可接受的成本稳定完成任务，并让人知道它做了什么、为什么做、出了问题如何停止和恢复的系统。

---

## 附录 A：关键术语

- **ReAct：** Reasoning + Acting，推理与行动交替的 Agent 模式。
- **Tool/Function Calling：** 模型选择并生成结构化参数，交由外部系统执行函数。
- **MCP：** Model Context Protocol，连接 AI 应用与工具/数据的开放协议。
- **A2A：** Agent2Agent Protocol，独立 Agent 之间通信和委派的开放协议。
- **Agent Skill：** 以说明、脚本和资源封装的可复用过程能力。
- **Harness：** 包围模型的 Agent 执行框架，包括循环、工具、上下文、状态和协调机制。
- **HITL：** Human-in-the-loop，在关键节点由人审核、补充或批准。
- **Computer Use：** 通过截图、鼠标和键盘等方式操作图形界面的能力。
- **Checkpoint：** 保存任务执行状态，以便暂停、恢复、重放或审计。
- **Indirect Prompt Injection：** 恶意指令藏在 Agent 读取的外部内容中，诱导其偏离用户目标。

## 附录 B：重点参考资料

### 基础论文

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation](https://www.microsoft.com/en-us/research/wp-content/uploads/2023/08/LLM_agent.pdf)
- [AgentScope: A Flexible yet Robust Multi-Agent Platform](https://arxiv.org/abs/2402.14034)

### 产业与工程

- [Anthropic: Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [OpenAI: New Tools for Building Agents](https://openai.com/index/new-tools-for-building-agents/)
- [OpenAI: Introducing the Agents API](https://openai.com/index/introducing-the-agents-api/)
- [LangChain and LangGraph 1.0](https://www.langchain.com/blog/langchain-langgraph-1dot0)
- [Microsoft Agent Framework](https://devblogs.microsoft.com/agent-framework/migrate-your-semantic-kernel-and-autogen-projects-to-microsoft-agent-framework-release-candidate/)
- [Google Agent Development Kit](https://google.github.io/adk-docs/)
- [Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
- [Google Agent2Agent Protocol](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)
- [Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)

### 评估、安全与市场

- [Stanford AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)
- [METR: Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)
- [Anthropic: Measuring AI Agent Autonomy in Practice](https://www.anthropic.com/research/measuring-agent-autonomy)
- [NIST: Insights into AI Agent Security](https://www.nist.gov/blogs/caisi-research-blog/insights-ai-agent-security-large-scale-red-teaming-competition)
- [McKinsey: The State of AI in 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai-2025)
- [Deloitte: Agentic AI Is Scaling Faster Than Guardrails](https://www.deloitte.com/us/en/insights/topics/emerging-technologies/ai-agents-scaling-faster.html)

## 附录 C：Agent 应用开发技术详解

本附录面向准备从事 Agent 应用开发的工程师。重点不是某一个框架的 API，而是不同框架背后共通的运行机制和生产约束。

### C.1 Agent 的最小运行循环

Agent 的本质可以简化为四件事：构造上下文、让模型决策、执行工具、把环境反馈写回状态。框架主要是在这个循环外增加持久化、恢复、权限、Trace 和并发能力。

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol
from uuid import uuid4

JsonObject = dict[str, Any]


@dataclass(slots=True)
class Action:
    kind: Literal["tool", "final"]
    name: str | None = None
    arguments: JsonObject = field(default_factory=dict)
    answer: str | None = None


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: JsonObject = field(default_factory=dict)
    error_code: str | None = None
    retryable: bool = False


class Model(Protocol):
    async def decide(self, context: list[JsonObject]) -> Action: ...


class Tool(Protocol):
    name: str
    has_side_effect: bool

    async def run(
        self,
        arguments: JsonObject,
        idempotency_key: str,
    ) -> ToolResult: ...


class Approval(Protocol):
    async def allow(self, tool_name: str, arguments: JsonObject) -> bool: ...


class StateStore(Protocol):
    async def save(self, state: AgentState) -> None: ...


@dataclass(slots=True)
class AgentState:
    goal: str
    run_id: str = field(default_factory=lambda: str(uuid4()))
    events: list[JsonObject] = field(default_factory=list)

    def context(self) -> list[JsonObject]:
        return [{"role": "user", "content": self.goal}, *self.events]


async def run_agent(
    state: AgentState,
    model: Model,
    tools: dict[str, Tool],
    approval: Approval,
    store: StateStore,
    max_steps: int = 12,
) -> str:
    for step in range(max_steps):
        action = await model.decide(state.context())

        if action.kind == "final":
            return action.answer or ""

        tool = tools.get(action.name or "")
        if tool is None:
            state.events.append({"type": "tool_error", "error_code": "unknown_tool"})
            await store.save(state)
            continue

        if tool.has_side_effect and not await approval.allow(
            tool.name,
            action.arguments,
        ):
            state.events.append({"type": "tool_denied", "tool": tool.name})
            await store.save(state)
            continue

        result = await tool.run(
            action.arguments,
            idempotency_key=f"{state.run_id}:{step}:{tool.name}",
        )
        state.events.append(
            {
                "type": "tool_result",
                "tool": tool.name,
                "ok": result.ok,
                "data": result.data,
                "error_code": result.error_code,
                "retryable": result.retryable,
            }
        )
        await store.save(state)

    raise RuntimeError("agent exceeded max_steps")
```

这段代码刻意保持简单，但包含了五个生产级关键点：

1. 模型只提出动作，不直接执行动作；
2. 工具通过注册表暴露，而不是让模型任意运行函数；
3. 有副作用的工具经过审批策略；
4. 每次写操作携带幂等键；
5. 每一步都持久化，可从最后一个检查点恢复。

实际项目还应增加超时、取消、重试退避、Token/金额预算、并发控制、凭据注入、Trace 传播和输出模式校验。不要把这些逻辑全部塞进系统提示词；它们属于确定性运行时。

### C.2 状态模型：会话、运行、步骤和产物要分开

常见错误是把全部状态等同于 `messages`。更稳妥的设计至少区分：

| 对象 | 含义 | 典型字段 |
|---|---|---|
| Session | 用户与 Agent 的长期交互容器 | 用户、租户、偏好、创建时间 |
| Run | 一次目标明确的任务执行 | 目标、状态、预算、开始/结束时间 |
| Step | 一次模型决策或工具执行 | 输入、输出、耗时、重试、错误 |
| Event | 追加式事实记录 | tool_requested、approved、completed |
| Artifact | 文件、报告、补丁等任务产物 | URI、哈希、MIME、版本、来源 |
| Checkpoint | 可恢复的状态快照 | 当前节点、变量、未决审批、版本 |

推荐使用“追加事件 + 周期性快照”的思路：事件用于审计与重放，快照用于快速恢复。数据库写入、队列消费和外部工具调用通常只能保证至少一次执行，因此所有有副作用工具都应支持幂等键或业务唯一约束。

需要明确区分三类失败：

- **模型失败：** 无效结构、选错工具、推理偏离；
- **工具失败：** 超时、限流、参数错误、下游不可用；
- **业务失败：** 库存不足、权限不够、状态已变化。

只有暂时性工具失败适合自动重试。业务失败应返回可理解的结构化状态，让模型改变方案或请求用户处理。

### C.3 工具设计：Agent 可靠性的主要接口

工具应像面向一个“聪明但不了解内部约定的新同事”设计。Anthropic 的上下文工程经验强调，工具集合应尽量小、职责不重叠、描述清晰；如果工程师自己都无法判断何时使用哪个工具，模型也很难稳定判断。[Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

#### 工具设计原则

1. 名称体现动作和对象，例如 `get_order`、`search_orders`、`cancel_order`，避免 `handle_order`。
2. 读工具与写工具分离，方便权限和审批。
3. 输入使用严格 JSON Schema，枚举、格式、范围和必填项写清楚。
4. 输出结构稳定且紧凑，不把整页 HTML 或上万行日志直接返回模型。
5. 返回证据引用，如记录 ID、来源 URI、版本和时间戳。
6. 错误使用机器可判定的 `error_code` 和 `retryable`，不要只返回自然语言。
7. 写操作支持 `dry_run`、`idempotency_key` 和结果查询。
8. 搜索工具支持分页、过滤、排序和字段选择。
9. 工具内部再次做鉴权和参数校验，不能因为调用来自模型就信任它。
10. 将工具版本写入 Trace，避免升级后无法复现历史行为。

推荐的工具返回结构：

```json
{
  "ok": true,
  "data": {
    "order_id": "o_123",
    "status": "shipped"
  },
  "evidence": [
    {
      "source": "orders-db",
      "record_id": "o_123",
      "observed_at": "2026-09-29T10:00:00Z"
    }
  ],
  "error_code": null,
  "retryable": false
}
```

#### 按副作用分级

| 等级 | 示例 | 默认策略 |
|---|---|---|
| L0 纯读取 | 搜索知识库、查订单 | 自动执行，记录审计 |
| L1 可逆写入 | 建草稿、加标签、创建工单 | 自动或批量审批，必须幂等 |
| L2 对外动作 | 发邮件、改生产配置、提交订单 | 执行前预览与明确审批 |
| L3 高风险动作 | 付款、删除数据、权限变更 | 强身份验证、双人审批或禁止 Agent 直接执行 |

### C.4 上下文工程：把上下文窗口当作有限缓存

上下文不是越长越好。长任务中，原始工具输出、重复历史和过期计划会造成“上下文腐化”。较好的上下文通常由以下部分动态组合：

1. 稳定且简洁的系统约束；
2. 当前用户目标和验收条件；
3. 当前阶段需要的工具定义，而非全部工具；
4. 结构化工作状态，如计划、已完成项和待解决问题；
5. 按需检索的领域事实；
6. 最近几步原始交互；
7. 为模型输出与工具结果预留的空间。

长任务常用三种技术：

- **Compaction：** 压缩早期历史，保留决策、约束、未解决问题和证据指针；
- **结构化笔记：** 把计划、发现、假设、文件路径和待办写到上下文外；
- **子 Agent 隔离：** 让子 Agent 在干净上下文中深挖，只返回短摘要和证据。

Anthropic 将上下文工程定义为对系统指令、工具、外部数据和历史等整体推理状态的持续策展，并建议采用 Just-in-Time 检索和渐进式披露，而非一次把所有资料放进提示词。[Effective Context Engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)

### C.5 记忆不是一个向量数据库

| 记忆类型 | 用途 | 推荐存储 | 写入策略 |
|---|---|---|---|
| 工作记忆 | 当前计划、变量、待办 | Run state/Checkpoint | 每一步确定性更新 |
| 情节记忆 | 过去任务和关键事件 | 事件库、关系数据库 | 任务结束后提炼 |
| 语义记忆 | 稳定事实与领域知识 | 文档库、搜索、向量/关键词索引 | 有来源和有效期 |
| 程序记忆 | 如何完成某类任务 | Skill、代码、工作流 | 版本控制与评审 |
| 用户记忆 | 偏好和长期设置 | 用户配置库 | 明确同意、可查看和删除 |

记忆写入本身也需要策略。不要让模型把每句话都写成长期记忆；应校验来源、敏感性、租户范围、有效期和重复内容。对事实型记忆，读取时仍要返回来源和更新时间。

### C.6 Agentic RAG：从“检索一次”到“边做边找”

传统 RAG 一般在生成前检索一次；Agentic RAG 允许模型根据中间发现改写查询、选择不同数据源、追踪引用和判断是否需要继续搜索。

典型流程：

1. 将用户问题拆成可验证的子问题；
2. 为每个子问题选择检索源；
3. 混合关键词、向量和结构化查询；
4. 对结果重排并去重；
5. 检查证据是否覆盖关键结论；
6. 若存在缺口，继续检索而非立即生成；
7. 输出结论、引用和不确定性。

工程上应保存 `query → retrieved_ids → reranked_ids → cited_ids` 的完整链路。这样才能区分是“没检索到”“检索到了但模型没使用”，还是“引用与结论不匹配”。

### C.7 编排模式及选择

| 模式 | 适合 | 主要风险 |
|---|---|---|
| Prompt Chain | 步骤固定、前后依赖清楚 | 前一步错误向后传播 |
| Router | 输入类型差异大 | 错误路由导致后续全错 |
| Parallel/Map-Reduce | 多来源研究、批量文档 | 合并冲突和重复成本 |
| Evaluator-Optimizer | 有清晰质量标准的内容/代码 | 评审模型偏差、无限迭代 |
| Planner-Executor | 路径未知但可分步 | 计划过时、执行偏离 |
| Supervisor-Workers | 专业工具或上下文需要隔离 | 协调成本、信息丢失 |
| Graph/State Machine | 合规流程、长任务、人工审批 | 图过度复杂、维护成本高 |

选择顺序应是：确定性代码 → 工作流 → 单 Agent → 多 Agent。每增加一层自主性，都应有评估数据证明它带来足够收益。

### C.8 多 Agent 的工程契约

多 Agent 不是让多个角色自由聊天。稳定的协作应有结构化交接合同：

- 子任务目标和验收条件；
- 可用工具、数据和权限范围；
- 时间、步骤和成本预算；
- 输出模式、证据和置信度；
- 失败原因和需要上级处理的问题；
- 是否允许继续委派。

共享完整对话会迅速污染上下文。更好的方式是每个 Agent 保持独立上下文，通过任务对象和产物 URI 交接。主管 Agent 只接收必要摘要、证据和状态。

### C.9 MCP、A2A 与 Skill 的技术分工

#### MCP

MCP Server 向 Host/Client 暴露三类核心能力：

- `Tools`：由模型选择的可执行动作；
- `Resources`：由应用选择并提供给模型的上下文数据；
- `Prompts`：由用户选择的模板化交互入口。

截至 2026-07-28 版规范，MCP 核心趋向无状态请求/响应，增加可缓存目录结果、路由头、授权强化和正式扩展机制；工具输入输出支持完整 JSON Schema 2020-12。状态应通过显式 handle 传递，而不是隐藏在传输会话里。[MCP 2026-07-28 规范说明](https://blog.modelcontextprotocol.io/posts/2026-07-28/)

实现 MCP Server 时仍需自行处理租户隔离、OAuth scope、工具级授权、限流、审计、网络出口和返回内容净化。MCP 是连接协议，不是安全沙箱。

#### A2A

A2A 用于独立 Agent 之间的协作。服务端通过 Agent Card 描述能力和接口，交互围绕 Message、Task、状态和 Artifact 展开。它更接近跨服务任务协议，而不是本地函数调用。[A2A Specification](https://a2a-protocol.org/dev/specification/)

#### Skill

Skill 是文件化的过程知识，通常包含 `SKILL.md`、脚本、模板和参考资料。工具回答“能做什么”，Skill 回答“按照什么方法完成”。它适合沉淀组织流程，但必须版本化、评审并防止其中的脚本获得过大权限。

### C.10 沙箱与权限模型

代码执行和计算机操作应默认运行在临时隔离环境中：

- 非特权用户、只读基础镜像；
- CPU、内存、进程数、磁盘和执行时限；
- 默认禁止网络，按域名和端口放行；
- 密钥由代理按调用注入，不写入上下文或磁盘；
- 输入文件只读挂载，输出写入独立目录；
- 高风险系统调用和设备访问禁用；
- 产物做类型、大小、恶意代码和敏感信息检查；
- 沙箱结束后销毁，必要产物按白名单持久化。

权限检查至少发生在四层：Host 决定是否暴露工具、运行时策略检查动作、工具服务验证调用者身份、下游业务系统执行最终授权。任何一层都不能只相信模型的自然语言解释。

### C.11 可观测性：一个 Trace 应回答什么

Agent Trace 至少要能回答：

- 用户目标和运行版本是什么；
- 每一步模型看到了哪些上下文摘要和工具定义；
- 模型选择了什么动作，结构化参数是什么；
- 谁批准了哪个动作；
- 工具在哪里执行、耗时和结果是什么；
- 使用了哪些证据和产物；
- 发生了几次重试、压缩和模型切换；
- 最终成本、延迟和任务结果是什么。

推荐保留以下关联 ID：`trace_id`、`session_id`、`run_id`、`step_id`、`tool_call_id`、`approval_id`、`artifact_id`。敏感输入不应原样进入日志，应在采集端脱敏，并为日志设置租户隔离和保留期。

### C.12 Agent 评估的工程方法

评估对象应是“模型 + 提示 + 工具 + 数据 + 运行时 + 策略”的整体。Anthropic 的 2026 评估实践也强调，多轮 Agent 的环境状态和完整轨迹是评估的一部分，而不只是最后一句回答。[Demystifying Evals for AI Agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents)

一个可执行的评估流水线：

1. 从真实流量抽取并去敏 50—200 个种子任务；
2. 标注最终验收条件和关键过程约束；
3. 为外部系统建立固定快照或仿真环境；
4. 同一配置运行多次，测量方差而非只跑一次；
5. 使用确定性检查器优先判断数据库状态、文件差异和引用；
6. 对开放式质量再使用 LLM Judge，并定期用人工校准；
7. 按失败类型聚类，针对失败修改系统；
8. 将修复过的事故加入永久回归集；
9. 在 CI 中跑小型冒烟集，定期跑完整集；
10. 上线后用生产指标验证离线得分是否真的转化为价值。

建议的失败分类：任务理解、规划、检索、工具选择、参数、权限、执行、状态同步、事实性、引用、停止条件、用户交接、成本超限和安全违规。

### C.13 性能与成本优化

优化顺序应先保证正确，再降低成本：

1. 删除无用步骤和重复模型调用；
2. 缩小工具集合，按需加载工具定义；
3. 截断或摘要大型工具结果，只保留证据指针；
4. 对稳定系统提示和工具目录使用缓存；
5. 独立任务并行，存在依赖的步骤保持顺序；
6. 用小模型做分类、抽取和简单路由，强模型处理规划和异常；
7. 对相同外部查询做业务缓存，但保存数据时间戳；
8. 为每个 Run 设置 Token、工具、沙箱时长和金额预算；
9. 比较每个成功任务成本，而非单次调用成本。

会修改外部状态的工具，需要确认并发执行不会冲突；如果下一步依赖模型判断中间结果，就应保留逐步处理，避免直接批量执行。

### C.14 生产部署检查表

- API 和工具调用都有超时、取消和重试策略；
- 重试不会造成重复副作用；
- Run 可在进程重启后恢复；
- 模型、提示、工具和知识库都可版本化；
- 每个租户的数据、缓存、记忆和日志隔离；
- 有离线评估集和上线门槛；
- 有最大步骤、成本和执行时间；
- 有人工接管、停止开关和降级到普通工作流的路径；
- 有提示注入、越权、数据外泄和依赖供应链测试；
- 有 SLO：成功率、P95 延迟、恢复时间和安全事件目标。

## 附录 D：Agent 应用开发求职指南

### D.1 岗位真正需要的能力

| 能力层 | 应掌握内容 | 能证明能力的产物 |
|---|---|---|
| Python 工程 | 类型、异步、测试、错误处理、日志、包管理 | 有测试和 CI 的服务仓库 |
| LLM 基础 | Token、上下文、采样、结构化输出、工具调用 | 可替换模型的 Adapter |
| Agent 编排 | 状态机、检查点、HITL、并行与恢复 | 可暂停恢复的任务流程 |
| 数据/RAG | 分块、混合检索、重排、引用、评估 | 带检索链路分析的问答系统 |
| 工具与协议 | JSON Schema、MCP、OAuth、幂等 | 自己实现的安全 MCP Server |
| 后端系统 | FastAPI、PostgreSQL、Redis、队列、对象存储 | 可部署 API 与任务 Worker |
| 可观测性 | Trace、指标、日志、成本和版本关联 | OpenTelemetry/LangSmith 等 Trace |
| 评估 | 数据集、Grader、回归、红队、线上监控 | 自动化 Eval 报告 |
| 安全 | 最小权限、沙箱、提示注入、数据隔离 | 威胁模型与安全测试用例 |
| 部署 | Docker、CI/CD、云基础知识 | 一键启动和部署文档 |

求职时，知道多少框架名称远不如能解释一个失败案例、定位它属于哪一层，并用测试证明修复有效。

### D.2 推荐学习顺序

1. 直接调用模型 API，掌握流式输出、结构化输出和工具调用；
2. 不使用框架手写一个最小 Agent Loop；
3. 加入 PostgreSQL/Redis 状态、重试、幂等和任务队列；
4. 用 LangGraph、OpenAI Agents SDK、Google ADK 或 Microsoft Agent Framework 复现同一项目；
5. 实现一个本地 MCP Server，再实现带 OAuth 的远程版本；
6. 加入 RAG、引用校验和检索评估；
7. 加入审批、沙箱、提示注入测试和审计；
8. 建立 Eval 数据集，在 CI 中比较不同模型和提示版本；
9. 最后再尝试多 Agent，并用数据证明它比单 Agent 更好。

### D.3 八周作品集计划

#### 第 1—2 周：基础 Agent

- Python 异步 API；
- 模型 Adapter 和结构化输出；
- 三个只读工具；
- 运行步骤、超时和成本限制；
- 单元测试和基础 Trace。

#### 第 3—4 周：持久化与 RAG

- PostgreSQL 保存 Session、Run、Event 和 Artifact；
- 检查点与进程重启恢复；
- 混合检索、重排和引用；
- 50 个真实风格评估任务。

#### 第 5—6 周：工具与安全

- 将工具发布为 MCP Server；
- 读写工具分权、审批和幂等；
- Docker 沙箱与网络白名单；
- 间接提示注入、越权和数据外泄测试。

#### 第 7—8 周：生产化展示

- 任务队列、并发和取消；
- OpenTelemetry Trace、成本和仪表盘；
- CI 中运行冒烟 Eval；
- 部署演示环境；
- 编写架构决策记录、威胁模型和故障复盘。

### D.4 推荐作品集项目：可审计的 IT 事件调查 Agent

这是一个比普通聊天机器人更能体现工程能力的项目。

#### 场景

输入一条“某服务近 30 分钟 5xx 升高”的事件，Agent 自动查询指标、日志、部署记录和依赖状态，生成带证据的根因假设与处置建议；只有经过人工批准，才能创建工单或执行低风险缓解动作。

#### 必备模块

- 主 Agent：负责计划、选择工具和汇总；
- 只读 MCP 工具：指标、日志、部署、服务依赖、知识库；
- 状态机：调查 → 证据检查 → 报告 → 审批 → 执行；
- PostgreSQL：Session、Run、Event、Approval、Artifact；
- Redis/队列：后台任务、超时、取消和恢复；
- 沙箱：运行分析脚本，但默认无生产网络权限；
- Eval：正常、数据缺失、工具超时、冲突证据和恶意日志；
- 可观测：完整 Trace、Token、工具耗时和每任务成本；
- Web UI：展示计划、证据、步骤、审批和最终报告。

#### 面试中可展示的技术取舍

- 为什么先用单 Agent 而不是多 Agent；
- 为什么把日志查询与工单创建分成不同权限工具；
- 如何避免重试导致重复创建工单；
- 如何在日志包含恶意指令时阻止提示注入；
- 如何从检查点恢复长任务；
- 如何证明新提示或新模型没有造成回归；
- 如何计算每个成功调查任务的成本。

### D.5 高频面试问题

1. 什么情况下不应该使用 Agent？
2. 工作流、单 Agent 和多 Agent 如何选择？
3. 如何设计可重试但不重复执行的写工具？
4. MCP 与 A2A 分别解决什么问题？
5. 长上下文为什么不能替代记忆与状态管理？
6. 如何处理工具超时、限流和部分成功？
7. 如何评估一个多轮 Agent，而不只看最终文本？
8. 什么是间接提示注入？为什么提示词不能单独防住它？
9. 如何设计人工审批而不让体验完全退化？
10. 如何让任务跨进程、跨小时恢复？
11. 如何定位 RAG 错误发生在召回、重排还是生成阶段？
12. 如何比较两个模型或两个 Agent 框架的真实成本？

回答这些问题时，最好遵循“场景约束 → 设计选择 → 失败模式 → 监控指标 → 验证结果”的结构，而不是背诵框架概念。

### D.6 简历表述建议

避免写“熟悉 LangChain、掌握 Prompt Engineering”这类难以验证的描述。可以写成：

> 构建可恢复的 IT 事件调查 Agent，集成 5 个只读 MCP 工具和人工审批写工具；使用 PostgreSQL 事件记录与幂等键保证任务恢复和写操作安全；建立 120 条回归评估集，将端到端成功率从 68% 提升至 87%，同时把每个成功任务成本降低 31%。

其中所有数字都必须来自真实测试。即使项目是个人作品，也应保留评估脚本、数据定义和可复现报告。
