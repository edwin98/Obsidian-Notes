# Prompt Engineering 核心技巧与进阶框架要点

Prompt Engineering（提示词工程）是通过精心设计和优化输入文本（Prompt），以引导大语言模型（LLM）生成高质量、符合预期输出的一门技术。以下从核心原则、经典框架到进阶策略进行全面总结。

---

## 1. 核心设计原则 (The Golden Rules)

无论使用什么模型，优秀的 Prompt 都应遵循以下三大基本原则：

1. **清晰明确 (Clarity & Specificity)**
   - **错误示范**：“写一篇关于AI的文章。”
   - **正确示范**：“写一篇 800 字的科普文章，面向初中生解答通用人工智能（AGI）是什么，要求语言风趣幽默，并举三个生活中的例子。”
   - **要点**：避免模糊的形容词，明确字数、风格、受众和具体包含的要素。

2. **提供充足上下文 (Context is King)**
   - 单次调用只看到输入上下文；持久记忆需要应用保存并选择性注入。长上下文不等于自动具备跨会话记忆。
   - 告诉模型“你是谁”、“面临什么问题”、“有哪些前置条件”。例如赋予它专家角色（Role-playing）。

3. **结构化与分隔符 (Use Separators and Structures)**
   - 使用 Markdown 标记、XML 标签（如 `<context>`, `<instructions>`, `<output_format>`）或引号（`"""`），将指令、背景数据和输出要求严格隔开。
   - 这有助于明确输入边界，但不能单独保证事实正确或消除指令遗忘。

---

## 2. 经典的 Prompt 结构框架

为了系统化地编写长篇 Prompt，业界总结了多个有效框架，最常用的是 **CRISPE** 和 **BROKE** 框架：

### CRISPE 框架（最通用）
- **C (Capacity & Role)**：能力与角色（例如：“你是一位拥有20年经验的资深 Python 架构师...”）
- **R (Rule & Context)**：规则与背景（例如：“由于新加入的实习生较多，我们需要重写代码规范...”）
- **I (Instruction)**：指令/任务（例如：“请输出一份 Python Restful API 开发的规范文档...”）
- **S (Style)**：风格（例如：“语言专业客观，采用分点论述，必要处包含代码示例...”）
- **P (Perferred Format)**：期望格式（例如：“以 Markdown 格式输出，包含目录和三个主要章节...”）
- **E (Example)**：示例（如果是Few-shot，可以在这里提供示范）。

---

## 3. 进阶提示词技巧 (Advanced Techniques)

随着任务复杂度的提升，基础的 Instruction 不够用时，需要引入以下高阶范式：

### 3.1 零样本与少样本提示 (Zero-shot & Few-shot Prompting)
- **Zero-shot**：仅提供任务描述，不给例子。（适用于简单任务和强大的底座模型，如 GPT-4, Claude 3.5 Sonnet）。
- **Few-shot**：在 Prompt 中塞入 2~5 个 (Input, Output) 的成对示例。
  - *原理*：利用 In-context Learning（上下文学习），让模型隐式学习输出的分布特征和格式约束。极大地提高输出结构的一致性。

### 3.2 思维链 (Chain-of-Thought, CoT)


- **核心句式**：“Let's think step by step.”（让我们一步一步地思考）。
- **适用场景**：数学计算、逻辑推演、复杂归纳。
- **原理**：将一个复杂问题拆解为多个中间步骤，在输出中组织中间步骤，可能改善某些多步任务；效果依模型与任务而定，额外 token 也增加延迟，不能视为普遍准确性保证。
- **进阶变体**：
  - **Zero-shot CoT**：直接在强迫语中加入“请逐步思考”。
  - **Few-shot CoT**：在给出的示例中，不仅包含最终答案，还包含完整的推理过程。

普通生成模型的 CoT 提示与模型内部推理机制是不同层面。对于内置推理模型，官方建议优先明确目标、约束、输入边界和验收条件，通常无需显式要求输出逐步思考过程。可先测试清晰的零样本任务，再判断是否需要示例。[官方提示建议](https://developers.openai.com/api/docs/guides/reasoning-best-practices)

推理强度/预算与最终回答长度分别控制不同部分。工具交互时，某些 API 还要求保留返回的 reasoning Items；业务端不应自行伪造模型内部推理内容。[推理接口文档](https://developers.openai.com/api/docs/guides/reasoning)

### 3.3 思维树 (Tree of Thoughts, ToT) & 检索增强提示 (RAG)
- **ToT (Tree of Thoughts)**：不仅让模型一步步想，还让模型在每一步探索多条路径（如思考方案A、B、C的优缺点），评估后再选择一条路径继续往下走。常用于长文写作和极度复杂的运筹规划。
- **RAG (检索增强生成)**：针对模型可能胡编乱造（幻觉）的问题，在 Prompt 中强行注入外部检索系统查找到的知识。
- *要求*：必须在 Prompt 中声明“*请**仅仅**基于以下 <context> 标签内提供的信息来回答问题，如果其中找不到答案，请回答‘我不知道’。”

### 3.4 结构化输出能力 (Structured Output)

在业务流中调用 LLM，通常要求 JSON 或特定 Schema。应先了解接口能保证到哪一层：

JSON 前缀预填、停止词或 few-shot 可能帮助特定模型，但 API 未必支持，不能作为通用保证。原生严格 Schema 可以约束结构，仍需业务校验；简短理由字段按任务需要设置。[结构化输出文档](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=responses)

详见 [[大模型结构化输出技术解析]]。

---

## 4. 常见陷阱与避坑指南

1. **幻觉陷阱 (Hallucination)**
   - **症状**：模型在一本正经地胡说八道。
   - **对策**：加入“知识库外拦截指令”（例如，“如果你不确定，请不要猜测，直接指出缺乏信息”）。
2. **负面指令悖论 (Negative Constraints)**
   - **症状**：告诉模型“不要输出 A”，结果它偏偏输出了 A（类似“不要想一头粉色的大象”）。
   - **对策**：将负面指令转化为正面指令。例如：将“不要输出除代码以外的废话”改为“**只**输出纯代码和必要的代码注释”。
3. **指令丢失 (Lost in the Middle)**
   - **症状**：在几千字的超长 Prompt 中，模型忽略了中间写的要求。
   - **对策**：把最核心、最强制的指令前置（写在第一段）或后置（写在最后一段，模型“记忆”最深刻的位置）；利用 `<important>` 标签包裹。

---

### 模型、模态与 API 的适配

DeepSeek V4.1 Flash 于 2026-09-10 更新原生视觉和模型别名；V4 API 路线支持推理强度设置。模型、输入模态、API 和预算不同，提示效果需要按对应条件解释；历史项目的提示模板保留原背景。[官方更新记录](https://api-docs.deepseek.com/updates/)、[V4 API 说明](https://api-docs.deepseek.com/news/news260813/)

模型架构与发布路线见 [[MoE混合专家模型技术解析]]。

## 5. 总结：优质 Prompt 设计流
一个成熟的 Prompt 调试流应该是：
**写出初版** $\rightarrow$ **测试边界 Case** $\rightarrow$ **发现模型不听话的地方** $\rightarrow$ **补充 Few-shot 或更严格的约束条件** $\rightarrow$ **固化模板（存入代码或 Prompt 库）**。
