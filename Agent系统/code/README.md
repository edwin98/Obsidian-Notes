# 5G Test Verification Agent

基于 LangGraph 构建的 5G 测试验证 Agent 系统，使用 LangSmith 进行链路追踪与评估。

## 架构

```
START → agent → guardrail → tools → agent (ReAct 循环)
                     ↓ 危险操作
                    hitl → agent
          ↓ 无工具调用 / 置信度足够
        result_judge → END
```

**节点说明**

| 节点 | 职责 |
|---|---|
| `agent` | LLM 推理，决定调用工具或输出最终结论 |
| `guardrail` | 检测危险操作，触发 HITL 拦截 |
| `tools` | 执行工具调用，缓存结果至 `tool_outputs` |
| `hitl` | 人工介入节点（生产模式暂停等待审批） |
| `result_judge` | 双轨判定：统计规则 + LLM 语义综合 |

**熔断机制**

- 硬熔断：`recursion_limit=15`，超出强制终止
- 软熔断：连续工具错误 >= 2 次，或置信度 < 0.65，触发 HITL

## 快速开始

```bash
cd Agent系统/code

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 运行（in-memory 模式）
python main.py

# 4. 自定义 query
python main.py --query "Test NAS authentication in roaming scenario"

# 5. 生产模式（需要 PostgreSQL）
python main.py --postgres

# 6. 运行 LangSmith 评估
python main.py --eval
```

## 环境变量

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API 密钥 |
| `LANGSMITH_API_KEY` | LangSmith API 密钥 |
| `LANGSMITH_PROJECT` | LangSmith 项目名，默认 `5g-test-agent` |
| `LANGCHAIN_TRACING_V2` | 开启链路追踪，设为 `true` |
| `POSTGRES_URI` | PostgreSQL 连接串（`--postgres` 模式必填） |

## 文件结构

```
code/
├── config.py        # 配置常量与环境变量
├── state.py         # AgentState 定义
├── tools.py         # 5 个 Mock 工具
├── nodes.py         # 图节点实现
├── graph.py         # LangGraph 状态机与路由
├── evaluation.py    # LangSmith 四维评估套件
└── main.py          # CLI 入口
```

## LangSmith 评估维度

| 评估器 | 对应维度 |
|---|---|
| `verdict_accuracy` | 格式合规 + 工具调用正确性 |
| `confidence_threshold` | 置信度是否满足阈值 |
| `no_hitl_triggered` | 轨迹效率（agent 是否自主完成） |
| `safety_compliance` | 安全合规（高危操作拦截） |

## HITL 生产接入

`graph.py` 中的 `create_graph_with_postgres()` 已配置 `interrupt_before=["hitl"]`。
生产流程：触发暂停 → State 序列化至 Postgres → 发送告警 → 人工审批后调用 `.invoke()` 并传入修改后的 State 恢复执行。
