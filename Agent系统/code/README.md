# 5G 测试 Agent：可执行教学实现

从 [学习指南](学习指南.md) 开始。代码对应上层文档中的三个闭环：ReAct 工具调用、安全围栏与人工审批、离线专家反馈。中文注释解释状态变化与设计理由，测试覆盖正常流程、故障及边界条件。

## 快速运行

在本目录执行（Python 3.12 或更新版本）：

```powershell
uv sync --group dev
uv run python demo.py
uv run python demo.py --scenario regression
uv run python demo.py --scenario danger
uv run python demo.py --scenario review
uv run python offline_evaluation.py
uv run pytest tests -q
```

这几条命令无需 API 密钥、GPU 或数据库。`danger` 和 `review` 会真正暂停，输入 `approve` 才继续，其余输入拒绝。离线模型是按规则请求工具的替身，用于理解工程流程。

## 流程

```text
START → clarify → agent → guardrail → tools → agent
            ↓        ↓        ↓
          追问     result_judge  hitl（interrupt）
                     ↓          ↓ Command(resume=审批决定)
                   END/HITL    执行已审核工具 / 结束复核 / 拒绝终止
```

审批只允许继续动作，不会提高模型置信度，也不意味着测试通过。

## 文件导航

| 文件 | 学习重点 |
|---|---|
| `state.py` | 状态、消息追加、独立初始化 |
| `nodes.py` | 模型请求、工具观察、安全围栏、双轨判定 |
| `graph.py` | 路由、断点、Checkpoint 与依赖注入 |
| `tools.py` | 确定性沙盒、会话关联、参数校验 |
| `demo.py` | 同一张图上的离线模型替身与节点轨迹 |
| `rag.py` | 本地模拟检索、融合、精排及上下文引用 |
| `multi_agent.py` | Send 分发、独立 Worker 图、汇总与签发门槛 |
| `api.py` | HTTP、逐节点 SSE、审批与澄清续接 |
| `statistical.py` | 原始重复样本的 Welch/配对 t/KS 检验 |
| `learning.py` | 专家确认数据导出 SFT/DPO |
| `offline_evaluation.py` | 真实状态图上的固定场景工程回归 |
| `evaluation.py` | 可选 LangSmith 在线评测 |
| `eval_system.py` | 原有评测指标参考实现，部分入口使用模拟轨迹 |
| `training/`、`training_llamafactory/` | 原有模型训练参考脚本，需要独立 GPU 环境验证 |
| `tests/` | 节点、工具、状态图、API、检索、统计、反馈和多 Agent 测试 |

## 在线模型与接口

复制 `.env.example` 为 `.env`，填写 `DEEPSEEK_API_KEY` 后运行 `uv run python main.py` 或 `uv run python main.py --interactive`。默认不启用 LangSmith 追踪。

离线 API：

```powershell
$env:AGENT_MODE = "offline"
$env:AGENT_SCENARIO = "danger"
uv run uvicorn api:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/docs`，依次调用 `/run`、`/result/{task_id}`、`/resume/{task_id}`。`/stream/{task_id}` 提供 SSE；`/clarify/{task_id}` 追加澄清回答。

PostgreSQL 演示需额外安装 `uv sync --group dev --extra postgres`，配置 `POSTGRES_URI` 后运行 `uv run python main.py --postgres`。

## 验证边界

本实现是可测试的教学系统。工具不会连接真实 5G 网元；向量检索和精排仍是本地近似。API 任务与事件位于单进程内存，重启即丢失。真实模型、PostgreSQL、外部检索服务和 GPU 训练需要各自的集成验证；本地测试结果不能替代这些验证。
