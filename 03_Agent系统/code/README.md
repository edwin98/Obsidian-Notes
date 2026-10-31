# 测试与排查 Agent：统一教学实现

## 一、直观理解

命令行、页面、API、图展示和任务评测共用一套 `TaskService`。系统支持按需测试、多轮调查、条件修改、查询、暂停、继续、取消与修复验证。

默认使用规则意图器、规则计划器和确定性沙盒，无需密钥或 GPU。在线模式只替换意图与计划的生成方式，执行仍使用同一个沙盒和业务检查。当前没有连接真实设备。

完整交互见 [系统运行全流程与交互拓扑](../18_系统运行全流程与交互拓扑.md)，源码阅读见 [学习指南](学习指南.md)，模块定位见 [代码文件索引](代码文件索引.md)。

### 安装与运行

在本目录使用 Python 3.12 或更新版本：

~~~powershell
uv sync --group dev
uv run python -X utf8 main.py
uv run python -X utf8 main.py --graph
uv run python -X utf8 main.py --interactive
uv run python -X utf8 main.py --query "验证新版本 A 到 B 的切换"
uv run python -X utf8 main.py --db tasks.sqlite --interactive
~~~

默认示例会经历条件比较、简化和独立确认，实际实验和运行都会保存。交互中可以依次输入“复现升级后移动视频卡顿”“现在怎么样了”“这是修复版，按上次方法验证”。`new` 开始新会话，`exit` 退出；新会话不会取消已有任务。

### 页面与图展示

~~~powershell
uv run python -X utf8 main.py --ui
uv run python -X utf8 visualize.py
~~~

页面地址为 http://127.0.0.1:8000 。取消“自动跑到本轮结束”可逐组推进，观察条件、运行、证据和真实状态。`visualize.py` 输出同一个任务工作流的 Mermaid 源码，不执行测试、不调用模型，也不访问图片渲染服务。

## 二、详细技术拆解

### 唯一运行主线

`main.py / api.py → tasks_cli.py / task_api.py → TaskService.message → prepare_step → commit_step → execute_step → 结果汇总与报告`。

`--graph` 通过 `task_workflow.py` 展示计划、提交、执行与读取节点，仍复用任务服务的同一套方法。后台执行和 CLI 循环只是不同的驱动方式，不另写业务规则。

### API

`POST /tasks/messages` 接收 `message_id`、`conversation_id`、`text`、可选 `target_task_id`、`expected_task_version` 和 `auto_run`。`auto_run=false` 只理解消息，随后使用逐组接口推进。

| 接口 | 作用 |
| --- | --- |
| `GET /tasks/{task_id}` | 读取任务与状态 |
| `POST /tasks/{task_id}/step` | 推进一组实验或检查是否结束 |
| `GET /tasks/{task_id}/experiments` | 读取实验与运行 |
| `GET /tasks/{task_id}/events` | 读取已发生事件，返回 JSON |
| `GET /tasks/{task_id}/report` | 读取报告或生成只读阶段视图 |
| `GET /health` | 读取服务状态与在途任务数量 |

暂停、继续、查询和取消都走消息入口。相同消息重传不重复执行，相同编号换内容返回冲突。查询和报告不触发新实验。API 仅供本机单进程教学，尚未接生产认证。

### 配置、保存与恢复

复制 `.env.example` 为 `.env` 后填写模型密钥，CLI 使用 `--online`：

~~~powershell
uv run python -X utf8 main.py --online --interactive
uv run python -X utf8 main.py --ui --online --db tasks.sqlite
~~~

CLI 和页面用 `--db` 指定 SQLite 文件。直接通过 `uvicorn api:app --host 127.0.0.1` 启动 API 时，使用 `AGENT_TASK_MODE` 和 `AGENT_TASK_DB`；默认离线并保存在内存。模型密钥与可选追踪参数由 `config.py` 读取。

重启后，未完成动作标记为需核查，不盲目重新执行。真实平台恢复需要实现按运行身份查询与收尾，本沙盒没有该连接。

### 评测与训练

~~~powershell
uv run python -X utf8 task_evaluation.py --output reports/evaluation.json
uv sync --extra training --group dev
uv run python -X utf8 -m training.inspect_batch
uv run python -X utf8 -m training.tiny_smoke
~~~

`task_evaluation.py` 驱动实际任务服务，`task_grading.py` 独立核验事实。训练使用专家审核、决策前快照和统一 `PlanDecision` 目标；数据准备与 SFT/DPO 入口均在 `training/`。数学演练与极小模型自检不代表任务质量，日常任务不会自动启动训练。详见 [训练说明](training/README.md)。

### 检查

~~~powershell
uv run python -m ruff format .
uv run python -m ruff check .
uv run python -X utf8 -m pytest tests -q
~~~

基础依赖可运行主流程与任务评测；没有可选训练依赖时，张量测试会明确跳过。Windows 也可用 `.venv/Scripts/python.exe` 代替 `uv run python`。
