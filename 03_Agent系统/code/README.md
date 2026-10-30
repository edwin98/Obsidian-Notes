# 测试与排查 Agent：可运行教学实现

默认使用新版统一入口。用户可以要求测试、复现异常、修改条件、查询、暂停、继续、取消，以及验证修复。脚本仍为确定性沙盒，不连接真实设备。

## 按文档阅读整套代码

先看 [学习指南](学习指南.md)，再用 [代码文件索引](代码文件索引.md) 定位每个模块。新版主流程、旧图对照、评测和训练的边界都在指南中说明。

新增的可观察入口：

~~~powershell
uv run python -X utf8 main.py --graph
uv run python -X utf8 task_evaluation.py --output reports/evaluation.json
uv run python -X utf8 -m training.inspect_batch
uv run python -X utf8 -m training.tiny_smoke
~~~

前两项使用基础依赖；后两项需要可选 training 依赖。tiny_smoke 构造本地极小随机模型，CPU各做一步SFT和DPO，不下载模型，不代表真实质量。

## 先跑一遍

在本目录使用 Python 3.12 或更新版本：

~~~powershell
uv sync --group dev
uv run python main.py
uv run python main.py --interactive
uv run python main.py --query "验证新版本 A 到 B 的切换"
uv run python main.py --db tasks.sqlite --interactive
~~~

默认是离线规则替身，无需密钥或 GPU。main.py 默认示例会经历条件比较、简化和独立复核。不是直接打印一份预写结果，实际每组数据都会保存。

交互中可以依次输入：

~~~text
升级后移动视频卡顿，请在实验室查到能复现
现在怎么样了
这是修复版，按上次方法验证
这是修复变慢版，按上次方法验证
~~~

new 开始新对话，exit 退出。取消和暂停最适合在页面的逐组模式或后台运行期间观察。

## 打开界面

~~~powershell
uv run python demo.py --ui
~~~

浏览器打开 http://127.0.0.1:8000 。首页是新版任务界面，取消“自动跑到本轮结束”可逐组执行。页面显示参数、运行、证据与真实状态。

旧版 LangGraph 单步界面位于 /legacy。/demo/run、/run 等旧接口保留用于原教学与回归；新功能使用 /tasks/messages。

## 新版 API

POST /tasks/messages 接收 message_id、conversation_id、text、可选 target_task_id、expected_task_version 和 auto_run。auto_run=false 只理解要求，之后用逐组接口推进。

- GET /tasks/{task_id}：任务与状态。
- POST /tasks/{task_id}/step：完成下一组实验，或评价是否结束。
- GET /tasks/{task_id}/experiments：实验记录。
- GET /tasks/{task_id}/events：已发生事件，返回 JSON。
- GET /tasks/{task_id}/report：已有报告或只读阶段报告。

暂停、继续、查询和取消仍通过统一消息入口表达。同一消息编号的重传不重复执行；同一编号换内容会返回冲突。

接口未接生产认证，仅供本机单进程使用，不应直接暴露到网络。读接口不构成多租户隔离。

## 保存和恢复

默认记录只在内存中。API 设置 AGENT_TASK_DB 为 SQLite 文件路径，CLI 使用 --db，即可保留任务与消息。

服务重新启动后，之前正在执行的任务会标记需人工核查，不会重新触发未知动作。真实平台恢复还需用实际 session_id 查询平台，本示例没有该连接。

## 使用在线模型

复制 .env.example 为 .env，配置 DEEPSEEK_API_KEY。

~~~powershell
uv run python main.py --online --interactive
~~~

API 则设置 AGENT_TASK_MODE=online。模型负责输出结构化意图与下一组计划，程序仍检查对象、条件、次数与证据。真实模型本轮未调用验证，所有执行工具仍为沙盒。

旧单次测试图使用 main.py --legacy；--postgres 也仍走原图。新任务保存到 SQLite，不把两种保存方式混为一谈。

## 文件怎么读

| 文件 | 内容 |
| --- | --- |
| task_models.py | 任务、计划、实验与运行记录 |
| intent.py | 离线识别和在线模型识别 |
| experiments.py | 计划选择、实际沙盒执行、数据与恢复 |
| task_service.py、task_workflow.py | 统一消息，计划/提交/执行三个阶段，直接循环与LangGraph共用 |
| task_store.py | SQLite、消息去重与历史 |
| task_api.py、tasks_cli.py | HTTP 与命令行 |
| web/tasks.* | 新统一界面 |
| learning.py | 专家反馈与实验决策训练数据导出 |
| task_evaluation.py、task_grading.py | 完整episode、独立事实核验、分项判分与轨迹导出 |
| skill_library.py、skills/ | 按需读取排查方法，记录内容版本 |
| graph.py、nodes.py、state.py、tools.py | 保留的旧单次测试图和工具示例 |
| tests/ | 新旧工程回归 |
| training/、training_llamafactory/ | 独立训练参考，不由日常任务自动启动 |

## 检查

~~~powershell
uv run python -m ruff format .
uv run python -m ruff check .
uv run python -m pytest tests -q
uv run python task_evaluation.py
uv run python offline_evaluation.py
~~~

如本机已有虚拟环境，Windows 也可用 .venv/Scripts/python.exe 代替 uv run python。

测试检查实际程序状态和沙盒返回，不是现场效果。真实模型、设备、数据库服务与训练需要另行验证。上手流程见 [学习指南](学习指南.md)，正文见 [项目导航](../README.md)。
