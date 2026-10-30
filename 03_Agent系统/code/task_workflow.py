"""新版调查图：把文档中的“计划、提交、执行”变成可观察节点。

阅读顺序：task_models → task_service.prepare_step/commit_step/execute_step → 本文件。
图状态只放本轮引用与轻量计划，完整任务和证据仍从 TaskStore 读取。
一次图调用最多推进一组实验；END 表示本轮调度返回，不等于任务完成。
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from task_models import ExecutionBatch, PreparedStep


class WorkflowState(TypedDict, total=False):
    task_id: str
    prepared: dict | None
    batch: dict | None
    status: str


def create_task_workflow(service):
    """与直接调用 service.step 共用业务方法，避免维护两套校验规则。"""

    def plan(state):
        # prepare_step 在锁外调用模型。没有可推进工作时返回 None。
        prepared = service.prepare_step(state["task_id"])
        return {
            "prepared": prepared.model_dump() if prepared else None,
            # 每轮覆盖临时字段；不使用列表追加 reducer 累积过期计划。
            "batch": None,
        }

    def commit(state):
        # 模型返回后重新读取数据库并检查版本，再预留额度、登记实验。
        prepared = PreparedStep.model_validate(state["prepared"])
        batch = service.commit_step(prepared)
        return {"batch": batch.model_dump() if batch else None}

    def execute(state):
        batch = ExecutionBatch.model_validate(state["batch"])
        task = service.execute_step(batch)
        return {"status": task.status}

    def read(state):
        # WAIT、完成、交接或版本冲突都需要读取实际状态，不能猜结果。
        return {"status": service.get(state["task_id"]).status}

    builder = StateGraph(WorkflowState)
    builder.add_node("plan", plan)
    builder.add_node("commit", commit)
    builder.add_node("execute", execute)
    builder.add_node("read", read)
    builder.add_edge(START, "plan")
    builder.add_conditional_edges(
        "plan", lambda state: "commit" if state["prepared"] else "read"
    )
    builder.add_conditional_edges(
        "commit", lambda state: "execute" if state["batch"] else "read"
    )
    builder.add_edge("execute", "read")
    builder.add_edge("read", END)
    # 本教学图每次从业务库重新加载，不叠加另一套持久化事实。
    # 旧 graph.py 单独演示 checkpointer 与 interrupt，两者不是同一个图。
    return builder.compile()


def run_task_workflow(service, task_id, *, on_node=None):
    """运行多轮图调用；打印节点可看到一次实验怎样穿过整个框架。"""
    graph = create_task_workflow(service)
    while service.get(task_id).status == "READY":
        before = service.get(task_id)
        for update in graph.stream({"task_id": task_id}, stream_mode="updates"):
            if on_node:
                for name, values in update.items():
                    on_node(name, values)
        after = service.get(task_id)
        # 防止重复调度在途运行时空转；真正事件到来后再推进。
        if after.model_dump() == before.model_dump():
            break
    return service.get(task_id)
