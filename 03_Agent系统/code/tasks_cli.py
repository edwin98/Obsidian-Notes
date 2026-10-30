"""【学习定位：新版终端适配】
用户输入经MessageRequest进入TaskService，收到可执行信号才run；打印报告不等于执行测试。
对应11、17；--graph与直接循环共用业务规则。

新版统一入口的离线/在线命令行。
"""

import json

from task_api import create_service
from task_models import MessageRequest, identifier


def run_tasks(query, *, interactive=False, online=False, db=":memory:", graph=False):
    """终端只收消息和展示结果，业务规则始终由 TaskService 管理。"""
    service = create_service(mode="online" if online else "offline", path=db)
    conversation = identifier("chat")
    target = None
    print("测试与排查助手：离线模式是规则替身，所有执行工具均为沙盒。")
    while True:
        if interactive:
            try:
                text = input("\n要求（new 新对话，exit 退出）：").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text in {"exit", "quit"}:
                break
            if text == "new":
                conversation, target = identifier("chat"), None
                continue
            if not text:
                continue
        else:
            text = query
        try:
            response = service.message(
                MessageRequest(
                    message_id=identifier("msg"),
                    conversation_id=conversation,
                    target_task_id=target,
                    text=text,
                )
            )
            target = response["task_id"] or target
            print(response["message"])
            if response["run_requested"]:
                if graph:
                    from task_workflow import run_task_workflow

                    result = run_task_workflow(
                        service, target, on_node=lambda name, _: print(f"[节点] {name}")
                    )
                else:
                    result = service.run(target)
            elif target:
                result = service.get(target)
            else:
                result = None
            if result:
                print(
                    json.dumps(
                        {
                            "task_id": result.task_id,
                            "status": result.status,
                            "report": result.report or {"question": result.question},
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
        except ValueError as exc:
            print(f"没有执行：{exc}")
        if not interactive:
            break
