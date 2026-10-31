"""输出当前任务工作流的 Mermaid 图；不执行实验或调用模型。"""

from task_service import TaskService
from task_workflow import create_task_workflow


def main():
    service = TaskService()
    try:
        print(create_task_workflow(service).get_graph().draw_mermaid())
    finally:
        service.store.close()


if __name__ == "__main__":
    main()
