"""统一命令行入口：CLI、页面和图展示共用同一套任务服务。"""

import argparse

from tasks_cli import run_tasks

DEFAULT_QUERY = "请在实验室复现升级后移动视频卡顿"


def main():
    parser = argparse.ArgumentParser(description="测试与排查助手")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="测试或调查请求")
    parser.add_argument("--interactive", "-i", action="store_true", help="多轮交互")
    parser.add_argument(
        "--online", action="store_true", help="使用在线模型；执行仍为沙盒"
    )
    parser.add_argument("--db", default=":memory:", help="SQLite 路径，默认保存在内存")
    parser.add_argument("--graph", action="store_true", help="显示计划、提交与执行节点")
    parser.add_argument("--ui", action="store_true", help="启动本机学习界面")
    parser.add_argument("--port", type=int, default=8000, help="页面服务端口")
    args = parser.parse_args()
    if args.ui:
        import uvicorn

        import task_api
        from api import app

        instance = task_api.create_service(
            mode="online" if args.online else "offline", path=args.db
        )
        task_api._service = instance
        print(f"学习界面：http://127.0.0.1:{args.port}/")
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port)
        finally:
            instance.store.close()
        return
    run_tasks(
        args.query,
        interactive=args.interactive,
        online=args.online,
        db=args.db,
        graph=args.graph,
    )


if __name__ == "__main__":
    main()
