"""
5G Test Verification Agent — CLI entry point.

Usage:
  # 单次查询（非交互）
  python main.py
  python main.py --query "Test NAS authentication in roaming scenario"

  # 多轮对话交互模式（支持意图澄清）
  python main.py --interactive

  # 使用 PostgreSQL checkpointer
  python main.py --interactive --postgres

  # 运行 LangSmith 评测套件
  python main.py --eval
"""

import argparse
import json
import uuid

from langchain_core.messages import HumanMessage

from config import RECURSION_LIMIT

DEFAULT_QUERY = "Run regression tests for 5G NR handover feature between gNB cells"

INITIAL_STATE = {
    "current_step": "start",
    "tool_outputs": {},
    "error_count": 0,
    "confidence_score": 1.0,
    "hitl_required": False,
    "hitl_feedback": "",
    "final_result": None,
    "needs_clarification": False,
    "clarification_question": "",
    "hitl_rejected": False,
}


def _build_graph(use_postgres: bool):
    if use_postgres:
        from graph import create_graph_with_postgres

        graph = create_graph_with_postgres()
        print("[Agent] Checkpointer: PostgreSQL")
    else:
        from graph import create_graph_in_memory

        graph = create_graph_in_memory()
        print("[Agent] Checkpointer: in-memory")
    return graph


def _print_chunk_update(node_name: str, update: dict) -> None:
    """打印单个节点更新的调试信息。"""
    print(f"\n[{node_name}]")

    msgs = update.get("messages", [])
    for msg in msgs:
        content = getattr(msg, "content", "")
        tool_calls = getattr(msg, "tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                print(f"  tool_call : {tc['name']}({json.dumps(tc['args'])})")
        elif content:
            preview = content[:300].replace("\n", " ")
            print(f"  content   : {preview}{'...' if len(content) > 300 else ''}")

    if "confidence_score" in update:
        print(f"  confidence: {update['confidence_score']:.2f}")
    if "hitl_required" in update and update["hitl_required"]:
        print("  ** HITL triggered **")
    if update.get("final_result"):
        try:
            result = json.loads(update["final_result"])
            print("\n  FINAL RESULT:")
            print(f"    verdict         : {result.get('verdict')}")
            print(f"    confidence_score: {result.get('confidence_score')}")
            print(f"    root_cause      : {result.get('root_cause')}")
            if result.get("issues"):
                print(f"    issues          : {result.get('issues')}")
        except json.JSONDecodeError:
            print(f"  FINAL RESULT (raw): {update['final_result']}")


def run_agent(
    query: str, use_postgres: bool = False, thread_id: str = "demo-001"
) -> None:
    graph = _build_graph(use_postgres)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    state = {**INITIAL_STATE, "messages": [HumanMessage(content=query)]}

    print(f"[Agent] Query : {query}")
    print("=" * 64)

    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
            _print_chunk_update(node_name, update)

    print("\n" + "=" * 64)


def run_interactive(use_postgres: bool = False) -> None:
    """
    多轮对话交互模式。

    澄清流程：
      1. clarify 节点判断请求不明确 → needs_clarification=True，图提前结束
      2. 主循环打印澄清问题，等待用户输入
      3. 用户回答追加进同一 thread 的 messages，图重新从 START 执行
      4. clarify 节点看到完整对话历史，判断是否已明确；明确则继续到 agent
      5. 循环直至完成或用户退出

    命令：
      quit / exit — 退出程序
      new         — 丢弃当前对话，开始全新请求
    """
    graph = _build_graph(use_postgres)

    print("[Agent] 5G 测试验证 Agent — 多轮对话模式")
    print("[Agent] 命令: quit=退出  new=开始新对话")
    print("=" * 64)

    def _new_session() -> tuple[str, dict]:
        tid = str(uuid.uuid4())
        cfg = {"configurable": {"thread_id": tid}, "recursion_limit": RECURSION_LIMIT}
        return tid, cfg

    thread_id, config = _new_session()
    is_new_query = True  # True: 期待新测试请求；False: 期待澄清回答

    while True:
        try:
            hint = "请输入测试请求" if is_new_query else "请提供补充信息"
            user_input = input(f"\n[用户] {hint}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Agent] 已退出。")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print("[Agent] 已退出。")
            break

        if user_input.lower() == "new":
            thread_id, config = _new_session()
            is_new_query = True
            print("[Agent] 已开始新对话。")
            continue

        # 首次请求携带完整初始状态；后续（澄清回答）只追加新消息
        if is_new_query:
            state = {**INITIAL_STATE, "messages": [HumanMessage(content=user_input)]}
        else:
            state = {"messages": [HumanMessage(content=user_input)]}

        print("\n" + "-" * 40)

        needs_clarification = False
        clarification_question = ""
        has_final_result = False

        for chunk in graph.stream(state, config=config, stream_mode="updates"):
            for node_name, update in chunk.items():
                _print_chunk_update(node_name, update)
                if update.get("needs_clarification"):
                    needs_clarification = True
                    clarification_question = update.get("clarification_question", "")
                if update.get("final_result"):
                    has_final_result = True

        print("-" * 40)

        if needs_clarification:
            # 打印澄清问题，下一轮继续同一会话
            print(f"\n[Agent] {clarification_question}")
            is_new_query = False
        else:
            # 本次对话完成，重置会话供下次新请求使用
            is_new_query = True
            thread_id, config = _new_session()
            if has_final_result:
                print("\n[Agent] 分析完成。输入新请求继续，或输入 'new' 开始新对话。")


def main() -> None:
    parser = argparse.ArgumentParser(description="5G Test Verification Agent")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="测试请求（非交互模式）")
    parser.add_argument(
        "--postgres", action="store_true", help="使用 PostgreSQL checkpointer"
    )
    parser.add_argument(
        "--thread-id", default="demo-001", help="会话 thread ID（非交互模式）"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="多轮对话交互模式"
    )
    parser.add_argument("--eval", action="store_true", help="运行 LangSmith 评测套件")
    args = parser.parse_args()

    if args.eval:
        from evaluation import run_evaluation

        run_evaluation()
    elif args.interactive:
        run_interactive(use_postgres=args.postgres)
    else:
        run_agent(args.query, use_postgres=args.postgres, thread_id=args.thread_id)


if __name__ == "__main__":
    main()
