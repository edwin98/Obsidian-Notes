"""
5G Test Verification Agent — CLI entry point.

Usage:
  # Demo run (in-memory checkpointer)
  python main.py

  # Custom query
  python main.py --query "Test NAS authentication in roaming scenario"

  # Use PostgreSQL checkpointer (requires running Postgres)
  python main.py --postgres

  # Run LangSmith evaluation suite
  python main.py --eval
"""

import argparse
import json

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
}


def run_agent(query: str, use_postgres: bool = False, thread_id: str = "demo-001") -> None:
    if use_postgres:
        from graph import create_graph_with_postgres
        graph = create_graph_with_postgres()
        print(f"[Agent] Checkpointer: PostgreSQL")
    else:
        from graph import create_graph_in_memory
        graph = create_graph_in_memory()
        print("[Agent] Checkpointer: in-memory")

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }
    state = {**INITIAL_STATE, "messages": [HumanMessage(content=query)]}

    print(f"[Agent] Query : {query}")
    print("=" * 64)

    for chunk in graph.stream(state, config=config, stream_mode="updates"):
        for node_name, update in chunk.items():
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
                    print(f"\n  FINAL RESULT:")
                    print(f"    verdict         : {result.get('verdict')}")
                    print(f"    confidence_score: {result.get('confidence_score')}")
                    print(f"    root_cause      : {result.get('root_cause')}")
                    if result.get("issues"):
                        print(f"    issues          : {result.get('issues')}")
                except json.JSONDecodeError:
                    print(f"  FINAL RESULT (raw): {update['final_result']}")

    print("\n" + "=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(description="5G Test Verification Agent")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="Natural language test request")
    parser.add_argument("--postgres", action="store_true", help="Use PostgreSQL checkpointer")
    parser.add_argument("--thread-id", default="demo-001", help="Conversation thread ID")
    parser.add_argument("--eval", action="store_true", help="Run LangSmith evaluation suite")
    args = parser.parse_args()

    if args.eval:
        from evaluation import run_evaluation
        run_evaluation()
    else:
        run_agent(args.query, use_postgres=args.postgres, thread_id=args.thread_id)


if __name__ == "__main__":
    main()
