"""
visualize.py — Agent 流程图生成

LangGraph 内置三种可视化方式，无需额外建模工具：

  1. ASCII art      draw_ascii()          无任何依赖，终端直接输出
  2. Mermaid 文本   draw_mermaid()        输出 Mermaid 语法，可粘贴至 Obsidian / mermaid.live
  3. PNG 图片       draw_mermaid_png()    通过 mermaid.ink 在线 API 渲染，保存为本地文件
                                         （需要网络连接，无需本地安装 graphviz）

用法：
  python visualize.py              # 输出 ASCII + Mermaid 文本，并保存 PNG
  python visualize.py --ascii      # 仅终端 ASCII
  python visualize.py --mermaid    # 仅 Mermaid 文本
  python visualize.py --png        # 仅保存 PNG
"""

import argparse

from graph import create_graph_in_memory

OUTPUT_PNG = "agent_flow.png"


def show_ascii(graph) -> None:
    print("\n── ASCII 流程图 " + "─" * 48)
    # draw_ascii() 直接返回字符串，适合快速在终端确认图结构
    print(graph.get_graph().draw_ascii())


def show_mermaid(graph) -> None:
    print("\n── Mermaid 源码（粘贴至 Obsidian 代码块或 mermaid.live）" + "─" * 20)
    # draw_mermaid() 输出标准 Mermaid flowchart 语法
    # Obsidian 使用方式：新建代码块，语言设为 mermaid，粘贴输出内容即可渲染
    print(graph.get_graph().draw_mermaid())


def save_png(graph, path: str = OUTPUT_PNG) -> None:
    # draw_mermaid_png() 将 Mermaid 文本提交至 https://mermaid.ink 渲染为 PNG 字节流
    # 返回 bytes，需手动写文件；draw_mermaid_png(output_file_path=...) 可直接保存
    graph.get_graph().draw_mermaid_png(output_file_path=path)
    print(f"\n[Visualize] PNG 已保存至 {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Agent flowchart")
    parser.add_argument("--ascii", action="store_true", help="Print ASCII art only")
    parser.add_argument("--mermaid", action="store_true", help="Print Mermaid source only")
    parser.add_argument("--png", action="store_true", help="Save PNG only")
    parser.add_argument("--out", default=OUTPUT_PNG, help=f"PNG output path (default: {OUTPUT_PNG})")
    args = parser.parse_args()

    graph = create_graph_in_memory()

    # 无参数时输出全部三种格式
    all_modes = not (args.ascii or args.mermaid or args.png)

    if args.ascii or all_modes:
        show_ascii(graph)

    if args.mermaid or all_modes:
        show_mermaid(graph)

    if args.png or all_modes:
        save_png(graph, path=args.out)


if __name__ == "__main__":
    main()
