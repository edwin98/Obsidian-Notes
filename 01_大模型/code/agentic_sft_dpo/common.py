"""公共部分：任务定义、工具执行、对话编码（损失掩码）、模型加载。

整条流水线只有一个 Agent 任务：
    用户给一道 GSM8K 小学数学应用题，模型可以调用 calculator 工具做算术，
    工具结果由程序真实执行后回填，模型最后输出 "Final answer: <数字>"。

所有脚本共享这里的系统提示、工具定义和编码方式，保证 训练 / 采样 / 评测 三处看到的格式完全一致。
"""

from __future__ import annotations

import ast
import json
import math
import operator
import os
import random
import re
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

DEFAULT_MODEL = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

SYSTEM_PROMPT = (
    "You are a careful math agent. Solve the problem step by step. "
    "Do NOT do arithmetic in your head: for every calculation, call the `calculator` tool "
    "and use the number it returns. When you are done, reply with a short explanation "
    "and finish with a line of the form `Final answer: <number>`."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an arithmetic expression. Supports + - * / // % ** and parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A pure arithmetic expression, e.g. '(48 + 24) / 2'",
                    }
                },
                "required": ["expression"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# 1. 工具：安全的计算器（只解析算术 AST，不执行任意代码）
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"unsupported syntax: {ast.dump(node)[:60]}")


def calculate(expression: str) -> float:
    expr = expression.replace(",", "").replace("×", "*").replace("÷", "/").replace("^", "**")
    expr = expr.replace("$", "").strip()
    return _eval_node(ast.parse(expr, mode="eval"))


def format_number(x: float) -> str:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.6f}".rstrip("0").rstrip(".")


def run_calculator(arguments: dict) -> str:
    """真实执行工具。出错时返回错误字符串，而不是抛异常——模型需要看到错误并自己修正。"""
    try:
        return format_number(calculate(str(arguments["expression"])))
    except Exception as e:  # noqa: BLE001
        return f"Error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# 2. 解析模型输出：工具调用、最终答案
# ---------------------------------------------------------------------------
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
FINAL_RE = re.compile(r"Final answer\s*[:：]\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)
NUMBER_RE = re.compile(r"-?[\d,]*\.?\d+")


def parse_assistant(text: str) -> tuple[dict, list[dict], bool]:
    """把一段模型生成的文本变成一条 assistant 消息。

    返回 (message, calls, ok)：
      - calls: 成功解析的工具调用 [{"name", "arguments"}]
      - ok:    所有 <tool_call> 块都是合法 JSON 且调用的是 calculator
    非法调用时 message 保留原始文本（这样 DPO 的 rejected 轨迹里就是模型真实写出来的错误）。
    """
    blocks = TOOL_CALL_RE.findall(text)
    has_open_tag = "<tool_call>" in text
    if not blocks:
        if has_open_tag:  # 写了开标签但没写完
            return {"role": "assistant", "content": text.strip()}, [], False
        return {"role": "assistant", "content": text.strip()}, [], True

    calls, ok = [], True
    for b in blocks:
        try:
            obj = json.loads(b)
            args = obj.get("arguments", {})
            if isinstance(args, str):
                args = json.loads(args)
            if obj.get("name") != "calculator" or "expression" not in args:
                raise ValueError
            calls.append({"name": "calculator", "arguments": {"expression": str(args["expression"])}})
        except Exception:  # noqa: BLE001
            ok = False
    if not ok:
        return {"role": "assistant", "content": text.strip()}, [], False

    content = text.split("<tool_call>")[0].strip()
    msg = {
        "role": "assistant",
        "content": content,
        "tool_calls": [{"type": "function", "function": c} for c in calls],
    }
    return msg, calls, True


def extract_answer(text: str) -> tuple[float | None, bool]:
    """返回 (数值, 是否按 'Final answer:' 格式给出)。没按格式时退而取最后一个数字，便于公平对比原模型。"""
    m = FINAL_RE.search(text)
    if m:
        return _to_float(m.group(1)), True
    nums = NUMBER_RE.findall(text)
    return (_to_float(nums[-1]) if nums else None), False


def _to_float(s: str) -> float | None:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def is_correct(pred: float | None, gold: float) -> bool:
    return pred is not None and abs(pred - gold) < 1e-4 * max(1.0, abs(gold))


# ---------------------------------------------------------------------------
# 3. 对话 → token，并生成“只在 assistant 段计算损失”的掩码
# ---------------------------------------------------------------------------
def render(tokenizer, messages, add_generation_prompt=False) -> str:
    return tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def encode_conversation(tokenizer, messages: list[dict], max_len: int) -> dict:
    """把整段多轮对话编码成 input_ids，并给出 loss_mask。

    loss_mask=1 的位置只有 assistant 自己写的内容（含 <tool_call> 与结尾 <|im_end|>）。
    system / user 问题 / 工具返回 都是“环境给的信息”，mask=0：模型要学的是怎么行动，而不是背工具输出。

    做法：逐个 assistant 消息，用“带生成提示的前缀”与“包含该消息的前缀”两次渲染，
    得到它在完整字符串里的字符区间，再用 offset_mapping 映射到 token。
    """
    full = render(tokenizer, messages)
    spans = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(render(tokenizer, messages[:i], add_generation_prompt=True))
        prefix = render(tokenizer, messages[: i + 1])
        assert full.startswith(prefix), "chat template 不满足前缀一致性，不能用这种方式算掩码"
        # prefix 以 "<|im_end|>\n" 结尾：<|im_end|> 要学（学会停），之后的换行推理时不会生成，不学
        end = len(prefix.rstrip("\n"))
        spans.append((start, end))

    enc = tokenizer(full, return_offsets_mapping=True, add_special_tokens=False)
    ids, offsets = enc["input_ids"], enc["offset_mapping"]
    mask = [0] * len(ids)
    for t, (s, e) in enumerate(offsets):
        for a, b in spans:
            if s >= a and e <= b and e > s:
                mask[t] = 1
                break
    ids, mask = ids[:max_len], mask[:max_len]
    return {"input_ids": ids, "loss_mask": mask, "truncated": len(enc["input_ids"]) > max_len}


def pad_batch(features: list[dict], pad_id: int) -> dict:
    """右侧 padding，训练用。labels 中不算损失的位置设为 -100。"""
    n = max(len(f["input_ids"]) for f in features)
    ids, attn, labels = [], [], []
    for f in features:
        pad = n - len(f["input_ids"])
        ids.append(f["input_ids"] + [pad_id] * pad)
        attn.append([1] * len(f["input_ids"]) + [0] * pad)
        labels.append([t if m else -100 for t, m in zip(f["input_ids"], f["loss_mask"])] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(attn),
        "labels": torch.tensor(labels),
    }


# ---------------------------------------------------------------------------
# 4. 模型加载：base →（合并 SFT LoRA）→（合并 DPO LoRA）
# ---------------------------------------------------------------------------
def get_device_dtype():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    print("[warn] 没检测到 CUDA，将在 CPU 上以 float32 运行（会很慢，仅适合冒烟测试）")
    return torch.device("cpu"), torch.float32


def load_tokenizer(model_name: str = DEFAULT_MODEL):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_name: str = DEFAULT_MODEL, adapters: list[str | Path] = ()):
    """加载基座，并按顺序把 LoRA 适配器合并进去（例如 [sft, dpo]）。"""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    device, dtype = get_device_dtype()
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    for path in adapters:
        model = PeftModel.from_pretrained(model, str(path)).merge_and_unload()
    return model.to(device)


# ---------------------------------------------------------------------------
# 5. 杂项
# ---------------------------------------------------------------------------
def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def lora_config(r: int = 16, alpha: int = 32, dropout: float = 0.05):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
