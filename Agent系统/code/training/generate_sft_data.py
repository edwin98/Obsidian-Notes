"""
generate_sft_data.py — SFT 训练数据生成流水线

目标：
  使用强 LLM（GPT-4o 或 Qwen2.5-72B）自动合成 5G 测试验证 Agent 的训练数据。
  采用 Self-Instruct + Magpie 思路，从种子数据出发，生成涵盖以下能力的对话：
    1. 5G 专有术语的正确理解与使用（RSRP/SINR/PUSCH/BWP 等）
    2. ReAct 推理范式（Thought → Action → Observation 的完整循环）
    3. 严格的 JSON 格式输出（confidence_score、verdict 等必填字段）
    4. 工具失败恢复（工具返回 500/空数据时的优雅降级）

数据流：
  种子数据（seed_examples.json）
    ↓ Step 1：Self-Instruct 指令扩增
  多样化测试请求指令（raw_instructions.jsonl）
    ↓ Step 2：Magpie 响应生成
  完整多轮对话（raw_dialogs.jsonl）
    ↓ Step 3：格式打包为 ChatML
  SFT 训练样本（sft_train.jsonl）

参考笔记：
  Agent系统/04_Agent模型后训练SFT_DPO.md § 3.1 数据来源与构建
"""

import asyncio
import hashlib
import json
import random
import re
from pathlib import Path

from openai import AsyncOpenAI  # pip install openai

# ── 配置区域 ────────────────────────────────────────────────────────────────

# 强模型 API（生产用 GPT-4o，测试用 DeepSeek）
STRONG_MODEL = "gpt-4o"
API_KEY = "your-api-key-here"
BASE_URL = None  # 若使用 OpenAI 原生，设为 None；DeepSeek 设为 "https://api.deepseek.com/v1"

# 生成数量目标
NUM_INSTRUCTIONS_PER_SEED = 10  # 每条种子数据扩增多少条指令
MAX_DIALOGS = 35_000             # 合成数据目标量（笔记：约 3.5 万条）

# 并发控制（防止 API 被打爆）
SEMAPHORE_LIMIT = 10

# 输出路径
OUTPUT_DIR = Path("data")

# ── 场景分布目标（笔记 §5.2）─────────────────────────────────────────────────
# 保证训练数据覆盖各类 5G 测试场景，避免模型只会处理"切换"场景
SCENARIO_DISTRIBUTION = {
    "handover": 0.25,        # 切换场景：最常见（LTE→NR 切换、gNB 间 Xn 切换）
    "interference": 0.20,    # 干扰场景：同频/异频干扰、SINR 劣化
    "capacity": 0.20,        # 容量场景：高并发压测、吞吐量测试
    "channel": 0.15,         # 信道场景：多径衰落、频率选择性衰落
    "auth_bearer": 0.10,     # 认证/承载场景：UE 接入、PDU Session 建立
    "error_recovery": 0.10,  # 工具失败恢复：体现容错能力（易被忽略但关键）
}

# ── 种子数据（Seed Examples）──────────────────────────────────────────────────
# 种子数据是专家手写的高质量示例，覆盖核心场景。
# 实际生产中从 seed_examples.json 加载，这里内嵌 3 条示例供演示。
SEED_EXAMPLES = [
    {
        "scenario": "handover",
        "user_input": "请验证 gNB-A 到 gNB-B 的 Xn 接口切换时延，近期做了 RRC 配置变更",
        "agent_thought": "需要先查询 Xn 切换相关测试用例，再执行仿真并收集时延指标",
        "tool_sequence": ["test_case_query", "simulation_runner", "metrics_collector", "baseline_comparator"],
        "verdict": "FAIL",
        "root_cause": "切换成功率 85% 低于基线 99%，根因：Xn 回传链路 SN Status Transfer 超时",
    },
    {
        "scenario": "interference",
        "user_input": "测试 5G NR 下行同频干扰抑制，当前站点覆盖重叠率约 35%",
        "agent_thought": "同频干扰测试需要在 interference 类别查询用例，关注 SINR 指标",
        "tool_sequence": ["test_case_query", "simulation_runner", "metrics_collector", "log_analyzer"],
        "verdict": "PASS",
        "root_cause": "SINR 均值 18.3 dB，满足基线要求 ≥ 15 dB",
    },
    {
        "scenario": "error_recovery",
        "user_input": "测试基站 B 的 PDCP 层 RLC 重传成功率",
        "agent_thought": "查询 PDCP 测试用例，注意工具可能返回空结果，需要重试逻辑",
        "tool_sequence": ["test_case_query", "test_case_query", "simulation_runner", "metrics_collector"],
        # 第二次 test_case_query 是重试（第一次工具失败场景）
        "verdict": "PASS",
        "root_cause": "RLC 重传成功率 99.2%，满足基线要求",
    },
]

# ── Step 1：Self-Instruct 指令扩增 ────────────────────────────────────────────

INSTRUCTION_GENERATION_PROMPT = """你是一位资深 5G 无线通信测试工程师。

基于以下测试场景背景，生成 {num_instructions} 条不同类型的 5G 测试验证请求。

测试场景类型：{scenario}
参考示例：{example}

要求：
1. 每条指令覆盖不同的测试角度（正向测试、边界场景、模糊诉求、异常恢复等）
2. 使用真实的 5G 术语（RSRP、SINR、PUSCH、BWP、A3 Event、gNB、UE、SN Status Transfer 等）
3. 难度分布：50% 标准场景，30% 边界场景，20% 模糊/异常场景
4. 禁止生成重复内容

请严格以 JSON 数组格式输出，每个元素是一条用户请求字符串：
["请求1", "请求2", ...]"""


async def generate_instructions(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    scenario: str,
    seed_example: dict,
    num_per_call: int = 10,
) -> list[str]:
    """
    使用 Self-Instruct 方法，基于种子示例批量生成多样化指令。

    Self-Instruct 核心思路（Wang et al., 2022）：
      - 给强 LLM 看几条专家写的高质量指令
      - 要求它"站在专家角度"生成更多相似但不同的指令
      - 再过滤低质量和重复的指令
    """
    async with semaphore:
        prompt = INSTRUCTION_GENERATION_PROMPT.format(
            num_instructions=num_per_call,
            scenario=scenario,
            example=json.dumps(seed_example["user_input"], ensure_ascii=False),
        )
        try:
            response = await client.chat.completions.create(
                model=STRONG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,   # 高温度 → 多样性更好，指令更丰富
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
            # 提取 JSON 数组（防止模型在 JSON 前后加解释性文字）
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                instructions = json.loads(match.group())
                return [str(i) for i in instructions if isinstance(i, str)]
        except Exception as e:
            print(f"[ERROR] 指令生成失败 (scenario={scenario}): {e}")
        return []


# ── Step 2：Magpie 响应生成 ────────────────────────────────────────────────────

# 工具定义（JSON Schema），注入到系统提示中
# 与 tools.py 保持一致，让强 LLM 生成符合平台格式的工具调用
TOOL_DEFINITIONS = """
可用工具：
1. test_case_query(feature: str, category: str) → 查询测试用例
   - feature: 功能名，如 "Xn_handover"、"interference_suppression"
   - category: "regression" | "sanity" | "stress"

2. simulation_runner(test_case_ids: list[str], env: str) → 执行仿真
   - test_case_ids: 从 test_case_query 返回的用例 ID 列表
   - env: "sandbox" | "staging"

3. metrics_collector(session_id: str) → 收集 KPI 指标
   - session_id: 从 simulation_runner 返回的 session_id

4. baseline_comparator(throughput: float, latency: float) → 与历史基线对比
   - throughput: 吞吐量（Mbps）
   - latency: P99 延迟（ms）

5. log_analyzer(log_type: str, session_id: str) → 信令日志分析
   - log_type: "signaling" | "rlc" | "pdcp"
   - session_id: 仿真会话 ID

6. fleet_manager(site_id: str, action: str, probe_count: int) → 多局点管理
"""

DIALOG_GENERATION_PROMPT = """你是一个 5G 测试验证 Agent，请严格遵循 ReAct 推理范式处理以下测试请求。

{tool_definitions}

ReAct 格式规范：
- 每步推理以 "Thought:" 开头，说明当前分析
- 工具调用以 "Action: <工具名>" 开头，参数用 JSON 对象表示
- 工具返回以 "[Tool Response]" 标注（你需要模拟合理的返回值）
- 最终结论输出 JSON 代码块，包含：confidence_score（0-1）、verdict（PASS/FAIL/INCONCLUSIVE）、summary

场景类型：{scenario}
请处理以下测试请求：{user_input}

注意事项：
1. 工具调用参数必须来自前一步工具的实际返回值（不能凭空捏造参数）
2. 若工具返回失败（模拟 {error_prob}% 概率），需重试一次后继续
3. confidence_score < 0.65 表示不确定，需触发人工审核
4. 最终输出必须是可解析的 JSON，不要加自然语言说明"""


async def generate_dialog(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    user_input: str,
    scenario: str,
    is_error_recovery: bool = False,
) -> dict | None:
    """
    使用 Magpie 思路生成完整的多轮 ReAct 对话。

    Magpie 核心思路（Xu et al., 2024）：
      - 直接给强 LLM 植入 Agent 的 System Prompt
      - 让它同时扮演"Agent"和"工具环境"两个角色
      - 生成端到端的完整对话轨迹（包含工具调用和返回值）

    is_error_recovery: 若为 True，强制在对话中加入工具失败场景
    """
    async with semaphore:
        # 工具失败概率：error_recovery 场景设为 50%，其他场景设为 10%
        error_prob = 50 if is_error_recovery else 10

        prompt = DIALOG_GENERATION_PROMPT.format(
            tool_definitions=TOOL_DEFINITIONS,
            scenario=scenario,
            user_input=user_input,
            error_prob=error_prob,
        )
        try:
            response = await client.chat.completions.create(
                model=STRONG_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,   # 适中温度：推理步骤需要准确，不能太随机
                max_tokens=3000,
            )
            raw_dialog = response.choices[0].message.content.strip()
            return {
                "user_input": user_input,
                "scenario": scenario,
                "raw_dialog": raw_dialog,
                "is_error_recovery": is_error_recovery,
            }
        except Exception as e:
            print(f"[ERROR] 对话生成失败 (input={user_input[:30]}...): {e}")
            return None


# ── Step 3：打包为 ChatML 格式 ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a 5G test verification agent. Follow the ReAct framework:
- Think step by step before each action
- Always start with test_case_query before simulation_runner
- Never invent tool parameters — use values from previous tool outputs
- Set confidence_score < 0.65 if uncertain — this triggers human review"""


def pack_to_chatml(dialog_record: dict) -> dict:
    """
    将生成的原始对话打包为标准 ChatML 格式（OpenAI messages 格式）。

    ChatML 格式是 Qwen 系列模型的原生训练格式，最终被 tokenizer 转化为：
      <|im_start|>system\n...<|im_end|>
      <|im_start|>user\n...<|im_end|>
      <|im_start|>assistant\n...<|im_end|>

    SFT 训练时只对 assistant 部分的 token 计算损失（user/system token 被 mask 掉）。
    """
    raw_dialog = dialog_record["raw_dialog"]

    # 简单解析：将原始对话封装为标准消息结构
    # 生产中应做更精细的解析（提取每个 Thought/Action/Observation）
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": dialog_record["user_input"]},
        {"role": "assistant", "content": raw_dialog},
    ]

    return {
        "messages": messages,
        "scenario": dialog_record["scenario"],
        "is_error_recovery": dialog_record["is_error_recovery"],
        # 用于后续数据污染检测
        "instruction_hash": hashlib.md5(
            dialog_record["user_input"].encode()
        ).hexdigest(),
    }


# ── 数据量分配计算 ────────────────────────────────────────────────────────────

def compute_scenario_counts(total: int, distribution: dict) -> dict:
    """
    根据目标分布计算每个场景需要生成的样本数量。

    笔记 §5.2：训练数据应覆盖各类测试场景，避免某类场景过拟合。
    实际数据分布偏离目标超过 5% 时，通过过采样/欠采样纠正。
    """
    counts = {}
    for scenario, ratio in distribution.items():
        counts[scenario] = int(total * ratio)
    # 修正由于取整导致的总量误差
    diff = total - sum(counts.values())
    first_key = list(counts.keys())[0]
    counts[first_key] += diff
    return counts


# ── 主流程 ───────────────────────────────────────────────────────────────────

async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    semaphore = asyncio.Semaphore(SEMAPHORE_LIMIT)

    # 计算各场景需要生成的数量
    scenario_counts = compute_scenario_counts(MAX_DIALOGS, SCENARIO_DISTRIBUTION)
    print(f"[INFO] 各场景数量目标: {scenario_counts}")

    # ── Step 1：为每个场景生成多样化指令 ────────────────────────────────────────
    print("\n[Step 1] 开始 Self-Instruct 指令扩增...")
    all_instructions: dict[str, list[str]] = {s: [] for s in SCENARIO_DISTRIBUTION}

    for seed in SEED_EXAMPLES:
        scenario = seed["scenario"]
        if scenario not in all_instructions:
            continue
        tasks = [
            generate_instructions(client, semaphore, scenario, seed, 10)
            for _ in range(scenario_counts[scenario] // 10 + 1)
        ]
        results = await asyncio.gather(*tasks)
        for batch in results:
            all_instructions[scenario].extend(batch)

    # 对指令做简单去重（基于文本相似度的精细去重在 clean_data.py 中处理）
    for scenario in all_instructions:
        seen = set()
        unique = []
        for inst in all_instructions[scenario]:
            h = hashlib.md5(inst.encode()).hexdigest()
            if h not in seen:
                seen.add(h)
                unique.append(inst)
        all_instructions[scenario] = unique[: scenario_counts[scenario]]
        print(f"  [{scenario}] 生成指令 {len(unique)} 条，取 {len(all_instructions[scenario])} 条")

    # 保存中间产物（方便 Debug 和增量重跑）
    with open(OUTPUT_DIR / "raw_instructions.jsonl", "w", encoding="utf-8") as f:
        for scenario, instructions in all_instructions.items():
            for inst in instructions:
                f.write(json.dumps({"scenario": scenario, "instruction": inst}, ensure_ascii=False) + "\n")
    print(f"[Step 1] 完成，指令保存至 {OUTPUT_DIR / 'raw_instructions.jsonl'}")

    # ── Step 2：Magpie 响应生成 ──────────────────────────────────────────────
    print("\n[Step 2] 开始 Magpie 对话生成...")
    dialog_tasks = []
    for scenario, instructions in all_instructions.items():
        for inst in instructions:
            is_error = (scenario == "error_recovery") or (random.random() < 0.1)
            dialog_tasks.append(
                generate_dialog(client, semaphore, inst, scenario, is_error)
            )

    # 分批执行，避免内存溢出
    BATCH_SIZE = 100
    raw_dialogs = []
    for i in range(0, len(dialog_tasks), BATCH_SIZE):
        batch = dialog_tasks[i: i + BATCH_SIZE]
        batch_results = await asyncio.gather(*batch)
        raw_dialogs.extend([r for r in batch_results if r is not None])
        print(f"  进度: {min(i + BATCH_SIZE, len(dialog_tasks))}/{len(dialog_tasks)}")

    print(f"[Step 2] 生成对话 {len(raw_dialogs)} 条")

    # ── Step 3：打包为 ChatML 格式 ────────────────────────────────────────────
    print("\n[Step 3] 打包为 ChatML 格式...")
    sft_samples = [pack_to_chatml(d) for d in raw_dialogs]

    with open(OUTPUT_DIR / "sft_raw.jsonl", "w", encoding="utf-8") as f:
        for sample in sft_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"[Step 3] 完成！原始 SFT 数据 {len(sft_samples)} 条，保存至 {OUTPUT_DIR / 'sft_raw.jsonl'}")
    print(f"\n[提示] 下一步运行 clean_data.py 对数据进行质量过滤和去重")


if __name__ == "__main__":
    asyncio.run(main())
