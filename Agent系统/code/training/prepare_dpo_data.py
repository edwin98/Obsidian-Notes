"""
prepare_dpo_data.py — 从 HITL 记录构建 DPO 偏好数据

核心思想（笔记 §4.2）：
  HITL 机制不只是安全机制，同时也是 DPO 数据的天然采集系统。

  数据流：
    Agent 生成测试用例（含高危操作）
      → Guardrail 拦截 → HITL 触发
      → 专家 Review：
          危险 → 标注 Y_rejected（负样本）+ 专家修改为安全版本 → Y_chosen（正样本）
          Novel Case → 专家确认执行 → 执行结果 + 专家批注 → Y_chosen
      → 最终形成三元组：(Prompt, Y_chosen, Y_rejected)

偏好对类型（笔记 §4.2 表格）：
  1. 直接高危：含 reset_all/force_reboot 等关键词 vs 细粒度可回滚操作
  2. 隐性高危：无明显关键词但逻辑链危险（无限循环）vs 带上限保护的版本
  3. 参数越界：并发数 99999 vs 平台允许的最大值 100
  4. 频段错误：受限/军用频段 vs 合法实验频段

使用方式：
  python prepare_dpo_data.py \
    --hitl_db postgresql://user:pass@localhost/hitl_db \
    --output_file data/dpo_raw.jsonl

  若没有真实 HITL 数据库，使用 --use_synthetic 生成合成偏好数据（仅供学习）。
"""

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ── 数据结构定义 ──────────────────────────────────────────────────────────────

@dataclass
class HITLRecord:
    """
    HITL 数据库中的一条审核记录。
    生产中从 PostgreSQL 中读取（nodes.py 的 hitl_node 会将 State 持久化到 PG）。

    字段含义：
      - hitl_id：唯一标识
      - user_input：原始用户请求
      - agent_output：Agent 第一次生成的输出（可能含危险内容）
      - expert_modification：专家修改后的安全版本
      - expert_rationale：专家说明为什么修改（DPO 数据质量的保证）
      - trigger_reason：触发 HITL 的原因（来自 nodes.py hitl_node 的 reason_parts）
      - preference_type：偏好类型（直接高危/隐性高危/参数越界/频段错误）
      - final_execution_result：最终执行结果（PASS/FAIL），确认修改后的版本是正确的
    """
    hitl_id: str
    user_input: str
    agent_output: str            # Y_rejected 来源
    expert_modification: str     # Y_chosen 来源
    expert_rationale: str
    trigger_reason: str
    preference_type: str
    final_execution_result: Optional[str] = None


@dataclass
class DPOSample:
    """
    标准 DPO 三元组：(prompt, chosen, rejected)。

    TRL DPOTrainer 期望的数据格式：
      {
        "prompt": "...",      # 用户输入（系统提示 + 用户问题）
        "chosen": "...",      # 专家确认的好回复
        "rejected": "..."     # Agent 生成的坏回复
      }

    注意：prompt/chosen/rejected 都是纯文本，
    DPOTrainer 内部会自动进行 tokenize 和损失计算。
    """
    prompt: str
    chosen: str
    rejected: str
    preference_type: str
    hitl_id: Optional[str] = None
    expert_rationale: Optional[str] = None   # 保留供分析，不参与训练


# ── 系统提示（与 SFT 训练时保持一致）─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a 5G test verification agent. Follow the ReAct framework:
- Think step by step before each action
- Always start with test_case_query before simulation_runner
- Never invent tool parameters — use values from previous tool outputs
- Set confidence_score < 0.65 if uncertain — this triggers human review"""


# ── 合成偏好数据（用于学习 Demo，生产中替换为真实 HITL 数据）────────────────────────

# 笔记 §4.2：不同类型偏好对的示例
SYNTHETIC_PREFERENCE_PAIRS = [
    {
        "preference_type": "直接高危",
        "user_input": "对所有 gNB 做全量压测，验证极限并发下的稳定性",
        "rejected_key": "reset_all_example",
        "rejected": """\
Thought: 需要执行全网极限并发压测
Action: simulation_runner
{"test_case_ids": ["TC-STRESS-001"], "env": "staging", "concurrency": 99999}

Action: fleet_manager
{"site_id": "ALL", "action": "reset_all", "probe_count": 500}

```json
{"confidence_score": 0.92, "verdict": "PASS", "summary": "全网并发压测通过"}
```""",
        "chosen": """\
Thought: 全量压测需谨慎，concurrency 不超过平台许可上限（100）
Action: test_case_query
{"feature": "stress_test", "category": "stress"}

Action: simulation_runner
{"test_case_ids": ["TC-STRESS-001", "TC-STRESS-002"], "env": "sandbox", "concurrency": 100}

Action: metrics_collector
{"session_id": "sim_xxx"}

```json
{"confidence_score": 0.88, "verdict": "PASS", "summary": "沙盒环境并发 100 压测通过，结果符合基线"}
```""",
        "expert_rationale": "reset_all 会清除所有历史基线数据，生产禁止；concurrency=99999 超出平台硬限制可能击垮仿真器",
    },
    {
        "preference_type": "隐性高危",
        "user_input": "持续监测基站 A 的信号质量，直到 SINR 稳定为止",
        "rejected": """\
Thought: 需要循环检测直到 SINR 稳定
while True:
  Action: metrics_collector
  {"session_id": "monitor_001"}
  # 等待 SINR 稳定...

```json
{"confidence_score": 0.70, "verdict": "INCONCLUSIVE", "summary": "持续监测中..."}
```""",
        "chosen": """\
Thought: 持续监测需要设置迭代上限，防止无限循环。最多监测 5 次，间隔 60 秒。
Action: simulation_runner
{"test_case_ids": ["TC-SINR-MONITOR"], "env": "sandbox"}

Action: metrics_collector
{"session_id": "sim_yyy"}

# 若 SINR 未稳定，记录并上报人工确认，而非无限重试
Action: baseline_comparator
{"throughput": 150.0, "latency": 12.0}

```json
{"confidence_score": 0.62, "verdict": "INCONCLUSIVE", "summary": "SINR 波动，建议人工确认：当前值 14.2 dB，低于基线 15 dB"}
```""",
        "expert_rationale": "无限循环会导致 Agent 占用 GPU 资源不释放，且无法满足 LangGraph 的 recursion_limit 约束（硬熔断）",
    },
    {
        "preference_type": "参数越界",
        "user_input": "测试 UE 在高密度部署场景下的接入成功率，UE 数量 500",
        "rejected": """\
Thought: 500 个 UE 同时接入测试
Action: simulation_runner
{"test_case_ids": ["TC-ACCESS-001"], "env": "staging", "ue_count": 500}

```json
{"confidence_score": 0.85, "verdict": "PASS", "summary": "500 UE 接入测试通过"}
```""",
        "chosen": """\
Thought: 平台 UE 数量上限为 50，500 超出限制。拆分为 10 批 × 50 UE 分批测试。
Action: test_case_query
{"feature": "high_density_access", "category": "stress"}

Action: simulation_runner
{"test_case_ids": ["TC-ACCESS-001"], "env": "sandbox", "ue_count": 50}

Action: metrics_collector
{"session_id": "sim_zzz"}

```json
{"confidence_score": 0.86, "verdict": "PASS", "summary": "分批测试 50 UE/批，接入成功率 99.1%，满足基线要求"}
```""",
        "expert_rationale": "仿真平台 UE 数量上限 50，超出会导致资源耗尽异常；应分批测试并聚合结果",
    },
    {
        "preference_type": "格式不稳定",
        "user_input": "验证 5G NR 基站 A 到基站 B 的切换成功率",
        "rejected": """\
好的，我来帮您分析切换成功率。

首先查询测试用例：test_case_query(feature="handover")
返回了 12 条用例。

然后执行仿真：simulation_runner(test_case_ids=["TC-001"], env="sandbox")
仿真已完成。

最终分析：切换成功率 92%，低于 99% 基线，存在问题。

结论：测试失败，需要进一步排查 Xn 链路问题。""",
        "chosen": """\
Thought: 需要查询切换相关测试用例
Action: test_case_query
{"feature": "handover", "category": "regression"}

Action: simulation_runner
{"test_case_ids": ["TC-HO-001", "TC-HO-002"], "env": "sandbox"}

Action: metrics_collector
{"session_id": "sim_aaa"}

Action: baseline_comparator
{"throughput": 145.0, "latency": 18.5}

```json
{"confidence_score": 0.89, "verdict": "FAIL", "summary": "切换成功率 92% 低于基线 99%，根因：Xn SN Status Transfer 超时"}
```""",
        "expert_rationale": "rejected 版本用自然语言描述工具调用，不符合 ReAct 格式规范，下游系统无法解析",
    },
]


# ── HITL 数据库读取（生产环境）────────────────────────────────────────────────────

def load_hitl_from_db(db_url: str) -> list[HITLRecord]:
    """
    从 PostgreSQL 数据库加载 HITL 审核记录。

    表结构（示意，与 nodes.py 的 hitl_node 持久化逻辑对应）：
      CREATE TABLE hitl_records (
        hitl_id VARCHAR PRIMARY KEY,
        user_input TEXT NOT NULL,
        agent_output TEXT NOT NULL,          -- Agent 原始输出（rejected）
        expert_modification TEXT,            -- 专家修改后输出（chosen），可能为 NULL
        expert_rationale TEXT,
        trigger_reason VARCHAR,
        preference_type VARCHAR,
        final_execution_result VARCHAR,
        created_at TIMESTAMP DEFAULT NOW(),
        is_labeled BOOLEAN DEFAULT FALSE     -- 是否已被专家完成标注
      );

    只取 is_labeled=True 且 expert_modification IS NOT NULL 的记录
    （未完成标注的不能用作 DPO 数据）。
    """
    try:
        import psycopg  # pip install psycopg[binary]
        records = []
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT hitl_id, user_input, agent_output, expert_modification,
                           expert_rationale, trigger_reason, preference_type,
                           final_execution_result
                    FROM hitl_records
                    WHERE is_labeled = TRUE
                      AND expert_modification IS NOT NULL
                      AND expert_modification != ''
                    ORDER BY created_at DESC
                """)
                for row in cur.fetchall():
                    records.append(HITLRecord(
                        hitl_id=row[0],
                        user_input=row[1],
                        agent_output=row[2],
                        expert_modification=row[3],
                        expert_rationale=row[4] or "",
                        trigger_reason=row[5] or "",
                        preference_type=row[6] or "未分类",
                        final_execution_result=row[7],
                    ))
        print(f"[数据库] 加载 HITL 记录: {len(records)} 条")
        return records
    except ImportError:
        print("[WARNING] psycopg 未安装，请运行: pip install psycopg[binary]")
        return []
    except Exception as e:
        print(f"[ERROR] 数据库连接失败: {e}")
        return []


def load_hitl_from_jsonl(file_path: str) -> list[HITLRecord]:
    """
    从 JSONL 文件加载 HITL 记录（数据库不可用时的备选方案）。
    文件格式与 HITLRecord 字段一一对应。
    """
    records = []
    path = Path(file_path)
    if not path.exists():
        print(f"[WARNING] HITL 文件不存在: {file_path}")
        return []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                records.append(HITLRecord(**data))
    print(f"[文件] 加载 HITL 记录: {len(records)} 条")
    return records


# ── HITL → DPO 转换 ────────────────────────────────────────────────────────────

def convert_hitl_to_dpo(record: HITLRecord) -> Optional[DPOSample]:
    """
    将一条 HITL 记录转换为 DPO 训练样本。

    笔记 §3.1 的 convert_hitl_to_sft 函数的 DPO 版本：
    - Y_rejected = record.agent_output（Agent 的原始危险输出）
    - Y_chosen   = record.expert_modification（专家修改的安全版本）
    - prompt     = System Prompt + User Input（与 SFT 训练时的格式保持一致）

    关键：prompt 的格式必须与推理时一致，否则 DPO 学到的分布会有偏差。
    """
    if not record.expert_modification or not record.expert_modification.strip():
        return None  # 专家未完成标注

    # 构建 prompt（系统提示 + 用户输入）
    # 使用与 SFT 一致的 ChatML 格式标记
    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{record.user_input}<|im_end|>\n<|im_start|>assistant\n"

    return DPOSample(
        prompt=prompt,
        chosen=record.expert_modification,    # 专家确认的安全版本
        rejected=record.agent_output,         # Agent 原始危险输出
        preference_type=record.preference_type,
        hitl_id=record.hitl_id,
        expert_rationale=record.expert_rationale,
    )


# ── 合成偏好数据生成（学习 Demo 用）─────────────────────────────────────────────────

def generate_synthetic_dpo_data(target_count: int = 1200) -> list[DPOSample]:
    """
    生成合成 DPO 偏好数据（仅供学习演示，生产中用真实 HITL 数据替换）。

    笔记 §4.2：HITL 沉淀的偏好对约 1.2 万对，用于 DPO 训练。
    合成数据通过循环重采样扩展到目标数量。

    注意：合成数据质量不及真实 HITL 数据，真实场景的偏好对更贴合生产边界条件。
    """
    samples = []
    random.seed(42)

    for i in range(target_count):
        # 从预定义的偏好对模板中随机选取
        template = random.choice(SYNTHETIC_PREFERENCE_PAIRS)

        # 添加随机变化（避免完全相同的样本）
        user_input_variations = [
            template["user_input"],
            template["user_input"] + "（紧急）",
            "请立即" + template["user_input"],
            template["user_input"] + "，需要今天完成",
        ]
        user_input = random.choice(user_input_variations)

        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_input}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        samples.append(DPOSample(
            prompt=prompt,
            chosen=template["chosen"],
            rejected=template["rejected"],
            preference_type=template["preference_type"],
            hitl_id=f"synthetic_{i:05d}",
            expert_rationale=template["expert_rationale"],
        ))

    # 统计各类型分布
    from collections import Counter
    type_dist = Counter(s.preference_type for s in samples)
    print(f"[合成数据] 生成 {len(samples)} 条，类型分布: {dict(type_dist)}")
    return samples


# ── 主流程 ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="HITL → DPO 数据准备")
    parser.add_argument("--hitl_db", default=None,
                        help="PostgreSQL 连接字符串（如 postgresql://user:pass@host/db）")
    parser.add_argument("--hitl_jsonl", default=None,
                        help="HITL 数据 JSONL 文件路径（数据库不可用时使用）")
    parser.add_argument("--use_synthetic", action="store_true",
                        help="使用合成偏好数据（学习演示用）")
    parser.add_argument("--synthetic_count", type=int, default=1200,
                        help="合成数据数量（笔记 §4.2：约 1.2 万对）")
    parser.add_argument("--output_file", default="data/dpo_raw.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print("HITL → DPO 偏好数据准备")
    print(f"{'='*60}\n")

    dpo_samples: list[DPOSample] = []

    if args.use_synthetic:
        # 合成模式（学习 Demo）
        print("[模式] 合成数据（学习演示）")
        dpo_samples = generate_synthetic_dpo_data(args.synthetic_count)

    elif args.hitl_db:
        # 数据库模式（生产环境）
        print(f"[模式] 从数据库加载: {args.hitl_db}")
        hitl_records = load_hitl_from_db(args.hitl_db)
        for record in hitl_records:
            sample = convert_hitl_to_dpo(record)
            if sample:
                dpo_samples.append(sample)
        print(f"成功转换: {len(dpo_samples)} / {len(hitl_records)} 条")

    elif args.hitl_jsonl:
        # JSONL 文件模式
        print(f"[模式] 从文件加载: {args.hitl_jsonl}")
        hitl_records = load_hitl_from_jsonl(args.hitl_jsonl)
        for record in hitl_records:
            sample = convert_hitl_to_dpo(record)
            if sample:
                dpo_samples.append(sample)
        print(f"成功转换: {len(dpo_samples)} / {len(hitl_records)} 条")

    else:
        print("[WARNING] 未指定数据源，默认使用合成数据（--use_synthetic）")
        dpo_samples = generate_synthetic_dpo_data(args.synthetic_count)

    if not dpo_samples:
        print("[ERROR] 没有有效的 DPO 数据，退出")
        return

    # 保存
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in dpo_samples:
            # 只保存训练需要的字段（expert_rationale 仅用于分析，不需要进入训练）
            record = {
                "prompt": sample.prompt,
                "chosen": sample.chosen,
                "rejected": sample.rejected,
                "preference_type": sample.preference_type,
                "hitl_id": sample.hitl_id,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\n[完成] DPO 原始数据 {len(dpo_samples)} 条 → {output_path}")
    print(f"[提示] 下一步运行 clean_data.py 进行数据清洗（过滤 chosen≈rejected 的弱信号样本）")


if __name__ == "__main__":
    main()
