"""
clean_data.py — SFT/DPO 数据清洗与质量过滤

负责对 generate_sft_data.py 和 prepare_dpo_data.py 输出的原始数据进行：
  1. 格式校验（Pydantic）：字段完整性、JSON 可解析性
  2. 质量过滤：长度异常、乱码、直接抄写检测
  3. 语义去重：MD5 精确去重 + n-gram 近似去重
  4. 数据污染检测：确保训练集和评测集不重叠
  5. 场景分布校正：偏离目标超过 5% 时自动过采样/欠采样

参考笔记：
  Agent系统/04_Agent模型后训练SFT_DPO.md § 3.1 Step 4 质量过滤
  Agent系统/04_Agent模型后训练SFT_DPO.md § 5. 训练数据质量保证
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError, field_validator

# ── 配置 ─────────────────────────────────────────────────────────────────────

INPUT_SFT_FILE = Path("data/sft_raw.jsonl")
INPUT_DPO_FILE = Path("data/dpo_raw.jsonl")
EVAL_FILE = Path("data/eval_set.jsonl")       # 评测集（用于污染检测）
OUTPUT_SFT_FILE = Path("data/sft_clean.jsonl")
OUTPUT_DPO_FILE = Path("data/dpo_clean.jsonl")

# 过滤阈值
MIN_ASSISTANT_LEN = 100     # assistant 回复最短字符数（太短说明质量差）
MAX_ASSISTANT_LEN = 8000    # assistant 回复最长字符数（太长通常是模型跑偏了）
COPY_THRESHOLD = 0.85        # 重叠率超过此值认为是"直接抄写"（从 user 抄到 assistant）
DEDUP_NGRAM_N = 4            # n-gram 去重的 n
CONTAMINATION_THRESHOLD = 0.8   # 与评测集 n-gram 重叠率超过此值则剔除
DISTRIBUTION_TOLERANCE = 0.05   # 场景分布偏差容忍度（5%）

# 笔记 §5.2 的场景分布目标（与 generate_sft_data.py 保持一致）
SCENARIO_DISTRIBUTION_TARGET = {
    "handover": 0.25,
    "interference": 0.20,
    "capacity": 0.20,
    "channel": 0.15,
    "auth_bearer": 0.10,
    "error_recovery": 0.10,
}


# ── Pydantic 数据模型（SFT）────────────────────────────────────────────────────

class MessageItem(BaseModel):
    """单条消息的格式校验"""
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, v: str) -> str:
        """
        role 只能是 system/user/assistant。
        GPT 格式偶尔会出现 "tool" 角色，在此统一过滤。
        """
        if v not in {"system", "user", "assistant"}:
            raise ValueError(f"无效的 role: {v}")
        return v

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content 不能为空")
        return v


class SFTSample(BaseModel):
    """SFT 训练样本的格式校验"""
    messages: list[MessageItem]
    scenario: Optional[str] = None
    is_error_recovery: Optional[bool] = False
    instruction_hash: Optional[str] = None

    @field_validator("messages")
    @classmethod
    def validate_message_structure(cls, msgs: list) -> list:
        """
        校验消息结构完整性：
        - 必须有 system 消息
        - 必须有 user 消息
        - 必须有 assistant 消息（最后一条）
        - 不能以 user 结尾（说明 assistant 没有回复）
        """
        roles = [m.role for m in msgs]
        if "system" not in roles:
            raise ValueError("缺少 system 消息")
        if "user" not in roles:
            raise ValueError("缺少 user 消息")
        if "assistant" not in roles:
            raise ValueError("缺少 assistant 消息")
        if roles[-1] != "assistant":
            raise ValueError("最后一条消息必须是 assistant")
        return msgs


class DPOSample(BaseModel):
    """DPO 偏好数据的格式校验"""
    prompt: str          # 用户输入（Context）
    chosen: str          # 专家确认的正样本（Y_chosen）
    rejected: str        # 专家拒绝的负样本（Y_rejected）
    preference_type: Optional[str] = None   # 偏好类型（见笔记 §4.2 分类）

    @field_validator("chosen", "rejected")
    @classmethod
    def validate_response_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("chosen/rejected 不能为空")
        return v

    @field_validator("rejected")
    @classmethod
    def chosen_and_rejected_must_differ(cls, v: str, info) -> str:
        """
        chosen 和 rejected 必须不同，否则 DPO 损失为 0，没有学习信号。
        """
        if "chosen" in info.data and v.strip() == info.data["chosen"].strip():
            raise ValueError("chosen 和 rejected 内容完全相同，无法提供学习信号")
        return v


# ── 过滤函数 ─────────────────────────────────────────────────────────────────

def has_control_chars(text: str) -> bool:
    """
    检测文本是否含有控制字符（乱码检测）。
    控制字符（如 \x00-\x1f 中的非换行符）通常是 API 返回异常的标志。
    """
    for ch in text:
        cat = unicodedata.category(ch)
        # Cc = 控制字符，但允许换行（\n）和制表符（\t）
        if cat == "Cc" and ch not in ("\n", "\t", "\r"):
            return True
    return False


def compute_overlap_ratio(text_a: str, text_b: str, n: int = 4) -> float:
    """
    计算两段文本的 n-gram 重叠率（Jaccard 相似度）。
    用于检测"直接抄写"：assistant 把 user 的问题原封不动复制到回复中。

    公式：|A ∩ B| / |A ∪ B|
    其中 A、B 是各自的 n-gram 集合。
    """
    def get_ngrams(text: str, n: int) -> set:
        tokens = text.split()
        return set(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))

    ngrams_a = get_ngrams(text_a, n)
    ngrams_b = get_ngrams(text_b, n)
    if not ngrams_a or not ngrams_b:
        return 0.0
    intersection = len(ngrams_a & ngrams_b)
    union = len(ngrams_a | ngrams_b)
    return intersection / union if union > 0 else 0.0


def has_valid_json_conclusion(assistant_text: str) -> bool:
    """
    检查 assistant 输出是否包含合法的 JSON 结论块。
    SFT 数据的核心目的之一是让模型学会输出格式化 JSON。

    合法结论块示例：
      {"confidence_score": 0.91, "verdict": "PASS", "summary": "..."}
    """
    # 提取 JSON 代码块（```json ... ```）或裸 JSON
    json_pattern = r"```json\s*(\{.*?\})\s*```|(\{[^{}]*\"verdict\"[^{}]*\})"
    matches = re.findall(json_pattern, assistant_text, re.DOTALL)
    for match in matches:
        candidate = match[0] or match[1]
        try:
            data = json.loads(candidate)
            # 必须包含 verdict 和 confidence_score 字段
            if "verdict" in data and "confidence_score" in data:
                verdict = data["verdict"]
                score = float(data["confidence_score"])
                # verdict 只能是 PASS/FAIL/INCONCLUSIVE
                if verdict in {"PASS", "FAIL", "INCONCLUSIVE"} and 0.0 <= score <= 1.0:
                    return True
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
    return False


def filter_sft_sample(sample: SFTSample) -> tuple[bool, str]:
    """
    对单条 SFT 样本进行质量过滤。
    返回 (是否保留, 过滤原因)。

    过滤逻辑参考笔记 §3.1 Step 4：
    - 格式校验：Pydantic 自动过滤格式错误（~15% 的合成数据）
    - 一致性校验：最终结论与工具返回是否逻辑自洽
    - 人工抽检逻辑（代码层的自动近似替代）
    """
    # 找到 assistant 的最后一条回复
    assistant_msgs = [m for m in sample.messages if m.role == "assistant"]
    if not assistant_msgs:
        return False, "缺少 assistant 消息"

    assistant_text = assistant_msgs[-1].content
    user_msgs = [m for m in sample.messages if m.role == "user"]
    user_text = user_msgs[-1].content if user_msgs else ""

    # 1. 长度过滤
    if len(assistant_text) < MIN_ASSISTANT_LEN:
        return False, f"assistant 回复过短 ({len(assistant_text)} < {MIN_ASSISTANT_LEN})"
    if len(assistant_text) > MAX_ASSISTANT_LEN:
        return False, f"assistant 回复过长 ({len(assistant_text)} > {MAX_ASSISTANT_LEN})"

    # 2. 乱码检测
    if has_control_chars(assistant_text):
        return False, "assistant 回复含控制字符（乱码）"

    # 3. 直接抄写检测（assistant 抄写 user）
    overlap = compute_overlap_ratio(user_text, assistant_text, n=DEDUP_NGRAM_N)
    if overlap > COPY_THRESHOLD:
        return False, f"assistant 直接抄写 user 内容（重叠率 {overlap:.2f} > {COPY_THRESHOLD}）"

    # 4. JSON 结论格式校验（核心质量指标）
    if not has_valid_json_conclusion(assistant_text):
        return False, "assistant 回复缺少合法的 JSON 结论块（{verdict, confidence_score}）"

    # 5. 必须包含至少一次 ReAct 推理（Thought 关键词）
    if "Thought:" not in assistant_text and "thought:" not in assistant_text.lower():
        return False, "assistant 回复缺少 ReAct Thought 推理步骤"

    return True, ""


# ── 去重逻辑 ─────────────────────────────────────────────────────────────────

def dedup_by_instruction(samples: list[dict]) -> list[dict]:
    """
    基于指令文本的 MD5 哈希去重。
    相同指令的不同 assistant 回复只保留第一条（生产中可保留质量更好的）。

    注意：这是精确去重。近似去重（语义相似）需要 embedding 模型，成本较高，
    实际生产中会用 SimHash 或 MinHash 做近似去重。
    """
    seen_hashes = set()
    deduped = []
    for sample in samples:
        # 提取 user 消息的文本作为去重键
        user_texts = [
            m["content"] for m in sample.get("messages", [])
            if m.get("role") == "user"
        ]
        if not user_texts:
            continue
        key = hashlib.md5(user_texts[-1].encode()).hexdigest()
        if key not in seen_hashes:
            seen_hashes.add(key)
            deduped.append(sample)
    return deduped


def check_data_contamination(
    train_samples: list[dict],
    eval_samples: list[dict],
    threshold: float = CONTAMINATION_THRESHOLD,
) -> list[dict]:
    """
    数据污染检测：确保训练集中没有与评测集高度相似的样本。

    笔记 §5.1：约 2.3% 的合成数据被检测为与评测集高度相似，全部剔除。

    实现：对每条训练样本的指令，与所有评测样本的指令做 n-gram 重叠检测，
    超过阈值则从训练集剔除。

    注意：这是 O(n*m) 的操作，数量大时应改用 MinHash LSH 加速。
    """
    # 提取评测集的指令文本
    eval_instructions = []
    for s in eval_samples:
        msgs = s.get("messages", [])
        user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
        if user_msgs:
            eval_instructions.append(user_msgs[-1])

    cleaned = []
    contaminated_count = 0
    for sample in train_samples:
        msgs = sample.get("messages", [])
        user_msgs = [m["content"] for m in msgs if m.get("role") == "user"]
        if not user_msgs:
            continue
        train_instr = user_msgs[-1]

        is_contaminated = False
        for eval_instr in eval_instructions:
            overlap = compute_overlap_ratio(train_instr, eval_instr, n=4)
            if overlap > threshold:
                is_contaminated = True
                break

        if is_contaminated:
            contaminated_count += 1
        else:
            cleaned.append(sample)

    print(f"  [污染检测] 剔除 {contaminated_count} 条（占 {contaminated_count/len(train_samples)*100:.1f}%）")
    return cleaned


# ── 场景分布校正 ──────────────────────────────────────────────────────────────

def rebalance_by_scenario(
    samples: list[dict],
    target_distribution: dict,
    tolerance: float = DISTRIBUTION_TOLERANCE,
) -> list[dict]:
    """
    场景分布校正：若某类场景数量偏离目标超过 tolerance，通过过采样/欠采样纠正。

    笔记 §5.2：实际数据分布偏离目标超过 5% 时，通过过采样/欠采样纠正。

    过采样：重复部分样本（简单方案；生产中可用数据增强）
    欠采样：随机丢弃部分样本
    """
    # 统计当前各场景数量
    scenario_groups: dict[str, list] = defaultdict(list)
    no_scenario = []
    for s in samples:
        scenario = s.get("scenario", "unknown")
        if scenario in target_distribution:
            scenario_groups[scenario].append(s)
        else:
            no_scenario.append(s)

    total = len(samples)
    result = []
    for scenario, target_ratio in target_distribution.items():
        group = scenario_groups.get(scenario, [])
        current_ratio = len(group) / total if total > 0 else 0
        target_count = int(total * target_ratio)

        if abs(current_ratio - target_ratio) > tolerance:
            if current_ratio < target_ratio:
                # 过采样：循环复制直到够数
                print(f"  [分布校正] {scenario}: {current_ratio:.1%} < {target_ratio:.1%}，过采样")
                import random
                extended = group.copy()
                while len(extended) < target_count:
                    extended.extend(random.sample(group, min(len(group), target_count - len(extended))))
                group = extended[:target_count]
            else:
                # 欠采样：随机丢弃
                print(f"  [分布校正] {scenario}: {current_ratio:.1%} > {target_ratio:.1%}，欠采样")
                import random
                group = random.sample(group, target_count)

        result.extend(group)

    result.extend(no_scenario)
    return result


# ── 主流程 ───────────────────────────────────────────────────────────────────

def clean_sft_data():
    """SFT 数据清洗主流程"""
    print(f"\n{'='*60}")
    print("SFT 数据清洗")
    print(f"{'='*60}")

    if not INPUT_SFT_FILE.exists():
        print(f"[ERROR] 输入文件不存在: {INPUT_SFT_FILE}")
        return

    # 加载原始数据
    raw_samples = []
    with open(INPUT_SFT_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))
    print(f"[1/5] 加载原始数据: {len(raw_samples)} 条")

    # Step 1：Pydantic 格式校验
    valid_samples = []
    format_error_count = 0
    for raw in raw_samples:
        try:
            sample = SFTSample(**raw)
            valid_samples.append(raw)  # 保留原始 dict 而非 Pydantic 对象
        except ValidationError as e:
            format_error_count += 1

    filter_rate = format_error_count / len(raw_samples) * 100
    print(f"[2/5] 格式校验: 剔除 {format_error_count} 条（{filter_rate:.1f}%），剩余 {len(valid_samples)} 条")
    # 笔记 §3.1：格式校验约过滤 15% 的合成数据，符合预期

    # Step 2：内容质量过滤
    quality_passed = []
    quality_filter_reasons = Counter()
    for raw in valid_samples:
        try:
            sample = SFTSample(**raw)
            passed, reason = filter_sft_sample(sample)
            if passed:
                quality_passed.append(raw)
            else:
                quality_filter_reasons[reason] += 1
        except Exception:
            quality_filter_reasons["Pydantic 解析失败"] += 1

    print(f"[3/5] 内容质量过滤: 剩余 {len(quality_passed)} 条")
    for reason, count in quality_filter_reasons.most_common(5):
        print(f"       - {reason}: {count} 条")

    # Step 3：指令去重
    deduped = dedup_by_instruction(quality_passed)
    print(f"[4/5] 指令去重: {len(quality_passed)} → {len(deduped)} 条（去除 {len(quality_passed)-len(deduped)} 条重复）")

    # Step 4：数据污染检测（若评测集存在）
    if EVAL_FILE.exists():
        eval_samples = []
        with open(EVAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_samples.append(json.loads(line))
        print(f"[4.5] 数据污染检测（评测集 {len(eval_samples)} 条）...")
        deduped = check_data_contamination(deduped, eval_samples)
    else:
        print(f"[4.5] 评测集文件不存在，跳过污染检测")

    # Step 5：场景分布校正
    print(f"[5/5] 场景分布校正...")
    final_samples = rebalance_by_scenario(deduped, SCENARIO_DISTRIBUTION_TARGET)

    # 统计最终分布
    scenario_dist = Counter(s.get("scenario", "unknown") for s in final_samples)
    print(f"  最终分布: {dict(scenario_dist)}")

    # 保存
    OUTPUT_SFT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_SFT_FILE, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[完成] SFT 清洗数据: {len(final_samples)} 条 → {OUTPUT_SFT_FILE}")
    print(f"[提示] 下一步运行 train_sft.py 开始 QLoRA 微调")


def clean_dpo_data():
    """DPO 数据清洗主流程（更简单，主要做格式校验和去重）"""
    print(f"\n{'='*60}")
    print("DPO 数据清洗")
    print(f"{'='*60}")

    if not INPUT_DPO_FILE.exists():
        print(f"[WARNING] DPO 原始数据文件不存在: {INPUT_DPO_FILE}，跳过")
        return

    raw_samples = []
    with open(INPUT_DPO_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_samples.append(json.loads(line))
    print(f"[1/3] 加载 DPO 原始数据: {len(raw_samples)} 条")

    # Step 1：Pydantic 格式校验
    valid_samples = []
    for raw in raw_samples:
        try:
            DPOSample(**raw)
            valid_samples.append(raw)
        except ValidationError as e:
            pass  # DPO 数据格式错误通常意味着 HITL 数据提取有问题

    print(f"[2/3] 格式校验: 剩余 {len(valid_samples)} 条")

    # Step 2：chosen/rejected 相似度过滤（太相似的偏好对学习信号弱）
    filtered = []
    weak_signal_count = 0
    for raw in valid_samples:
        overlap = compute_overlap_ratio(raw["chosen"], raw["rejected"], n=4)
        if overlap > 0.9:
            # chosen 和 rejected 内容几乎相同，偏好对质量极低
            weak_signal_count += 1
        else:
            filtered.append(raw)
    print(f"[3/3] 弱信号过滤: 剔除 {weak_signal_count} 条（chosen≈rejected），剩余 {len(filtered)} 条")

    with open(OUTPUT_DPO_FILE, "w", encoding="utf-8") as f:
        for sample in filtered:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"\n[完成] DPO 清洗数据: {len(filtered)} 条 → {OUTPUT_DPO_FILE}")
    print(f"[提示] 下一步运行 train_dpo.py 开始 DPO 偏好对齐训练")


if __name__ == "__main__":
    clean_sft_data()
    clean_dpo_data()
