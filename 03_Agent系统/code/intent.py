"""【学习定位：自然语言到结构化操作】
控制消息优先，条件修改产生补丁；未知信息需要澄清。离线词法规则不能覆盖所有否定与复合语义。
对应11；interpret返回候选，TaskService才真正修改业务状态。

统一入口：离线规则便于演示，在线模型用同一结构输出。
"""

import re

from task_models import ConditionPatch, IntentDecision, TaskRecord


def control_actions(text: str) -> list[str] | None:
    # 这是有限规则的快速通道；生产级否定、指代与复合意图仍需结构化理解，不能把关键词当完整语义。
    """控制请求优先，查询不得因为含有“测试”二字而启动测试。"""
    query = any(
        w in text
        for w in (
            "进度",
            "怎么样了",
            "已有结果",
            "报告",
            "查到哪",
            "是什么意思",
            "怎么理解",
            "解释一下",
        )
    )
    if any(w in text for w in ("取消", "终止任务", "停止任务", "结束这次")):
        return ["CANCEL"] + (["QUERY"] if query else [])
    if any(w in text for w in ("暂停", "先别跑", "先停", "不要继续跑")):
        return ["PAUSE"] + (["QUERY"] if query else [])
    if query and not any(w in text for w in ("再测", "复测", "验证修复", "修复版")):
        return ["QUERY"]
    if text.strip() in {"继续", "继续上次的", "恢复", "接着跑"}:
        return ["RESUME"]
    return None


class OfflineIntent:
    """有限中文词法规则，不冒充通用语言理解能力；未知请求会追问。"""

    def interpret(self, text: str, task: TaskRecord | None) -> IntentDecision:
        """只做有限词法识别，输出动作、字段补丁和待澄清问题，不写任务。

        task 只用于理解“其他不变”等上下文；缺失字段不能由默认值覆盖。
        返回后仍由 TaskService 核对任务类型、状态与执行范围。
        """
        # 只把用户明确提到的字段放入patch；未提到的字段保留，缺少具体数值时返回question。
        control = control_actions(text)
        if control:
            return IntentDecision(actions=control, reason="按用户控制请求处理")
        patch = {}
        pair = re.search(r"([A-Z])\s*(?:到|→|->)\s*([A-Z])", text)
        if pair:
            patch.update(source_cell=pair[1], target_cell=pair[2])
        bandwidth = re.search(
            r"(?:带宽[^\d]{0,5})(\d+)\s*(?:MHz|兆)?", text, re.IGNORECASE
        )
        if bandwidth:
            patch["bandwidth_mhz"] = int(bandwidth[1])
        band = re.search(
            r"(?<![A-Za-z0-9])n\d{1,3}(?![A-Za-z0-9])", text, re.IGNORECASE
        )
        if band:
            patch["band"] = band[0].lower()
        count = re.search(r"(?:切换[^\d]{0,3}|连续)(\d+)次", text)
        if count:
            patch["handovers"] = int(count[1])
        speed = re.search(r"速度[^\d]{0,5}(\d+)", text)
        if speed:
            patch["speed_kmh"] = int(speed[1])
        if "高负载" in text:
            patch["load"] = "high"
        elif any(w in text for w in ("普通负载", "低负载")):
            patch["load"] = "normal"
        if "修复" in text or "fixed" in text:
            patch["version"] = (
                "slow_fix"
                if any(w in text for w in ("变慢版", "slow_fix"))
                else "fixed"
            )
        elif "旧版本" in text:
            patch["version"] = "old"
        elif "新版本" in text or "升级" in text:
            patch["version"] = "new"
        budget = re.search(r"最多\s*(\d+)\s*次", text)
        readonly = any(w in text for w in ("只读", "只看日志", "不要运行", "不跑测试"))
        fix = "修复" in text or "fixed" in text
        investigation = any(
            w in text
            for w in (
                "卡顿",
                "卡住",
                "异常",
                "断线",
                "复现",
                "排查",
                "查原因",
                "继续查",
            )
        )
        conditional = any(w in text for w in ("失败再", "失败后", "失败就"))
        modifying = task is not None and (
            any(
                w in text
                for w in (
                    "改成",
                    "改为",
                    "改小",
                    "带宽改",
                    "继续查",
                    "其他不变",
                    "补充",
                )
            )
            or task.status == "WAITING_INFO"
            and bool(patch)
        )
        kind = "VERIFY_FIX" if fix else "INVESTIGATE" if investigation else "TEST"
        if conditional:
            kind = "TEST"
        goal = {"TEST": "TEST", "INVESTIGATE": "REPRODUCE", "VERIFY_FIX": "VERIFY_FIX"}[
            kind
        ]
        if investigation and any(w in text for w in ("查原因", "验证原因", "原因范围")):
            goal = "CHECK_CAUSE"
        known = (
            fix
            or investigation
            or bool(patch)
            or any(
                w in text.lower()
                for w in ("测试", "验证", "测一下", "handover", "test ")
            )
        )
        question = "" if known else "你是要执行测试、复现异常，还是只查看已有结果？"
        if modifying and not patch and not investigation and not fix:
            question = (
                "请说明需要修改的具体数值，例如“带宽改为40 MHz”；尚未更改或执行。"
            )
        if re.search(r"(?<![A-Za-z0-9])v\d+(?:\.\d+)*", text, re.IGNORECASE):
            question = "沙盒只登记 old/new/fixed/slow_fix，不能把实际版本号静默替换；请选演示版本或接入真实平台。"
        if any(word in text for word in ("干扰", "认证", "容量极限")):
            question = "当前新沙盒只支持移动业务及相关吞吐、时延示例，该测试还需要对应的用例与执行器。"
        return IntentDecision(
            actions=["QUERY", "CREATE"]
            if fix and "报告" in text
            else ["UPDATE" if modifying else "CREATE"],
            task_type=kind
            if not (modifying and not investigation and not fix)
            else task.task_type,
            goal=goal
            if not (modifying and not investigation and not fix)
            else task.goal,
            patch=ConditionPatch(**patch),
            read_only=readonly if readonly else None,
            investigate_on_failure=conditional if conditional else None,
            max_runs=int(budget[1]) if budget else None,
            unsupported_environment=any(w in text for w in ("生产环境", "现场设备")),
            question=question,
            reason="离线教学规则识别；未识别内容需要补充说明",
        )


class ModelIntent:
    """在线意图适配器；结构化输出约束字段形状，业务约束仍留在服务层。"""

    def __init__(self, model):
        self.model = model.with_structured_output(
            IntentDecision, method="function_calling"
        )

    def interpret(self, text: str, task: TaskRecord | None) -> IntentDecision:
        """先识别明确控制消息，再用当前任务的有限投影解释其余自然语言。"""
        control = control_actions(text)
        if control:
            return IntentDecision(actions=control, reason="明确控制请求优先")
        context = (
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "goal": task.goal,
                "status": task.status,
                "conditions": task.conditions.model_dump(),
                "query": task.query,
                "question": task.question,
                "has_reproduction": task.reproduction is not None,
                "read_only": task.read_only,
                "max_runs": task.max_runs,
            }
            if task
            else None
        )
        prompt = (
            "理解测试助手收到的消息。区分新建/修改/查询/暂停/继续/取消，"
            "和 TEST/INVESTIGATE/VERIFY_FIX。目标只复现时不要扩展为确认原因。"
            "未修改的条件保留；不猜设备编号。只读不执行。生产或现场标记"
            "unsupported_environment。版本只支持 old/new/fixed/slow_fix，其他版本追问。"
            "不清楚时填写 question，不凭高置信度跳过检查。\n"
            f"当前任务：{context}\n用户：{text}"
        )
        return IntentDecision.model_validate(self.model.invoke(prompt))
