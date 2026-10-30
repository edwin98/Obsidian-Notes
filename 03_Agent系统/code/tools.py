"""【学习定位：旧图的沙盒工具注册表】
每个工厂持有独立会话；后续工具必须使用先前真实返回的session_id。外部网元未接入。
对应03、04；看清工具schema、函数执行与ToolMessage三者的区别。

确定性的内存沙盒工具，不会连接网元或执行真实仿真。

工厂为每次教学运行分配独立会话存储；未知 session_id 必须报错。
这让学习者看到真实的数据依赖，而不是每一步都生成无关的随机数字。
线上接入时保留工具契约，把函数体替换为真实平台适配器。
"""

import math
import uuid

from langchain_core.tools import tool

from rag import retrieve


def create_sandbox_tools(*, regression=False):
    """返回隔离的工具注册表；regression 注入稳定可重现的回归故障。"""
    sessions = {}
    fleets = {}

    def session(session_id):
        if session_id not in sessions:
            raise ValueError(f"未知仿真会话：{session_id}")
        return sessions[session_id]

    @tool
    def test_case_query(feature: str, category: str = "regression") -> dict:
        """按功能查询沙盒用例，并返回本地 RAG 检索引用。"""
        if not feature.strip():
            raise ValueError("feature 不能为空")
        return {
            "feature": feature,
            "category": category,
            "cases": [
                {"id": "TC-0001", "name": f"{feature} 基础验证", "priority": "P0"}
            ],
            "total": 1,
            "references": [r.doc.doc_id for r in retrieve(feature)],
        }

    @tool
    def simulation_runner(test_case_ids: list[str], env: str = "sandbox") -> dict:
        """提交沙盒仿真任务，返回会话 ID；仅允许 sandbox 环境。"""
        if not test_case_ids or any(tc != "TC-0001" for tc in test_case_ids):
            raise ValueError("请使用查询工具返回的有效用例 ID")
        if env != "sandbox":
            raise ValueError("教学工具仅允许 sandbox 环境")
        sid = f"sim-{uuid.uuid4().hex[:8]}"
        sessions[sid] = {
            "throughput_avg_mbps": 60.0 if regression else 150.0,
            "latency_p99_ms": 40.0 if regression else 10.0,
            "packet_loss_rate": 0.03 if regression else 0.001,
            "handover_success_rate": 0.90 if regression else 0.995,
        }
        return {
            "session_id": sid,
            "env": env,
            "status": "queued",
            "pass_rate": 0.6 if regression else 1.0,
        }

    @tool
    def metrics_collector(session_id: str) -> dict:
        """从指定沙盒会话采集 KPI，教学环境立即完成。"""
        return {"session_id": session_id, "status": "completed", **session(session_id)}

    @tool
    def baseline_comparator(
        throughput_avg: float, latency_p99: float, baseline_version: str = "v1.0"
    ) -> dict:
        """与具名业务阈值比较；本工具不输出统计显著性 p 值。"""
        if baseline_version != "v1.0":
            raise ValueError("沙盒仅登记了基线 v1.0")
        if any(not math.isfinite(x) or x < 0 for x in (throughput_avg, latency_p99)):
            raise ValueError("KPI 必须是非负有限数值")
        degradations = []
        if throughput_avg < 90:
            degradations.append("吞吐量低于 90 Mbps")
        if latency_p99 > 25:
            degradations.append("P99 时延高于 25 ms")
        return {
            "baseline_version": baseline_version,
            "degradations": degradations,
            "overall_status": "REGRESSION" if degradations else "STABLE",
        }

    @tool
    def log_analyzer(log_type: str, session_id: str) -> dict:
        """读取沙盒会话的信令异常；支持 RRC/NAS/PDCP。"""
        session(session_id)
        if log_type.upper() not in {"RRC", "NAS", "PDCP"}:
            raise ValueError("不支持的日志类型")
        anomalies = ["切换准备超时（故障注入）"] if regression else []
        return {
            "log_type": log_type,
            "session_id": session_id,
            "anomalies": anomalies,
            "severity": "HIGH" if anomalies else "NORMAL",
        }

    @tool
    def fleet_manager(site_id: str, action: str, probe_count: int = 4) -> dict:
        """部署、查询、采集或释放沙盒探针；状态只存在于当前进程。"""
        if not site_id.strip() or not 1 <= probe_count <= 100:
            raise ValueError("局点不能为空，探针数必须在 1~100 之间")
        if action == "deploy":
            fleets[site_id] = [f"{site_id}-probe-{i:02d}" for i in range(probe_count)]
        elif action not in {"status", "collect", "teardown"}:
            raise ValueError(f"未知操作：{action}")
        elif site_id not in fleets:
            raise ValueError("局点尚未部署探针")
        probes = list(fleets[site_id])
        if action == "teardown":
            del fleets[site_id]
        return {
            "site_id": site_id,
            "action": action,
            "probes": probes,
            "status": "released" if action == "teardown" else "running",
        }

    return {
        t.name: t
        for t in (
            test_case_query,
            simulation_runner,
            metrics_collector,
            baseline_comparator,
            log_analyzer,
            fleet_manager,
        )
    }


# 保留原有导入方式，已有 CLI 和在线模型可以直接使用这些沙盒工具。
TOOL_MAP = create_sandbox_tools()
TOOLS = list(TOOL_MAP.values())
test_case_query = TOOL_MAP["test_case_query"]
simulation_runner = TOOL_MAP["simulation_runner"]
metrics_collector = TOOL_MAP["metrics_collector"]
baseline_comparator = TOOL_MAP["baseline_comparator"]
log_analyzer = TOOL_MAP["log_analyzer"]
fleet_manager = TOOL_MAP["fleet_manager"]
