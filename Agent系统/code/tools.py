import random
import uuid

from langchain_core.tools import tool


@tool
def test_case_query(feature: str, category: str = "regression") -> dict:
    """Query test cases from the 5G knowledge base for a given feature and category."""
    cases = [
        {
            "id": f"TC-{i:04d}",
            "name": f"{feature} scenario {i}",
            "priority": random.choice(["P0", "P1", "P2"]),
            "tags": random.sample(
                ["handover", "auth", "bearer", "kpi", "signaling"], k=2
            ),
        }
        for i in range(1, random.randint(5, 12))
    ]
    return {
        "feature": feature,
        "category": category,
        "cases": cases,
        "total": len(cases),
    }


@tool
def simulation_runner(test_case_ids: list, env: str = "sandbox") -> dict:
    """Run 5G simulations for the given test case IDs in a sandboxed environment."""
    results = {
        tc_id: {
            "status": random.choice(["PASS", "PASS", "PASS", "FAIL"]),
            "duration_ms": random.randint(200, 8000),
            "kpi": {
                "throughput_mbps": round(random.uniform(50, 200), 2),
                "latency_ms": round(random.uniform(1, 25), 2),
            },
        }
        for tc_id in test_case_ids
    }
    pass_count = sum(1 for r in results.values() if r["status"] == "PASS")
    pass_rate = pass_count / len(results) if results else 0.0
    session_id = f"sim-{uuid.uuid4().hex[:8]}"
    return {
        "session_id": session_id,
        "env": env,
        "results": results,
        "pass_rate": round(pass_rate, 4),
    }


@tool
def metrics_collector(session_id: str) -> dict:
    """Collect KPI metrics for a given test session ID."""
    return {
        "session_id": session_id,
        "throughput_avg_mbps": round(random.uniform(80, 180), 2),
        "latency_p99_ms": round(random.uniform(5, 35), 2),
        "packet_loss_rate": round(random.uniform(0, 0.05), 4),
        "handover_success_rate": round(random.uniform(0.92, 1.0), 4),
    }


@tool
def baseline_comparator(
    throughput_avg: float, latency_p99: float, baseline_version: str = "v1.0"
) -> dict:
    """Compare current KPI metrics against the registered baseline version."""
    degradations = []
    if throughput_avg < 90:
        degradations.append(
            f"throughput {throughput_avg:.1f} Mbps below threshold 90 Mbps"
        )
    if latency_p99 > 25:
        degradations.append(f"P99 latency {latency_p99:.1f} ms exceeds threshold 25 ms")
    return {
        "baseline_version": baseline_version,
        "degradations": degradations,
        "overall_status": "REGRESSION" if degradations else "STABLE",
    }


@tool
def log_analyzer(log_type: str, session_id: str) -> dict:
    """Analyze 5G signaling logs (RRC/NAS/PDCP) for protocol anomalies."""
    all_anomalies = [
        "RRC connection setup timeout",
        "NAS authentication failure",
        "Bearer setup rejected by network",
        "Handover preparation timeout",
        "PDCP reorder timeout exceeded",
    ]
    anomalies = random.sample(all_anomalies, k=random.randint(0, 2))
    return {
        "log_type": log_type,
        "session_id": session_id,
        "anomalies": anomalies,
        "severity": "HIGH" if anomalies else "NORMAL",
    }


@tool
def fleet_manager(
    site_id: str,
    action: str,
    probe_count: int = 4,
) -> dict:
    """
    Manage distributed test probe clusters for multi-site acceptance testing.

    action: 'deploy'  — activate probes on the target site
            'status'  — query running probe status
            'collect' — aggregate KPI results from all probes
            'teardown'— release probe resources after testing
    """
    if action == "deploy":
        probe_ids = [f"{site_id}-probe-{i:02d}" for i in range(1, probe_count + 1)]
        return {
            "site_id": site_id,
            "action": "deploy",
            "deployed_probes": probe_ids,
            "status": "running",
        }
    elif action == "status":
        import random

        probes = [
            {
                "probe_id": f"{site_id}-probe-{i:02d}",
                "status": random.choice(["running", "running", "running", "idle"]),
                "progress_pct": random.randint(40, 100),
            }
            for i in range(1, probe_count + 1)
        ]
        return {"site_id": site_id, "action": "status", "probes": probes}
    elif action == "collect":
        import random

        results = {
            f"{site_id}-probe-{i:02d}": {
                "vonr_call_success_rate": round(random.uniform(0.95, 1.0), 4),
                "ftp_throughput_mbps": round(random.uniform(80, 200), 2),
                "handover_success_rate": round(random.uniform(0.97, 1.0), 4),
                "paging_success_rate": round(random.uniform(0.98, 1.0), 4),
            }
            for i in range(1, probe_count + 1)
        }
        return {"site_id": site_id, "action": "collect", "probe_results": results}
    elif action == "teardown":
        return {
            "site_id": site_id,
            "action": "teardown",
            "status": "released",
            "probe_count": probe_count,
        }
    else:
        return {
            "site_id": site_id,
            "action": action,
            "error": f"unknown action: {action}",
        }


TOOLS = [
    test_case_query,
    simulation_runner,
    metrics_collector,
    baseline_comparator,
    log_analyzer,
    fleet_manager,
]
TOOL_MAP = {t.name: t for t in TOOLS}
