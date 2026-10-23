"""原始重复样本的正式统计比较，与业务阈值判断分开。

Welch t 检验比较独立样本均值，配对 t 检验要求调用者确认一一配对，
KS 双样本检验比较分布。多指标采用 Bonferroni 控制家族错误率。
不显著只表示尚未检测出差异，绝不直接等价于业务通过或统计等效。
"""

import math

import numpy as np
from scipy import stats


def compare_samples(current: dict, baseline: dict, *, method="welch", alpha=0.05):
    """每个指标至少两个有限观测，返回 p 值、均值差、效应量与置信区间。

    t 检验的假设条件须由采样设计保障；KS 假设独立且主要适用于连续分布。
    这里的“差异”无好坏方向，业务回归仍需指标方向与阈值定义。
    """
    if method not in {"welch", "paired", "ks"} or not 0 < alpha < 1:
        raise ValueError("method 必须为 welch/paired/ks，alpha 必须在 (0, 1)")
    if not current or set(current) != set(baseline):
        raise ValueError("两组必须包含相同且非空的指标集合")
    corrected_alpha = alpha / len(current)
    results = {}
    for metric, values in current.items():
        a, b = (
            np.asarray(values, dtype=float),
            np.asarray(baseline[metric], dtype=float),
        )
        if (
            a.ndim != 1
            or b.ndim != 1
            or min(a.size, b.size) < 2
            or not (np.isfinite(a).all() and np.isfinite(b).all())
        ):
            raise ValueError(f"{metric} 需要至少两个有限的一维原始观测")
        delta = float(a.mean() - b.mean())
        interval = None
        effect = None
        if method == "ks":
            test = stats.ks_2samp(a, b)
        elif method == "paired":
            if a.size != b.size:
                raise ValueError("配对检验要求等长且逐项配对")
            differences = a - b
            sd = float(differences.std(ddof=1))
            if sd == 0:
                raise ValueError("配对差值方差为零，不能进行常规 t 推断")
            test = stats.ttest_rel(a, b)
            margin = float(
                stats.t.ppf(1 - corrected_alpha / 2, a.size - 1)
                * sd
                / math.sqrt(a.size)
            )
            interval, effect = [delta - margin, delta + margin], delta / sd
        else:
            va, vb = float(a.var(ddof=1)), float(b.var(ddof=1))
            sa, sb = va / a.size, vb / b.size
            if sa + sb == 0:
                raise ValueError("两组方差均为零，不能进行常规 t 推断")
            test = stats.ttest_ind(a, b, equal_var=False)
            df = (sa + sb) ** 2 / (sa**2 / (a.size - 1) + sb**2 / (b.size - 1))
            margin = float(
                stats.t.ppf(1 - corrected_alpha / 2, df) * math.sqrt(sa + sb)
            )
            pooled = math.sqrt(
                ((a.size - 1) * va + (b.size - 1) * vb) / (a.size + b.size - 2)
            )
            interval, effect = [delta - margin, delta + margin], delta / pooled
        results[metric] = {
            "method": method,
            "statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "adjusted_p_value": min(1.0, float(test.pvalue) * len(current)),
            "mean_delta": delta,
            "effect_size": effect,
            "confidence_interval": interval,
            "interval_level": 1 - corrected_alpha if interval else None,
            "decision": "DIFFERENT"
            if test.pvalue < corrected_alpha
            else "NO_DIFFERENCE_DETECTED",
            "n_current": int(a.size),
            "n_baseline": int(b.size),
        }
    return {"alpha": alpha, "correction": "bonferroni", "results": results}
