"""统计模块与学习飞轮数据入口均不需要模型、GPU 或云端服务。"""

import json

import pytest
from scipy import stats

from learning import ExpertReview, export_reviews
from statistical import compare_samples


@pytest.mark.parametrize("method", ["welch", "paired", "ks"])
def test_statistics_agree_with_scipy(method):
    a, b = [12, 14, 15, 18, 19], [3, 6, 8, 9, 10]
    result = compare_samples({"kpi": a}, {"kpi": b}, method=method)["results"]["kpi"]
    reference = {
        "welch": lambda: stats.ttest_ind(a, b, equal_var=False),
        "paired": lambda: stats.ttest_rel(a, b),
        "ks": lambda: stats.ks_2samp(a, b),
    }[method]()
    assert result["p_value"] == pytest.approx(reference.pvalue)
    if method != "ks":
        assert (
            result["confidence_interval"][0]
            < result["mean_delta"]
            < result["confidence_interval"][1]
        )


def test_no_significance_is_not_pass_and_multiple_comparison_is_corrected():
    result = compare_samples(
        {"a": [1, 2, 3], "b": [1, 2, 3]}, {"a": [1, 2, 3], "b": [1, 2, 3]}
    )
    assert result["results"]["a"]["decision"] == "NO_DIFFERENCE_DETECTED"
    assert result["results"]["a"]["interval_level"] == 0.975


@pytest.mark.parametrize(
    "a,b,method",
    [
        ([1], [1, 2], "welch"),
        ([float("nan"), 2], [1, 2], "ks"),
        ([1, 2, 3], [1, 2], "paired"),
        ([1, 1], [2, 2], "welch"),
        ([1, 2], [0, 1], "paired"),
    ],
)
def test_invalid_samples_are_rejected(a, b, method):
    with pytest.raises(ValueError):
        compare_samples({"kpi": a}, {"kpi": b}, method=method)


def test_expert_feedback_exports_only_confirmed_deduplicated_pairs(tmp_path):
    confirmed = ExpertReview("a", "验证切换", "通过", "失败：丢包超限", "expert", True)
    unconfirmed = ExpertReview("b", "验证鉴权", "通过", "失败", "expert", False)
    result = export_reviews([confirmed, confirmed, unconfirmed], tmp_path)
    assert result["sft_count"] == result["dpo_count"] == 1
    sft = json.loads((tmp_path / "sft.jsonl").read_text(encoding="utf-8"))
    dpo = json.loads((tmp_path / "dpo.jsonl").read_text(encoding="utf-8"))
    assert sft["messages"][-1]["content"] == dpo["chosen"] == "失败：丢包超限"
    assert dpo["rejected"] == "通过"
    assert sft["messages"][0]["role"] == "system"
    with pytest.raises(FileExistsError):
        export_reviews([confirmed], tmp_path)


def test_conflicting_expert_labels_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        export_reviews(
            [
                ExpertReview("a", "q", "a", "b", "e", True),
                ExpertReview("a", "q", "a", "c", "e", True),
            ],
            tmp_path,
        )


def test_same_answer_only_produces_sft(tmp_path):
    result = export_reviews([ExpertReview("a", "q", "a", "a", "e", True)], tmp_path)
    assert result["sft_count"] == 1 and result["dpo_count"] == 0
