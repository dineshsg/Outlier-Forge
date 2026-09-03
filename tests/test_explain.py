"""Tests for src/explain.py."""
import pandas as pd
import pytest

from src.explain import explain_flagged_point


def test_explain_flagged_point_ranks_extreme_feature_first():
    means = pd.Series({"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0, "e": 0.0})
    stds = pd.Series({"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0, "e": 1.0})
    # "c" is 10 std above its calibration mean; everything else is near 0.
    row = pd.Series({"a": 0.1, "b": -0.2, "c": 10.0, "d": 0.05, "e": -0.15})

    result = explain_flagged_point(row, means, stds, top_k=3)

    assert len(result) == 3
    assert result[0]["feature"] == "c"
    assert result[0]["z_score"] == pytest.approx(10.0)
    assert result[0]["value"] == pytest.approx(10.0)
    assert result[0]["calibration_mean"] == pytest.approx(0.0)

    abs_z_scores = [abs(entry["z_score"]) for entry in result]
    assert abs_z_scores == sorted(abs_z_scores, reverse=True)


def test_explain_flagged_point_respects_top_k():
    means = pd.Series({f"f{i}": 0.0 for i in range(10)})
    stds = pd.Series({f"f{i}": 1.0 for i in range(10)})
    row = pd.Series({f"f{i}": float(i) for i in range(10)})

    result = explain_flagged_point(row, means, stds, top_k=5)

    assert len(result) == 5
    # Highest-index features have the largest values -> largest |z|.
    assert [entry["feature"] for entry in result] == ["f9", "f8", "f7", "f6", "f5"]


def test_explain_flagged_point_guards_zero_calibration_std():
    means = pd.Series({"a": 5.0, "b": 0.0})
    stds = pd.Series({"a": 0.0, "b": 1.0})  # "a" was constant during calibration
    row = pd.Series({"a": 5.0, "b": 3.0})

    result = explain_flagged_point(row, means, stds, top_k=2)

    by_feature = {entry["feature"]: entry for entry in result}
    assert by_feature["a"]["z_score"] == pytest.approx(0.0)  # value == mean, no div-by-zero
    assert by_feature["b"]["z_score"] == pytest.approx(3.0)
