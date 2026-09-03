"""Tests for src/detectors.py.

The sign-convention test is the single most important test in this
project (see the module docstring in src/detectors.py): it is exactly
the check that would have caught a silent score-sign bug before it
reached a report, for every one of the five detectors.
"""
import numpy as np
import pandas as pd
import pytest

from src.detectors import (
    SVM_MAX_FIT_ROWS,
    build_zscore_matrix,
    compute_line_zscore_stats,
    fit_isolation_forest,
    fit_local_outlier_factor,
    fit_one_class_svm,
    fit_pca_reconstruction,
    fit_rolling_zscore_baseline,
)

N_INLIERS = 195
N_OUTLIERS = 5
CONTAMINATION = N_OUTLIERS / (N_INLIERS + N_OUTLIERS)
SEED = 42


def _make_inlier_outlier_matrix(n_features: int, seed: int = SEED):
    rng = np.random.default_rng(seed)
    inliers = rng.normal(loc=0.0, scale=1.0, size=(N_INLIERS, n_features))
    # every feature +15 standard deviations from the inlier cluster
    outliers = rng.normal(loc=15.0, scale=1.0, size=(N_OUTLIERS, n_features))
    X = np.vstack([inliers, outliers])
    inlier_idx = np.arange(N_INLIERS)
    outlier_idx = np.arange(N_INLIERS, N_INLIERS + N_OUTLIERS)
    return X, inlier_idx, outlier_idx


@pytest.mark.parametrize(
    "name, fit_fn, needs_contamination",
    [
        ("isolation_forest", fit_isolation_forest, True),
        ("local_outlier_factor", fit_local_outlier_factor, True),
        ("one_class_svm", fit_one_class_svm, True),
        ("pca_reconstruction", fit_pca_reconstruction, False),
    ],
)
def test_sign_convention_higher_score_means_more_anomalous(name, fit_fn, needs_contamination):
    X, inlier_idx, outlier_idx = _make_inlier_outlier_matrix(n_features=24)

    if needs_contamination:
        detector = fit_fn(X, CONTAMINATION, SEED)
    else:
        detector = fit_fn(X, None, SEED)

    scores = detector.predict_fn(X)

    assert detector.name == name
    assert scores.shape == (len(X),)
    assert scores[outlier_idx].mean() > scores[inlier_idx].mean(), (
        f"{name}: known outliers scored lower than known inliers on average - "
        f"this is exactly the sign-convention bug this test exists to catch"
    )


def test_sign_convention_rolling_zscore_baseline():
    # 6 columns to mirror the real per-line z-score matrix shape, but the
    # arithmetic under test (max(abs(z)) per row) doesn't care what built it.
    Z, inlier_idx, outlier_idx = _make_inlier_outlier_matrix(n_features=6)

    detector = fit_rolling_zscore_baseline(Z)
    scores = detector.predict_fn(Z)

    assert scores[outlier_idx].mean() > scores[inlier_idx].mean()


def test_one_class_svm_subsamples_large_input_without_erroring():
    n_total = SVM_MAX_FIT_ROWS + 1_000
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(n_total, 24))

    detector = fit_one_class_svm(X, contamination=0.03, seed=SEED)

    assert detector.fit_seconds >= 0
    # Scoring the full (unsampled) input should also work without erroring.
    scores = detector.predict_fn(X[:50])
    assert scores.shape == (50,)


def test_compute_line_zscore_stats_and_build_zscore_matrix():
    calib_df = pd.DataFrame(
        {
            "line_id": ["L1"] * 4 + ["L2"] * 4,
            "temperature_c": [10.0, 12.0, 10.0, 12.0, 100.0, 104.0, 100.0, 104.0],
            "pressure_psi": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "vibration_mm_s": [0.0] * 8,
            "motor_current_a": [0.0] * 8,
            "flow_rate_lpm": [0.0] * 8,
            "humidity_pct": [0.0] * 8,
        }
    )
    stats = compute_line_zscore_stats(calib_df)

    assert stats["L1"]["temperature_c"][0] == pytest.approx(11.0)
    assert stats["L2"]["temperature_c"][0] == pytest.approx(102.0)
    # pressure_psi is constant per line (zero variance) - std should be 0.
    assert stats["L1"]["pressure_psi"][1] == pytest.approx(0.0)

    # A row on L1 at exactly its own line's mean should z-score to ~0,
    # even though L2's mean/std for the same sensor is wildly different.
    monitor_row = pd.DataFrame(
        {
            "line_id": ["L1"],
            "temperature_c": [11.0],
            "pressure_psi": [1.0],
            "vibration_mm_s": [0.0],
            "motor_current_a": [0.0],
            "flow_rate_lpm": [0.0],
            "humidity_pct": [0.0],
        }
    )
    Z = build_zscore_matrix(monitor_row, stats)
    assert Z[0, 0] == pytest.approx(0.0)  # temperature_c z-score
    assert Z[0, 1] == pytest.approx(0.0)  # zero-variance sensor guarded, not NaN/inf
