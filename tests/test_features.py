"""Tests for src/features.py.

The no-lookahead test is the important one here: it is a real
mutation-based regression check (mutate a future row, recompute, assert
an earlier row's features are bit-identical), not an assumption read off
the code. A centered or lookahead rolling window would pass a casual code
review and still fail this test.
"""
import numpy as np
import pandas as pd
import pytest

from src.features import (
    ROLL_WINDOW,
    ROLLING_FEATURE_COLUMNS,
    calibration_monitoring_split,
    compute_calibration_contamination,
    engineer_features,
)
from src.generate_data import SENSORS


def _make_line_frame(line_id: str, n: int, start_time: pd.Timestamp, offset: float = 0.0) -> pd.DataFrame:
    """A small, deterministic (no randomness) synthetic frame for one line."""
    idx = np.arange(n, dtype=float)
    data = {
        "line_id": line_id,
        "timestamp": start_time + pd.to_timedelta(idx * 5, unit="m"),
    }
    for i, sensor in enumerate(SENSORS):
        # A distinct, deterministic pattern per sensor so bugs that mix
        # sensors or lines together are easy to notice.
        data[sensor] = offset + idx * (i + 1) + i
    data["is_anomaly"] = 0
    data["anomaly_type"] = "none"
    return pd.DataFrame(data)


def test_no_lookahead_mutating_future_row_leaves_earlier_row_unchanged():
    df = _make_line_frame("L1", 20, pd.Timestamp("2026-01-01"))
    out_before, _ = engineer_features(df)

    df_mutated = df.copy()
    last_idx = df_mutated.index[-1]
    df_mutated.loc[last_idx, SENSORS] = df_mutated.loc[last_idx, SENSORS] + 10_000.0
    out_after, _ = engineer_features(df_mutated)

    # Row 15 is valid (window=12 => rows 0-10 are NaN, 11+ are valid) and
    # strictly before the mutated row (19) with no overlap in the rolling
    # window (which only looks backward 12 readings).
    target_ts = df["timestamp"].iloc[15]
    row_before = out_before.loc[out_before["timestamp"] == target_ts].reset_index(drop=True)
    row_after = out_after.loc[out_after["timestamp"] == target_ts].reset_index(drop=True)

    pd.testing.assert_frame_equal(row_before, row_after)


def test_rolling_features_never_mix_across_lines():
    start = pd.Timestamp("2026-01-01")
    line1_alone = _make_line_frame("L1", 25, start, offset=0.0)
    line2 = _make_line_frame("L2", 25, start, offset=1_000_000.0)  # wildly different scale

    out_alone, _ = engineer_features(line1_alone)
    out_combined, _ = engineer_features(pd.concat([line1_alone, line2], ignore_index=True))

    out_combined_l1 = out_combined[out_combined["line_id"] == "L1"].reset_index(drop=True)
    out_alone = out_alone.reset_index(drop=True)

    pd.testing.assert_frame_equal(out_alone, out_combined_l1)


def test_leading_nan_row_count_matches_window_minus_one_per_line():
    # engineer_features uses a current-inclusive rolling window
    # (window=12, min_periods=12), so the first `window - 1` = 11 rows of
    # each line are dropped, not 12 — verified directly against pandas'
    # own rolling behavior (see the stage-2 commit message for the
    # arithmetic check). This corrects an off-by-one in the original
    # build plan, the same kind of slip already caught and fixed in
    # generate_data.py's episode-injection slicing.
    start = pd.Timestamp("2026-01-01")
    n_per_line = 20
    df = pd.concat(
        [
            _make_line_frame("L1", n_per_line, start),
            _make_line_frame("L2", n_per_line, start),
            _make_line_frame("L3", n_per_line, start),
        ],
        ignore_index=True,
    )
    n_lines = df["line_id"].nunique()

    _, n_dropped = engineer_features(df)

    assert n_dropped == (ROLL_WINDOW - 1) * n_lines


def test_calibration_monitoring_split_is_70_30_and_chronological():
    start = pd.Timestamp("2026-01-01")
    df = pd.concat(
        [
            _make_line_frame("L1", 100, start),
            _make_line_frame("L2", 100, start),
        ],
        ignore_index=True,
    )

    calib_df, monitor_df = calibration_monitoring_split(df, calib_frac=0.7)

    for line_id in ["L1", "L2"]:
        n_calib = (calib_df["line_id"] == line_id).sum()
        n_monitor = (monitor_df["line_id"] == line_id).sum()
        assert n_calib == 70
        assert n_monitor == 30

        calib_max_ts = calib_df.loc[calib_df["line_id"] == line_id, "timestamp"].max()
        monitor_min_ts = monitor_df.loc[monitor_df["line_id"] == line_id, "timestamp"].min()
        assert calib_max_ts < monitor_min_ts


def test_compute_calibration_contamination_matches_label_mean():
    df = pd.DataFrame(
        {
            "line_id": ["L1"] * 10,
            "is_anomaly": [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        }
    )
    assert compute_calibration_contamination(df) == pytest.approx(0.2)


def test_engineered_frame_has_no_nans_in_rolling_columns():
    start = pd.Timestamp("2026-01-01")
    df = _make_line_frame("L1", 30, start)
    out, _ = engineer_features(df)
    assert not out[ROLLING_FEATURE_COLUMNS].isna().any().any()
