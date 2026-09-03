"""Causal feature engineering + calibration/monitoring split.

Every engineered feature at row t depends only on rows <= t within the
same line — never on a later row, and never on another line's rows. This
matters because the whole evaluation downstream assumes the monitoring
period was scored the way a real deployment would score it: one row at a
time, using only what was known up to that point. tests/test_features.py
verifies this with an actual mutation-based check, not just code review.

**Honest simplification, stated plainly (not hidden):** the calibration
period below still contains some injected anomalies — a real deployment's
"known-good" history isn't perfectly clean either. `compute_calibration_
contamination` measures the real anomaly rate within the calibration
period *using labels that exist only because this is a synthetic
dataset*. In a real deployment you would not have that number and would
need to estimate or tune the `contamination` hyperparameter some other
way. Detectors.py uses this value directly — it is a label-derived
hyperparameter, not an unsupervised one, and the README says so.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SENSORS = [
    "temperature_c",
    "pressure_psi",
    "vibration_mm_s",
    "motor_current_a",
    "flow_rate_lpm",
    "humidity_pct",
]

ROLL_WINDOW = 12  # 1 hour at 5-minute intervals

ROLL_MEAN_COLS = [f"{s}_roll_mean_1h" for s in SENSORS]
ROLL_STD_COLS = [f"{s}_roll_std_1h" for s in SENSORS]
DELTA_COLS = [f"{s}_delta" for s in SENSORS]
ROLLING_FEATURE_COLUMNS = ROLL_MEAN_COLS + ROLL_STD_COLS + DELTA_COLS  # 18

# 6 raw sensor columns + 18 engineered rolling columns = 24 feature columns,
# the matrix every detector in src/detectors.py is fit and scored on.
FEATURE_COLUMNS = SENSORS + ROLLING_FEATURE_COLUMNS  # 24


def engineer_features(
    df: pd.DataFrame, window: int = ROLL_WINDOW
) -> tuple[pd.DataFrame, int]:
    """Add causal rolling features per line; drop leading-NaN rows.

    For each line (grouped by `line_id`, sorted by `timestamp`) and each
    sensor: a rolling mean and rolling std over the trailing `window`
    readings (current reading included, `min_periods=window` so there
    are no partial windows), plus a row-to-row delta. Rows where any
    rolling feature is still NaN (the first `window - 1` rows of each
    line, since the window is current-inclusive) are dropped.

    Returns (engineered_df, n_dropped) — the caller is expected to log
    or report n_dropped rather than let it pass silently.
    """
    df = df.sort_values(["line_id", "timestamp"]).reset_index(drop=True)

    line_frames = []
    for _, g in df.groupby("line_id", sort=False):
        g = g.sort_values("timestamp").copy()
        for sensor in SENSORS:
            roll = g[sensor].rolling(window=window, min_periods=window)
            g[f"{sensor}_roll_mean_1h"] = roll.mean()
            g[f"{sensor}_roll_std_1h"] = roll.std()
            g[f"{sensor}_delta"] = g[sensor].diff()
        line_frames.append(g)

    out = pd.concat(line_frames, ignore_index=True)
    out = out.sort_values(["line_id", "timestamp"]).reset_index(drop=True)

    n_before = len(out)
    out = out.dropna(subset=ROLLING_FEATURE_COLUMNS).reset_index(drop=True)
    n_dropped = n_before - len(out)

    n_lines = df["line_id"].nunique()
    logger.info(
        "engineer_features: dropped %d leading-NaN rows across %d lines "
        "(%d per line, window=%d)",
        n_dropped,
        n_lines,
        window - 1,
        window,
    )
    return out, n_dropped


def calibration_monitoring_split(
    df: pd.DataFrame, calib_frac: float = 0.7
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-line, time-based split: first `calib_frac` of rows = calibration.

    Calibration is used to fit/scale detectors; monitoring is used only
    for scoring and evaluation, never for fitting. Splitting is done per
    line so every line contributes to both periods.
    """
    calib_parts = []
    monitor_parts = []
    for _, g in df.groupby("line_id", sort=False):
        g = g.sort_values("timestamp")
        split_idx = int(round(len(g) * calib_frac))
        calib_parts.append(g.iloc[:split_idx])
        monitor_parts.append(g.iloc[split_idx:])

    calib_df = pd.concat(calib_parts).sort_values(["line_id", "timestamp"]).reset_index(drop=True)
    monitor_df = pd.concat(monitor_parts).sort_values(["line_id", "timestamp"]).reset_index(drop=True)
    return calib_df, monitor_df


def compute_calibration_contamination(calib_df: pd.DataFrame) -> float:
    """The calibration period's measured anomaly rate (label-derived).

    Used as the `contamination` hyperparameter for IsolationForest/
    LocalOutlierFactor. See the module docstring: this is only available
    because this is a synthetic, labeled dataset.
    """
    return float(calib_df["is_anomaly"].mean())


def scale_features(
    calib_df: pd.DataFrame,
    monitor_df: pd.DataFrame,
    feature_cols: list[str] = FEATURE_COLUMNS,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on calibration features only; apply to both.

    IsolationForest is technically scale-invariant (it splits on raw
    thresholds), but scaling is applied uniformly anyway because
    LocalOutlierFactor, OneClassSVM, and PCA reconstruction error all
    require it, and a single shared feature matrix keeps the detector
    interface in detectors.py uniform.
    """
    scaler = StandardScaler()
    calib_scaled = scaler.fit_transform(calib_df[feature_cols])
    monitor_scaled = scaler.transform(monitor_df[feature_cols])
    return calib_scaled, monitor_scaled, scaler
