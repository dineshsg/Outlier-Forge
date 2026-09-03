"""Five anomaly detectors behind one uniform, higher-is-more-anomalous interface.

**The pitfall this module is built around:** scikit-learn's anomaly-
detection APIs are inconsistent about score sign convention.
`IsolationForest.score_samples` and `LocalOutlierFactor.score_samples`
return *lower = more abnormal*. `OneClassSVM.decision_function` returns
*lower = more abnormal* too. A hand-rolled PCA-reconstruction-error score
is *higher = more abnormal* natively. Silently mixing these up produces a
detector that is confidently wrong — it flags normal points as anomalies
and vice versa, and every downstream metric still "computes" without
erroring, because precision/recall/PR-AUC don't know the sign is
inverted; they just report a bad number that looks like a bad model
instead of a sign bug.

Every `fit_*` function below returns a `FittedDetector` whose `predict_fn`
is guaranteed, by construction and by tests/test_detectors.py's
sign-convention regression test, to return scores where **higher always
means more anomalous** — the three methods that need a sign flip
(`isolation_forest`, `local_outlier_factor`, `one_class_svm`) apply it
inline, right next to the model call, so the fix is never more than one
line away from the API it corrects.

**One documented, deliberate exception to the "same matrix for every
detector" pattern:** `rolling_zscore_baseline` is a *per-line* statistic
(each production line has its own operating point) computed from raw
sensor values, not from the shared, globally-scaled 24-column feature
matrix the other four detectors use — a single StandardScaler fit across
all six lines together would wash out exactly the per-line baseline
differences this detector is supposed to catch. So it takes its own
per-line z-score matrix (`build_zscore_matrix`, 6 columns) instead of the
shared `X` (24 columns). Both are still plain `np.ndarray` in, `np.ndarray`
out — `evaluate.py`/`run_pipeline.py` just have to know which matrix goes
with which detector; see `DETECTOR_REGISTRY`'s docstring below.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

N_ESTIMATORS_IF = 200
N_NEIGHBORS_LOF = 35
SVM_MAX_FIT_ROWS = 5_000  # SVM training is O(n^2)-O(n^3); see fit_one_class_svm
PCA_N_COMPONENTS = 10

RAW_SENSORS = [
    "temperature_c",
    "pressure_psi",
    "vibration_mm_s",
    "motor_current_a",
    "flow_rate_lpm",
    "humidity_pct",
]


@dataclass
class FittedDetector:
    name: str
    fit_seconds: float
    predict_fn: Callable[[np.ndarray], np.ndarray]  # returns scores, HIGHER = MORE ANOMALOUS, always


def fit_isolation_forest(
    X_calib: np.ndarray, contamination: float, seed: int
) -> FittedDetector:
    model = IsolationForest(
        contamination=contamination, random_state=seed, n_estimators=N_ESTIMATORS_IF
    )
    t0 = time.perf_counter()
    model.fit(X_calib)
    fit_seconds = time.perf_counter() - t0

    # score_samples: lower = more abnormal -> negate so higher = more anomalous.
    def predict_fn(X: np.ndarray) -> np.ndarray:
        return -model.score_samples(X)

    return FittedDetector("isolation_forest", fit_seconds, predict_fn)


def fit_local_outlier_factor(
    X_calib: np.ndarray, contamination: float, seed: int
) -> FittedDetector:
    # LocalOutlierFactor has no random_state parameter; seed is accepted
    # only so every fit_* function in this module shares one call signature.
    del seed
    model = LocalOutlierFactor(
        novelty=True, contamination=contamination, n_neighbors=N_NEIGHBORS_LOF
    )
    t0 = time.perf_counter()
    model.fit(X_calib)
    fit_seconds = time.perf_counter() - t0

    # score_samples: lower = more abnormal -> negate so higher = more anomalous.
    def predict_fn(X: np.ndarray) -> np.ndarray:
        return -model.score_samples(X)

    return FittedDetector("local_outlier_factor", fit_seconds, predict_fn)


def fit_one_class_svm(
    X_calib: np.ndarray, contamination: float, seed: int
) -> FittedDetector:
    # SVM training is O(n^2)-O(n^3) and does not scale to tens of
    # thousands of rows. This subsample is a deliberate, documented
    # engineering tradeoff, not an oversight: it keeps this detector
    # runnable on the full calibration set without silently hanging or
    # blowing up build time, at the cost of fitting on a representative
    # slice of calibration data rather than all of it.
    if len(X_calib) > SVM_MAX_FIT_ROWS:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_calib), size=SVM_MAX_FIT_ROWS, replace=False)
        X_fit = X_calib[idx]
    else:
        X_fit = X_calib

    model = OneClassSVM(kernel="rbf", nu=contamination, gamma="scale")
    t0 = time.perf_counter()
    model.fit(X_fit)
    fit_seconds = time.perf_counter() - t0

    # decision_function: lower = more abnormal -> negate so higher = more anomalous.
    def predict_fn(X: np.ndarray) -> np.ndarray:
        return -model.decision_function(X)

    return FittedDetector("one_class_svm", fit_seconds, predict_fn)


def fit_pca_reconstruction(
    X_calib: np.ndarray, contamination: Optional[float] = None, seed: Optional[int] = None
) -> FittedDetector:
    # contamination is unused (PCA reconstruction error needs no
    # contamination hyperparameter) and accepted only for calling-
    # convention uniformity with the other fit_* functions.
    del contamination
    model = PCA(n_components=PCA_N_COMPONENTS, random_state=seed)
    t0 = time.perf_counter()
    model.fit(X_calib)
    fit_seconds = time.perf_counter() - t0

    # Per-row mean squared reconstruction error: already higher = more
    # anomalous natively, no sign correction needed.
    def predict_fn(X: np.ndarray) -> np.ndarray:
        reconstructed = model.inverse_transform(model.transform(X))
        return np.mean((X - reconstructed) ** 2, axis=1)

    return FittedDetector("pca_reconstruction", fit_seconds, predict_fn)


def compute_line_zscore_stats(calib_df: pd.DataFrame) -> dict[str, dict[str, tuple[float, float]]]:
    """Per-line, per-sensor (mean, std) from the calibration period's raw readings.

    This is deliberately *not* derived from the shared, globally-scaled
    24-column feature matrix — see the module docstring.
    """
    stats: dict[str, dict[str, tuple[float, float]]] = {}
    for line_id, g in calib_df.groupby("line_id"):
        stats[line_id] = {
            sensor: (float(g[sensor].mean()), float(g[sensor].std(ddof=0)))
            for sensor in RAW_SENSORS
        }
    return stats


def build_zscore_matrix(
    df: pd.DataFrame, line_stats: dict[str, dict[str, tuple[float, float]]]
) -> np.ndarray:
    """Per-row raw-sensor z-scores, each row scored against its own line's
    calibration-period mean/std (not the rolling features)."""
    n = len(df)
    Z = np.zeros((n, len(RAW_SENSORS)), dtype=float)
    line_ids = df["line_id"].to_numpy()
    for j, sensor in enumerate(RAW_SENSORS):
        values = df[sensor].to_numpy()
        means = np.array([line_stats[line_id][sensor][0] for line_id in line_ids])
        stds = np.array([line_stats[line_id][sensor][1] for line_id in line_ids])
        stds = np.where(stds == 0, 1.0, stds)  # guard against a degenerate zero-variance line/sensor
        Z[:, j] = (values - means) / stds
    return Z


def fit_rolling_zscore_baseline(
    Z_calib: np.ndarray, contamination: Optional[float] = None, seed: Optional[int] = None
) -> FittedDetector:
    """The floor every learned detector should beat.

    Not an sklearn model: there is nothing to fit beyond the per-line
    z-score stats already baked into `Z_calib` (see `compute_line_
    zscore_stats`/`build_zscore_matrix`). `contamination`/`seed` are
    unused, accepted only for calling-convention uniformity.
    """
    del contamination, seed
    t0 = time.perf_counter()
    fit_seconds = time.perf_counter() - t0  # no real fitting step

    # Already higher = more anomalous natively: the worst (largest
    # magnitude) single-sensor deviation this row shows, in either direction.
    def predict_fn(Z: np.ndarray) -> np.ndarray:
        return np.max(np.abs(Z), axis=1)

    return FittedDetector("rolling_zscore_baseline", fit_seconds, predict_fn)


# Every entry takes (calibration_matrix, contamination, seed) and returns a
# FittedDetector. Four entries expect the shared, globally-scaled 24-column
# feature matrix from features.scale_features(); "rolling_zscore_baseline"
# expects its own per-line z-score matrix from build_zscore_matrix() instead
# (see the module docstring for why) - callers must pass the right one.
DETECTOR_REGISTRY: dict[str, Callable[..., FittedDetector]] = {
    "isolation_forest": fit_isolation_forest,
    "local_outlier_factor": fit_local_outlier_factor,
    "one_class_svm": fit_one_class_svm,
    "pca_reconstruction": fit_pca_reconstruction,
    "rolling_zscore_baseline": fit_rolling_zscore_baseline,
}
