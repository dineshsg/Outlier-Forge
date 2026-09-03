"""Evaluation metrics: PR-AUC as the headline metric, by-anomaly-type
recall, and a cost-minimizing decision threshold.

**Why PR-AUC (average precision), not just ROC-AUC:** at this dataset's
~3% anomaly rate, ROC-AUC can look deceptively good even for a mediocre
detector, because the false-positive *rate* denominator is dominated by
the huge number of true negatives — a detector can rack up hundreds of
false positives and barely move its false-positive rate. Precision, and
therefore PR-AUC, is far more sensitive to exactly that failure mode,
which is what an analyst staring at a flagged-anomaly queue actually
experiences. Both are reported below, but PR-AUC is the one this
project's README treats as the headline number.

**Why a cost-minimizing threshold, not just the library default:** the
`contamination` hyperparameter implies a threshold, but the "right"
threshold for a real deployment is a business decision — the cost of a
wasted inspection (a false alarm) versus the cost of a missed equipment
failure (a false negative) are not equal, and treating them as equal by
defaulting to `contamination`'s implied cutoff is itself a choice, just
an unexamined one. `sweep_cost_minimizing_threshold` makes that choice
explicit and searchable. The `cost_fp=50`/`cost_fn=2000` defaults below
are **illustrative placeholder values**, not researched figures — they
stand in for "a wasted inspection" vs. "an unplanned equipment failure"
to demonstrate the cost-based threshold-selection *methodology*, which
is the actual point; the README repeats this caveat next to the numbers
it produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

DEFAULT_COST_FP = 50.0
DEFAULT_COST_FN = 2000.0
DEFAULT_N_CANDIDATES = 200


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """ROC-AUC, PR-AUC (average precision), and precision/recall/F1 at `threshold`.

    ROC-AUC and average_precision are threshold-independent (rank-based);
    precision/recall/f1 use `scores >= threshold` as the positive
    prediction. `threshold` is echoed back in the result so a downstream
    report can label which operating point these were computed at.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def metrics_by_anomaly_type(
    y_true: np.ndarray, anomaly_type: np.ndarray, scores: np.ndarray, threshold: float
) -> pd.DataFrame:
    """Recall broken down by anomaly type, one row per type.

    "none" (the negative class) is included as a reference row - for it,
    "recall" is the fraction of true-negative rows incorrectly flagged
    (i.e. the false-positive rate), computed with the exact same formula
    as every other row: of the rows actually of this type, what fraction
    scored >= threshold.
    """
    anomaly_type = np.asarray(anomaly_type)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)

    rows = []
    for a_type in sorted(np.unique(anomaly_type)):
        mask = anomaly_type == a_type
        n_rows = int(mask.sum())
        n_flagged = int(y_pred[mask].sum())
        recall = (n_flagged / n_rows) if n_rows > 0 else float("nan")
        rows.append(
            {
                "anomaly_type": a_type,
                "n_rows": n_rows,
                "n_flagged": n_flagged,
                "recall": recall,
            }
        )
    return pd.DataFrame(rows)


def naive_threshold(scores: np.ndarray, contamination: float) -> float:
    """The score at the (1 - contamination) quantile of `scores`.

    Flags however many points as anomalous as the calibration
    contamination rate implies - the "naive" operating point every
    detector gets evaluated at before the cost-minimizing sweep below.
    """
    return float(np.quantile(np.asarray(scores), 1.0 - contamination))


def sweep_cost_minimizing_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    n_candidates: int = DEFAULT_N_CANDIDATES,
) -> dict:
    """Search score quantiles for the threshold minimizing total cost.

    total_cost(t) = (# false positives at t) * cost_fp
                  + (# false negatives at t) * cost_fn

    Candidate thresholds are `n_candidates` evenly spaced percentiles of
    `scores` (deduplicated). Returns the minimizing threshold plus the
    full sweep table (as a list of dicts, ready to serialize) for a plot.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, n_candidates)))

    sweep = []
    best = None
    for t in candidates:
        y_pred = (scores >= t).astype(int)
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        total_cost = fp * cost_fp + fn * cost_fn
        row = {
            "threshold": float(t),
            "fp": fp,
            "fn": fn,
            "total_cost": float(total_cost),
        }
        sweep.append(row)
        if best is None or total_cost < best["total_cost"]:
            best = row

    return {
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "best_threshold": best["threshold"],
        "best_total_cost": best["total_cost"],
        "best_fp": best["fp"],
        "best_fn": best["fn"],
        "sweep": sweep,
    }
