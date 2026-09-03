"""Per-feature "why was this flagged" approximation.

**What this is, stated plainly:** a deviation-from-baseline approximation
— for a flagged point, which features deviate the most (in z-score terms)
from the calibration period's mean for that feature — computed identically
regardless of which detector flagged the point. **What this is not:** a
model-specific attribution method. It is not SHAP values, not permutation
importance, and does not explain any individual model's decision boundary
— it explains the data, not the model. The README repeats this caveat
next to the example outputs it shows, so it isn't mistaken for something
more rigorous than it is.
"""
from __future__ import annotations

import pandas as pd


def explain_flagged_point(
    feature_row: pd.Series,
    calibration_means: pd.Series,
    calibration_stds: pd.Series,
    top_k: int = 3,
) -> list[dict]:
    """The `top_k` features whose calibration z-score has the largest magnitude.

    Returns a list of {"feature", "z_score", "value", "calibration_mean"}
    dicts, sorted by abs(z_score) descending. A calibration std of 0 (a
    constant feature during calibration) is treated as 1 to avoid a
    division by zero — any deviation from a constant baseline is then
    reported as its raw z-score-equivalent difference rather than an
    undefined value.
    """
    safe_stds = calibration_stds.where(calibration_stds != 0, 1.0)
    z_scores = (feature_row - calibration_means) / safe_stds
    ranked = z_scores.abs().sort_values(ascending=False)

    explanations = []
    for feature in ranked.index[:top_k]:
        explanations.append(
            {
                "feature": feature,
                "z_score": float(z_scores[feature]),
                "value": float(feature_row[feature]),
                "calibration_mean": float(calibration_means[feature]),
            }
        )
    return explanations
