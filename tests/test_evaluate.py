"""Tests for src/evaluate.py.

Per the project's acceptance checklist, these metrics are checked against
hand-computed toy examples, not only against sklearn's own functions
computing the same thing.
"""
import numpy as np
import pytest

from src.evaluate import (
    compute_metrics,
    metrics_by_anomaly_type,
    naive_threshold,
    sweep_cost_minimizing_threshold,
)


def test_compute_metrics_precision_recall_f1_hand_computed():
    # 8 rows, 4 true positives (idx 2,3,5,7), threshold=0.5.
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.4, 0.05, 0.7])

    # By hand: predicted positive (score >= 0.5) = idx 2 (0.9), 3 (0.8), 7 (0.7).
    # All three are true positives -> TP=3, FP=0.
    # True positives total = idx 2,3,5,7 = 4; idx 5 (score 0.4) is missed -> FN=1.
    # precision = 3/3 = 1.0, recall = 3/4 = 0.75,
    # f1 = 2*1.0*0.75/(1.0+0.75) = 1.5/1.75 = 0.8571428571428571
    metrics = compute_metrics(y_true, scores, threshold=0.5)

    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(0.75)
    assert metrics["f1"] == pytest.approx(6 / 7)
    assert metrics["threshold"] == pytest.approx(0.5)

    # Cross-check roc_auc/average_precision against sklearn directly too
    # (supplementing, not replacing, the hand-computed checks above).
    from sklearn.metrics import average_precision_score, roc_auc_score

    assert metrics["roc_auc"] == pytest.approx(roc_auc_score(y_true, scores))
    assert metrics["average_precision"] == pytest.approx(
        average_precision_score(y_true, scores)
    )


def test_average_precision_higher_for_informative_scores_than_uninformative():
    rng = np.random.default_rng(0)
    n_normal, n_anomaly = 90, 10
    y_true = np.array([0] * n_normal + [1] * n_anomaly)

    # Anomalies score clearly higher than normals -> near-perfect ranking.
    informative_scores = np.concatenate(
        [rng.normal(0.0, 1.0, n_normal), rng.normal(8.0, 1.0, n_anomaly)]
    )
    # Same values, randomly reassigned to rows -> uninformative w.r.t. labels.
    uninformative_scores = rng.permutation(informative_scores)

    metrics_informative = compute_metrics(y_true, informative_scores, threshold=4.0)
    metrics_uninformative = compute_metrics(y_true, uninformative_scores, threshold=4.0)

    assert metrics_informative["average_precision"] > metrics_uninformative["average_precision"]
    assert metrics_informative["average_precision"] > 0.9


def test_metrics_by_anomaly_type_recall_breakdown():
    anomaly_type = np.array(["none", "none", "spike", "spike", "drift"])
    y_true = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.6, 0.9, 0.2, 0.8])  # threshold=0.5
    threshold = 0.5

    df = metrics_by_anomaly_type(y_true, anomaly_type, scores, threshold)
    by_type = df.set_index("anomaly_type")

    # "none": 2 rows, 1 flagged (score 0.6 >= 0.5) -> recall (here, FPR) = 0.5
    assert by_type.loc["none", "n_rows"] == 2
    assert by_type.loc["none", "recall"] == pytest.approx(0.5)

    # "spike": 2 rows, only score 0.9 flagged -> recall = 0.5
    assert by_type.loc["spike", "n_rows"] == 2
    assert by_type.loc["spike", "recall"] == pytest.approx(0.5)

    # "drift": 1 row, score 0.8 >= 0.5 flagged -> recall = 1.0
    assert by_type.loc["drift", "n_rows"] == 1
    assert by_type.loc["drift", "recall"] == pytest.approx(1.0)


def test_naive_threshold_is_1_minus_contamination_quantile():
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # 20% contamination -> 80th percentile of [1,2,3,4,5] = 4.2
    t = naive_threshold(scores, contamination=0.2)
    assert t == pytest.approx(np.quantile(scores, 0.8))


def test_sweep_cost_minimizing_threshold_hand_computed():
    # 5 rows, scores already at clean quantile points so the 5-candidate
    # sweep lands exactly on [1,2,3,4,5].
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = np.array([0, 0, 1, 0, 1])  # positives at scores 3 and 5

    # Hand-computed total cost at each of the 5 candidate thresholds
    # (cost_fp=50, cost_fn=2000):
    #   t=1: FP=3 (idx0,1,3), FN=0            -> cost = 150
    #   t=2: FP=2 (idx1,3),   FN=0            -> cost = 100
    #   t=3: FP=1 (idx3),     FN=0            -> cost = 50   <- minimum
    #   t=4: FP=1 (idx3),     FN=1 (idx2)     -> cost = 2050
    #   t=5: FP=0,            FN=1 (idx2)     -> cost = 2000
    result = sweep_cost_minimizing_threshold(
        y_true, scores, cost_fp=50.0, cost_fn=2000.0, n_candidates=5
    )

    assert result["best_threshold"] == pytest.approx(3.0)
    assert result["best_total_cost"] == pytest.approx(50.0)
    assert result["best_fp"] == 1
    assert result["best_fn"] == 0
    assert len(result["sweep"]) == 5

    costs_by_threshold = {row["threshold"]: row["total_cost"] for row in result["sweep"]}
    assert costs_by_threshold[1.0] == pytest.approx(150.0)
    assert costs_by_threshold[2.0] == pytest.approx(100.0)
    assert costs_by_threshold[3.0] == pytest.approx(50.0)
    assert costs_by_threshold[4.0] == pytest.approx(2050.0)
    assert costs_by_threshold[5.0] == pytest.approx(2000.0)
