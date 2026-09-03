"""Orchestration CLI: raw data -> features -> five detectors -> reports/*.

Loads (or generates) `data/sensor_readings.csv`, engineers features and
splits into calibration/monitoring periods, fits all five detectors from
`detectors.py` on the calibration period, scores the monitoring period
with each, evaluates every detector at both a naive and a cost-minimizing
threshold (overall and broken down by anomaly type), explains each
detector's top-scoring monitoring points, and writes every `reports/*`
file this project's README draws its real numbers from.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.detectors import DETECTOR_REGISTRY, build_zscore_matrix, compute_line_zscore_stats
from src.evaluate import compute_metrics, metrics_by_anomaly_type, naive_threshold, sweep_cost_minimizing_threshold
from src.explain import explain_flagged_point
from src.features import (
    FEATURE_COLUMNS,
    calibration_monitoring_split,
    compute_calibration_contamination,
    engineer_features,
    scale_features,
)
from src.generate_data import RANDOM_SEED, generate_dataset

DATA_PATH = Path("data/sensor_readings.csv")
REPORTS_DIR = Path("reports")
TOP_N_EXPLANATIONS = 10


def load_or_generate_data(path: Path, seed: int) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, parse_dates=["timestamp"])
    df = generate_dataset(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def run_pipeline(
    data_path: Path = DATA_PATH, reports_dir: Path = REPORTS_DIR, seed: int = RANDOM_SEED
) -> dict:
    reports_dir.mkdir(parents=True, exist_ok=True)

    df = load_or_generate_data(data_path, seed=seed)
    engineered, n_dropped = engineer_features(df)
    calib_df, monitor_df = calibration_monitoring_split(engineered)
    contamination = compute_calibration_contamination(calib_df)
    X_calib, X_monitor, _scaler = scale_features(calib_df, monitor_df)

    line_stats = compute_line_zscore_stats(calib_df)
    Z_calib = build_zscore_matrix(calib_df, line_stats)
    Z_monitor = build_zscore_matrix(monitor_df, line_stats)

    y_monitor = monitor_df["is_anomaly"].to_numpy()
    anomaly_type_monitor = monitor_df["anomaly_type"].to_numpy()

    calib_means = calib_df[FEATURE_COLUMNS].mean()
    calib_stds = calib_df[FEATURE_COLUMNS].std()

    eval_summary = {}
    cost_sweeps = {}
    by_type_frames = []
    sample_explanations = {}
    pr_curve_data = {}
    summary_rows = []

    for name, fit_fn in DETECTOR_REGISTRY.items():
        if name == "rolling_zscore_baseline":
            detector = fit_fn(Z_calib)
            scores = detector.predict_fn(Z_monitor)
        else:
            detector = fit_fn(X_calib, contamination, seed)
            scores = detector.predict_fn(X_monitor)

        naive_t = naive_threshold(scores, contamination)
        naive_metrics = compute_metrics(y_monitor, scores, naive_t)

        cost_result = sweep_cost_minimizing_threshold(y_monitor, scores)
        cost_metrics = compute_metrics(y_monitor, scores, cost_result["best_threshold"])
        cost_sweeps[name] = cost_result

        eval_summary[name] = {
            "fit_seconds": detector.fit_seconds,
            "roc_auc": naive_metrics["roc_auc"],
            "average_precision": naive_metrics["average_precision"],
            "naive_threshold": naive_t,
            "naive": naive_metrics,
            "cost_minimizing_threshold": cost_result["best_threshold"],
            "cost_minimizing": cost_metrics,
            "cost_sweep_summary": {
                "cost_fp": cost_result["cost_fp"],
                "cost_fn": cost_result["cost_fn"],
                "best_total_cost": cost_result["best_total_cost"],
                "best_fp": cost_result["best_fp"],
                "best_fn": cost_result["best_fn"],
            },
        }

        by_type_df = metrics_by_anomaly_type(y_monitor, anomaly_type_monitor, scores, naive_t)
        by_type_df.insert(0, "detector", name)
        by_type_frames.append(by_type_df)

        top_idx = np.argsort(scores)[::-1][:TOP_N_EXPLANATIONS]
        detector_explanations = []
        for i in top_idx:
            row_meta = monitor_df.iloc[i]
            feat_row = monitor_df.iloc[i][FEATURE_COLUMNS]
            top_features = explain_flagged_point(feat_row, calib_means, calib_stds, top_k=3)
            detector_explanations.append(
                {
                    "line_id": row_meta["line_id"],
                    "timestamp": str(row_meta["timestamp"]),
                    "true_anomaly_type": row_meta["anomaly_type"],
                    "score": float(scores[i]),
                    "top_features": top_features,
                }
            )
        sample_explanations[name] = detector_explanations

        precision, recall, _ = precision_recall_curve(y_monitor, scores)
        pr_curve_data[name] = (recall, precision, naive_metrics["average_precision"])

        summary_rows.append(
            {
                "detector": name,
                "pr_auc": naive_metrics["average_precision"],
                "roc_auc": naive_metrics["roc_auc"],
                "f1_naive": naive_metrics["f1"],
                "f1_cost": cost_metrics["f1"],
                "fit_seconds": detector.fit_seconds,
            }
        )

    with open(reports_dir / "eval_summary.json", "w") as f:
        json.dump(eval_summary, f, indent=2)

    by_type_all = pd.concat(by_type_frames, ignore_index=True)
    by_type_all.to_csv(reports_dir / "eval_by_anomaly_type.csv", index=False)

    with open(reports_dir / "cost_threshold_sweep.json", "w") as f:
        json.dump(cost_sweeps, f, indent=2)

    with open(reports_dir / "sample_explanations.json", "w") as f:
        json.dump(sample_explanations, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, (recall, precision, ap) in pr_curve_data.items():
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves by detector (monitoring period)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(reports_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows).sort_values("pr_auc", ascending=False)
    print(f"\n{'detector':22s} {'PR-AUC':>8s} {'ROC-AUC':>8s} {'F1(naive)':>10s} {'F1(cost)':>9s} {'fit_s':>8s}")
    for _, row in summary_df.iterrows():
        print(
            f"{row['detector']:22s} {row['pr_auc']:8.4f} {row['roc_auc']:8.4f} "
            f"{row['f1_naive']:10.4f} {row['f1_cost']:9.4f} {row['fit_seconds']:8.3f}"
        )
    print(f"\nn_dropped leading-NaN rows: {n_dropped}")
    print(f"calibration rows: {len(calib_df)}  monitoring rows: {len(monitor_df)}")
    print(f"calibration contamination: {contamination:.4f}")
    print(f"\nWrote reports to {reports_dir}/")

    return {
        "eval_summary": eval_summary,
        "cost_sweeps": cost_sweeps,
        "sample_explanations": sample_explanations,
        "summary_table": summary_df,
        "n_dropped": n_dropped,
        "n_calibration": len(calib_df),
        "n_monitoring": len(monitor_df),
        "calibration_contamination": contamination,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=str, default=str(DATA_PATH))
    parser.add_argument("--reports-dir", type=str, default=str(REPORTS_DIR))
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    t0 = time.perf_counter()
    run_pipeline(Path(args.data_path), Path(args.reports_dir), args.seed)
    print(f"\nTotal pipeline time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
