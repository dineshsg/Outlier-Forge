"""Synthetic multivariate sensor telemetry with labeled, categorized anomalies.

Generates readings for several production lines, each with a stable but
distinct per-sensor baseline, a shared latent "load factor" that couples
vibration/motor_current/temperature together (so a multivariate detector
has something real to exploit), daily seasonality on temperature/humidity,
and independent Gaussian noise on every sensor.

Four fault signatures are injected as discrete, non-overlapping episodes
per line: spike, drift, flatline (stuck sensor), and correlated_fault
(a multi-sensor bearing/motor-failure-style signature). Every row is
labeled with ground truth (`is_anomaly`, `anomaly_type`), which is what
lets `evaluate.py` compute real precision/recall/PR-AUC instead of the
circular "we told the model the contamination rate" pattern most
anomaly-detection portfolios fall into.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

LINE_IDS = ["Line-A", "Line-B", "Line-C", "Line-D", "Line-E", "Line-F"]

SAMPLE_INTERVAL_MINUTES = 5
N_DAYS = 45
READINGS_PER_DAY = 24 * 60 // SAMPLE_INTERVAL_MINUTES  # 288
READINGS_PER_LINE = N_DAYS * READINGS_PER_DAY  # 12,960 at N_DAYS=45

SENSORS = [
    "temperature_c",
    "pressure_psi",
    "vibration_mm_s",
    "motor_current_a",
    "flow_rate_lpm",
    "humidity_pct",
]

# Per-line baseline is drawn once per line at generation time (so each line
# has a stable, distinct operating point), uniformly from these ranges.
BASELINE_MEAN_RANGES = {
    "temperature_c": (60.0, 75.0),
    "pressure_psi": (40.0, 55.0),
    "vibration_mm_s": (2.0, 3.5),
    "motor_current_a": (10.0, 18.0),
    "flow_rate_lpm": (80.0, 120.0),
    "humidity_pct": (45.0, 60.0),
}
BASELINE_STD = {
    "temperature_c": 1.5,
    "pressure_psi": 2.0,
    "vibration_mm_s": 0.3,
    "motor_current_a": 0.8,
    "flow_rate_lpm": 4.0,
    "humidity_pct": 3.0,
}

ANOMALY_TYPES = ["spike", "drift", "flatline", "correlated_fault"]

# Episode counts per line (see §4.3 of the build plan for the rationale
# behind each fault signature).
N_SPIKE_EPISODES = 8
N_DRIFT_EPISODES = 5
N_FLATLINE_EPISODES = 4
N_CORRELATED_EPISODES = 4
MIN_GAP_READINGS = 12  # 1 hour at 5-minute intervals


def _generate_line_readings(
    line_id: str, rng: np.random.Generator, n_readings: int, start_time: pd.Timestamp
) -> pd.DataFrame:
    """Generate the normal (pre-anomaly) signal for one line."""
    baseline_mean = {
        sensor: rng.uniform(*BASELINE_MEAN_RANGES[sensor]) for sensor in SENSORS
    }

    # Shared latent load factor: AR(1) process.
    load = np.zeros(n_readings)
    for t in range(1, n_readings):
        load[t] = 0.9 * load[t - 1] + rng.normal(0, 1.0)

    timestamps = start_time + pd.to_timedelta(
        np.arange(n_readings) * SAMPLE_INTERVAL_MINUTES, unit="m"
    )
    minute_of_day = (
        timestamps.hour * 60 + timestamps.minute
    ).to_numpy()  # for seasonality phase

    data = {"line_id": line_id, "timestamp": timestamps}
    for sensor in SENSORS:
        mean = baseline_mean[sensor]
        std = BASELINE_STD[sensor]
        noise = rng.normal(0, std, size=n_readings)

        seasonality_term = np.zeros(n_readings)
        if sensor in ("temperature_c", "humidity_pct"):
            amplitude = 0.08 * mean
            phase = 2 * np.pi * minute_of_day / (24 * 60)
            seasonality_term = amplitude * np.sin(phase)

        load_term = np.zeros(n_readings)
        if sensor == "vibration_mm_s":
            load_term = load * 0.15 * std
        elif sensor == "motor_current_a":
            load_term = load * 0.12 * std
        elif sensor == "temperature_c":
            load_term = load * 0.05 * std

        data[sensor] = mean + seasonality_term + load_term + noise

    df = pd.DataFrame(data)
    df["is_anomaly"] = 0
    df["anomaly_type"] = "none"
    return df


def _sample_episode_starts(
    rng: np.random.Generator,
    n_readings: int,
    episode_specs: list[tuple[str, int]],
    max_attempts: int = 10_000,
) -> list[tuple[int, int, str]]:
    """Sample non-overlapping (start, duration, type) episodes for one line.

    Episodes must not overlap and must be separated by at least
    MIN_GAP_READINGS. Start times are sampled sequentially with
    reject/resample on collision.
    """
    placed: list[tuple[int, int, str]] = []
    for anomaly_type, duration in episode_specs:
        for _ in range(max_attempts):
            start = rng.integers(0, n_readings - duration)
            end = start + duration
            collision = any(
                start < (p_end + MIN_GAP_READINGS)
                and (p_start - MIN_GAP_READINGS) < end
                for p_start, p_end, _ in ((s, s + d, t) for s, d, t in placed)
            )
            if not collision:
                placed.append((start, duration, anomaly_type))
                break
        else:
            raise RuntimeError(
                f"could not place a non-overlapping {anomaly_type} episode "
                f"after {max_attempts} attempts"
            )
    return placed


def _build_episode_specs(rng: np.random.Generator) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    for _ in range(N_SPIKE_EPISODES):
        specs.append(("spike", 1))
    for _ in range(N_DRIFT_EPISODES):
        specs.append(("drift", int(rng.integers(24, 73))))
    for _ in range(N_FLATLINE_EPISODES):
        specs.append(("flatline", int(rng.integers(12, 37))))
    for _ in range(N_CORRELATED_EPISODES):
        specs.append(("correlated_fault", int(rng.integers(6, 19))))
    rng.shuffle(specs)  # avoid a fixed ordering biasing placement
    return specs


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n_readings = len(df)
    episode_specs = _build_episode_specs(rng)
    episodes = _sample_episode_starts(rng, n_readings, episode_specs)

    is_anomaly_col = df.columns.get_loc("is_anomaly")
    anomaly_type_col = df.columns.get_loc("anomaly_type")

    for start, duration, anomaly_type in episodes:
        idx = slice(start, start + duration)  # positional (iloc) — half-open
        df.iloc[idx, is_anomaly_col] = 1
        df.iloc[idx, anomaly_type_col] = anomaly_type

        if anomaly_type == "spike":
            sensor = rng.choice(SENSORS)
            sensor_col = df.columns.get_loc(sensor)
            sign = rng.choice([-1.0, 1.0])
            magnitude = BASELINE_STD[sensor] * rng.uniform(6, 10)
            df.iloc[idx, sensor_col] = df.iloc[idx, sensor_col] + sign * magnitude

        elif anomaly_type == "drift":
            sensor = rng.choice(SENSORS)
            sensor_col = df.columns.get_loc(sensor)
            peak = BASELINE_STD[sensor] * rng.uniform(4, 7)
            ramp = np.linspace(0, peak, duration)
            df.iloc[idx, sensor_col] = df.iloc[idx, sensor_col].to_numpy() + ramp

        elif anomaly_type == "flatline":
            sensor = rng.choice(SENSORS)
            sensor_col = df.columns.get_loc(sensor)
            stuck_value = df[sensor].iloc[start - 1] if start > 0 else df[sensor].iloc[start]
            df.iloc[idx, sensor_col] = stuck_value

        elif anomaly_type == "correlated_fault":
            for sensor in ("vibration_mm_s", "motor_current_a", "temperature_c"):
                sensor_col = df.columns.get_loc(sensor)
                magnitude = BASELINE_STD[sensor] * rng.uniform(3, 6)
                df.iloc[idx, sensor_col] = df.iloc[idx, sensor_col] + magnitude

    return df


def generate_dataset(
    n_days: int = N_DAYS, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    readings_per_line = n_days * READINGS_PER_DAY
    start_time = pd.Timestamp("2026-01-01 00:00:00")

    line_frames = []
    for i, line_id in enumerate(LINE_IDS):
        # Independent, reproducible stream per line, derived from the master seed.
        line_rng = np.random.default_rng(seed + i * 1000 + 1)
        line_df = _generate_line_readings(line_id, line_rng, readings_per_line, start_time)
        line_df = _inject_anomalies(line_df, line_rng)
        line_frames.append(line_df)

    df = pd.concat(line_frames, ignore_index=True)
    df = df[
        [
            "line_id",
            "timestamp",
            "temperature_c",
            "pressure_psi",
            "vibration_mm_s",
            "motor_current_a",
            "flow_rate_lpm",
            "humidity_pct",
            "is_anomaly",
            "anomaly_type",
        ]
    ]
    return df


def summarize(df: pd.DataFrame, n_days: int) -> dict:
    n_rows = len(df)
    overall_anomaly_rate = float(df["is_anomaly"].mean())
    anomaly_count_by_type = (
        df.loc[df["is_anomaly"] == 1, "anomaly_type"].value_counts().to_dict()
    )
    rows_per_line = df.groupby("line_id").size().to_dict()
    return {
        "n_rows": int(n_rows),
        "n_lines": len(LINE_IDS),
        "n_days": n_days,
        "date_range": [
            str(df["timestamp"].min()),
            str(df["timestamp"].max()),
        ],
        "overall_anomaly_rate": overall_anomaly_rate,
        "anomaly_count_by_type": {k: int(v) for k, v in anomaly_count_by_type.items()},
        "rows_per_line": {k: int(v) for k, v in rows_per_line.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-days", type=int, default=N_DAYS)
    parser.add_argument("--out", type=str, default="data/sensor_readings.csv")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--summary-out", type=str, default="reports/data_summary.json"
    )
    args = parser.parse_args()

    df = generate_dataset(n_days=args.n_days, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    summary = summarize(df, args.n_days)
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
