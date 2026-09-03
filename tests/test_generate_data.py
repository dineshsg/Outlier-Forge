"""Tests for src/generate_data.py.

Structural checks (row/line/type coverage, non-overlapping episodes,
seed reproducibility) run at a small size for speed. The anomaly-rate
check runs at the real full 45-day size - see the note in
test_full_size_row_count_and_anomaly_rate for why.
"""
import pandas as pd
import pytest

from src.generate_data import (
    ANOMALY_TYPES,
    LINE_IDS,
    MIN_GAP_READINGS,
    N_CORRELATED_EPISODES,
    N_DRIFT_EPISODES,
    N_FLATLINE_EPISODES,
    N_SPIKE_EPISODES,
    READINGS_PER_DAY,
    generate_dataset,
)

SMALL_N_DAYS = 10
FULL_N_DAYS = 45  # the real, committed dataset's size
EXPECTED_EPISODES_PER_LINE = (
    N_SPIKE_EPISODES + N_DRIFT_EPISODES + N_FLATLINE_EPISODES + N_CORRELATED_EPISODES
)


def _episode_blocks(line_df: pd.DataFrame) -> list[dict]:
    """Contiguous is_anomaly==1 runs for one line's rows, as {start,end,types}."""
    line_df = line_df.reset_index(drop=True)
    block_id = (line_df["is_anomaly"] != line_df["is_anomaly"].shift()).cumsum()
    blocks = []
    for _, block in line_df.groupby(block_id):
        if block["is_anomaly"].iloc[0] != 1:
            continue
        blocks.append(
            {
                "start": block.index[0],
                "end": block.index[-1],
                "types": set(block["anomaly_type"].unique()),
            }
        )
    return blocks


def test_small_run_row_count_matches_lines_times_readings_per_line():
    df = generate_dataset(n_days=SMALL_N_DAYS, seed=1)
    expected_per_line = SMALL_N_DAYS * READINGS_PER_DAY
    assert len(df) == len(LINE_IDS) * expected_per_line
    for line_id in LINE_IDS:
        assert (df["line_id"] == line_id).sum() == expected_per_line


def test_small_run_every_line_present():
    df = generate_dataset(n_days=SMALL_N_DAYS, seed=1)
    assert set(df["line_id"].unique()) == set(LINE_IDS)


def test_small_run_all_anomaly_types_present():
    df = generate_dataset(n_days=SMALL_N_DAYS, seed=1)
    present_types = set(df.loc[df["is_anomaly"] == 1, "anomaly_type"].unique())
    assert present_types == set(ANOMALY_TYPES)


def test_small_run_episodes_do_not_overlap_and_respect_min_gap():
    df = generate_dataset(n_days=SMALL_N_DAYS, seed=1)

    for line_id in LINE_IDS:
        line_df = df[df["line_id"] == line_id]
        blocks = _episode_blocks(line_df)

        # If any two episodes had overlapped or merged into one run, this
        # count would come in lower than the number actually injected -
        # an explicit, recomputed check, not an assumption about the
        # injection code's own reject/resample logic.
        assert len(blocks) == EXPECTED_EPISODES_PER_LINE, (
            f"{line_id}: expected {EXPECTED_EPISODES_PER_LINE} distinct "
            f"episodes, found {len(blocks)} contiguous anomalous blocks "
            f"- a sign two episodes overlapped or merged"
        )

        blocks = sorted(blocks, key=lambda b: b["start"])
        for prev, curr in zip(blocks, blocks[1:]):
            assert prev["end"] < curr["start"], f"{line_id}: overlapping episodes"
            gap = curr["start"] - prev["end"] - 1
            assert gap >= MIN_GAP_READINGS, (
                f"{line_id}: episodes only {gap} readings apart, "
                f"expected >= {MIN_GAP_READINGS}"
            )


def test_same_seed_reproduces_identical_output():
    df1 = generate_dataset(n_days=SMALL_N_DAYS, seed=123)
    df2 = generate_dataset(n_days=SMALL_N_DAYS, seed=123)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seeds_produce_different_output():
    df1 = generate_dataset(n_days=SMALL_N_DAYS, seed=1)
    df2 = generate_dataset(n_days=SMALL_N_DAYS, seed=2)
    assert not df1["temperature_c"].equals(df2["temperature_c"])


def test_full_size_row_count_and_anomaly_rate():
    # The build plan's [0.02, 0.06] target band is calibrated for the
    # full 45-day run specifically: episode counts are fixed *per line*
    # (see §4.3), not scaled by n_days, so a short run's anomaly rate is
    # mechanically much higher just from having a smaller denominator -
    # verified directly, a 10-day run lands around 13-14%, nowhere near
    # this band. The build plan's own wording ("generate with a small
    # --n-days... assert rate lands in [0.02, 0.06]") doesn't hold
    # arithmetically at small size, so - matching the off-by-one already
    # caught and documented in features.py's stage-2 commit - this test
    # checks the rate at the actual size the band was designed for
    # (which is also the real committed dataset's size) instead of
    # forcing an assertion that would fail at a small size regardless of
    # whether generation is correct.
    df = generate_dataset(n_days=FULL_N_DAYS, seed=42)

    assert len(df) == len(LINE_IDS) * FULL_N_DAYS * READINGS_PER_DAY
    assert len(df) == 77_760

    rate = df["is_anomaly"].mean()
    assert 0.02 <= rate <= 0.06

    present_types = set(df.loc[df["is_anomaly"] == 1, "anomaly_type"].unique())
    assert present_types == set(ANOMALY_TYPES)
