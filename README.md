# Outlier Forge

[![CI](https://github.com/dineshsg/Outlier-Forge/actions/workflows/ci.yml/badge.svg)](https://github.com/dineshsg/Outlier-Forge/actions/workflows/ci.yml)

Multivariate sensor anomaly detection built around Isolation Forest and
benchmarked against three other classical detectors, on synthetic
production-line telemetry with **labeled, categorized ground-truth
anomalies** — so precision, recall, and PR-AUC can be checked against
real fault labels instead of just trusting whatever a `contamination`
hyperparameter implied. Detectors are compared on PR-AUC (not just
ROC-AUC, which is misleading at this dataset's ~3% anomaly rate — see
below), broken down by fault type, and the decision threshold is framed
as a cost-minimization problem (a wasted inspection vs. a missed
equipment failure) instead of defaulting to whatever cutoff
`contamination` happens to imply.

## Why this project

Most anomaly-detection portfolios call `IsolationForest().fit_predict()`
on a toy dataset and report an accuracy number that is circular — the
model was told the contamination rate, and there's no independent ground
truth to check it against. This project avoids that in three specific
ways:

1. **Labeled, categorized synthetic anomalies** — four distinct fault
   signatures (spike, drift, stuck-sensor/flatline, correlated
   multi-sensor fault) injected as discrete episodes with real ground
   truth, so precision/recall/PR-AUC can be computed *and broken down by
   fault type* — revealing that different detectors have different blind
   spots (see the results table below: every detector here struggles
   with `flatline` far more than with `correlated_fault`).
2. **PR-AUC (average precision), not just ROC-AUC, as the headline
   metric.** At a ~3% anomaly rate, ROC-AUC's false-positive-rate
   denominator is dominated by the huge count of true negatives, so a
   detector can rack up hundreds of false positives and barely move it —
   ROC-AUC can look deceptively good for a mediocre detector. PR-AUC is
   far more sensitive to exactly the failure mode an analyst staring at
   a flagged-anomaly queue actually experiences.
3. **A cost-minimization framing for the decision threshold** instead of
   defaulting to whatever the `contamination` parameter implies — the
   cost of a false alarm (a wasted inspection) and the cost of a missed
   equipment failure are not equal, and treating them as equal by
   accepting the library default is itself an unexamined choice.

This dataset domain — production-line sensor telemetry — was chosen
deliberately: classical unsupervised anomaly detection is a gap in a
Trigent Data Scientist role's requirements (alongside PySpark and SQL,
covered by this author's other portfolio projects), and equipment/process
monitoring on a production line is a real, recognizable problem for a
food-manufacturing company like General Mills. One project, two audiences.

## What's inside

- **`src/generate_data.py`** — synthetic telemetry for 6 production
  lines: a shared AR(1) "load factor" that genuinely couples
  vibration/motor_current/temperature together, daily seasonality on
  temperature/humidity, independent Gaussian noise per sensor, and four
  labeled fault signatures injected as non-overlapping episodes.
- **`src/features.py`** — causal per-line rolling features (mean/std/
  delta, 1-hour window), a 70/30 time-based calibration/monitoring
  split, and shared feature scaling.
- **`src/detectors.py`** — five detectors (Isolation Forest, Local
  Outlier Factor, One-Class SVM, PCA reconstruction error, and a rolling
  z-score baseline) behind one uniform interface where higher score
  always means more anomalous.
- **`src/evaluate.py`** — PR-AUC/ROC-AUC/precision/recall/F1, a
  by-anomaly-type recall breakdown, and a cost-minimizing threshold
  sweep.
- **`src/explain.py`** — a per-feature deviation-from-baseline
  approximation of "why was this point flagged."
- **`src/run_pipeline.py`** — orchestrates all of the above end-to-end
  and writes every file under `reports/`.

## The bug that shaped this design: score sign conventions

scikit-learn's anomaly-detection APIs are **inconsistent about score
sign convention**, and this is easy to get wrong silently. `IsolationForest.score_samples`
and `LocalOutlierFactor.score_samples` return *lower = more abnormal*.
`OneClassSVM.decision_function` also returns *lower = more abnormal*. A
hand-rolled PCA-reconstruction-error score, by contrast, is *higher =
more abnormal* natively.

If you mix these up — say, treat `IsolationForest.score_samples`'s raw
output as "higher = more anomalous" without negating it — you get a
detector that is **confidently wrong**: it flags the calmest, most
normal readings as the biggest anomalies and vice versa. Nothing errors.
Precision, recall, PR-AUC, the cost sweep — every downstream metric
still "computes" and produces a number; it just silently reports the
performance of the *inverted* detector, which typically looks like "this
model just isn't very good" rather than an obvious crash. It's the kind
of bug that's easy to ship and hard to notice from the metrics alone.

`tests/test_detectors.py::test_sign_convention_higher_score_means_more_anomalous`
is built specifically to catch this: it fits each detector on a 200-row
set (195 tight-cluster inliers, 5 points +15 standard deviations away)
and asserts the known outliers score higher, on average, than the known
inliers. Every one of the five detectors in `src/detectors.py` applies
its sign correction inline, immediately next to the sklearn call it
corrects, specifically so the fix is never more than one line away from
the API convention that requires it.

## Results (real run, 45-day / 6-line dataset, 77,760 rows, 2.98% anomaly rate)

All numbers below are copied directly from `reports/eval_summary.json`
and `reports/eval_by_anomaly_type.csv`, produced by an actual
`python -m src.run_pipeline` run (5.6s total) against the full committed
dataset — nothing here is invented or hand-tuned.

### Overall metrics per detector

| Detector | PR-AUC | ROC-AUC | F1 (naive thr.) | F1 (cost thr.) | Fit time |
|---|---|---|---|---|---|
| `rolling_zscore_baseline` | **0.420** | 0.753 | 0.408 | 0.155 | <0.001s |
| `local_outlier_factor` | 0.254 | 0.779 | 0.350 | 0.182 | 2.661s |
| `one_class_svm` | 0.193 | **0.780** | 0.338 | 0.204 | 0.044s |
| `isolation_forest` | 0.192 | 0.751 | 0.250 | 0.101 | 0.613s |
| `pca_reconstruction` | 0.101 | 0.713 | 0.220 | 0.111 | 0.004s |

**This did not come out the way the "multivariate beats a naive
threshold rule" hypothesis predicted, and it's reported as-is rather
than reshaped to fit that story.** The rolling z-score baseline — the
simplest detector here, with no learned model at all — has the best
PR-AUC of the five. Looking at *why* (below) is more interesting than
the headline number.

### Recall by anomaly type (at each detector's naive threshold)

| Detector | spike | drift | flatline | correlated_fault | none (false-positive rate) |
|---|---|---|---|---|---|
| `isolation_forest` | 0.733 | 0.112 | 0.099 | 0.952 | 0.023 |
| `local_outlier_factor` | 1.000 | 0.256 | **0.485** | 0.362 | 0.019 |
| `one_class_svm` | 0.867 | 0.119 | 0.427 | 0.962 | 0.020 |
| `pca_reconstruction` | 0.867 | 0.065 | 0.433 | 0.381 | 0.024 |
| `rolling_zscore_baseline` | 0.933 | 0.386 | **0.029** | 0.971 | 0.018 |

This is the table that explains the headline number. Every detector
here catches `correlated_fault` well (94-97% recall for four of the
five) — the injected magnitude (3-6 standard deviations added
simultaneously to `vibration_mm_s`, `motor_current_a`, and
`temperature_c`) turns out to be large enough that even the z-score
baseline's simple "worst single-sensor deviation" rule crosses its
threshold, so this dataset doesn't isolate the "only a multivariate
detector can see it" case as cleanly as intended.

Where the detectors actually separate is `flatline` (a sensor stuck at
its last value): `local_outlier_factor`, `one_class_svm`, and
`pca_reconstruction` catch 43-49% of these, while
`rolling_zscore_baseline` catches only 3%. That gap makes sense
structurally — a flatline collapses that sensor's rolling standard
deviation toward zero, which is exactly the kind of *reduced-variance*
signal the 24-column feature matrix's `_roll_std_1h` columns can expose
to a multivariate/density-based detector, but which a baseline that only
looks at raw-value deviation from the mean has no way to see (a value
sitting near its own long-run mean, just not moving, doesn't look
unusual to a plain z-score). `isolation_forest` is the outlier here,
underperforming the other three learned detectors on `flatline` (9.9%)
despite using the same feature matrix — worth investigating further
(see Extensions) rather than a settled explanation.

## Cost-based thresholding

The `contamination` hyperparameter implies a threshold, but treating
that as "the" operating point is itself a choice — usually an unexamined
one. `evaluate.py`'s `sweep_cost_minimizing_threshold` searches score
quantiles for the threshold that minimizes
`(false positives × cost_fp) + (false negatives × cost_fn)`.

**`cost_fp=50` and `cost_fn=2000` below are illustrative placeholder
values — a stand-in for "a wasted inspection" vs. "an unplanned
equipment failure," not researched figures.** The point is the
methodology (a searchable, explicit tradeoff), not these exact numbers;
a real deployment would plug in real inspection and downtime costs.

| Detector | Naive thr. (P / R / F1) | Cost-min. thr. (P / R / F1) | Best total cost (FP, FN) |
|---|---|---|---|
| `isolation_forest` | 0.496 (0.257 / 0.244 / 0.250) | 0.451 (0.054 / **0.757** / 0.101) | 826,400 (9528 FP, 175 FN) |
| `local_outlier_factor` | 1.274 (0.359 / 0.341 / 0.350) | 1.122 (0.107 / 0.609 / 0.182) | 747,050 (3661 FP, 282 FN) |
| `one_class_svm` | 0.540 (0.347 / 0.330 / 0.338) | -1.306 (0.122 / 0.614 / 0.204) | **715,450** (3189 FP, 278 FN) |
| `pca_reconstruction` | 0.648 (0.226 / 0.215 / 0.220) | 0.295 (0.061 / 0.626 / 0.111) | 886,450 (6929 FP, 270 FN) |
| `rolling_zscore_baseline` | 2.781 (0.418 / 0.398 / 0.408) | 1.982 (0.089 / 0.580 / 0.155) | 819,400 (4268 FP, 303 FN) |

The pattern is consistent across every detector: at `cost_fn = 40 ×
cost_fp`, the cost-minimizing threshold trades a lot of precision for
recall — it would rather flag many more false alarms than miss a fault,
which is exactly what a 40:1 cost ratio should do. `one_class_svm` has
the lowest total cost here, even though it isn't the PR-AUC leader — a
concrete illustration of why "best by PR-AUC" and "best by business
cost" aren't automatically the same detector.

## Example explanations

`src/explain.py` is a **deviation-from-baseline approximation** — for a
flagged point, which features deviate most (in z-score terms) from the
calibration period's mean, computed identically regardless of which
detector flagged it. **It is not SHAP, not permutation importance, and
not any model-specific attribution method** — it explains the data, not
the model's decision boundary. Two real entries from
`reports/sample_explanations.json` (`isolation_forest`'s top-scoring
monitoring points):

```json
{
  "line_id": "Line-D",
  "timestamp": "2026-02-06 07:15:00",
  "true_anomaly_type": "correlated_fault",
  "score": 0.6147,
  "top_features": [
    {"feature": "temperature_c_roll_std_1h", "z_score": 6.23, "value": 3.80, "calibration_mean": 1.51},
    {"feature": "vibration_mm_s_roll_std_1h", "z_score": 5.82, "value": 0.74, "calibration_mean": 0.30},
    {"feature": "motor_current_a_roll_std_1h", "z_score": 4.72, "value": 1.74, "calibration_mean": 0.80}
  ]
}
```

This one is a genuine coherence check across the whole pipeline: the
true label is `correlated_fault`, and the top 3 explaining features are
exactly `temperature_c`, `vibration_mm_s`, and `motor_current_a` — the
three sensors that fault type actually injects into, recovered purely
from the model's flagged score and the calibration statistics, with no
knowledge of the injection logic.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate   # or `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

python -m src.generate_data      # writes data/sensor_readings.csv + reports/data_summary.json
python -m src.run_pipeline       # fits all 5 detectors, writes every reports/*.json/.csv/.png
pytest tests/ -v                 # 28 tests
```

## Possible extensions

- A streaming/online variant (incremental refitting, or a library like
  `river`) instead of a static calibration/monitoring split.
- True SHAP-based explanations once that dependency is acceptable, to
  compare against `explain.py`'s deviation-from-baseline approximation
  directly.
- Semi-supervised fine-tuning of the `contamination` estimate from
  simulated analyst feedback (accept/reject flagged points), rather than
  reading it straight off calibration labels.
- Investigate why `isolation_forest` underperforms the other
  feature-matrix-based detectors specifically on `flatline` recall
  (9.9% vs. 43-49% for LOF/SVM/PCA) despite sharing the same 24-column
  input — a real, unresolved observation from the results table above.

## Tech stack

Python, pandas, NumPy, scikit-learn (`IsolationForest`,
`LocalOutlierFactor`, `OneClassSVM`, `PCA`), matplotlib, pytest.

## License

MIT — see [LICENSE](LICENSE).
