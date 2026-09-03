# Build Plan & Stage Tracker

Development is split into stages, completed one after another, each
committed on its own so the history mirrors the build order laid out in
the project's build plan (multivariate sensor anomaly detection with
Isolation Forest, benchmarked against three other classical detectors).

| # | Stage | Status |
|---|-------|--------|
| 1 | Repo scaffold + `generate_data.py` — synthetic multi-line sensor telemetry with labeled, categorized anomalies (spike/drift/flatline/correlated_fault), run for real at full 45-day size | ✅ done |
| 2 | `features.py` — causal rolling features + calibration/monitoring split + scaling, with a real mutation-based no-lookahead regression test | ✅ done |
| 3 | `detectors.py` — five detectors (Isolation Forest, LOF, One-Class SVM, PCA reconstruction error, rolling z-score baseline) behind one uniform higher-is-more-anomalous interface, with the sign-convention regression test | ✅ done |
| 4 | `evaluate.py` — PR-AUC/ROC-AUC/precision/recall/F1, by-anomaly-type breakdown, cost-minimizing threshold sweep, checked against hand-computed toy examples | ✅ done |
| 5 | `explain.py` — deviation-from-baseline "why was this flagged" approximation (not SHAP/LIME) | ✅ done |
| 6 | `run_pipeline.py` — full end-to-end real run producing all `reports/*` outputs, including the PR-curve comparison plot | ✅ done |
| 7 | `test_generate_data.py` finalized against the real full-size run's actual statistics; full test suite green | ✅ done |
| 8 | `README.md` written last, using only real numbers from the committed `reports/*` files | ✅ done |

All 8 stages complete. Every number in `README.md` was cross-checked
against `reports/eval_summary.json` and `reports/eval_by_anomaly_type.csv`
programmatically before committing (see the stage-8 commit message).
