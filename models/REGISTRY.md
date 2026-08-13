# Model registry

One entry per scripts/train.py run. Metrics are Task A (P(arrival delay > 180s)): PR-AUC and recall at precision >= 0.80, on a temporal (not random) train/test split. Models themselves (`*.joblib`) are gitignored -- this file is the durable record.

## v20260813-051201

**PRELIMINARY - insufficient data for the report deliverable** (window is 0.16 days, need >= 7). route_hour_historical_late_rate is drawn from a window nearly identical to the one it's scoring against this early -- treat a high PR-AUC as evidence of that leakage-adjacency, not of model skill.

- git sha: `49a28b0d5a1815d4ac24b2216a94f14a137376b3`
- data window: 2026-08-12 21:17:00-04:00 to 2026-08-13 01:01:00-04:00 (0.16 days)
- train/test rows: 6546 / 2806 (temporal split, 70% train)
- best real model: `logistic_regression`, saved to `v20260813-051201-logistic_regression.joblib` (gitignored binary)
- features: route_hour_historical_late_rate, headway_seconds, hour_of_day, day_of_week, current_delay_persistence_seconds

| candidate | pr_auc | recall@precision>=0.80 |
|---|---|---|
| baseline_always_on_time | 0.2331 | n/a (never reaches target precision) |
| baseline_route_hour_rate | 0.5665 | 0.2462 |
| baseline_persistence | 0.7737 | 0.8058 |
| logistic_regression | 0.9245 | 0.8670 |
| gradient_boosting | 0.9204 | 0.8349 |

