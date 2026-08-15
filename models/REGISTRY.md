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

## v20260813-051522

**PRELIMINARY - insufficient data for the report deliverable** (window is 0.16 days, need >= 7). route_hour_historical_late_rate is drawn from a window nearly identical to the one it's scoring against this early -- treat a high PR-AUC as evidence of that leakage-adjacency, not of model skill.

- git sha: `cf86a58138951e0d18f4277bf6ecfa191d7d82e3`
- data window: 2026-08-12 21:17:00-04:00 to 2026-08-13 01:13:00-04:00 (0.16 days)
- train/test rows: 7037 / 3017 (temporal split, 70% train)
- best real model: `logistic_regression`, saved to `v20260813-051522-logistic_regression.joblib` (gitignored binary)
- features: route_hour_historical_late_rate, headway_seconds, hour_of_day, day_of_week, current_delay_persistence_seconds

| candidate | pr_auc | recall@precision>=0.80 |
|---|---|---|
| baseline_always_on_time | 0.2343 | n/a (never reaches target precision) |
| baseline_route_hour_rate | 0.5852 | 0.2645 |
| baseline_persistence | 0.8303 | 0.8444 |
| logistic_regression | 0.9570 | 0.9066 |
| gradient_boosting | 0.9366 | 0.8642 |

## v20260815-205642

**PRELIMINARY - insufficient data for the report deliverable** (window is 2.79 days, need >= 7). See docs/report.md for what this window can and cannot support.

- git sha: `bde6ed5606d49e87ab1f950faf755ff3f47c6096` (the code state this run came from, not the commit carrying it)
- ingestion window: 2026-08-12 21:45:01.657904-04:00 to 2026-08-15 16:07:41.980657-04:00 (2.77 days), 10532354 stop_events rows, 65.0% uptime
- labels: 185247 closed / 142258 training-usable / 32047 gap_abutted / 10942 no_arrival_signal
- training window (scheduled_arrival): 2026-08-12 21:17:00-04:00 to 2026-08-15 16:08:00-04:00 (2.79 days)
- train split: 99590 rows [2026-08-12 21:17:00-04:00 .. 2026-08-15 00:04:00-04:00], late rate 38.69%
- test split: 42649 rows [2026-08-15 00:05:00-04:00 .. 2026-08-15 16:08:00-04:00], late rate 34.54%
- best real model: `gradient_boosting`, saved to `v20260815-205642-gradient_boosting.joblib` (gitignored binary)
- features: route_hour_historical_late_rate, headway_seconds, hour_of_day, day_of_week, current_delay_persistence_seconds

| candidate | pr_auc | roc_auc | recall@P>=0.80 (oracle) | train threshold | test precision | test recall | test accuracy | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_always_on_time | 0.3454 | 0.5000 | n/a (never reaches P>=0.80) | n/a | n/a | n/a | n/a | - | - | - | - |
| baseline_route_hour_rate | 0.4913 | 0.6711 | n/a (never reaches P>=0.80) | 0.7660 | 0.6514 | 0.0222 | 0.6582 | 27745 | 175 | 14402 | 327 |
| baseline_persistence | 0.8636 | 0.9224 | 0.8758 | 1.0000 | 0.9371 | 0.8758 | 0.9368 | 27054 | 866 | 1830 | 12899 |
| logistic_regression | 0.9678 | 0.9799 | 0.9373 | 0.2208 | 0.7999 | 0.9374 | 0.8974 | 24466 | 3454 | 922 | 13807 |
| gradient_boosting | 0.9711 | 0.9819 | 0.9557 | 0.1762 | 0.7782 | 0.9645 | 0.8928 | 23871 | 4049 | 523 | 14206 |

Reference: predicting on-time for every test row scores 0.6546 accuracy and 0.3454 PR-AUC.

Sensitivity, route-hour aggregate frozen at the train/test boundary:

| candidate | pr_auc | pr_auc delta vs per-row as-of | recall@P>=0.80 (oracle) |
|---|---|---|---|
| logistic_regression_frozen | 0.9672 | -0.0007 | 0.9386 |
| gradient_boosting_frozen | 0.9703 | -0.0007 | 0.9545 |

