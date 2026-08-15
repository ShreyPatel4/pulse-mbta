"""Tests for pulse.train: temporal split and baseline scoring, pure pandas
-- no Postgres, no model fitting, no MLflow."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pulse import train


def _df(n: int, **overrides) -> pd.DataFrame:
    base = {
        "late": [False] * n,
        "route_hour_historical_late_rate": [None] * n,
        "current_delay_persistence_seconds": [None] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_temporal_split_takes_first_fraction_as_train():
    df = pd.DataFrame({"scheduled_arrival": range(10), "late": [False] * 10})
    train_df, test_df = train.temporal_split(df, train_fraction=0.7)
    assert list(train_df["scheduled_arrival"]) == list(range(7))
    assert list(test_df["scheduled_arrival"]) == list(range(7, 10))


def test_temporal_split_never_puts_one_timestamp_on_both_sides():
    # Six rows, all at the same three timestamps. A naive 0.5 cut at index 3
    # would land inside the pair at t=2, putting that timestamp in train and
    # in test. The cut moves forward to the next change instead.
    df = pd.DataFrame({"scheduled_arrival": [1, 1, 2, 2, 3, 3], "late": [False] * 6})
    train_df, test_df = train.temporal_split(df, train_fraction=0.5)
    assert list(train_df["scheduled_arrival"]) == [1, 1, 2, 2]
    assert list(test_df["scheduled_arrival"]) == [3, 3]
    assert set(train_df["scheduled_arrival"]).isdisjoint(set(test_df["scheduled_arrival"]))


def test_temporal_split_puts_everything_in_train_when_the_window_is_one_timestamp():
    # Degenerate but reachable (one busy minute, nothing else): pushing the
    # cut forward runs off the end, so test is empty. Empty is the honest
    # outcome -- there is no later data to hold out. Callers see 0 test rows
    # rather than a split that silently shares a timestamp.
    df = pd.DataFrame({"scheduled_arrival": [5, 5, 5, 5], "late": [False] * 4})
    train_df, test_df = train.temporal_split(df, train_fraction=0.5)
    assert len(train_df) == 4
    assert len(test_df) == 0


def test_temporal_split_does_not_mutate_input():
    df = pd.DataFrame({"scheduled_arrival": range(10), "late": [False] * 10})
    before = df.copy()
    train.temporal_split(df, train_fraction=0.5)
    pd.testing.assert_frame_equal(df, before)


def test_temporal_split_returned_frames_are_independent_copies():
    df = pd.DataFrame({"scheduled_arrival": range(4), "late": [False] * 4})
    train_df, _test_df = train.temporal_split(df, train_fraction=0.5)
    train_df.loc[0, "scheduled_arrival"] = 999
    assert df.loc[0, "scheduled_arrival"] == 0  # original untouched


def test_baseline_always_on_time_is_all_zero():
    train_df = _df(3, late=[True, False, True])
    test_df = _df(4)
    scores = train.baseline_scores(train_df, test_df)
    assert list(scores["baseline_always_on_time"]) == [0.0, 0.0, 0.0, 0.0]


def test_baseline_route_hour_rate_uses_the_feature_when_present():
    train_df = _df(2, late=[False, False])
    test_df = _df(2, route_hour_historical_late_rate=[0.3, 0.9])
    scores = train.baseline_scores(train_df, test_df)
    assert list(scores["baseline_route_hour_rate"]) == [0.3, 0.9]


def test_baseline_route_hour_rate_fills_nan_with_train_late_rate_not_test():
    # train late rate: 2/4 = 0.5. If the fallback leaked from test instead
    # (test late rate would be 1/1=1.0), this would be 1.0.
    train_df = _df(4, late=[True, True, False, False])
    test_df = _df(1, route_hour_historical_late_rate=[None], late=[True])
    scores = train.baseline_scores(train_df, test_df)
    assert scores["baseline_route_hour_rate"][0] == 0.5


def test_baseline_persistence_thresholds_at_180_seconds():
    train_df = _df(1)
    test_df = _df(4, current_delay_persistence_seconds=[0, 180, 181, None])
    scores = train.baseline_scores(train_df, test_df)
    # 0 -> not late, 180 -> not late (strictly greater), 181 -> late,
    # None -> filled with 0 -> not late.
    assert list(scores["baseline_persistence"]) == [0.0, 0.0, 1.0, 0.0]


def test_split_summary_reports_boundaries_and_class_balance():
    df = pd.DataFrame(
        {
            "scheduled_arrival": pd.to_datetime(
                ["2026-08-13T10:00Z", "2026-08-13T11:00Z", "2026-08-13T12:00Z", "2026-08-13T13:00Z"]
            ),
            "late": [True, False, True, False],
        }
    )
    s = train.split_summary(df)
    assert s["rows"] == 4
    assert s["late_rows"] == 2
    assert s["late_rate"] == 0.5
    assert s["first_scheduled_arrival"] == pd.Timestamp("2026-08-13T10:00Z")
    assert s["last_scheduled_arrival"] == pd.Timestamp("2026-08-13T13:00Z")


def test_frozen_route_hour_rate_comes_only_from_the_train_split():
    # Train: route 1 hour 8 is 1 late of 2 -> 0.5. Test rows in that bucket
    # are all late; if the frozen rate were recomputed over train+test it
    # would climb toward 1.0. It must stay 0.5.
    train_df = pd.DataFrame(
        {"route_id": ["1", "1", "2"], "hour_of_day": [8, 8, 8], "late": [True, False, True]}
    )
    test_df = pd.DataFrame({"route_id": ["1", "1"], "hour_of_day": [8, 8], "late": [True, True]})
    rates = train.frozen_route_hour_late_rate(train_df, test_df)
    assert list(rates) == [0.5, 0.5]


def test_frozen_route_hour_rate_is_nan_for_a_bucket_train_never_saw():
    # Never fabricate a default for an unseen bucket -- same rule
    # pulse.features follows. NaN is the honest "nothing is known yet".
    train_df = pd.DataFrame({"route_id": ["1"], "hour_of_day": [8], "late": [True]})
    test_df = pd.DataFrame({"route_id": ["1", "9"], "hour_of_day": [8, 23], "late": [True, False]})
    rates = train.frozen_route_hour_late_rate(train_df, test_df)
    assert rates[0] == 1.0
    assert np.isnan(rates[1])


def test_build_models_returns_logistic_regression_and_gradient_boosting():
    models = train.build_models()
    assert set(models) == {"logistic_regression", "gradient_boosting"}


def test_build_models_logistic_pipeline_handles_nan_features_via_imputer():
    # LogisticRegression itself can't handle NaN -- the pipeline's imputer
    # must make that transparent. A fit on features with NaN shouldn't raise.
    models = train.build_models()
    X = pd.DataFrame(
        {
            "route_hour_historical_late_rate": [0.1, np.nan, 0.5, 0.2],
            "headway_seconds": [600, 700, np.nan, 650],
            "hour_of_day": [8, 9, 10, 11],
            "day_of_week": [1, 1, 2, 2],
            "current_delay_persistence_seconds": [np.nan, 30, 60, np.nan],
        }
    )
    y = [0, 1, 0, 1]
    models["logistic_regression"].fit(X, y)  # must not raise
    preds = models["logistic_regression"].predict_proba(X)
    assert preds.shape == (4, 2)


def test_build_models_gradient_boosting_handles_nan_features_natively():
    models = train.build_models()
    X = pd.DataFrame(
        {
            "route_hour_historical_late_rate": [0.1, np.nan, 0.5, 0.2],
            "headway_seconds": [600, 700, np.nan, 650],
            "hour_of_day": [8, 9, 10, 11],
            "day_of_week": [1, 1, 2, 2],
            "current_delay_persistence_seconds": [np.nan, 30, 60, np.nan],
        }
    )
    y = [0, 1, 0, 1]
    models["gradient_boosting"].fit(X, y)  # must not raise
    preds = models["gradient_boosting"].predict_proba(X)
    assert preds.shape == (4, 2)
