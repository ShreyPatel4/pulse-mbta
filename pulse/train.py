"""Training-scaffold logic reused by scripts/train.py: the temporal split,
baseline scoring, and model construction. Split out (rather than living
inline in the script) so it's importable and testable without a live
Postgres, MLflow, or a real model-fitting run -- see tests/test_train.py.

Task A (docs/2026-08-13-pulse-design.md): P(arrival delay > 180s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "route_hour_historical_late_rate",
    "headway_seconds",
    "hour_of_day",
    "day_of_week",
    "current_delay_persistence_seconds",
]

TRAIN_FRACTION = 0.70
LATE_PERSISTENCE_THRESHOLD_SECONDS = 180


def temporal_split(df: pd.DataFrame, train_fraction: float = TRAIN_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by row order, not randomly: the caller is responsible for
    passing df already sorted by scheduled_arrival (scripts/train.py's
    training-set query does this in SQL). A random split would let a
    route-hour bucket's rows scatter across both sides and would evaluate
    the model on data chronologically interleaved with its own training
    data -- not how this model would ever actually be used (always
    predicting forward from what's known so far)."""
    split_at = int(len(df) * train_fraction)
    return df.iloc[:split_at].copy(), df.iloc[split_at:].copy()


def baseline_scores(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Scores (P(late) proxies) for the 3 design-doc baselines, evaluated on
    test_df. Baseline 2's NaN fallback uses ONLY train_df's late rate --
    computing it from test_df, or the whole set, would leak test-set
    outcomes into the fallback value itself."""
    n = len(test_df)
    always_on_time = np.zeros(n)

    train_late_rate = float(train_df["late"].mean()) if len(train_df) else 0.0
    # pd.to_numeric first: a column that arrived as Python None/NaN mix (via
    # psycopg's raw rows, or a test fixture's plain list) can carry object
    # dtype, where .fillna's implicit downcast is deprecated -- coercing to
    # a real float dtype up front sidesteps that and is correct either way.
    route_hour_numeric = pd.to_numeric(test_df["route_hour_historical_late_rate"], errors="coerce")
    route_hour_rate = route_hour_numeric.fillna(train_late_rate).to_numpy()

    persistence_numeric = pd.to_numeric(test_df["current_delay_persistence_seconds"], errors="coerce")
    persistence = (persistence_numeric.fillna(0) > LATE_PERSISTENCE_THRESHOLD_SECONDS).astype(float).to_numpy()

    return {
        "baseline_always_on_time": always_on_time,
        "baseline_route_hour_rate": route_hour_rate,
        "baseline_persistence": persistence,
    }


def build_models() -> dict[str, object]:
    """LogisticRegression (imputed + scaled -- it can't handle NaN) and
    HistGradientBoostingClassifier ("GradientBoosting" -- this specific
    sklearn class handles NaN features natively, which several engineered
    features legitimately are when there's no history yet)."""
    logistic = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    gradient_boosting = HistGradientBoostingClassifier(random_state=0)
    return {"logistic_regression": logistic, "gradient_boosting": gradient_boosting}
