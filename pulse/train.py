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
    training-set query does this in SQL, with a full natural-key tiebreak so
    the order is total and the same on every run). A random split would let a
    route-hour bucket's rows scatter across both sides and would evaluate
    the model on data chronologically interleaved with its own training
    data -- not how this model would ever actually be used (always
    predicting forward from what's known so far).

    The cut is then pushed forward to the next change of scheduled_arrival,
    so no single timestamp lands on both sides. MBTA schedules are
    minute-granular and a busy minute carries dozens of trip-stops across 13
    routes; cutting through one puts the same minute in train and in test.
    Running this twice on the same table originally produced train late rates
    of 38.69% and 38.68% because the boundary minute's rows were ordered
    differently by Postgres each time. A report whose numbers move between
    runs cannot claim its numbers are reproducible, so both halves of that
    are fixed: total order in SQL, and a cut that only falls between
    timestamps.
    """
    split_at = int(len(df) * train_fraction)
    if 0 < split_at < len(df):
        times = df["scheduled_arrival"].to_numpy()
        boundary = times[split_at - 1]
        while split_at < len(df) and times[split_at] == boundary:
            split_at += 1
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


def split_summary(df: pd.DataFrame) -> dict[str, object]:
    """Boundaries and class balance for one split, so the report can state
    them rather than assert them. A temporal split on a short window can hand
    the test half a materially different late rate than train saw; that
    divergence explains metric shifts on its own and belongs next to the
    metrics, not in a footnote."""
    late = df["late"].astype(bool)
    return {
        "rows": int(len(df)),
        "first_scheduled_arrival": df["scheduled_arrival"].min(),
        "last_scheduled_arrival": df["scheduled_arrival"].max(),
        "late_rows": int(late.sum()),
        "late_rate": float(late.mean()) if len(df) else 0.0,
    }


def frozen_route_hour_late_rate(train_df: pd.DataFrame, target_df: pd.DataFrame) -> np.ndarray:
    """The strict "training window only" reading of the route-hour aggregate:
    one late rate per (route_id, hour_of_day) bucket, computed once from the
    TRAIN split, frozen, and applied unchanged to every row scored.

    This is a deliberate second reading, not a replacement. The feature
    pulse.features builds is a per-row as-of aggregate (every row sees the
    history strictly earlier than itself, including earlier rows inside the
    test window). That is point-in-time correct and it is what a deployed
    service would actually see, since by prediction time yesterday's and this
    morning's trips have already settled. But on a window this short, "the
    history strictly earlier than this test row" overlaps heavily with the
    test period itself, and the M2 report already flagged that adjacency as
    the thing a high PR-AUC might really be measuring.

    Freezing the aggregate at the train/test boundary removes that overlap
    completely: a test row's rate now comes only from data that existed
    before the split. Running both and reporting the delta turns "the
    aggregate might be too adjacent to what it scores" from a hedge into a
    number. NaN for a (route, hour) bucket the train split never saw -- the
    same never-fabricate-a-default rule pulse.features follows.
    """
    rates = train_df.groupby(["route_id", "hour_of_day"])["late"].mean()
    keys = pd.MultiIndex.from_arrays(
        [target_df["route_id"], target_df["hour_of_day"]], names=["route_id", "hour_of_day"]
    )
    return rates.reindex(keys).to_numpy(dtype=float)


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
