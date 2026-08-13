"""Task A training scaffold (docs/2026-08-13-pulse-design.md):
P(arrival delay > 180s), scored by PR-AUC and recall at precision >= 0.80.

Usage: uv run python scripts/train.py

Trains sklearn LogisticRegression + HistGradientBoostingClassifier (see
pulse.train.build_models for why HistGradientBoosting specifically) against
the 3 design-doc baselines (always-on-time, route-hour historical rate,
delay-persistence -- pulse.train.baseline_scores), on whatever's in
features_trip_stop JOIN trip_stop_labels_training today.

DE-shaped, code-complete scaffold: this is meant to run correctly on M2's
~1-day data window, not to produce a defensible model yet. See the
PRELIMINARY banner below and docs/lineage.md for the honest caveats.

MLflow tracking is local-only (file:./mlruns, gitignored). The better of
the two real models by test PR-AUC is saved to ./models/ via joblib
(gitignored) with an entry appended to models/REGISTRY.md (committed --
see .gitignore: only the *.joblib binaries and ./mlruns are excluded).
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

# MLflow >=3.x's filesystem tracking backend ("./mlruns") is in maintenance
# mode and refuses to start without this opt-out (set before importing
# mlflow -- it's read at import/first-use time). The task calls for local
# file-store tracking specifically (no server, no external DB dependency),
# so this is a deliberate choice, not a workaround for a bug: a sqlite/db
# backend would work too but adds a stateful file this project doesn't
# otherwise need.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pulse import db, metrics, train

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "REGISTRY.md"
MLRUNS_DIR = REPO_ROOT / "mlruns"

# A data window shorter than this means the "beats guessing" report
# deliverable (M3/M4) isn't supportable yet -- print the banner rather than
# let a metrics table imply more confidence than the data can carry.
MIN_DAYS_FOR_REPORT = 7
MIN_PRECISION = 0.80

_TRAINING_SET_SQL = """
SELECT
    f.service_date_norm, f.route_id, f.direction_id, f.stop_id, f.trip_id,
    l.scheduled_arrival, l.delay_seconds, l.late,
    f.route_hour_historical_late_rate, f.headway_seconds, f.hour_of_day, f.day_of_week,
    f.current_delay_persistence_seconds
FROM features_trip_stop f
JOIN trip_stop_labels_training l USING (service_date_norm, route_id, direction_id, stop_id, trip_id)
ORDER BY l.scheduled_arrival
"""


def _load_training_set(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(_TRAINING_SET_SQL)
        columns = [c.name for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _evaluate(y_true, y_score) -> dict[str, float | None]:
    return {
        "pr_auc": metrics.pr_auc(y_true, y_score),
        "recall_at_precision_80": metrics.recall_at_precision(y_true, y_score, min_precision=MIN_PRECISION),
    }


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
    return result.stdout.strip()


def _format_metric(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a (never reaches target precision)"


def main() -> int:
    conn = db.connect()
    try:
        df = _load_training_set(conn)
    finally:
        conn.close()

    if df.empty:
        print(
            "train: no training-usable rows in features_trip_stop JOIN trip_stop_labels_training. "
            "Run scripts/build-labels.py and scripts/build-features.py first.",
            file=sys.stderr,
        )
        return 1

    window_start = df["scheduled_arrival"].min()
    window_end = df["scheduled_arrival"].max()
    window_days = (window_end - window_start).total_seconds() / 86400.0
    preliminary = window_days < MIN_DAYS_FOR_REPORT

    print(f"train: {len(df)} training-usable rows, window [{window_start}, {window_end}] ({window_days:.2f} days)")
    if preliminary:
        print()
        print("=" * 78)
        print("PRELIMINARY - insufficient data for the report deliverable")
        print(f"Data window is {window_days:.2f} days; the report deliverable (M3/M4) needs >= {MIN_DAYS_FOR_REPORT}.")
        print("route_hour_historical_late_rate in particular is drawn from a window nearly")
        print("identical to the one it's scoring against this early -- a HIGH PR-AUC here is")
        print("evidence of that leakage-adjacency, not of model skill. Treat every number below")
        print("as a correctness check on the pipeline, not as a claim the model beats guessing.")
        print("=" * 78)
        print()

    train_df, test_df = train.temporal_split(df)
    print(f"train: temporal split -- {len(train_df)} train rows, {len(test_df)} test rows")

    y_train = train_df["late"].astype(int)
    y_test = test_df["late"].astype(int)
    X_train = train_df[train.FEATURE_COLUMNS]
    X_test = test_df[train.FEATURE_COLUMNS]

    results: dict[str, dict[str, float | None]] = {}
    fitted_models: dict[str, object] = {}

    for name, scores in train.baseline_scores(train_df, test_df).items():
        results[name] = _evaluate(y_test, scores)

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("pulse-delay-classification")

    for name, model in train.build_models().items():
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        result = _evaluate(y_test, y_score)
        results[name] = result
        fitted_models[name] = model

        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("features", ",".join(train.FEATURE_COLUMNS))
            mlflow.log_param("train_rows", len(train_df))
            mlflow.log_param("test_rows", len(test_df))
            mlflow.log_param("window_days", round(window_days, 2))
            mlflow.log_metric("pr_auc", result["pr_auc"])
            if result["recall_at_precision_80"] is not None:
                mlflow.log_metric("recall_at_precision_80", result["recall_at_precision_80"])
            # serialization_format="pickle": mlflow's default (skops) refuses
            # to serialize a numpy.dtype it encounters in this pipeline's
            # fitted state (SimpleImputer statistics_ / HistGradientBoosting
            # internals) unless explicitly trusted. Plain pickle is well
            # understood for a local, single-operator tracking store like
            # this one; skops' extra trust-listing isn't buying anything
            # here since nothing ever loads an mlflow-tracked model from an
            # untrusted source in this project.
            mlflow.sklearn.log_model(model, name=name, serialization_format="pickle")

    # Log baselines as their own runs too, for side-by-side comparison in
    # the MLflow UI (tagged so they're not mistaken for trained models).
    for name in ("baseline_always_on_time", "baseline_route_hour_rate", "baseline_persistence"):
        result = results[name]
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("kind", "baseline")
            mlflow.log_param("model", name)
            mlflow.log_metric("pr_auc", result["pr_auc"])
            if result["recall_at_precision_80"] is not None:
                mlflow.log_metric("recall_at_precision_80", result["recall_at_precision_80"])

    print()
    print(f"{'candidate':<28} {'pr_auc':>10} {'recall@P>=0.80':>18}")
    for name, result in results.items():
        print(f"{name:<28} {result['pr_auc']:>10.4f} {_format_metric(result['recall_at_precision_80']):>18}")

    best_name = max(fitted_models, key=lambda n: results[n]["pr_auc"])
    best_model = fitted_models[best_name]
    print()
    print(f"train: best real model by test PR-AUC = {best_name} ({results[best_name]['pr_auc']:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    version = dt.datetime.now(dt.timezone.utc).strftime("v%Y%m%d-%H%M%S")
    model_path = MODELS_DIR / f"{version}-{best_name}.joblib"
    joblib.dump(best_model, model_path)
    print(f"train: saved best model to {model_path}")

    git_sha = _git_sha()
    _append_registry_entry(
        version=version,
        best_name=best_name,
        window_start=window_start,
        window_end=window_end,
        window_days=window_days,
        preliminary=preliminary,
        train_rows=len(train_df),
        test_rows=len(test_df),
        results=results,
        git_sha=git_sha,
        model_path=model_path,
    )
    print(f"train: appended entry to {REGISTRY_PATH}")

    return 0


def _append_registry_entry(
    *,
    version: str,
    best_name: str,
    window_start,
    window_end,
    window_days: float,
    preliminary: bool,
    train_rows: int,
    test_rows: int,
    results: dict[str, dict[str, float | None]],
    git_sha: str,
    model_path: Path,
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []
    if not REGISTRY_PATH.exists():
        lines.append("# Model registry\n\n")
        lines.append(
            "One entry per scripts/train.py run. Metrics are Task A (P(arrival delay > 180s)): "
            "PR-AUC and recall at precision >= 0.80, on a temporal (not random) train/test split. "
            "Models themselves (`*.joblib`) are gitignored -- this file is the durable record.\n\n"
        )

    lines.append(f"## {version}\n\n")
    if preliminary:
        lines.append(
            f"**PRELIMINARY - insufficient data for the report deliverable** "
            f"(window is {window_days:.2f} days, need >= {MIN_DAYS_FOR_REPORT}). "
            "route_hour_historical_late_rate is drawn from a window nearly identical to the one "
            "it's scoring against this early -- treat a high PR-AUC as evidence of that "
            "leakage-adjacency, not of model skill.\n\n"
        )
    lines.append(f"- git sha: `{git_sha}`\n")
    lines.append(f"- data window: {window_start} to {window_end} ({window_days:.2f} days)\n")
    lines.append(
        f"- train/test rows: {train_rows} / {test_rows} (temporal split, {train.TRAIN_FRACTION:.0%} train)\n"
    )
    lines.append(f"- best real model: `{best_name}`, saved to `{model_path.name}` (gitignored binary)\n")
    lines.append(f"- features: {', '.join(train.FEATURE_COLUMNS)}\n\n")
    lines.append("| candidate | pr_auc | recall@precision>=0.80 |\n")
    lines.append("|---|---|---|\n")
    for name, result in results.items():
        recall_str = _format_metric(result["recall_at_precision_80"])
        lines.append(f"| {name} | {result['pr_auc']:.4f} | {recall_str} |\n")
    lines.append("\n")

    with REGISTRY_PATH.open("a") as f:
        f.writelines(lines)


if __name__ == "__main__":
    sys.exit(main())
