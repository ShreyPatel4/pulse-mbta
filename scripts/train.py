"""Task A training run (docs/2026-08-13-pulse-design.md):
P(arrival delay > 180s), scored by PR-AUC and recall at precision >= 0.80.

Usage: uv run python scripts/train.py

Trains sklearn LogisticRegression + HistGradientBoostingClassifier (see
pulse.train.build_models for why HistGradientBoosting specifically) against
the 3 design-doc baselines (always-on-time, route-hour historical rate,
delay-persistence -- pulse.train.baseline_scores), on features_trip_stop
JOIN trip_stop_labels_training, split by time.

What this run prints, in order: the data regime it is training on (ingestion
window, label counts by closed_reason, ingestion uptime derived from the
poll_runs gap ledger), the temporal split boundaries and each split's class
balance, the metrics table, a confusion matrix per candidate at a threshold
chosen on TRAIN, and a sensitivity run with the route-hour aggregate frozen
at the train/test boundary. docs/report.md is written from this output.

Two recall numbers are reported for every candidate and they mean different
things. `recall@P>=0.80 (oracle)` searches thresholds on the test set itself,
so it is the best operating point in hindsight and nothing deployable
reaches it. `recall @ train threshold` fixes the cutoff on train and reports
what it actually did on test. The second is the honest one; the first is
kept because it is the metric the design doc names.

MLflow tracking is local-only (file:./mlruns, gitignored). The better of the
two real models by test PR-AUC is saved to ./models/ via joblib (gitignored)
with an entry appended to models/REGISTRY.md (committed -- see .gitignore:
only the *.joblib binaries and ./mlruns are excluded).
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

from pulse import db, labels, metrics, train

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "REGISTRY.md"
MLRUNS_DIR = REPO_ROOT / "mlruns"

# A data window shorter than this means the "beats guessing" report
# deliverable (M3/M4) isn't supportable yet -- print the banner rather than
# let a metrics table imply more confidence than the data can carry. This
# gate fires on the M3/M4 run and is quoted in docs/report.md rather than
# tuned away; lowering it to make the banner disappear would be the exact
# dishonesty the report is supposed to guard against.
MIN_DAYS_FOR_REPORT = 7
MIN_PRECISION = 0.80

BASELINE_NAMES = ("baseline_always_on_time", "baseline_route_hour_rate", "baseline_persistence")

_TRAINING_SET_SQL = """
SELECT
    f.service_date_norm, f.route_id, f.direction_id, f.stop_id, f.trip_id,
    l.scheduled_arrival, l.delay_seconds, l.late,
    f.route_hour_historical_late_rate, f.headway_seconds, f.hour_of_day, f.day_of_week,
    f.current_delay_persistence_seconds
FROM features_trip_stop f
JOIN trip_stop_labels_training l USING (service_date_norm, route_id, direction_id, stop_id, trip_id)
ORDER BY l.scheduled_arrival, service_date_norm, route_id, direction_id, stop_id, trip_id
"""

# The ingestion window is taken from trip_stop_labels' own observed spans, not
# from a live max(polled_at) on stop_events: the poller keeps running while
# this script does, so a live reading would drift between runs and no number
# in the report would reproduce. The label table's spans are pinned by the
# build that produced it.
_REGIME_SQL = """
SELECT
    (SELECT min(observed_span_start) FROM trip_stop_labels),
    (SELECT max(observed_span_end) FROM trip_stop_labels),
    (SELECT count(*) FROM trip_stop_labels),
    (SELECT count(*) FROM trip_stop_labels WHERE closed_reason IS NULL),
    (SELECT count(*) FROM trip_stop_labels WHERE closed_reason = 'gap_abutted'),
    (SELECT count(*) FROM trip_stop_labels WHERE closed_reason = 'no_arrival_signal')
"""

_STOP_EVENTS_IN_WINDOW_SQL = "SELECT count(*) FROM stop_events WHERE polled_at <= %(until)s"

# Ingestion uptime, derived from the same ledger and the same threshold that
# pulse.labels' gap-exclusion rule uses (GAP_THRESHOLD_SECONDS), so the
# reported uptime and the gap_abutted count come from one definition rather
# than two. Note that scripts/publish-status.py's public ledger uses a
# different rule (3x median cadence) for the public page, so its gap COUNT
# will not match this one; the definition used here is the one that decides
# which labels get excluded.
_UPTIME_SQL = """
WITH bounded AS (
    SELECT polled_at FROM poll_runs WHERE polled_at <= %(until)s
), consecutive AS (
    SELECT polled_at, lag(polled_at) OVER (ORDER BY polled_at) AS prev FROM bounded
), gaps AS (
    SELECT extract(epoch FROM (polled_at - prev)) AS secs
    FROM consecutive
    WHERE prev IS NOT NULL AND extract(epoch FROM (polled_at - prev)) > %(threshold)s
)
SELECT
    (SELECT count(*) FROM bounded),
    (SELECT extract(epoch FROM (max(polled_at) - min(polled_at))) FROM bounded),
    (SELECT count(*) FROM gaps),
    (SELECT coalesce(sum(secs), 0) FROM gaps),
    (SELECT coalesce(max(secs), 0) FROM gaps)
"""


def _load_training_set(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(_TRAINING_SET_SQL)
        columns = [c.name for c in cur.description]
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


def _regime(conn) -> dict[str, object]:
    with conn.cursor() as cur:
        cur.execute(_REGIME_SQL)
        ingest_start, ingest_end, labels_total, usable, gap_abutted, no_signal = cur.fetchone()
        cur.execute(_STOP_EVENTS_IN_WINDOW_SQL, {"until": ingest_end})
        (stop_events_rows,) = cur.fetchone()
        cur.execute(_UPTIME_SQL, {"until": ingest_end, "threshold": labels.GAP_THRESHOLD_SECONDS})
        cycles, ledger_span_s, gap_count, gap_seconds, longest_gap_s = cur.fetchone()

    # extract(epoch ...) and sum() over it come back as decimal.Decimal from
    # psycopg; coerce before any float arithmetic.
    ledger_span_s = float(ledger_span_s or 0.0)
    gap_seconds = float(gap_seconds or 0.0)
    longest_gap_s = float(longest_gap_s or 0.0)
    uptime_pct = 100.0 * (1.0 - gap_seconds / ledger_span_s) if ledger_span_s else 0.0
    return {
        "ingest_start": ingest_start,
        "ingest_end": ingest_end,
        "ingest_days": (ingest_end - ingest_start).total_seconds() / 86400.0,
        "stop_events_rows": stop_events_rows,
        "labels_total": labels_total,
        "labels_usable": usable,
        "labels_gap_abutted": gap_abutted,
        "labels_no_arrival_signal": no_signal,
        "poll_cycles": cycles,
        "ledger_span_hours": ledger_span_s / 3600.0,
        "gap_count": gap_count,
        "gap_hours": gap_seconds / 3600.0,
        "longest_gap_hours": longest_gap_s / 3600.0,
        "uptime_pct": uptime_pct,
    }


def _evaluate(y_train, train_score, y_test, test_score) -> dict[str, object]:
    """Every number this project reports about one candidate, in one place.

    The threshold is chosen on TRAIN and then applied, fixed, to test. Picking
    it on test to hit precision 0.80 would tune on the held-out split and the
    confusion matrix underneath it would be meaningless.
    """
    threshold = metrics.threshold_for_precision(y_train, train_score, min_precision=MIN_PRECISION)
    confusion = (
        metrics.confusion_at_threshold(y_test, test_score, threshold) if threshold is not None else None
    )
    return {
        "pr_auc": metrics.pr_auc(y_test, test_score),
        # roc_auc is reported, never ranked on. It is here so the report's
        # case against it is made with these numbers rather than in the
        # abstract.
        "roc_auc": metrics.roc_auc(y_test, test_score),
        "recall_at_precision_80_oracle": metrics.recall_at_precision(
            y_test, test_score, min_precision=MIN_PRECISION
        ),
        "train_threshold": threshold,
        "confusion": confusion,
    }


def _git_sha() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
    return result.stdout.strip()


def _fmt(value: float | None, spec: str = ".4f") -> str:
    return format(value, spec) if value is not None else "n/a"


def _fmt_recall(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a (never reaches P>=0.80)"


def main() -> int:
    conn = db.connect()
    try:
        regime = _regime(conn)
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

    _print_regime(regime)

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
        print("A window this short sees no weekly seasonality, one weather regime, and one")
        print("segment of the service calendar. Nothing below generalizes past this window on")
        print("its own evidence. See docs/report.md, 'What this cannot say'.")
        print("=" * 78)
        print()

    train_df, test_df = train.temporal_split(df)
    train_summary = train.split_summary(train_df)
    test_summary = train.split_summary(test_df)
    _print_split(train_summary, test_summary)

    y_train = train_df["late"].astype(int)
    y_test = test_df["late"].astype(int)
    X_train = train_df[train.FEATURE_COLUMNS]
    X_test = test_df[train.FEATURE_COLUMNS]

    results: dict[str, dict[str, object]] = {}
    fitted_models: dict[str, object] = {}

    train_baselines = train.baseline_scores(train_df, train_df)
    test_baselines = train.baseline_scores(train_df, test_df)
    for name in BASELINE_NAMES:
        results[name] = _evaluate(y_train, train_baselines[name], y_test, test_baselines[name])

    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR}")
    mlflow.set_experiment("pulse-delay-classification")

    for name, model in train.build_models().items():
        model.fit(X_train, y_train)
        result = _evaluate(
            y_train, model.predict_proba(X_train)[:, 1], y_test, model.predict_proba(X_test)[:, 1]
        )
        results[name] = result
        fitted_models[name] = model
        _log_mlflow_run(name, result, train_summary, test_summary, window_days, model=model)

    for name in BASELINE_NAMES:
        _log_mlflow_run(name, results[name], train_summary, test_summary, window_days, model=None)

    # Sensitivity: the route-hour aggregate frozen at the train/test boundary
    # (pulse.train.frozen_route_hour_late_rate) instead of per-row as-of.
    frozen_results = _run_frozen_aggregate_sensitivity(
        train_df, test_df, y_train, y_test, train_summary, test_summary, window_days
    )

    _print_results(results, frozen_results, test_summary)

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
        regime=regime,
        window_start=window_start,
        window_end=window_end,
        window_days=window_days,
        preliminary=preliminary,
        train_summary=train_summary,
        test_summary=test_summary,
        results=results,
        frozen_results=frozen_results,
        git_sha=git_sha,
        model_path=model_path,
    )
    print(f"train: appended entry to {REGISTRY_PATH}")

    return 0


def _run_frozen_aggregate_sensitivity(
    train_df, test_df, y_train, y_test, train_summary, test_summary, window_days
) -> dict[str, dict[str, object]]:
    frozen_train = train_df.copy()
    frozen_test = test_df.copy()
    frozen_train["route_hour_historical_late_rate"] = train.frozen_route_hour_late_rate(train_df, train_df)
    frozen_test["route_hour_historical_late_rate"] = train.frozen_route_hour_late_rate(train_df, test_df)

    out: dict[str, dict[str, object]] = {}
    for name, model in train.build_models().items():
        model.fit(frozen_train[train.FEATURE_COLUMNS], y_train)
        result = _evaluate(
            y_train,
            model.predict_proba(frozen_train[train.FEATURE_COLUMNS])[:, 1],
            y_test,
            model.predict_proba(frozen_test[train.FEATURE_COLUMNS])[:, 1],
        )
        out[name] = result
        _log_mlflow_run(
            f"{name}_frozen_route_hour", result, train_summary, test_summary, window_days, model=None
        )
    return out


def _log_mlflow_run(name, result, train_summary, test_summary, window_days, model) -> None:
    with mlflow.start_run(run_name=name):
        if name.startswith("baseline_"):
            mlflow.set_tag("kind", "baseline")
        if name.endswith("_frozen_route_hour"):
            mlflow.set_tag("kind", "sensitivity")
        mlflow.log_param("model", name)
        mlflow.log_param("features", ",".join(train.FEATURE_COLUMNS))
        mlflow.log_param("train_rows", train_summary["rows"])
        mlflow.log_param("test_rows", test_summary["rows"])
        mlflow.log_param("train_late_rate", round(train_summary["late_rate"], 4))
        mlflow.log_param("test_late_rate", round(test_summary["late_rate"], 4))
        mlflow.log_param("window_days", round(window_days, 2))
        mlflow.log_metric("pr_auc", result["pr_auc"])
        mlflow.log_metric("roc_auc", result["roc_auc"])
        if result["recall_at_precision_80_oracle"] is not None:
            mlflow.log_metric("recall_at_precision_80_oracle", result["recall_at_precision_80_oracle"])
        if result["train_threshold"] is not None:
            mlflow.log_metric("train_threshold", result["train_threshold"])
        confusion = result["confusion"]
        if confusion is not None:
            for key in ("tn", "fp", "fn", "tp"):
                mlflow.log_metric(f"test_{key}", confusion[key])
            if confusion["precision"] is not None:
                mlflow.log_metric("test_precision_at_train_threshold", confusion["precision"])
            if confusion["recall"] is not None:
                mlflow.log_metric("test_recall_at_train_threshold", confusion["recall"])
        if model is not None:
            # serialization_format="pickle": mlflow's default (skops) refuses
            # to serialize a numpy.dtype it encounters in this pipeline's
            # fitted state (SimpleImputer statistics_ / HistGradientBoosting
            # internals) unless explicitly trusted. Plain pickle is well
            # understood for a local, single-operator tracking store like
            # this one; skops' extra trust-listing isn't buying anything
            # here since nothing ever loads an mlflow-tracked model from an
            # untrusted source in this project.
            mlflow.sklearn.log_model(model, name=name, serialization_format="pickle")


def _print_regime(regime) -> None:
    print("-- data regime --")
    print(
        f"ingestion window: {regime['ingest_start']} to {regime['ingest_end']} "
        f"({regime['ingest_days']:.2f} days), {regime['stop_events_rows']} stop_events rows"
    )
    print(
        f"labels: {regime['labels_total']} closed, {regime['labels_usable']} training-usable, "
        f"{regime['labels_gap_abutted']} gap_abutted, {regime['labels_no_arrival_signal']} no_arrival_signal"
    )
    print(
        f"ingestion uptime: {regime['uptime_pct']:.1f}% "
        f"({regime['poll_cycles']} poll cycles over {regime['ledger_span_hours']:.1f}h ledger span; "
        f"{regime['gap_count']} gaps > {labels.GAP_THRESHOLD_SECONDS:.0f}s totalling "
        f"{regime['gap_hours']:.1f}h, longest {regime['longest_gap_hours']:.2f}h)"
    )
    print()


def _print_split(train_summary, test_summary) -> None:
    print("-- temporal split (by scheduled_arrival, never random) --")
    for label, s in (("train", train_summary), ("test", test_summary)):
        print(
            f"{label:<6} {s['rows']:>7} rows  "
            f"[{s['first_scheduled_arrival']} .. {s['last_scheduled_arrival']}]  "
            f"late {s['late_rows']} ({s['late_rate']:.2%})"
        )
    print()


def _print_results(results, frozen_results, test_summary) -> None:
    print()
    print("-- test-split metrics (threshold chosen on train, applied fixed to test) --")
    header = (
        f"{'candidate':<28} {'pr_auc':>8} {'roc_auc':>8} {'recall@P>=.80':>15} {'thresh':>8} "
        f"{'test_P':>8} {'test_R':>8} {'test_acc':>9} {'tn':>7} {'fp':>7} {'fn':>7} {'tp':>7}"
    )
    print(header)
    for name, result in results.items():
        c = result["confusion"]
        print(
            f"{name:<28} {result['pr_auc']:>8.4f} {result['roc_auc']:>8.4f} "
            f"{_fmt_recall(result['recall_at_precision_80_oracle']):>15} "
            f"{_fmt(result['train_threshold'], '.4f'):>8} "
            f"{_fmt(c['precision'] if c else None):>8} {_fmt(c['recall'] if c else None):>8} "
            f"{_fmt(c['accuracy'] if c else None):>9} "
            f"{(c['tn'] if c else '-'):>7} {(c['fp'] if c else '-'):>7} "
            f"{(c['fn'] if c else '-'):>7} {(c['tp'] if c else '-'):>7}"
        )
    # The do-nothing reference the metrics defense turns on: predict "on time"
    # for every row and never fire an alert at all. It has no threshold, so it
    # has no confusion row above; its accuracy is just the on-time share of
    # the test split.
    print(
        f"reference: predicting on-time for every test row scores "
        f"{1.0 - test_summary['late_rate']:.4f} accuracy and "
        f"{test_summary['late_rate']:.4f} PR-AUC (the positive-class prevalence)"
    )
    print()
    print("-- sensitivity: route-hour aggregate frozen at the train/test boundary --")
    for name, result in frozen_results.items():
        delta = result["pr_auc"] - results[name]["pr_auc"]
        print(
            f"{name + '_frozen':<28} {result['pr_auc']:>8.4f} "
            f"{_fmt_recall(result['recall_at_precision_80_oracle']):>15} "
            f"(pr_auc delta vs per-row as-of: {delta:+.4f})"
        )


def _append_registry_entry(
    *,
    version: str,
    best_name: str,
    regime: dict,
    window_start,
    window_end,
    window_days: float,
    preliminary: bool,
    train_summary: dict,
    test_summary: dict,
    results: dict,
    frozen_results: dict,
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
            "See docs/report.md for what this window can and cannot support.\n\n"
        )
    lines.append(f"- git sha: `{git_sha}` (the code state this run came from, not the commit carrying it)\n")
    lines.append(
        f"- ingestion window: {regime['ingest_start']} to {regime['ingest_end']} "
        f"({regime['ingest_days']:.2f} days), {regime['stop_events_rows']} stop_events rows, "
        f"{regime['uptime_pct']:.1f}% uptime\n"
    )
    lines.append(
        f"- labels: {regime['labels_total']} closed / {regime['labels_usable']} training-usable / "
        f"{regime['labels_gap_abutted']} gap_abutted / {regime['labels_no_arrival_signal']} no_arrival_signal\n"
    )
    lines.append(f"- training window (scheduled_arrival): {window_start} to {window_end} ({window_days:.2f} days)\n")
    lines.append(
        f"- train split: {train_summary['rows']} rows "
        f"[{train_summary['first_scheduled_arrival']} .. {train_summary['last_scheduled_arrival']}], "
        f"late rate {train_summary['late_rate']:.2%}\n"
    )
    lines.append(
        f"- test split: {test_summary['rows']} rows "
        f"[{test_summary['first_scheduled_arrival']} .. {test_summary['last_scheduled_arrival']}], "
        f"late rate {test_summary['late_rate']:.2%}\n"
    )
    lines.append(f"- best real model: `{best_name}`, saved to `{model_path.name}` (gitignored binary)\n")
    lines.append(f"- features: {', '.join(train.FEATURE_COLUMNS)}\n\n")
    lines.append(
        "| candidate | pr_auc | roc_auc | recall@P>=0.80 (oracle) | train threshold | "
        "test precision | test recall | test accuracy | tn | fp | fn | tp |\n"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for name, result in results.items():
        c = result["confusion"]
        lines.append(
            f"| {name} | {result['pr_auc']:.4f} | {result['roc_auc']:.4f} | "
            f"{_fmt_recall(result['recall_at_precision_80_oracle'])} | "
            f"{_fmt(result['train_threshold'], '.4f')} | {_fmt(c['precision'] if c else None)} | "
            f"{_fmt(c['recall'] if c else None)} | {_fmt(c['accuracy'] if c else None)} | "
            f"{c['tn'] if c else '-'} | {c['fp'] if c else '-'} | "
            f"{c['fn'] if c else '-'} | {c['tp'] if c else '-'} |\n"
        )
    lines.append(
        f"\nReference: predicting on-time for every test row scores "
        f"{1.0 - test_summary['late_rate']:.4f} accuracy and {test_summary['late_rate']:.4f} PR-AUC.\n"
    )
    lines.append("\nSensitivity, route-hour aggregate frozen at the train/test boundary:\n\n")
    lines.append("| candidate | pr_auc | pr_auc delta vs per-row as-of | recall@P>=0.80 (oracle) |\n")
    lines.append("|---|---|---|---|\n")
    for name, result in frozen_results.items():
        delta = result["pr_auc"] - results[name]["pr_auc"]
        lines.append(
            f"| {name}_frozen | {result['pr_auc']:.4f} | {delta:+.4f} | "
            f"{_fmt_recall(result['recall_at_precision_80_oracle'])} |\n"
        )
    lines.append("\n")

    with REGISTRY_PATH.open("a") as f:
        f.writelines(lines)


if __name__ == "__main__":
    sys.exit(main())
