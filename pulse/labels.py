"""Derives trip_stop_labels rows from stop_events + poll_runs.

M2 must-address items 1, 5, 6 -- the core of M2. A trip-stop (one
(trip_id, stop_id) pair) is labeled once it has SETTLED: its last-seen
snapshot is far enough in the past (SETTLE_MARGIN_SECONDS -- roughly 3 poll
cycles) that a prediction which hasn't reappeared is treated as having
disappeared from the feed, not merely "still in flight" between two
snapshots that happen to straddle this build's cutoff.

CLOSING RULE (item 1, the carried M1 obligation): a settled trip-stop closes
normally -- a real label, usable for training -- UNLESS a recorded polling
gap (a poll_runs row with error set, or a stretch with no poll_runs row at
all -- see compute_gap_intervals) ABUTS its disappearance: overlaps the
window [last_seen, last_seen + SETTLE_MARGIN_SECONDS]. A gap there means a
missing prediction is indistinguishable from a missing observation -- the
trip-stop might still have been actively predicted during the gap and we
simply didn't see it. Those rows get closed_reason='gap_abutted' and are
excluded from trip_stop_labels_training (migrations/005), not silently
treated as a real disappearance.

ORIGIN-STOP FILTERING (item 6): MBTA never gives an arrival_time at a
trip's first stop (departure_time only -- see pulse/mbta.py's map_rows
docstring), so scheduled_arrival and/or final_predicted_arrival end up null
for those trip-stops. closed_reason='no_arrival_signal' whenever EITHER is
null (not just "both", though the M1 review's cross-tab found zero
mixed-null rows in the actual data -- this is a defensive guard against a
case the observed data doesn't currently exercise, e.g. an ADDED trip with a
live prediction but no GTFS schedule counterpart, documented in
migrations/005's header). Never imputed as on-time.

SERVICE_DATE_NORM (item 5): the GTFS service day runs to ~3 AM, so a raw
polled_at calendar date splits late-night trips at midnight. service_date is
normalized to America/New_York local time with hours < 3 belonging to the
PRIOR service date. RERUN-STABILITY (required for scripts/build-labels.py's
idempotent upsert, keyed on service_date_norm among other columns): this
anchors on scheduled_arrival when present -- a GTFS-schedule-derived value,
unaffected by which window a given build run happens to scan -- and only
falls back to the trip-stop's observed_span_start (its first sighting,
computed over stop_events' FULL history for that trip-stop, never truncated
to the current build's window -- see scripts/build-labels.py's aggregation
query) when scheduled_arrival is null. A version of this that anchored on
"first sighting within the current window" would silently duplicate rows
under a different service_date_norm on a rerun over a different, overlapping
window -- exactly the bug idempotency is supposed to prevent.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import psycopg

_EASTERN = ZoneInfo("America/New_York")

# Measured cadence (README): launchd's StartInterval is 60s but batch-fetch
# work pushes the real cadence to ~66s.
EXPECTED_CYCLE_SECONDS = 66.0

# A gap between two consecutive poll_runs rows wider than this means at
# least one cycle's worth of time passed with no recorded cycle at all.
# Comfortably above a single jittery cycle (~66-90s observed in practice --
# a slow batch, one retried request) so that never false-positives as a
# missing cycle, but comfortably below the ~132s two consecutive rows would
# be apart if exactly one cycle in between never got recorded, so an actual
# single missed cycle is reliably caught.
GAP_THRESHOLD_SECONDS = 100.0

# How many consecutive missing cycles' worth of quiet time must pass after a
# trip-stop's last sighting before its disappearance is trusted enough to
# close the label. 3 cycles (~200s) rather than 1: a single missed poll is
# unremarkable at ~66s cadence and shouldn't force a premature close.
SETTLE_MARGIN_CYCLES = 3
SETTLE_MARGIN_SECONDS = SETTLE_MARGIN_CYCLES * EXPECTED_CYCLE_SECONDS

# Task A's classification threshold (docs/2026-08-13-pulse-design.md):
# P(arrival delay > 180s).
LATE_THRESHOLD_SECONDS = 180

# Sentinel for an unbounded "since" (all of history is eligible). NOT
# datetime.min: Postgres/psycopg round-trips a year-1 timestamptz through
# America/New_York's pre-1883 Local Mean Time zone rule (a real, if obscure,
# tzdata behavior), producing a value psycopg's loader rejects outright
# ("timestamp too small") if it's ever read back, and the raw offset is
# genuinely confusing to log/debug. A boring, safely-after-year-1 epoch
# avoids all of that while still comfortably predating this project's first
# real data (2026-08-12).
_NO_LOWER_BOUND = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)


def service_date_norm(instant: dt.datetime) -> dt.date:
    """GTFS 3AM rule. `instant` must be timezone-aware (any zone -- this
    converts to America/New_York before applying the rule); local hours < 3
    belong to the PRIOR calendar date."""
    local = instant.astimezone(_EASTERN)
    if local.hour < 3:
        return (local - dt.timedelta(days=1)).date()
    return local.date()


def compute_gap_intervals(
    poll_runs_rows: Sequence[Mapping[str, Any]],
    gap_threshold_seconds: float = GAP_THRESHOLD_SECONDS,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Derive (start, end) intervals of "the feed's state during this time is
    not trustworthy" from a list of poll_runs rows (each a mapping with at
    least `polled_at` and `error`; order doesn't matter, this sorts).

    Two sources, matching the ledger's documented meaning (README's Ops
    section): an errored cycle (poll_runs.error is not None) contributes a
    zero-width point interval at its own polled_at -- it ran, but the cycle
    is known-bad. A gap between two consecutive rows wider than
    gap_threshold_seconds means at least one cycle never got recorded at all
    (poller down, crashed, or the DB write itself failed) -- that whole
    span between the two known-good neighbors is untrustworthy.

    Deliberate simplification, stated rather than left implicit: any row
    with error set is treated as a gap for the WHOLE cycle (all routes),
    even though batches_ok < batches_total means some batches within that
    cycle may have actually succeeded. Distinguishing per-route gaps would
    need per-route ledger granularity poll_runs doesn't have; treating the
    whole cycle as suspect is the conservative (safe) direction to be wrong
    in for a gap-exclusion rule.
    """
    rows = sorted(poll_runs_rows, key=lambda r: r["polled_at"])
    intervals: list[tuple[dt.datetime, dt.datetime]] = []

    for row in rows:
        if row.get("error") is not None:
            intervals.append((row["polled_at"], row["polled_at"]))

    for prev, cur in zip(rows, rows[1:]):
        elapsed = (cur["polled_at"] - prev["polled_at"]).total_seconds()
        if elapsed > gap_threshold_seconds:
            intervals.append((prev["polled_at"], cur["polled_at"]))

    return intervals


def gap_abuts(
    observed_span_end: dt.datetime,
    gap_intervals: Sequence[tuple[dt.datetime, dt.datetime]],
    settle_margin_seconds: float = SETTLE_MARGIN_SECONDS,
) -> bool:
    """True if any gap interval overlaps the trailing window
    [observed_span_end, observed_span_end + settle_margin_seconds] -- the
    exact window a trip-stop's disappearance is judged against. A gap
    starting before observed_span_end but still open at it also counts
    (the interval overlap test below is symmetric)."""
    window_start = observed_span_end
    window_end = observed_span_end + dt.timedelta(seconds=settle_margin_seconds)
    for gap_start, gap_end in gap_intervals:
        if gap_start <= window_end and gap_end >= window_start:
            return True
    return False


def is_settled(
    observed_span_end: dt.datetime,
    build_as_of: dt.datetime,
    settle_margin_seconds: float = SETTLE_MARGIN_SECONDS,
) -> bool:
    """True once enough quiet time has passed after a trip-stop's last
    sighting that its disappearance can be evaluated at all -- a trip-stop
    seen in the last settle_margin_seconds is still plausibly "in flight"
    (about to show up in the very next cycle) and shouldn't be judged yet."""
    return (build_as_of - observed_span_end).total_seconds() >= settle_margin_seconds


def derive_label_row(
    group: Mapping[str, Any],
    gap_intervals: Sequence[tuple[dt.datetime, dt.datetime]],
    build_as_of: dt.datetime,
    settle_margin_seconds: float = SETTLE_MARGIN_SECONDS,
) -> dict[str, Any] | None:
    """Derive one trip_stop_labels row from one trip-stop's full-history
    aggregate (see scripts/build-labels.py's query for exactly what `group`
    must contain: route_id, direction_id, stop_id, trip_id, scheduled_arrival,
    final_predicted_arrival, observed_span_start, observed_span_end,
    n_snapshots -- all computed over the trip-stop's ENTIRE stop_events
    history, not the current build window, for rerun-stability).

    Returns None when the trip-stop isn't settled yet (still open -- not
    written to trip_stop_labels this run; a later, later-triggered build
    will pick it up once it settles). Otherwise returns a dict shaped for
    trip_stop_labels' columns, with delay_seconds/late populated whenever
    both arrival times are present (even for a gap_abutted row -- excluded
    from training via the view, not blanked out here) and closed_reason set
    per the module docstring's precedence: no_arrival_signal first (a
    structural data fact, independent of timing), then gap_abutted, else
    None (normal, training-usable close).
    """
    observed_span_end = group["observed_span_end"]
    if not is_settled(observed_span_end, build_as_of, settle_margin_seconds):
        return None

    scheduled_arrival = group["scheduled_arrival"]
    final_predicted_arrival = group["final_predicted_arrival"]

    anchor = scheduled_arrival if scheduled_arrival is not None else group["observed_span_start"]
    norm_date = service_date_norm(anchor)

    row: dict[str, Any] = {
        "service_date_norm": norm_date,
        "route_id": group["route_id"],
        "direction_id": group["direction_id"],
        "stop_id": group["stop_id"],
        "trip_id": group["trip_id"],
        "scheduled_arrival": scheduled_arrival,
        "final_predicted_arrival": final_predicted_arrival,
        "observed_span_start": group["observed_span_start"],
        "observed_span_end": observed_span_end,
        "n_snapshots": group["n_snapshots"],
        "delay_seconds": None,
        "late": None,
        "closed_reason": None,
    }

    if scheduled_arrival is None or final_predicted_arrival is None:
        row["closed_reason"] = "no_arrival_signal"
        return row

    delay_seconds = round((final_predicted_arrival - scheduled_arrival).total_seconds())
    row["delay_seconds"] = delay_seconds
    row["late"] = delay_seconds > LATE_THRESHOLD_SECONDS

    if gap_abuts(observed_span_end, gap_intervals, settle_margin_seconds):
        row["closed_reason"] = "gap_abutted"

    return row


# -- DB orchestration (scripts/build-labels.py's CLI wraps run_build) ------

# Two-step query, deliberately: step 1 (the `touched` CTE) finds which
# trip-stops to CONSIDER this run (any snapshot in [since, until)); step 2
# aggregates each touched trip-stop's ENTIRE stop_events history, not just
# the window slice. This is what makes derive_label_row's output
# rerun-stable across two different, overlapping backfill windows (see the
# module docstring's SERVICE_DATE_NORM section) -- a trip-stop reprocessed
# from two different windows gets byte-identical aggregates both times,
# because the aggregation itself never depends on the window boundaries,
# only on which trip-stops get selected for consideration.
_TOUCHED_GROUPS_SQL = """
WITH touched AS (
    SELECT DISTINCT trip_id, stop_id
    FROM stop_events
    WHERE polled_at >= %(since)s AND polled_at < %(until)s
)
SELECT
    se.trip_id,
    se.stop_id,
    (array_agg(se.route_id ORDER BY se.polled_at DESC))[1] AS route_id,
    (array_agg(se.direction_id ORDER BY se.polled_at DESC))[1] AS direction_id,
    (array_agg(se.scheduled_arrival ORDER BY se.polled_at DESC)
        FILTER (WHERE se.scheduled_arrival IS NOT NULL))[1] AS scheduled_arrival,
    (array_agg(se.predicted_arrival ORDER BY se.polled_at DESC)
        FILTER (WHERE se.predicted_arrival IS NOT NULL))[1] AS final_predicted_arrival,
    min(se.polled_at) AS observed_span_start,
    max(se.polled_at) AS observed_span_end,
    count(*) AS n_snapshots
FROM stop_events se
JOIN touched t ON t.trip_id = se.trip_id AND t.stop_id = se.stop_id
GROUP BY se.trip_id, se.stop_id
"""

_POLL_RUNS_SQL = "SELECT polled_at, error FROM poll_runs ORDER BY polled_at"

_MAX_POLL_RUNS_POLLED_AT_SQL = "SELECT max(polled_at) FROM poll_runs"

_UPSERT_LABEL_SQL = """
INSERT INTO trip_stop_labels (
    service_date_norm, route_id, direction_id, stop_id, trip_id,
    scheduled_arrival, final_predicted_arrival, delay_seconds, late,
    observed_span_start, observed_span_end, n_snapshots, closed_reason, built_at
) VALUES (
    %(service_date_norm)s, %(route_id)s, %(direction_id)s, %(stop_id)s, %(trip_id)s,
    %(scheduled_arrival)s, %(final_predicted_arrival)s, %(delay_seconds)s, %(late)s,
    %(observed_span_start)s, %(observed_span_end)s, %(n_snapshots)s, %(closed_reason)s, now()
)
ON CONFLICT (service_date_norm, route_id, direction_id, stop_id, trip_id) DO UPDATE SET
    scheduled_arrival = EXCLUDED.scheduled_arrival,
    final_predicted_arrival = EXCLUDED.final_predicted_arrival,
    delay_seconds = EXCLUDED.delay_seconds,
    late = EXCLUDED.late,
    observed_span_start = EXCLUDED.observed_span_start,
    observed_span_end = EXCLUDED.observed_span_end,
    n_snapshots = EXCLUDED.n_snapshots,
    closed_reason = EXCLUDED.closed_reason,
    built_at = now()
"""

_INSERT_TRANSFORM_RUN_SQL = """
INSERT INTO transform_runs (
    transform_name, started_at, finished_at, window_since, window_until,
    as_of, groups_considered, rows_written, gate_results, passed, git_sha
) VALUES (
    %(transform_name)s, %(started_at)s, %(finished_at)s, %(window_since)s, %(window_until)s,
    %(as_of)s, %(groups_considered)s, %(rows_written)s, %(gate_results)s, %(passed)s, %(git_sha)s
)
"""

# Quality gate thresholds. Named and centralized here (not buried in
# run_build) so scripts/build-labels.py's --help / this module's docstring
# is the one place a reviewer needs to look to know what "PASS" means.
MIN_ROWS_WRITTEN = 1
MAX_NULL_DELAY_RATE_PCT = 0.0  # among closed_reason IS NULL rows -- should never happen by construction
MIN_TRAINING_LABEL_RATE_PCT = 50.0  # rows_written that are training-usable (closed_reason IS NULL)


def fetch_touched_groups(
    conn: psycopg.Connection, since: dt.datetime | None, until: dt.datetime
) -> list[dict[str, Any]]:
    """Every trip-stop with at least one stop_events row in [since, until),
    each aggregated over its FULL stop_events history (see _TOUCHED_GROUPS_SQL's
    comment). since=None means no lower bound (all of history is eligible to
    be "touched")."""
    since_value = since or _NO_LOWER_BOUND
    with conn.cursor() as cur:
        cur.execute(_TOUCHED_GROUPS_SQL, {"since": since_value, "until": until})
        columns = [c.name for c in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def fetch_poll_runs(conn: psycopg.Connection) -> list[dict[str, Any]]:
    with conn.cursor() as cur:
        cur.execute(_POLL_RUNS_SQL)
        return [{"polled_at": polled_at, "error": error} for polled_at, error in cur.fetchall()]


def fetch_max_poll_runs_polled_at(conn: psycopg.Connection) -> dt.datetime | None:
    with conn.cursor() as cur:
        cur.execute(_MAX_POLL_RUNS_POLLED_AT_SQL)
        (value,) = cur.fetchone()
        return value


def upsert_label_rows(conn: psycopg.Connection, rows: Sequence[Mapping[str, Any]]) -> int:
    """Idempotent upsert -- a rerun over the same or an overlapping window
    updates existing rows in place rather than duplicating them (the natural
    key is trip_stop_labels' primary key)."""
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_LABEL_SQL, rows)
    return len(rows)


def compute_quality_gates(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Volume / null-rate / label-rate gates over this run's written rows
    (not the whole table -- a backfill of one quiet hour shouldn't fail
    because the table's all-time label-rate happens to differ). Returns
    {gate_name: {"passed": bool, "detail": str}}."""
    total = len(rows)
    normal = [r for r in rows if r["closed_reason"] is None]
    null_delay_among_normal = sum(1 for r in normal if r["delay_seconds"] is None)

    volume_passed = total >= MIN_ROWS_WRITTEN
    volume_detail = f"{total} rows written (threshold: >= {MIN_ROWS_WRITTEN})"

    if normal:
        null_rate_pct = 100.0 * null_delay_among_normal / len(normal)
    else:
        null_rate_pct = 0.0
    null_rate_passed = null_rate_pct <= MAX_NULL_DELAY_RATE_PCT
    null_rate_detail = (
        f"{null_delay_among_normal}/{len(normal)} normal-closed rows have a null delay_seconds "
        f"({null_rate_pct:.2f}%, threshold: <= {MAX_NULL_DELAY_RATE_PCT}%)"
    )

    if total:
        label_rate_pct = 100.0 * len(normal) / total
    else:
        label_rate_pct = 0.0
    label_rate_passed = label_rate_pct >= MIN_TRAINING_LABEL_RATE_PCT
    label_rate_detail = (
        f"{len(normal)}/{total} rows are training-usable ({label_rate_pct:.2f}%, "
        f"threshold: >= {MIN_TRAINING_LABEL_RATE_PCT}%)"
    )

    return {
        "volume": {"passed": volume_passed, "detail": volume_detail},
        "null_rate": {"passed": null_rate_passed, "detail": null_rate_detail},
        "label_rate": {"passed": label_rate_passed, "detail": label_rate_detail},
    }


def run_build(
    conn: psycopg.Connection,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
    as_of: dt.datetime | None = None,
    settle_margin_seconds: float = SETTLE_MARGIN_SECONDS,
    gap_threshold_seconds: float = GAP_THRESHOLD_SECONDS,
) -> dict[str, Any]:
    """Run one build: find touched trip-stops in [since, until), derive a
    label for each settled one, upsert, run quality gates, and return a
    summary dict (also what gets recorded into transform_runs -- the caller,
    scripts/build-labels.py, does that insert plus stdout PASS/FAIL printing
    and the non-zero exit on breach).

    until defaults to now (UTC); as_of defaults to the latest poll_runs row
    (see pulse.labels' module docstring -- either choice is safe because the
    gap-abut check is what actually protects a premature close, not the
    as_of source, but anchoring on the data's own latest confirmed cycle
    rather than wall-clock time makes backfills reproducible without needing
    to mock the clock).
    """
    started_at = dt.datetime.now(dt.timezone.utc)
    until = until or started_at
    if as_of is None:
        as_of = fetch_max_poll_runs_polled_at(conn) or started_at

    groups = fetch_touched_groups(conn, since, until)
    poll_runs_rows = fetch_poll_runs(conn)
    gap_intervals = compute_gap_intervals(poll_runs_rows, gap_threshold_seconds)

    rows = []
    for group in groups:
        row = derive_label_row(group, gap_intervals, as_of, settle_margin_seconds)
        if row is not None:
            rows.append(row)

    rows_written = upsert_label_rows(conn, rows)
    gate_results = compute_quality_gates(rows)
    passed = all(g["passed"] for g in gate_results.values())
    finished_at = dt.datetime.now(dt.timezone.utc)

    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "window_since": since,
        "window_until": until,
        "as_of": as_of,
        "groups_considered": len(groups),
        "rows_written": rows_written,
        "gate_results": gate_results,
        "passed": passed,
        "rows": rows,
    }


def record_transform_run(conn: psycopg.Connection, summary: Mapping[str, Any], git_sha: str | None) -> None:
    """Append one transform_runs row from run_build's summary dict."""
    import json

    with conn.cursor() as cur:
        cur.execute(
            _INSERT_TRANSFORM_RUN_SQL,
            {
                "transform_name": "build_labels",
                "started_at": summary["started_at"],
                "finished_at": summary["finished_at"],
                "window_since": summary["window_since"],
                "window_until": summary["window_until"],
                "as_of": summary["as_of"],
                "groups_considered": summary["groups_considered"],
                "rows_written": summary["rows_written"],
                "gate_results": json.dumps(summary["gate_results"]),
                "passed": summary["passed"],
                "git_sha": git_sha,
            },
        )
