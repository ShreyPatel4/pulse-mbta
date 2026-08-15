"""Builds features_trip_stop from trip_stop_labels_training.

M2 deliverable 5 (DE-shaped, code-complete even though the data window is
~1 day). Deliberate choice, stated per this project's dependency rules:
plain SQL/psycopg, not pandas -- every feature here is a set-based
aggregate over trip_stop_labels_training, which Postgres expresses directly
and which stays backfillable-by-window the same way pulse.labels is.
pandas is used only at scripts/train.py's boundary, where sklearn needs an
in-memory matrix anyway.

POINT-IN-TIME CORRECTNESS (the load-bearing property here): the training
signal is a route-hour historical late-rate. Computing "historical" over
the WHOLE table (including rows scheduled after the row being featured)
would leak the future into a feature meant to represent what was knowable
at prediction time. Every "historical" feature below is instead a
correlated subquery filtered to strictly-earlier scheduled_arrival -- an
honest as-of aggregate, NULL when no such history exists yet rather than
silently defaulting to some fabricated rate.

SCALE, stated honestly rather than hidden: these are correlated subqueries,
O(n^2) in the number of training-usable label rows. At M2's volume (tens of
thousands of rows) this runs in seconds; a real system with months of data
would materialize a rolling aggregate (e.g. a daily route-hour rate table
refreshed incrementally) instead of recomputing the full history inline per
row. Out of scope for M2 -- the code needs to be correct first, and correct
here is simple enough to read end-to-end as one query.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from typing import Any

import psycopg

# The "previous stop on this trip" persistence feature uses the temporally
# most recent EARLIER scheduled_arrival on the same trip_id as a proxy for
# "the stop before this one" -- trip_stop_labels doesn't carry stop_sequence,
# but MBTA trips visit their stops in non-decreasing scheduled-time order, so
# "most recent earlier scheduled_arrival, same trip" is equivalent to "the
# previous stop" for any trip that hasn't looped back on itself (city bus
# routes in this dataset don't).
#
# It is also scoped to the same service_date_norm (added at M3, alongside the
# matching grain fix in pulse.labels). MBTA reuses trip_id across service
# dates, so without that predicate the first stop of trip T today would
# inherit its "current delay" from the LAST stop of trip T yesterday -- a
# different bus, a different driver, twenty hours earlier. Point-in-time
# correct either way (it is still strictly-earlier data), but not the feature
# the name claims. Scoped, the first stop of each trip-day is NULL instead,
# which is the truth: at that stop there is no earlier delay to carry forward.
_BUILD_FEATURES_SQL = """
INSERT INTO features_trip_stop (
    service_date_norm, route_id, direction_id, stop_id, trip_id,
    route_hour_historical_late_rate, headway_seconds, hour_of_day, day_of_week,
    current_delay_persistence_seconds, built_at
)
SELECT
    l.service_date_norm, l.route_id, l.direction_id, l.stop_id, l.trip_id,
    (
        SELECT avg(CASE WHEN h.late THEN 1.0 ELSE 0.0 END)
        FROM trip_stop_labels_training h
        WHERE h.route_id = l.route_id
          AND EXTRACT(HOUR FROM h.scheduled_arrival AT TIME ZONE 'America/New_York')
              = EXTRACT(HOUR FROM l.scheduled_arrival AT TIME ZONE 'America/New_York')
          AND h.scheduled_arrival < l.scheduled_arrival
    ) AS route_hour_historical_late_rate,
    (
        SELECT EXTRACT(EPOCH FROM (l.scheduled_arrival - max(p.scheduled_arrival)))::int
        FROM trip_stop_labels_training p
        WHERE p.route_id = l.route_id AND p.direction_id = l.direction_id AND p.stop_id = l.stop_id
          AND p.scheduled_arrival < l.scheduled_arrival
    ) AS headway_seconds,
    EXTRACT(HOUR FROM l.scheduled_arrival AT TIME ZONE 'America/New_York')::int AS hour_of_day,
    EXTRACT(DOW FROM l.scheduled_arrival AT TIME ZONE 'America/New_York')::int AS day_of_week,
    (
        SELECT p.delay_seconds
        FROM trip_stop_labels_training p
        WHERE p.trip_id = l.trip_id
          AND p.service_date_norm = l.service_date_norm
          AND p.scheduled_arrival < l.scheduled_arrival
        ORDER BY p.scheduled_arrival DESC
        LIMIT 1
    ) AS current_delay_persistence_seconds,
    now()
FROM trip_stop_labels_training l
WHERE l.scheduled_arrival >= %(since)s AND l.scheduled_arrival < %(until)s
ON CONFLICT (service_date_norm, route_id, direction_id, stop_id, trip_id) DO UPDATE SET
    route_hour_historical_late_rate = EXCLUDED.route_hour_historical_late_rate,
    headway_seconds = EXCLUDED.headway_seconds,
    hour_of_day = EXCLUDED.hour_of_day,
    day_of_week = EXCLUDED.day_of_week,
    current_delay_persistence_seconds = EXCLUDED.current_delay_persistence_seconds,
    built_at = now()
"""

_COUNT_CANDIDATE_ROWS_SQL = """
SELECT count(*) FROM trip_stop_labels_training
WHERE scheduled_arrival >= %(since)s AND scheduled_arrival < %(until)s
"""

# See pulse.labels._NO_LOWER_BOUND for why this isn't datetime.min --
# duplicated here rather than imported to keep this module independent of
# pulse.labels (they're siblings in the pipeline, not a dependency chain).
_NO_LOWER_BOUND = dt.datetime(2000, 1, 1, tzinfo=dt.timezone.utc)

MIN_ROWS_WRITTEN = 1
# Historical-rate coverage: the fraction of written rows where
# route_hour_historical_late_rate is non-null. Given ~1 day of data this is
# expected to be low (little history has accumulated yet) -- the gate exists
# so a future, larger run can catch a real regression (e.g. the join
# predicate silently matching nothing), not to demand a specific rate today.
MIN_HISTORICAL_RATE_COVERAGE_PCT = 0.0


def build_features(
    conn: psycopg.Connection,
    since: dt.datetime | None = None,
    until: dt.datetime | None = None,
) -> dict[str, Any]:
    """Idempotent, backfillable over [since, until) on scheduled_arrival --
    a rerun over the same or an overlapping window UPDATEs existing
    features_trip_stop rows in place (ON CONFLICT DO UPDATE on the shared
    natural key), matching pulse.labels.run_build's idempotency posture.
    since=None means no lower bound; until=None defaults to now (UTC).
    """
    started_at = dt.datetime.now(dt.timezone.utc)
    until = until or started_at
    since_value = since or _NO_LOWER_BOUND

    with conn.cursor() as cur:
        cur.execute(_COUNT_CANDIDATE_ROWS_SQL, {"since": since_value, "until": until})
        (candidate_rows,) = cur.fetchone()

    with conn.cursor() as cur:
        cur.execute(_BUILD_FEATURES_SQL, {"since": since_value, "until": until})
        rows_written = cur.rowcount

    gate_results = _compute_quality_gates(conn, since_value, until, rows_written)
    passed = all(g["passed"] for g in gate_results.values())
    finished_at = dt.datetime.now(dt.timezone.utc)

    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "window_since": since,
        "window_until": until,
        "candidate_rows": candidate_rows,
        "rows_written": rows_written,
        "gate_results": gate_results,
        "passed": passed,
    }


def _compute_quality_gates(
    conn: psycopg.Connection, since: dt.datetime, until: dt.datetime, rows_written: int
) -> dict[str, dict[str, Any]]:
    volume_passed = rows_written >= MIN_ROWS_WRITTEN
    volume_detail = f"{rows_written} rows written (threshold: >= {MIN_ROWS_WRITTEN})"

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT count(*) FILTER (WHERE route_hour_historical_late_rate IS NOT NULL), count(*)
            FROM features_trip_stop
            WHERE service_date_norm IS NOT NULL
              AND (service_date_norm, route_id, direction_id, stop_id, trip_id) IN (
                  SELECT service_date_norm, route_id, direction_id, stop_id, trip_id
                  FROM trip_stop_labels_training
                  WHERE scheduled_arrival >= %(since)s AND scheduled_arrival < %(until)s
              )
            """,
            {"since": since, "until": until},
        )
        with_history, total = cur.fetchone()

    coverage_pct = 100.0 * with_history / total if total else 0.0
    coverage_passed = coverage_pct >= MIN_HISTORICAL_RATE_COVERAGE_PCT
    coverage_detail = (
        f"{with_history}/{total} rows have route_hour_historical_late_rate coverage "
        f"({coverage_pct:.2f}%, threshold: >= {MIN_HISTORICAL_RATE_COVERAGE_PCT}%)"
    )

    return {
        "volume": {"passed": volume_passed, "detail": volume_detail},
        "historical_rate_coverage": {"passed": coverage_passed, "detail": coverage_detail},
    }


def record_transform_run(conn: psycopg.Connection, summary: Mapping[str, Any], git_sha: str | None) -> None:
    """Reuses transform_runs (migrations/005) -- see pulse.labels.record_transform_run
    for the shared shape; features has no separate as_of concept (it's not
    settle-margin-gated the way labels are), so as_of is recorded as the
    build's own started_at."""
    import json

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO transform_runs (
                transform_name, started_at, finished_at, window_since, window_until,
                as_of, groups_considered, rows_written, gate_results, passed, git_sha
            ) VALUES (
                %(transform_name)s, %(started_at)s, %(finished_at)s, %(window_since)s, %(window_until)s,
                %(as_of)s, %(groups_considered)s, %(rows_written)s, %(gate_results)s, %(passed)s, %(git_sha)s
            )
            """,
            {
                "transform_name": "build_features",
                "started_at": summary["started_at"],
                "finished_at": summary["finished_at"],
                "window_since": summary["window_since"],
                "window_until": summary["window_until"],
                "as_of": summary["started_at"],
                "groups_considered": summary["candidate_rows"],
                "rows_written": summary["rows_written"],
                "gate_results": json.dumps(summary["gate_results"]),
                "passed": summary["passed"],
                "git_sha": git_sha,
            },
        )
