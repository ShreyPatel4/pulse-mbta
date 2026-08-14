"""Push a public-status snapshot + the derived gap ledger to Cloudflare D1.

Usage: uv run python scripts/publish-status.py

Pass 1.1 of the public uptime clock. ARCHITECTURAL RULE: this Mac never
serves a visitor. This script only PUSHES rows outward, every 5 minutes
(org.coconutlabs.pulse-publish, StartInterval 300), to the D1 database
`pulse-status` (969a7193-e4fd-4acd-85cc-04536f29790d). Public surfaces read
D1 only.

Failure handling is deliberate: if Postgres is unreachable (laptop asleep,
DSN bad, whatever), this script logs to stderr and exits 1 WITHOUT touching
D1 at all -- no partial write, no synthesized "error" snapshot. D1 keeps
whatever it last held, so a public page reading it degrades to a
stale-but-honest last-publish timestamp, never a connection error. That is
the whole point of the architecture; do not "fix" this by writing an error
row on failure.

Two tables, both written every run (see d1/schema.sql for the contract):

  snapshots -- one row per publish cycle. published_at is truncated to the
    minute (not full precision) specifically so a re-run within the same
    minute (e.g. a manual re-run right after a cron fire) REPLACEs instead
    of duplicating -- that's the idempotency the PK is for. Every other
    TEXT timestamp column keeps full microsecond precision so distinct
    events don't collide on their PK.

  gaps -- the gap ledger, derived fresh from the FULL poll_runs history on
    every run and upserted (INSERT OR REPLACE on started_at). Recomputing
    the whole thing each time rather than tracking "new gaps since last
    run" is deliberately simple and self-healing: the first-ever run of
    this script IS the one-time backfill the build asked for, and every
    run after that just re-affirms unchanged history plus whatever's new.
    Cheap at today's ~1,300 poll_runs rows/day; if poll_runs grows into the
    millions this may need a lookback window -- not needed yet.

  A gap is either:
    (a) the span between two consecutive poll_runs rows exceeding 3x the
        median cadence (computed over the same run's full history) -- the
        "no poll recorded" case, unless the row that closes the gap also
        carries a non-null error, in which case that error text IS the
        reason (more specific than the generic label).
    (b) a poll_runs row with a non-null error that does NOT also close a
        timing gap -- a cycle that ran on schedule but degraded. These
        never double up with (a) for the same row: if a row satisfies both,
        it's emitted once, as a timing gap, per the merge rule above.

  `reason` is never the raw error text -- this ledger renders on a public
  hiring surface, and a raw Python traceback there reads as sloppy, not
  transparent. _classify_reason() reduces the raw poll_runs.error string to
  one of a small set of short human sentences via plain string matching (no
  cleverness), optionally suffixed with the affected scope (e.g. "(2 of 3
  batches)") when poll_runs.batches_ok/batches_total say so. The original
  raw text always survives, untouched, in gaps.detail -- one query away.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from typing import Any

from pulse import db

# -- Cloudflare -------------------------------------------------------------

WRANGLER_BIN = "/opt/homebrew/bin/npx"
D1_DATABASE = "pulse-status"
WRANGLER_TIMEOUT_SECONDS = 60
MAX_SNAPSHOTS = 2000

# -- Gap derivation -----------------------------------------------------------

GAP_THRESHOLD_MULTIPLIER = 3.0
REASON_MAX_CHARS = 200
DETAIL_MAX_CHARS = 2000

# Plain substring matches against poll_runs.error.lower(), checked in order
# -- first match wins. Deliberately mechanical: no heuristics, no per-case
# regex beyond the http-5xx digit check. Anything that matches nothing falls
# through to "poll failed", with the raw text preserved in detail.
_UNREACHABLE_MARKERS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "failed to establish a new connection",
    "max retries exceeded",
    "name or service not known",
    "nodename nor servname",
    "getaddrinfo failed",
    "timed out",
    "read timed out",
)
_RATE_LIMIT_MARKERS = ("429", "too many requests")
_WAREHOUSE_MARKERS = (
    "db.connect failed",
    "connection to server at",
    "could not receive data from server",
    "psycopg",
)
_HTTP_5XX_RE = re.compile(r"\b5\d{2}\b")
_SCOPED_CATEGORIES = {
    "upstream unreachable",
    "rate limited by the upstream API",
    "upstream server error",
    "poll failed",
}


def _classify_reason(raw: str, batches_ok: int | None, batches_total: int | None) -> str:
    """Raw poll_runs.error text -> a short, classified, public-facing
    sentence. String matching only -- see the module docstring."""
    lower = raw.lower()

    if any(marker in lower for marker in _UNREACHABLE_MARKERS):
        category = "upstream unreachable"
    elif any(marker in lower for marker in _RATE_LIMIT_MARKERS):
        category = "rate limited by the upstream API"
    elif _HTTP_5XX_RE.search(raw) and any(w in lower for w in ("http", "status", "predictions", "mbta")):
        category = "upstream server error"
    elif any(marker in lower for marker in _WAREHOUSE_MARKERS):
        category = "warehouse unreachable"
    else:
        category = "poll failed"

    if (
        category in _SCOPED_CATEGORIES
        and batches_total
        and batches_ok is not None
        and batches_ok < batches_total
    ):
        failed = batches_total - batches_ok
        category = f"{category} ({failed} of {batches_total} batches)"

    return category

# -- M1 quality gates, kept in sync with scripts/check.py --------------------
# (duplicated rather than imported: scripts/ isn't a package, and these four
# thresholds are simple enough that duplication is less fragile than an
# importlib.util.spec_from_file_location load of a sibling script.)

DISK_FREE_MIN_BYTES = 20 * 1024**3
FRESHNESS_MAX_MINUTES = 10
NULL_RATE_MAX_PCT = 15.0
_DISK_PATHS = ("/System/Volumes/Data", "/")

_CADENCE_QUERY = """
SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM gap)) AS median_seconds
FROM (
    SELECT polled_at - LAG(polled_at) OVER (ORDER BY polled_at) AS gap
    FROM poll_runs
) d
WHERE gap IS NOT NULL
"""

_RUNS_QUERY = "SELECT min(polled_at) AS first_cycle_at, count(*) AS cycles FROM poll_runs"

_LAST_HOUR_QUERY = """
SELECT COALESCE(sum(inserted), 0)
FROM poll_runs
WHERE polled_at >= now() - interval '1 hour'
"""

_ROWS_TOTAL_QUERY = "SELECT count(*) FROM stop_events"

_DISTINCT_QUERY = """
SELECT count(DISTINCT route_id), count(DISTINCT trip_id), count(DISTINCT stop_id)
FROM stop_events
"""

_LABELS_QUERY = """
SELECT
    count(*) AS labels_total,
    count(*) FILTER (WHERE closed_reason = 'gap_abutted') AS labels_gap_abutted,
    count(*) FILTER (WHERE closed_reason = 'no_arrival_signal') AS labels_no_arrival
FROM trip_stop_labels
"""

_FRESHNESS_QUERY = "SELECT max(polled_at) FROM poll_runs"

_NULL_RATE_QUERY = """
SELECT
    count(*) FILTER (WHERE scheduled_arrival IS NULL) AS nulls,
    count(*) AS total
FROM stop_events
WHERE polled_at >= now() - interval '1 hour'
"""

_GAP_SOURCE_QUERY = """
SELECT polled_at, started_at, finished_at, error, batches_ok, batches_total
FROM poll_runs
ORDER BY polled_at
"""


# -- formatting / escaping ---------------------------------------------------


def _iso(value: dt.datetime) -> str:
    """Fixed-width UTC ISO8601 with a literal Z, full microsecond precision.

    Postgres hands back tz-aware datetimes with a local (-04:00 / -05:00)
    offset; normalizing every TEXT timestamp column to this one form is
    what lets both this script's own pruning (ORDER BY published_at) and a
    future Worker's "how stale is this" math rely on plain lexicographic
    string order matching chronological order.
    """
    return value.astimezone(dt.timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sql_str(value: str | None) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def _sql_num(value: float | int | None) -> str:
    if value is None:
        return "NULL"
    return repr(value)


def _truncate(text: str, limit: int = REASON_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 15] + "... [truncated]"


def _disk_free_bytes() -> tuple[int, str]:
    last_exc: OSError | None = None
    for path in _DISK_PATHS:
        try:
            usage = shutil.disk_usage(path)
            return usage.free, path
        except OSError as exc:
            last_exc = exc
            continue
    assert last_exc is not None
    raise last_exc


# -- snapshot -----------------------------------------------------------------


def compute_snapshot(conn, now: dt.datetime) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute(_CADENCE_QUERY)
        (median_cadence,) = cur.fetchone()

        cur.execute(_RUNS_QUERY)
        first_cycle_at, cycles = cur.fetchone()

        cur.execute(_LAST_HOUR_QUERY)
        (rows_last_hour,) = cur.fetchone()

        cur.execute(_ROWS_TOTAL_QUERY)
        (rows_total,) = cur.fetchone()

        distinct_start = time.monotonic()
        cur.execute(_DISTINCT_QUERY)
        distinct_routes, distinct_trips, distinct_stops = cur.fetchone()
        distinct_elapsed = time.monotonic() - distinct_start
        print(f"pulse.publish: distinct-count query took {distinct_elapsed:.2f}s", file=sys.stderr)

        cur.execute(_LABELS_QUERY)
        labels_total, labels_gap_abutted, labels_no_arrival = cur.fetchone()

        cur.execute(_FRESHNESS_QUERY)
        (last_run_at,) = cur.fetchone()

        cur.execute(_NULL_RATE_QUERY)
        null_nulls, null_total = cur.fetchone()

    free_bytes, disk_path = _disk_free_bytes()
    free_gib = free_bytes / 1024**3
    disk_pass = free_bytes >= DISK_FREE_MIN_BYTES

    if last_run_at is None:
        freshness_pass = False
        freshness_detail = "no poll_runs rows at all"
    else:
        age_minutes = (now - last_run_at).total_seconds() / 60
        freshness_pass = age_minutes <= FRESHNESS_MAX_MINUTES
        freshness_detail = f"last poll_runs row {age_minutes:.1f} min ago"

    volume_pass = rows_last_hour > 0
    volume_detail = f"{rows_last_hour} rows inserted (poll_runs) in the last hour"

    if null_total == 0:
        null_pass = False
        null_detail = "no stop_events rows in the last hour to evaluate"
    else:
        null_pct = 100.0 * null_nulls / null_total
        null_pass = null_pct <= NULL_RATE_MAX_PCT
        null_detail = f"scheduled_arrival null-rate {null_pct:.2f}% over last hour ({null_nulls}/{null_total})"

    gate_detail = {
        "disk": {"passed": disk_pass, "detail": f"{free_gib:.1f} GiB free on {disk_path}"},
        "freshness": {"passed": freshness_pass, "detail": freshness_detail},
        "volume": {"passed": volume_pass, "detail": volume_detail},
        "null_rate": {"passed": null_pass, "detail": null_detail},
    }
    gates_pass = sum(1 for g in gate_detail.values() if g["passed"])

    published_at = now.replace(second=0, microsecond=0)
    days_running = (now - first_cycle_at).total_seconds() / 86400.0 if first_cycle_at else None

    return {
        "published_at": _iso(published_at),
        "first_cycle_at": _iso(first_cycle_at) if first_cycle_at else None,
        "days_running": days_running,
        "cycles": cycles,
        "rows_total": rows_total,
        "rows_last_hour": rows_last_hour,
        "cadence_seconds": float(median_cadence) if median_cadence is not None else None,
        "distinct_routes": distinct_routes,
        "distinct_trips": distinct_trips,
        "distinct_stops": distinct_stops,
        "labels_total": labels_total,
        "labels_gap_abutted": labels_gap_abutted,
        "labels_no_arrival": labels_no_arrival,
        "disk_free_gib": free_gib,
        "gates_pass": gates_pass,
        "gates_total": len(gate_detail),
        "gate_detail": json.dumps(gate_detail),
    }, float(median_cadence) if median_cadence is not None else None


# -- gaps -----------------------------------------------------------------


def compute_gaps(conn, median_cadence: float | None) -> list[dict[str, Any]]:
    if not median_cadence or median_cadence <= 0:
        return []

    threshold_seconds = GAP_THRESHOLD_MULTIPLIER * median_cadence

    with conn.cursor() as cur:
        cur.execute(_GAP_SOURCE_QUERY)
        rows = cur.fetchall()

    gaps: list[dict[str, Any]] = []
    prev_polled_at: dt.datetime | None = None

    for polled_at, started_at, finished_at, error, batches_ok, batches_total in rows:
        gap_seconds = (polled_at - prev_polled_at).total_seconds() if prev_polled_at is not None else None

        if gap_seconds is not None and gap_seconds > threshold_seconds:
            if error:
                reason = _classify_reason(error, batches_ok, batches_total)
                detail = _truncate(error, DETAIL_MAX_CHARS)
            else:
                reason = "no poll recorded"
                detail = None
            cycles_missed = max(0, round(gap_seconds / median_cadence) - 1)
            gaps.append(
                {
                    "started_at": _iso(prev_polled_at),
                    "ended_at": _iso(polled_at),
                    "duration_seconds": round(gap_seconds),
                    "reason": _truncate(reason),
                    "cycles_missed": cycles_missed,
                    "detail": detail,
                }
            )
        elif error:
            duration_seconds = round((finished_at - started_at).total_seconds())
            gaps.append(
                {
                    "started_at": _iso(started_at),
                    "ended_at": _iso(finished_at),
                    "duration_seconds": duration_seconds,
                    "reason": _truncate(_classify_reason(error, batches_ok, batches_total)),
                    "cycles_missed": 0,
                    "detail": _truncate(error, DETAIL_MAX_CHARS),
                }
            )

        prev_polled_at = polled_at

    return gaps


# -- D1 -----------------------------------------------------------------


def build_sql(snapshot: dict[str, Any], gaps: list[dict[str, Any]]) -> str:
    cols = list(snapshot.keys())
    numeric_cols = {
        "days_running",
        "cycles",
        "rows_total",
        "rows_last_hour",
        "cadence_seconds",
        "distinct_routes",
        "distinct_trips",
        "distinct_stops",
        "labels_total",
        "labels_gap_abutted",
        "labels_no_arrival",
        "disk_free_gib",
        "gates_pass",
        "gates_total",
    }
    values = ", ".join(
        _sql_num(snapshot[c]) if c in numeric_cols else _sql_str(snapshot[c]) for c in cols
    )
    statements = [
        f"INSERT OR REPLACE INTO snapshots ({', '.join(cols)}) VALUES ({values})",
        f"DELETE FROM snapshots WHERE published_at NOT IN "
        f"(SELECT published_at FROM snapshots ORDER BY published_at DESC LIMIT {MAX_SNAPSHOTS})",
    ]

    if gaps:
        gap_cols = ["started_at", "ended_at", "duration_seconds", "reason", "cycles_missed", "detail"]
        rows_sql = ", ".join(
            "("
            + ", ".join(
                [
                    _sql_str(g["started_at"]),
                    _sql_str(g["ended_at"]),
                    _sql_num(g["duration_seconds"]),
                    _sql_str(g["reason"]),
                    _sql_num(g["cycles_missed"]),
                    _sql_str(g["detail"]),
                ]
            )
            + ")"
            for g in gaps
        )
        statements.append(f"INSERT OR REPLACE INTO gaps ({', '.join(gap_cols)}) VALUES {rows_sql}")

    return ";\n".join(statements) + ";"


def push_to_d1(sql: str) -> None:
    result = subprocess.run(
        [WRANGLER_BIN, "wrangler", "d1", "execute", D1_DATABASE, "--remote", "--command", sql],
        capture_output=True,
        text=True,
        timeout=WRANGLER_TIMEOUT_SECONDS,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"wrangler d1 execute failed (exit {result.returncode}): "
            f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}"
        )


def main() -> int:
    now = dt.datetime.now(dt.timezone.utc)

    try:
        conn = db.connect()
    except Exception as exc:  # noqa: BLE001 - stale-but-honest: log, exit, never touch D1
        print(f"pulse.publish: {now.isoformat()} Postgres unreachable, D1 NOT touched: {exc}", file=sys.stderr)
        return 1

    try:
        snapshot, median_cadence = compute_snapshot(conn, now)
        gaps = compute_gaps(conn, median_cadence)
    except Exception as exc:  # noqa: BLE001 - same stale-but-honest contract on any query failure
        print(f"pulse.publish: {now.isoformat()} snapshot computation failed, D1 NOT touched: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()

    sql = build_sql(snapshot, gaps)

    try:
        push_to_d1(sql)
    except Exception as exc:  # noqa: BLE001 - stale-but-honest: D1 keeps its last good row
        print(f"pulse.publish: {now.isoformat()} D1 push failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"published_at={snapshot['published_at']} days_running={snapshot['days_running']:.3f} "
        f"cycles={snapshot['cycles']} rows_total={snapshot['rows_total']} "
        f"gates={snapshot['gates_pass']}/{snapshot['gates_total']} gaps={len(gaps)}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
