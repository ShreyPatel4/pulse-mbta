"""Ingestion summary + M1 quality gate.

Usage: uv run python scripts/check.py

Prints: total rows, rows in the last hour, distinct trips/routes/stops,
min/max polled_at, and the null-rate of scheduled_arrival / predicted_arrival
(the ~3% origin-stop rows noted in README are expected here, not a bug) --
then four PASS/FAIL gates against the live poll_runs ledger and stop_events:

  disk        free space on the data volume >= 20 GiB
  freshness   a poll_runs row within the last 10 minutes
  volume      > 0 rows inserted (poll_runs) in the last hour
  null-rate   scheduled_arrival null-rate <= 15% over the last hour

Exit code is 1 if any gate fails, 0 if all pass.
"""

from __future__ import annotations

import datetime as dt
import shutil
import sys

from pulse import db

_SUMMARY_QUERY = """
SELECT
    count(*) AS total_rows,
    count(*) FILTER (WHERE polled_at >= now() - interval '1 hour') AS last_hour_rows,
    count(DISTINCT trip_id) AS distinct_trips,
    count(DISTINCT route_id) AS distinct_routes,
    count(DISTINCT stop_id) AS distinct_stops,
    min(polled_at) AS min_polled_at,
    max(polled_at) AS max_polled_at,
    count(*) FILTER (WHERE scheduled_arrival IS NULL) AS scheduled_arrival_null,
    count(*) FILTER (WHERE predicted_arrival IS NULL) AS predicted_arrival_null
FROM stop_events
"""

_FRESHNESS_QUERY = "SELECT max(polled_at) FROM poll_runs"

_VOLUME_QUERY = """
SELECT COALESCE(sum(inserted), 0)
FROM poll_runs
WHERE polled_at >= now() - interval '1 hour'
"""

_NULL_RATE_QUERY = """
SELECT
    count(*) FILTER (WHERE scheduled_arrival IS NULL) AS nulls,
    count(*) AS total
FROM stop_events
WHERE polled_at >= now() - interval '1 hour'
"""

# Data volume, not stop_events row count: this is the disk holding Postgres's
# data directory on macOS (APFS's separate Data volume), which is what
# actually runs out when stop_events keeps growing. Falls back to '/' for
# any layout where that volume doesn't exist (e.g. non-APFS, non-macOS).
_DISK_PATHS = ("/System/Volumes/Data", "/")

DISK_FREE_MIN_BYTES = 20 * 1024**3  # 20 GiB
FRESHNESS_MAX_MINUTES = 10
NULL_RATE_MAX_PCT = 15.0


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def _disk_free_bytes() -> tuple[int, str]:
    """Free bytes on the data volume, plus which path answered."""
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


def _gate(label: str, passed: bool, detail: str) -> bool:
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return passed


def main() -> int:
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_SUMMARY_QUERY)
            row = cur.fetchone()

        (
            total_rows,
            last_hour_rows,
            distinct_trips,
            distinct_routes,
            distinct_stops,
            min_polled_at,
            max_polled_at,
            scheduled_arrival_null,
            predicted_arrival_null,
        ) = row

        print(f"total_rows             = {total_rows}")
        print(f"last_hour_rows         = {last_hour_rows}")
        print(f"distinct_trips         = {distinct_trips}")
        print(f"distinct_routes        = {distinct_routes}")
        print(f"distinct_stops         = {distinct_stops}")
        print(f"min_polled_at          = {min_polled_at}")
        print(f"max_polled_at          = {max_polled_at}")
        print(
            f"scheduled_arrival_null = {scheduled_arrival_null} "
            f"({_pct(scheduled_arrival_null, total_rows)})"
        )
        print(
            f"predicted_arrival_null = {predicted_arrival_null} "
            f"({_pct(predicted_arrival_null, total_rows)})"
        )

        print()
        print("-- M1 quality gates --")
        gates_passed: list[bool] = []

        free_bytes, disk_path = _disk_free_bytes()
        free_gib = free_bytes / 1024**3
        gates_passed.append(
            _gate(
                "disk",
                free_bytes >= DISK_FREE_MIN_BYTES,
                f"{free_gib:.1f} GiB free on {disk_path} (threshold: >= 20.0 GiB)",
            )
        )

        with conn.cursor() as cur:
            cur.execute(_FRESHNESS_QUERY)
            (last_run_at,) = cur.fetchone()
        if last_run_at is None:
            gates_passed.append(
                _gate("freshness", False, f"no poll_runs rows at all (threshold: a row within {FRESHNESS_MAX_MINUTES} min)")
            )
        else:
            age_minutes = (dt.datetime.now(dt.timezone.utc) - last_run_at).total_seconds() / 60
            gates_passed.append(
                _gate(
                    "freshness",
                    age_minutes <= FRESHNESS_MAX_MINUTES,
                    f"last poll_runs row {age_minutes:.1f} min ago (threshold: <= {FRESHNESS_MAX_MINUTES} min)",
                )
            )

        with conn.cursor() as cur:
            cur.execute(_VOLUME_QUERY)
            (last_hour_inserted,) = cur.fetchone()
        gates_passed.append(
            _gate(
                "volume",
                last_hour_inserted > 0,
                f"{last_hour_inserted} rows inserted (poll_runs) in the last hour (threshold: > 0)",
            )
        )

        with conn.cursor() as cur:
            cur.execute(_NULL_RATE_QUERY)
            nulls, total = cur.fetchone()
        if total == 0:
            gates_passed.append(
                _gate("null-rate", False, f"no stop_events rows in the last hour to evaluate (threshold: <= {NULL_RATE_MAX_PCT}%)")
            )
        else:
            null_pct = 100.0 * nulls / total
            gates_passed.append(
                _gate(
                    "null-rate",
                    null_pct <= NULL_RATE_MAX_PCT,
                    f"scheduled_arrival null-rate {null_pct:.2f}% over last hour ({nulls}/{total}) "
                    f"(threshold: <= {NULL_RATE_MAX_PCT}%)",
                )
            )
    finally:
        conn.close()

    return 0 if all(gates_passed) else 1


if __name__ == "__main__":
    sys.exit(main())
