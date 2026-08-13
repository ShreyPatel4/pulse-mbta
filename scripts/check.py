"""First-night verification: summarize stop_events ingestion so far.

Usage: uv run python scripts/check.py

Prints: total rows, rows in the last hour, distinct trips/routes/stops,
min/max polled_at, and the null-rate of scheduled_arrival / predicted_arrival
(the ~4.2% origin-stop rows noted in README are expected here, not a bug).
"""

from __future__ import annotations

import sys

from pulse import db

_QUERY = """
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


def _pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{100.0 * numerator / denominator:.2f}%"


def main() -> int:
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_QUERY)
            row = cur.fetchone()
    finally:
        conn.close()

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
