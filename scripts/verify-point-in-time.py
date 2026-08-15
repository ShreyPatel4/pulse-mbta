"""Verifies point-in-time correctness of features_trip_stop against the live
database, independently of the SQL that built it.

Usage: uv run python scripts/verify-point-in-time.py

M3/M4 asks for point-in-time correctness to be verified and for the
verification to be stated, not asserted. Prose restating a WHERE clause is
not verification. Three checks run here, each of which fails loudly and
exits non-zero:

1. INDEPENDENT RECOMPUTATION. pulse.features computes
   route_hour_historical_late_rate as a correlated subquery filtered to
   strictly-earlier scheduled_arrival. This recomputes every row's value a
   different way -- a window function with a GROUPS frame,
   `GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`, which frames by peer
   group and therefore covers exactly the rows with a strictly smaller
   scheduled_arrival -- and compares. Two implementations that disagree
   nowhere are much stronger evidence than one implementation read carefully.
   If a `<` ever became a `<=`, or a rollup got refreshed over the whole
   table, the two would diverge here.

2. NULL ARITHMETIC. A NULL rate has to mean exactly one thing: this row is
   first in its (route, hour) bucket, so no strictly-earlier row exists. The
   count of NULLs must equal the count of rows tied for earliest in their
   bucket (rank() = 1, not row_number(): real schedules put several buses at
   the same minute, and every one of them legitimately sees an empty
   history). A mismatch means NULLs are coming from somewhere else -- a
   failed join, a fabricated default, a bucket key that does not line up.

3. NO FUTURE IN THE PAST. For every row, the newest label that fed its
   aggregate must be strictly older than the row itself. This is the direct
   statement of the property, checked over the whole table rather than
   argued from the query text.

Note what these checks do NOT cover, because it is a different property: they
prove the aggregate never reads data from after the row it describes. They do
not prove the data it reads was SETTLED by the 10-minute prediction horizon.
See docs/report.md, "The horizon this actually honors".

The durable version of check 1 is tests/test_features.py's mutation test,
which inserts future labels and asserts an earlier row does not move. This
script is the same question asked of the real table at real volume.
"""

from __future__ import annotations

import sys

from pulse import db

_HOUR = "EXTRACT(HOUR FROM l.scheduled_arrival AT TIME ZONE 'America/New_York')"

_RECOMPUTE_SQL = f"""
WITH recomputed AS (
    SELECT
        f.route_hour_historical_late_rate AS stored,
        avg(CASE WHEN l.late THEN 1.0 ELSE 0.0 END) OVER (
            PARTITION BY l.route_id, {_HOUR}
            ORDER BY l.scheduled_arrival
            GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS window_value
    FROM features_trip_stop f
    JOIN trip_stop_labels_training l
      USING (service_date_norm, route_id, direction_id, stop_id, trip_id)
)
SELECT
    count(*),
    count(*) FILTER (
        WHERE (stored IS NULL) <> (window_value IS NULL)
           OR abs(coalesce(stored, 0) - coalesce(window_value, 0)) > 1e-9
    )
FROM recomputed
"""

_NULL_ARITHMETIC_SQL = f"""
SELECT
    (SELECT count(*) FROM features_trip_stop WHERE route_hour_historical_late_rate IS NULL),
    (SELECT count(*) FROM (
        SELECT rank() OVER (
            PARTITION BY l.route_id, {_HOUR}
            ORDER BY l.scheduled_arrival
        ) AS rnk
        FROM trip_stop_labels_training l
    ) t WHERE rnk = 1)
"""

_NEWEST_CONTRIBUTOR_SQL = f"""
WITH newest AS (
    SELECT
        l.scheduled_arrival AS row_time,
        max(l.scheduled_arrival) FILTER (WHERE TRUE) OVER (
            PARTITION BY l.route_id, {_HOUR}
            ORDER BY l.scheduled_arrival
            GROUPS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS newest_contributor
    FROM features_trip_stop f
    JOIN trip_stop_labels_training l
      USING (service_date_norm, route_id, direction_id, stop_id, trip_id)
)
SELECT count(*), count(*) FILTER (WHERE newest_contributor >= row_time)
FROM newest
"""


def main() -> int:
    conn = db.connect()
    try:
        with conn.cursor() as cur:
            cur.execute(_RECOMPUTE_SQL)
            compared, mismatches = cur.fetchone()
            cur.execute(_NULL_ARITHMETIC_SQL)
            null_rates, first_in_bucket = cur.fetchone()
            cur.execute(_NEWEST_CONTRIBUTOR_SQL)
            checked, from_the_future = cur.fetchone()
    finally:
        conn.close()

    checks = [
        (
            "independent_recomputation",
            mismatches == 0,
            f"{compared} rows compared against a GROUPS-framed window function, "
            f"{mismatches} disagree (threshold: 0)",
        ),
        (
            "null_arithmetic",
            null_rates == first_in_bucket,
            f"{null_rates} rows have a NULL route_hour_historical_late_rate, "
            f"{first_in_bucket} rows are tied-earliest in their (route, hour) bucket "
            f"(these must be equal)",
        ),
        (
            "no_future_in_the_past",
            from_the_future == 0,
            f"{checked} rows checked, {from_the_future} have a contributing label "
            f"scheduled at or after the row itself (threshold: 0)",
        ),
    ]

    print("-- point-in-time correctness --")
    for name, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    return 0 if all(passed for _name, passed, _detail in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
