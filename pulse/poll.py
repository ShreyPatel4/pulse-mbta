"""One MBTA /predictions -> stop_events polling cycle.

Usage: uv run python -m pulse.poll

Always exits 0. launchd's KeepAlive re-fires this on its 60s StartInterval,
not on crash-loop detection, so a failed cycle logs to stderr and gets out
of the way rather than aborting the agent.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
import time

import psycopg
import requests

from pulse import db, mbta

ROUTE_IDS = ["1", "15", "22", "23", "28", "32", "39", "57", "66", "71", "73", "77", "111"]
BATCH_SIZES = (5, 5, 3)
BATCH_SLEEP_SECONDS = 1.5


def _prefilter(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop rows that would violate a stop_events NOT NULL column.

    pulse.db.connect() runs autocommit=True, so a NotNullViolation partway
    through executemany would otherwise commit the rows ahead of the bad one
    and silently drop the rest of the batch. Filtering here -- paired with
    wrapping each batch's upsert in an explicit conn.transaction() below --
    means one malformed row can't take out (or partially commit) its whole
    batch: it's counted as skipped and every valid row in the batch still
    lands atomically.
    """
    valid = []
    skipped = 0
    for row in rows:
        if any(row.get(field) is None for field in mbta.REQUIRED_FIELDS):
            skipped += 1
            continue
        valid.append(row)
    return valid, skipped


def run_cycle(conn: psycopg.Connection, session: requests.Session, api_key: str | None) -> str:
    """Run one poll cycle across all routes in 5/5/3 batches. Returns the summary line."""
    polled_at = dt.datetime.now(dt.timezone.utc)
    batches = mbta.batched(ROUTE_IDS, BATCH_SIZES)

    total_rows = 0
    total_inserted = 0
    total_skipped = 0
    routes_ok = 0

    for i, batch in enumerate(batches):
        try:
            payload = mbta.fetch_predictions(batch, session, api_key=api_key)
            rows = mbta.map_rows(payload, polled_at)
            valid_rows, skipped = _prefilter(rows)
            with conn.transaction():
                inserted = db.upsert_stop_events(conn, valid_rows)
            total_rows += len(valid_rows)
            total_inserted += inserted
            total_skipped += skipped
            routes_ok += len(batch)
        except Exception as exc:  # noqa: BLE001 - batch failure logs and the cycle continues
            print(f"pulse.poll: batch {batch} failed: {exc}", file=sys.stderr)

        if i < len(batches) - 1:
            time.sleep(BATCH_SLEEP_SECONDS)

    return (
        f"polled_at={polled_at.isoformat()} rows={total_rows} inserted={total_inserted} "
        f"routes_ok={routes_ok}/{len(ROUTE_IDS)} skipped={total_skipped}"
    )


def main() -> int:
    api_key = os.environ.get("MBTA_API_KEY") or None
    try:
        conn = db.connect()
        try:
            with requests.Session() as session:
                summary = run_cycle(conn, session, api_key)
        finally:
            conn.close()
        print(summary)
    except Exception as exc:  # noqa: BLE001 - never crash-loop launchd
        print(f"pulse.poll: cycle failed to run: {exc}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
