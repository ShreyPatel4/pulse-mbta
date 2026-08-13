"""Database layer: connection + idempotent upserts for stop_events."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from typing import Any

import psycopg

DEFAULT_DSN = "postgresql://localhost/pulse"

_UPSERT_SQL = """
INSERT INTO stop_events (
    route_id, direction_id, stop_id, trip_id, vehicle_id,
    service_date, scheduled_arrival, predicted_arrival, status, polled_at
) VALUES (
    %(route_id)s, %(direction_id)s, %(stop_id)s, %(trip_id)s, %(vehicle_id)s,
    %(service_date)s, %(scheduled_arrival)s, %(predicted_arrival)s, %(status)s, %(polled_at)s
)
ON CONFLICT (trip_id, stop_id, polled_at) DO NOTHING
"""


def connect(dsn: str | None = None) -> psycopg.Connection:
    """Open a connection to Postgres.

    dsn defaults to the PULSE_DSN env var, falling back to the local
    ``pulse`` database over TCP localhost (trust auth).
    """
    dsn = dsn or os.environ.get("PULSE_DSN", DEFAULT_DSN)
    conn = psycopg.connect(dsn)
    conn.autocommit = True
    return conn


def upsert_stop_events(conn: psycopg.Connection, rows: Iterable[Mapping[str, Any]]) -> int:
    """Idempotently upsert stop_event rows.

    Rows are dicts with keys matching the stop_events columns. Duplicate
    (trip_id, stop_id, polled_at) rows are silently skipped (ON CONFLICT DO
    NOTHING) since snapshots are immutable facts. Returns the number of
    rows actually inserted.
    """
    rows = list(rows)
    if not rows:
        return 0
    with conn.cursor() as cur:
        cur.executemany(_UPSERT_SQL, rows)
        return cur.rowcount
