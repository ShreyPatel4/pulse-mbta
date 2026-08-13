"""Tests for pulse.db against a real local Postgres (pulse_test, created/dropped per test)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import psycopg
import pytest

from pulse import db

ADMIN_DSN = "postgresql://localhost/postgres"
TEST_DSN = "postgresql://localhost/pulse_test"
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
# Every migrations/*.sql in name order, not just 001 -- a fixture pinned to
# 001 alone drifts the moment a later migration changes the live shape (e.g.
# 004's partitioning), and tests would keep passing against a schema the
# poller no longer writes to. sorted() matches scripts/migrate.py's own
# apply order exactly.
MIGRATION_PATHS = sorted(MIGRATIONS_DIR.glob("*.sql"))


@pytest.fixture()
def conn():
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS pulse_test")
        admin.execute("CREATE DATABASE pulse_test")

    with psycopg.connect(TEST_DSN, autocommit=True) as setup:
        for path in MIGRATION_PATHS:
            setup.execute(path.read_text())

    test_conn = db.connect(TEST_DSN)
    try:
        yield test_conn
    finally:
        test_conn.close()
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS pulse_test")


def _row(polled_at: dt.datetime, **overrides) -> dict:
    row = {
        "route_id": "1",
        "direction_id": 0,
        "stop_id": "110",
        "trip_id": "trip-1",
        "vehicle_id": "y1234",
        "service_date": dt.date(2026, 8, 13),
        "scheduled_arrival": dt.datetime(2026, 8, 13, 12, 0, tzinfo=dt.timezone.utc),
        "predicted_arrival": dt.datetime(2026, 8, 13, 12, 3, tzinfo=dt.timezone.utc),
        "status": "ON_TIME",
        "polled_at": polled_at,
    }
    row.update(overrides)
    return row


def _count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM stop_events")
        return cur.fetchone()[0]


def test_upsert_same_key_twice_second_call_inserts_zero(conn):
    polled_at = dt.datetime(2026, 8, 13, 11, 0, tzinfo=dt.timezone.utc)
    row = _row(polled_at)

    assert db.upsert_stop_events(conn, [row]) == 1
    assert db.upsert_stop_events(conn, [row]) == 0
    assert _count(conn) == 1


def test_upsert_distinct_polled_at_inserts_new_row(conn):
    row1 = _row(dt.datetime(2026, 8, 13, 11, 0, tzinfo=dt.timezone.utc))
    row2 = _row(dt.datetime(2026, 8, 13, 11, 1, tzinfo=dt.timezone.utc))

    assert db.upsert_stop_events(conn, [row1]) == 1
    assert db.upsert_stop_events(conn, [row2]) == 1
    assert _count(conn) == 2


def test_upsert_batch_mixed_new_and_duplicate(conn):
    polled_at = dt.datetime(2026, 8, 13, 11, 0, tzinfo=dt.timezone.utc)
    row = _row(polled_at)
    assert db.upsert_stop_events(conn, [row]) == 1

    other_stop = _row(polled_at, stop_id="111")
    assert db.upsert_stop_events(conn, [row, other_stop]) == 1
    assert _count(conn) == 2


def test_upsert_empty_rows_returns_zero_and_no_op(conn):
    assert db.upsert_stop_events(conn, []) == 0
    assert _count(conn) == 0


def test_connect_passes_connect_timeout_kwarg(monkeypatch):
    """Pure test (no real Postgres): db.connect() must pass connect_timeout
    as a psycopg.connect kwarg, not bake it into DEFAULT_DSN -- kwargs still
    apply when PULSE_DSN overrides the DSN string entirely."""
    captured: dict = {}

    class _FakeConn:
        autocommit = False

    def fake_connect(dsn, **kwargs):
        captured["dsn"] = dsn
        captured["kwargs"] = kwargs
        return _FakeConn()

    monkeypatch.setattr(db.psycopg, "connect", fake_connect)

    conn = db.connect("postgresql://example/db")

    assert captured["dsn"] == "postgresql://example/db"
    assert captured["kwargs"] == {"connect_timeout": db.CONNECT_TIMEOUT_SECONDS}
    assert conn.autocommit is True


def test_upsert_allows_null_optional_fields(conn):
    polled_at = dt.datetime(2026, 8, 13, 11, 0, tzinfo=dt.timezone.utc)
    row = _row(
        polled_at,
        vehicle_id=None,
        scheduled_arrival=None,
        predicted_arrival=None,
        status=None,
    )

    assert db.upsert_stop_events(conn, [row]) == 1
    assert _count(conn) == 1
