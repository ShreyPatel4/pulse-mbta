"""Tests for pulse.partitions: pure month-math, plus ensure_partitions
against a real local Postgres (pulse_test_partitions, created/dropped per
test, migrated through the full migrations/*.sql chain including 004's
partition swap -- so these tests exercise ensure_partitions against exactly
the partitioned shape the live database has after M2)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import psycopg
import pytest

from pulse import db, partitions

ADMIN_DSN = "postgresql://localhost/postgres"
TEST_DSN = "postgresql://localhost/pulse_test_partitions"
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
MIGRATION_PATHS = sorted(MIGRATIONS_DIR.glob("*.sql"))


@pytest.fixture()
def conn():
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS pulse_test_partitions")
        admin.execute("CREATE DATABASE pulse_test_partitions")

    with psycopg.connect(TEST_DSN, autocommit=True) as setup:
        for path in MIGRATION_PATHS:
            setup.execute(path.read_text())

    test_conn = db.connect(TEST_DSN)
    try:
        yield test_conn
    finally:
        test_conn.close()
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS pulse_test_partitions")


def _partition_exists(conn, name: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_class WHERE relname = %s AND relkind = 'r'", (name,))
        return cur.fetchone() is not None


# -- pure month-math --------------------------------------------------------


def test_target_months_current_plus_two():
    assert partitions._target_months(dt.date(2026, 8, 13), months_ahead=2) == [
        (2026, 8),
        (2026, 9),
        (2026, 10),
    ]


def test_target_months_handles_december_rollover():
    assert partitions._target_months(dt.date(2026, 12, 15), months_ahead=2) == [
        (2026, 12),
        (2027, 1),
        (2027, 2),
    ]


def test_target_months_zero_ahead_is_just_this_month():
    assert partitions._target_months(dt.date(2026, 8, 13), months_ahead=0) == [(2026, 8)]


def test_month_bounds_mid_year():
    assert partitions._month_bounds(2026, 8) == (dt.date(2026, 8, 1), dt.date(2026, 9, 1))


def test_month_bounds_december_rolls_into_next_year():
    assert partitions._month_bounds(2026, 12) == (dt.date(2026, 12, 1), dt.date(2027, 1, 1))


def test_partition_name_zero_pads_month():
    assert partitions.partition_name(2026, 8) == "stop_events_y2026m08"
    assert partitions.partition_name(2026, 11) == "stop_events_y2026m11"


# -- ensure_partitions against a live, already-partitioned stop_events -----


def test_ensure_partitions_noop_when_already_provisioned(conn):
    # migrations/004 already provisions Aug/Sep/Oct 2026 at swap time.
    created = partitions.ensure_partitions(conn, months_ahead=2, today=dt.date(2026, 8, 13))
    assert created == []


def test_ensure_partitions_creates_missing_future_months(conn):
    created = partitions.ensure_partitions(conn, months_ahead=5, today=dt.date(2026, 8, 13))

    assert created == ["stop_events_y2026m11", "stop_events_y2026m12", "stop_events_y2027m01"]
    for name in created:
        assert _partition_exists(conn, name)


def test_ensure_partitions_idempotent_second_call_creates_nothing(conn):
    first = partitions.ensure_partitions(conn, months_ahead=5, today=dt.date(2026, 8, 13))
    second = partitions.ensure_partitions(conn, months_ahead=5, today=dt.date(2026, 8, 13))

    assert first == ["stop_events_y2026m11", "stop_events_y2026m12", "stop_events_y2027m01"]
    assert second == []


def test_ensure_partitions_new_partition_accepts_an_insert_in_its_range(conn):
    partitions.ensure_partitions(conn, months_ahead=5, today=dt.date(2026, 8, 13))

    row = {
        "route_id": "1",
        "direction_id": 0,
        "stop_id": "110",
        "trip_id": "trip-nov",
        "vehicle_id": None,
        "service_date": dt.date(2026, 11, 5),
        "scheduled_arrival": None,
        "predicted_arrival": None,
        "status": None,
        "polled_at": dt.datetime(2026, 11, 5, 12, 0, tzinfo=dt.timezone.utc),
    }
    # Would raise "no partition of relation ... found for row" if November
    # weren't provisioned -- this is the regression ensure_partitions exists
    # to prevent.
    assert db.upsert_stop_events(conn, [row]) == 1


def test_ensure_partitions_row_outside_any_provisioned_month_lands_in_default(conn):
    # A polled_at far outside the provisioned window (no ensure_partitions
    # call for it) must still insert successfully via stop_events_default --
    # the backstop documented in migrations/004's header.
    row = {
        "route_id": "1",
        "direction_id": 0,
        "stop_id": "110",
        "trip_id": "trip-far-future",
        "vehicle_id": None,
        "service_date": dt.date(2030, 1, 1),
        "scheduled_arrival": None,
        "predicted_arrival": None,
        "status": None,
        "polled_at": dt.datetime(2030, 1, 1, 12, 0, tzinfo=dt.timezone.utc),
    }
    assert db.upsert_stop_events(conn, [row]) == 1

    with conn.cursor() as cur:
        cur.execute("SELECT tableoid::regclass::text FROM stop_events WHERE trip_id = 'trip-far-future'")
        (partition,) = cur.fetchone()
    assert partition == "stop_events_default"
