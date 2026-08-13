"""End-to-end tests for pulse.labels.run_build against a real local Postgres
(pulse_test_labels, created/dropped per test, migrated through the full
migrations/*.sql chain) with synthetic stop_events + poll_runs fixtures.
Proves: gap exclusion, 3AM service_date_norm, origin-stop filtering, and
idempotent rerun over a DIFFERENT, OVERLAPPING window (not just a same-window
rerun -- see pulse.labels' module docstring for why that distinction
matters: a same-window rerun can't catch a window-dependent bug that a
different-but-overlapping window can)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import psycopg
import pytest

from pulse import db, labels

ADMIN_DSN = "postgresql://localhost/postgres"
TEST_DSN = "postgresql://localhost/pulse_test_labels"
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
MIGRATION_PATHS = sorted(MIGRATIONS_DIR.glob("*.sql"))

_UTC = dt.timezone.utc
_EASTERN = ZoneInfo("America/New_York")
_CYCLE = dt.timedelta(seconds=66)  # measured cadence


@pytest.fixture()
def conn():
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS pulse_test_labels")
        admin.execute("CREATE DATABASE pulse_test_labels")

    with psycopg.connect(TEST_DSN, autocommit=True) as setup:
        for path in MIGRATION_PATHS:
            setup.execute(path.read_text())

    test_conn = db.connect(TEST_DSN)
    try:
        yield test_conn
    finally:
        test_conn.close()
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS pulse_test_labels")


def _insert_stop_event(
    conn,
    *,
    trip_id: str,
    stop_id: str = "110",
    route_id: str = "1",
    direction_id: int = 0,
    polled_at: dt.datetime,
    scheduled_arrival: dt.datetime | None,
    predicted_arrival: dt.datetime | None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO stop_events (
                route_id, direction_id, stop_id, trip_id, vehicle_id,
                service_date, scheduled_arrival, predicted_arrival, status, polled_at
            ) VALUES (%s, %s, %s, %s, NULL, %s, %s, %s, NULL, %s)
            ON CONFLICT DO NOTHING
            """,
            (
                route_id,
                direction_id,
                stop_id,
                trip_id,
                polled_at.astimezone(_EASTERN).date(),
                scheduled_arrival,
                predicted_arrival,
                polled_at,
            ),
        )


def _insert_poll_run(conn, polled_at: dt.datetime, error: str | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO poll_runs (polled_at, started_at, finished_at, batches_ok, batches_total,
                                    pages_fetched, rows, inserted, skipped, error)
            VALUES (%s, %s, %s, 3, 3, 3, 100, 100, 0, %s)
            ON CONFLICT DO NOTHING
            """,
            (polled_at, polled_at, polled_at + dt.timedelta(seconds=2), error),
        )


def _label_row(conn, trip_id: str, stop_id: str = "110") -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT service_date_norm, delay_seconds, late, closed_reason, n_snapshots "
            "FROM trip_stop_labels WHERE trip_id = %s AND stop_id = %s",
            (trip_id, stop_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = ["service_date_norm", "delay_seconds", "late", "closed_reason", "n_snapshots"]
        return dict(zip(cols, row))


def _seed_continuous_poll_runs(conn, start: dt.datetime, count: int) -> None:
    for i in range(count):
        _insert_poll_run(conn, start + i * _CYCLE)


# -- gap exclusion ------------------------------------------------------------


def test_normal_disappearance_with_continuous_polling_closes_clean(conn):
    base = dt.datetime(2026, 8, 13, 12, 0, 0, tzinfo=_UTC)
    scheduled = base
    # 5 snapshots, last sighting at base + 4*_CYCLE, predicted arrival 90s late.
    for i in range(5):
        _insert_stop_event(
            conn,
            trip_id="trip-clean",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=scheduled,
            predicted_arrival=scheduled + dt.timedelta(seconds=90),
        )
    last_seen = base + 4 * _CYCLE
    # Continuous polling well past the settle margin, no gaps.
    _seed_continuous_poll_runs(conn, base, count=15)  # spans ~15*66=990s past base

    as_of = last_seen + dt.timedelta(seconds=600)
    summary = labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    assert summary["passed"] is True
    row = _label_row(conn, "trip-clean")
    assert row is not None
    assert row["closed_reason"] is None
    assert row["delay_seconds"] == 90
    assert row["late"] is False
    assert row["n_snapshots"] == 5


def test_missing_poll_runs_cycle_abutting_disappearance_excludes(conn):
    base = dt.datetime(2026, 8, 13, 13, 0, 0, tzinfo=_UTC)
    scheduled = base
    for i in range(5):
        _insert_stop_event(
            conn,
            trip_id="trip-gapped",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=scheduled,
            predicted_arrival=scheduled + dt.timedelta(seconds=90),
        )
    last_seen = base + 4 * _CYCLE

    # poll_runs continuous up through last_seen, then a cycle goes missing
    # right after (no row recorded for base+5*_CYCLE), resuming afterward --
    # this gap abuts the trip-stop's disappearance.
    for i in range(5):
        _insert_poll_run(conn, base + i * _CYCLE)
    _insert_poll_run(conn, base + 7 * _CYCLE)  # skipped index 5 -- a real gap
    for i in range(8, 15):
        _insert_poll_run(conn, base + i * _CYCLE)

    as_of = last_seen + dt.timedelta(seconds=600)
    summary = labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    row = _label_row(conn, "trip-gapped")
    assert row is not None
    assert row["closed_reason"] == "gap_abutted"
    assert row["delay_seconds"] == 90  # still populated, informational
    # Excluded from the training view.
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM trip_stop_labels_training WHERE trip_id = 'trip-gapped'")
        assert cur.fetchone()[0] == 0
    # But still queryable directly.
    assert summary["rows_written"] >= 1


def test_gap_far_from_disappearance_does_not_exclude(conn):
    base = dt.datetime(2026, 8, 13, 14, 0, 0, tzinfo=_UTC)
    scheduled = base
    for i in range(3):
        _insert_stop_event(
            conn,
            trip_id="trip-far-gap",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=scheduled,
            predicted_arrival=scheduled + dt.timedelta(seconds=30),
        )
    last_seen = base + 2 * _CYCLE

    # Continuous polling well past the settle window after last_seen (up to
    # ~924s -- comfortably clears last_seen + SETTLE_MARGIN_SECONDS ~= 330s).
    _seed_continuous_poll_runs(conn, base, count=15)
    # A gap far in the future, well outside the settle window -- must not
    # affect this trip-stop's close.
    _insert_poll_run(conn, base + 40 * _CYCLE)
    _insert_poll_run(conn, base + 50 * _CYCLE)

    as_of = last_seen + dt.timedelta(seconds=600)
    labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    row = _label_row(conn, "trip-far-gap")
    assert row is not None
    assert row["closed_reason"] is None


def test_errored_poll_run_cycle_abutting_disappearance_excludes(conn):
    base = dt.datetime(2026, 8, 13, 15, 0, 0, tzinfo=_UTC)
    for i in range(3):
        _insert_stop_event(
            conn,
            trip_id="trip-errored",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=base,
            predicted_arrival=base + dt.timedelta(seconds=45),
        )
    last_seen = base + 2 * _CYCLE

    for i in range(3):
        _insert_poll_run(conn, base + i * _CYCLE)
    _insert_poll_run(conn, base + 3 * _CYCLE, error="batch ['1'] failed: fabricated")
    for i in range(4, 10):
        _insert_poll_run(conn, base + i * _CYCLE)

    as_of = last_seen + dt.timedelta(seconds=600)
    labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    row = _label_row(conn, "trip-errored")
    assert row["closed_reason"] == "gap_abutted"


# -- 3AM service_date_norm ----------------------------------------------------


def test_late_night_trip_normalizes_to_prior_service_date(conn):
    # 2:00am America/New_York -- must normalize to the PRIOR calendar date.
    scheduled = dt.datetime(2026, 8, 14, 2, 0, tzinfo=_EASTERN).astimezone(_UTC)
    base = scheduled - dt.timedelta(minutes=5)
    for i in range(3):
        _insert_stop_event(
            conn,
            trip_id="trip-latenight",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=scheduled,
            predicted_arrival=scheduled + dt.timedelta(seconds=20),
        )
    last_seen = base + 2 * _CYCLE
    _seed_continuous_poll_runs(conn, base, count=15)

    as_of = last_seen + dt.timedelta(seconds=600)
    labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    row = _label_row(conn, "trip-latenight")
    assert row["service_date_norm"] == dt.date(2026, 8, 13)  # NOT the 14th


def test_daytime_trip_normalizes_to_the_same_calendar_date(conn):
    scheduled = dt.datetime(2026, 8, 14, 14, 0, tzinfo=_EASTERN).astimezone(_UTC)
    base = scheduled - dt.timedelta(minutes=5)
    for i in range(3):
        _insert_stop_event(
            conn,
            trip_id="trip-daytime",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=scheduled,
            predicted_arrival=scheduled + dt.timedelta(seconds=20),
        )
    last_seen = base + 2 * _CYCLE
    _seed_continuous_poll_runs(conn, base, count=15)

    as_of = last_seen + dt.timedelta(seconds=600)
    labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    row = _label_row(conn, "trip-daytime")
    assert row["service_date_norm"] == dt.date(2026, 8, 14)


# -- origin-stop filtering -----------------------------------------------------


def test_origin_stop_rows_both_null_are_filtered_no_arrival_signal(conn):
    base = dt.datetime(2026, 8, 13, 16, 0, 0, tzinfo=_UTC)
    for i in range(4):
        _insert_stop_event(
            conn,
            trip_id="trip-origin",
            stop_id="999",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=None,
            predicted_arrival=None,
        )
    last_seen = base + 3 * _CYCLE
    _seed_continuous_poll_runs(conn, base, count=15)

    as_of = last_seen + dt.timedelta(seconds=600)
    summary = labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    row = _label_row(conn, "trip-origin", stop_id="999")
    assert row is not None
    assert row["closed_reason"] == "no_arrival_signal"
    assert row["delay_seconds"] is None
    assert row["late"] is None
    # Never imputed as on-time, and excluded from the training view.
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM trip_stop_labels_training WHERE trip_id = 'trip-origin'")
        assert cur.fetchone()[0] == 0
    # This fixture's only trip-stop is 100% excluded -- label_rate correctly
    # FAILs (it's the gate's job to flag a run that's mostly unusable rows;
    # see test_quality_gates_fail_when_nothing_gets_written for the volume
    # gate and the live build-labels run for a realistic, mixed-fixture PASS).
    assert summary["gate_results"]["label_rate"]["passed"] is False
    assert summary["passed"] is False


def test_not_yet_settled_trip_stop_is_skipped_this_run(conn):
    base = dt.datetime(2026, 8, 13, 17, 0, 0, tzinfo=_UTC)
    _insert_stop_event(
        conn, trip_id="trip-inflight", polled_at=base, scheduled_arrival=base, predicted_arrival=base
    )
    _insert_poll_run(conn, base)

    as_of = base + dt.timedelta(seconds=30)  # well within the settle margin
    labels.run_build(conn, since=base - dt.timedelta(minutes=1), until=as_of, as_of=as_of)

    assert _label_row(conn, "trip-inflight") is None


# -- idempotent rerun over a DIFFERENT, OVERLAPPING window --------------------


def test_rerun_over_a_different_overlapping_window_updates_not_duplicates(conn):
    base = dt.datetime(2026, 8, 13, 18, 0, 0, tzinfo=_UTC)
    scheduled = base
    for i in range(5):
        _insert_stop_event(
            conn,
            trip_id="trip-idempotent",
            polled_at=base + i * _CYCLE,
            scheduled_arrival=scheduled,
            predicted_arrival=scheduled + dt.timedelta(seconds=45),
        )
    last_seen = base + 4 * _CYCLE
    _seed_continuous_poll_runs(conn, base, count=20)

    as_of = last_seen + dt.timedelta(seconds=900)

    # First run: a narrow window starting right at base.
    labels.run_build(conn, since=base, until=as_of, as_of=as_of)
    first = _label_row(conn, "trip-idempotent")
    assert first is not None
    assert first["closed_reason"] is None
    assert first["delay_seconds"] == 45

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM trip_stop_labels WHERE trip_id = 'trip-idempotent'")
        count_after_first = cur.fetchone()[0]
    assert count_after_first == 1

    # Second run: a DIFFERENT window -- starts later (only overlaps the tail
    # of the trip-stop's snapshots) but still "touches" it since it has a
    # snapshot inside [since2, until). Per pulse.labels' module docstring,
    # the aggregation is computed over the trip-stop's FULL history
    # regardless of window, so this must produce byte-identical output, not
    # a second row under a different service_date_norm.
    since2 = base + 2 * _CYCLE
    labels.run_build(conn, since=since2, until=as_of, as_of=as_of)
    second = _label_row(conn, "trip-idempotent")

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM trip_stop_labels WHERE trip_id = 'trip-idempotent'")
        count_after_second = cur.fetchone()[0]

    assert count_after_second == 1  # updated in place, not duplicated
    assert second == first  # byte-identical: service_date_norm, delay, late, closed_reason, n_snapshots


def test_rerun_after_new_snapshots_arrive_updates_n_snapshots(conn):
    # A rerun that picks up NEW data for an already-labeled trip-stop should
    # update it (this only happens in practice if a trip-stop reappears
    # after being closed, which the settle margin makes rare, but the upsert
    # must still behave correctly rather than silently keeping stale data).
    base = dt.datetime(2026, 8, 13, 19, 0, 0, tzinfo=_UTC)
    for i in range(3):
        _insert_stop_event(
            conn, trip_id="trip-updates", polled_at=base + i * _CYCLE, scheduled_arrival=base,
            predicted_arrival=base + dt.timedelta(seconds=10),
        )
    _seed_continuous_poll_runs(conn, base, count=20)
    as_of = base + 2 * _CYCLE + dt.timedelta(seconds=900)

    labels.run_build(conn, since=base, until=as_of, as_of=as_of)
    first = _label_row(conn, "trip-updates")
    assert first["n_snapshots"] == 3

    # A late-arriving 4th snapshot (e.g. a delayed write) shows up before the
    # next build.
    _insert_stop_event(
        conn, trip_id="trip-updates", polled_at=base + 3 * _CYCLE, scheduled_arrival=base,
        predicted_arrival=base + dt.timedelta(seconds=10),
    )
    as_of2 = base + 3 * _CYCLE + dt.timedelta(seconds=900)
    labels.run_build(conn, since=base, until=as_of2, as_of=as_of2)
    second = _label_row(conn, "trip-updates")

    assert second["n_snapshots"] == 4
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM trip_stop_labels WHERE trip_id = 'trip-updates'")
        assert cur.fetchone()[0] == 1


# -- quality gates + transform_runs -------------------------------------------


def test_run_build_records_a_transform_run_row(conn):
    base = dt.datetime(2026, 8, 13, 20, 0, 0, tzinfo=_UTC)
    for i in range(3):
        _insert_stop_event(
            conn, trip_id="trip-gate", polled_at=base + i * _CYCLE, scheduled_arrival=base,
            predicted_arrival=base + dt.timedelta(seconds=10),
        )
    _seed_continuous_poll_runs(conn, base, count=15)
    as_of = base + 2 * _CYCLE + dt.timedelta(seconds=600)

    summary = labels.run_build(conn, since=base, until=as_of, as_of=as_of)
    labels.record_transform_run(conn, summary, git_sha="deadbeef")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT transform_name, rows_written, passed, git_sha FROM transform_runs "
            "ORDER BY started_at DESC LIMIT 1"
        )
        row = cur.fetchone()
    assert row[0] == "build_labels"
    assert row[1] == summary["rows_written"]
    assert row[2] == summary["passed"]
    assert row[3] == "deadbeef"


def test_quality_gates_fail_when_nothing_gets_written(conn):
    # An empty window (no stop_events touched at all) -- volume gate must FAIL.
    base = dt.datetime(2026, 8, 13, 21, 0, 0, tzinfo=_UTC)
    summary = labels.run_build(conn, since=base, until=base + dt.timedelta(seconds=1), as_of=base + dt.timedelta(hours=1))
    assert summary["rows_written"] == 0
    assert summary["gate_results"]["volume"]["passed"] is False
    assert summary["passed"] is False
