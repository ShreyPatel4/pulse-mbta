"""End-to-end tests for pulse.features.build_features against a real local
Postgres (pulse_test_features, created/dropped per test, migrated through
the full migrations/*.sql chain), with synthetic trip_stop_labels rows
inserted directly (bypassing pulse.labels -- these tests are about the
feature SQL, not label derivation)."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import psycopg
import pytest

from pulse import db, features

ADMIN_DSN = "postgresql://localhost/postgres"
TEST_DSN = "postgresql://localhost/pulse_test_features"
MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"
MIGRATION_PATHS = sorted(MIGRATIONS_DIR.glob("*.sql"))

_UTC = dt.timezone.utc


@pytest.fixture()
def conn():
    with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
        admin.execute("DROP DATABASE IF EXISTS pulse_test_features")
        admin.execute("CREATE DATABASE pulse_test_features")

    with psycopg.connect(TEST_DSN, autocommit=True) as setup:
        for path in MIGRATION_PATHS:
            setup.execute(path.read_text())

    test_conn = db.connect(TEST_DSN)
    try:
        yield test_conn
    finally:
        test_conn.close()
        with psycopg.connect(ADMIN_DSN, autocommit=True) as admin:
            admin.execute("DROP DATABASE IF EXISTS pulse_test_features")


def _insert_label(
    conn,
    *,
    trip_id: str,
    stop_id: str = "110",
    route_id: str = "1",
    direction_id: int = 0,
    service_date_norm: dt.date,
    scheduled_arrival: dt.datetime,
    delay_seconds: int | None,
    late: bool | None,
    closed_reason: str | None = None,
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO trip_stop_labels (
                service_date_norm, route_id, direction_id, stop_id, trip_id,
                scheduled_arrival, final_predicted_arrival, delay_seconds, late,
                observed_span_start, observed_span_end, n_snapshots, closed_reason
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 5, %s)
            """,
            (
                service_date_norm,
                route_id,
                direction_id,
                stop_id,
                trip_id,
                scheduled_arrival,
                scheduled_arrival + dt.timedelta(seconds=delay_seconds) if delay_seconds is not None else None,
                delay_seconds,
                late,
                scheduled_arrival - dt.timedelta(minutes=10),
                scheduled_arrival,
                closed_reason,
            ),
        )


def _feature_row(conn, trip_id: str, stop_id: str = "110") -> dict | None:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT route_hour_historical_late_rate, headway_seconds, hour_of_day, day_of_week, "
            "current_delay_persistence_seconds FROM features_trip_stop WHERE trip_id = %s AND stop_id = %s",
            (trip_id, stop_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        cols = ["route_hour_historical_late_rate", "headway_seconds", "hour_of_day", "day_of_week", "current_delay_persistence_seconds"]
        return dict(zip(cols, row))


def test_first_row_has_no_history_null_rate_and_null_headway(conn):
    t = dt.datetime(2026, 8, 13, 14, 0, 0, tzinfo=_UTC)
    _insert_label(
        conn, trip_id="trip-first", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t,
        delay_seconds=30, late=False,
    )
    summary = features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))
    assert summary["rows_written"] == 1

    row = _feature_row(conn, "trip-first")
    assert row["route_hour_historical_late_rate"] is None  # no earlier same route+hour rows
    assert row["headway_seconds"] is None  # no earlier same route/direction/stop row
    assert row["current_delay_persistence_seconds"] is None  # no earlier same trip row


def test_historical_late_rate_only_uses_strictly_earlier_rows_same_route_hour(conn):
    # Two earlier rows at the same route+hour: one late, one on-time -> 0.5.
    # A LATER row at the same route+hour must NOT be included (would leak
    # the future into an "earlier" aggregate).
    hour = dt.datetime(2026, 8, 13, 14, 0, 0, tzinfo=_UTC)
    _insert_label(conn, trip_id="t1", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=hour, delay_seconds=200, late=True)
    _insert_label(conn, trip_id="t2", stop_id="111", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=hour + dt.timedelta(minutes=5), delay_seconds=10, late=False)
    target = hour + dt.timedelta(minutes=10)
    _insert_label(conn, trip_id="t3", stop_id="112", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=target, delay_seconds=0, late=False)
    # A row scheduled LATER than the target, same route+hour, would skew the
    # rate to 2/4=0.5 if leaked in -- assert it stays 1/2=0.5 either way is
    # ambiguous, so make the "future" row unambiguously different (all late).
    future = target + dt.timedelta(minutes=20)
    _insert_label(conn, trip_id="t4", stop_id="113", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=future, delay_seconds=500, late=True)

    features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))

    row = _feature_row(conn, "t3", stop_id="112")
    # Only t1 (late) and t2 (on-time) are strictly earlier -> 1/2 = 0.5.
    # If the future row (t4, late) leaked in it would be 2/3 = 0.667.
    assert row["route_hour_historical_late_rate"] == pytest.approx(0.5)


def test_future_label_inserted_after_the_fact_cannot_change_an_earlier_rows_feature(conn):
    """The point-in-time mutation test. Prose about a WHERE clause is not
    verification -- this changes the future and asserts the past does not
    move.

    Build features for a target row. Then insert a batch of same-route,
    same-hour labels scheduled AFTER it, all late, enough to swing that
    route-hour bucket hard. Rebuild the target row's features over the same
    window. Its route_hour_historical_late_rate must be byte-identical. If
    the aggregate ever stopped being strictly-earlier -- a `<=` slipping in,
    a window frame including peers, a materialized rollup refreshed over the
    whole table -- this is the test that fails.
    """
    hour = dt.datetime(2026, 8, 14, 14, 0, 0, tzinfo=_UTC)
    day = dt.date(2026, 8, 14)
    build_until = dt.datetime(2026, 8, 16, tzinfo=_UTC)

    # Two strictly-earlier rows: one late, one on time -> the target sees 0.5.
    _insert_label(conn, trip_id="pit-a", stop_id="1", service_date_norm=day, scheduled_arrival=hour, delay_seconds=400, late=True)
    _insert_label(conn, trip_id="pit-b", stop_id="2", service_date_norm=day, scheduled_arrival=hour + dt.timedelta(minutes=1), delay_seconds=10, late=False)
    target_time = hour + dt.timedelta(minutes=2)
    _insert_label(conn, trip_id="pit-target", stop_id="3", service_date_norm=day, scheduled_arrival=target_time, delay_seconds=0, late=False)

    features.build_features(conn, until=build_until)
    before = _feature_row(conn, "pit-target", stop_id="3")
    assert before["route_hour_historical_late_rate"] == pytest.approx(0.5)

    # Now the future happens: 20 more trip-stops in the same route-hour, all
    # late, all scheduled after the target. A leaky aggregate would drag the
    # target's rate from 0.50 toward 21/22 = 0.95.
    for i in range(20):
        _insert_label(
            conn, trip_id=f"pit-future-{i}", stop_id=f"9{i:02d}", service_date_norm=day,
            scheduled_arrival=target_time + dt.timedelta(seconds=10 * (i + 1)),
            delay_seconds=900, late=True,
        )

    features.build_features(conn, until=build_until)
    after = _feature_row(conn, "pit-target", stop_id="3")

    assert after["route_hour_historical_late_rate"] == pytest.approx(0.5)
    assert after == before  # every feature on the row, not just the aggregate


def test_null_historical_rate_means_no_strictly_earlier_row_in_that_route_hour(conn):
    """The arithmetic that backs the mutation test: a NULL rate is not a
    silent failure, it is exactly the first row of its (route, hour) bucket.
    Fabricating a default there (0.0, or the global late rate) would be a
    quiet leak of the population mean into a feature that is supposed to say
    "nothing is known yet".

    rank(), not row_number(): real schedules put several buses at the same
    minute, and every one of a tied-earliest group legitimately sees an empty
    history. scripts/verify-point-in-time.py runs the same identity against
    the live table, where those ties actually occur."""
    day = dt.date(2026, 8, 14)
    # route 1 hour 14, two rows; route 2 hour 14, one row; route 1 hour 15, one row.
    h14 = dt.datetime(2026, 8, 14, 14, 0, 0, tzinfo=_UTC)
    h15 = dt.datetime(2026, 8, 14, 15, 0, 0, tzinfo=_UTC)
    _insert_label(conn, trip_id="n1", stop_id="1", route_id="1", service_date_norm=day, scheduled_arrival=h14, delay_seconds=0, late=False)
    _insert_label(conn, trip_id="n2", stop_id="2", route_id="1", service_date_norm=day, scheduled_arrival=h14 + dt.timedelta(minutes=1), delay_seconds=0, late=False)
    _insert_label(conn, trip_id="n3", stop_id="3", route_id="2", service_date_norm=day, scheduled_arrival=h14, delay_seconds=0, late=False)
    _insert_label(conn, trip_id="n4", stop_id="4", route_id="1", service_date_norm=day, scheduled_arrival=h15, delay_seconds=0, late=False)

    features.build_features(conn, until=dt.datetime(2026, 8, 16, tzinfo=_UTC))

    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM features_trip_stop WHERE route_hour_historical_late_rate IS NULL")
        (nulls,) = cur.fetchone()
        # Rows that are first in their (route_id, hour_of_day) bucket by
        # scheduled_arrival, computed independently of the feature SQL.
        cur.execute(
            """
            SELECT count(*) FROM (
                SELECT rank() OVER (
                    PARTITION BY route_id, EXTRACT(HOUR FROM scheduled_arrival AT TIME ZONE 'America/New_York')
                    ORDER BY scheduled_arrival
                ) AS rnk
                FROM trip_stop_labels_training
            ) t WHERE rnk = 1
            """
        )
        (firsts,) = cur.fetchone()

    assert nulls == firsts == 3  # (route 1, h14), (route 2, h14), (route 1, h15)


def test_persistence_does_not_carry_across_service_dates(conn):
    """MBTA reuses trip_id across service dates (see pulse.labels'
    _TOUCHED_GROUPS_SQL). Without the service_date_norm predicate, the first
    stop of trip T today inherits its "current delay" from the last stop of
    trip T yesterday: a different bus, twenty hours earlier. Still
    point-in-time correct, still the wrong feature."""
    yesterday = dt.datetime(2026, 8, 13, 20, 0, 0, tzinfo=_UTC)
    today = dt.datetime(2026, 8, 14, 12, 0, 0, tzinfo=_UTC)

    _insert_label(conn, trip_id="recur", stop_id="A", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=yesterday, delay_seconds=1200, late=True)
    _insert_label(conn, trip_id="recur", stop_id="A", service_date_norm=dt.date(2026, 8, 14), scheduled_arrival=today, delay_seconds=30, late=False)
    _insert_label(conn, trip_id="recur", stop_id="B", service_date_norm=dt.date(2026, 8, 14), scheduled_arrival=today + dt.timedelta(minutes=4), delay_seconds=45, late=False)

    features.build_features(conn, until=dt.datetime(2026, 8, 16, tzinfo=_UTC))

    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_delay_persistence_seconds FROM features_trip_stop "
            "WHERE trip_id = 'recur' AND stop_id = 'A' AND service_date_norm = %s",
            (dt.date(2026, 8, 14),),
        )
        (first_stop_today,) = cur.fetchone()
    # NULL, not 1200: today's first stop has no earlier delay to carry.
    assert first_stop_today is None

    # Within the same service date it still carries normally.
    with conn.cursor() as cur:
        cur.execute(
            "SELECT current_delay_persistence_seconds FROM features_trip_stop "
            "WHERE trip_id = 'recur' AND stop_id = 'B' AND service_date_norm = %s",
            (dt.date(2026, 8, 14),),
        )
        (second_stop_today,) = cur.fetchone()
    assert second_stop_today == 30


def test_headway_is_seconds_since_previous_scheduled_arrival_same_route_direction_stop(conn):
    t1 = dt.datetime(2026, 8, 13, 15, 0, 0, tzinfo=_UTC)
    t2 = t1 + dt.timedelta(minutes=12)
    _insert_label(conn, trip_id="h1", stop_id="200", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t1, delay_seconds=0, late=False)
    _insert_label(conn, trip_id="h2", stop_id="200", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t2, delay_seconds=0, late=False)

    features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))

    row = _feature_row(conn, "h2", stop_id="200")
    assert row["headway_seconds"] == 12 * 60


def test_headway_does_not_mix_different_stops_or_directions(conn):
    t1 = dt.datetime(2026, 8, 13, 15, 0, 0, tzinfo=_UTC)
    t2 = t1 + dt.timedelta(minutes=12)
    _insert_label(conn, trip_id="d1", stop_id="200", direction_id=0, service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t1, delay_seconds=0, late=False)
    # Same stop, opposite direction -- must not count as the "previous" headway.
    _insert_label(conn, trip_id="d2", stop_id="200", direction_id=1, service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t2, delay_seconds=0, late=False)

    features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))

    row = _feature_row(conn, "d2", stop_id="200")
    assert row["headway_seconds"] is None


def test_current_delay_persistence_is_the_trips_prior_earlier_stop_delay(conn):
    t1 = dt.datetime(2026, 8, 13, 16, 0, 0, tzinfo=_UTC)
    t2 = t1 + dt.timedelta(minutes=5)
    _insert_label(conn, trip_id="trip-multi", stop_id="A", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t1, delay_seconds=240, late=True)
    _insert_label(conn, trip_id="trip-multi", stop_id="B", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t2, delay_seconds=250, late=True)

    features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))

    row = _feature_row(conn, "trip-multi", stop_id="B")
    assert row["current_delay_persistence_seconds"] == 240  # stop A's delay, not stop B's own


def test_hour_of_day_and_day_of_week_are_america_new_york_local(conn):
    # 2026-08-13T02:30:00Z = 2026-08-12 22:30 America/New_York (EDT) --
    # hour_of_day must be 22 (NY local), not 2 (UTC).
    t = dt.datetime(2026, 8, 13, 2, 30, 0, tzinfo=_UTC)
    _insert_label(conn, trip_id="tz-check", service_date_norm=dt.date(2026, 8, 12), scheduled_arrival=t, delay_seconds=0, late=False)

    features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))

    row = _feature_row(conn, "tz-check")
    assert row["hour_of_day"] == 22
    assert row["day_of_week"] == 3  # Wednesday 2026-08-12 (DOW: Sun=0..Sat=6)


def test_excluded_labels_never_get_a_features_row(conn):
    t = dt.datetime(2026, 8, 13, 17, 0, 0, tzinfo=_UTC)
    _insert_label(
        conn, trip_id="trip-excluded", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t,
        delay_seconds=None, late=None, closed_reason="no_arrival_signal",
    )
    summary = features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))
    assert summary["rows_written"] == 0
    assert _feature_row(conn, "trip-excluded") is None


def test_rerun_over_different_overlapping_window_updates_not_duplicates(conn):
    t1 = dt.datetime(2026, 8, 13, 18, 0, 0, tzinfo=_UTC)
    t2 = t1 + dt.timedelta(minutes=10)
    _insert_label(conn, trip_id="r1", stop_id="X", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t1, delay_seconds=0, late=False)
    _insert_label(conn, trip_id="r2", stop_id="X", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t2, delay_seconds=0, late=False)

    features.build_features(conn, since=t1 - dt.timedelta(minutes=1), until=t2 + dt.timedelta(minutes=1))
    first = _feature_row(conn, "r2", stop_id="X")

    features.build_features(conn, since=t1 + dt.timedelta(minutes=5), until=t2 + dt.timedelta(minutes=1))
    second = _feature_row(conn, "r2", stop_id="X")

    assert first == second
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM features_trip_stop WHERE trip_id = 'r2'")
        assert cur.fetchone()[0] == 1


def test_run_build_features_records_a_transform_run_row(conn):
    t = dt.datetime(2026, 8, 13, 19, 0, 0, tzinfo=_UTC)
    _insert_label(conn, trip_id="gate-row", service_date_norm=dt.date(2026, 8, 13), scheduled_arrival=t, delay_seconds=0, late=False)

    summary = features.build_features(conn, until=dt.datetime(2026, 8, 14, tzinfo=_UTC))
    features.record_transform_run(conn, summary, git_sha="cafef00d")

    with conn.cursor() as cur:
        cur.execute(
            "SELECT transform_name, rows_written, passed, git_sha FROM transform_runs "
            "ORDER BY started_at DESC LIMIT 1"
        )
        row = cur.fetchone()
    assert row[0] == "build_features"
    assert row[1] == summary["rows_written"]
    assert row[3] == "cafef00d"
