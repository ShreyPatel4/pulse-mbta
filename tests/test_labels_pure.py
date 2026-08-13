"""Pure-function tests for pulse.labels: service_date_norm, gap-interval
math, settle-margin logic, and derive_label_row. No Postgres needed -- see
tests/test_labels.py for the DB-orchestrated build (run_build,
fetch_touched_groups, idempotent rerun) against synthetic fixtures."""

from __future__ import annotations

import datetime as dt

from pulse import labels

_UTC = dt.timezone.utc


def _et(year, month, day, hour, minute=0, second=0):
    """Construct a UTC-aware datetime from America/New_York wall-clock
    components -- August is EDT (UTC-4) in this repo's data window."""
    from zoneinfo import ZoneInfo

    return dt.datetime(year, month, day, hour, minute, second, tzinfo=ZoneInfo("America/New_York")).astimezone(_UTC)


# -- service_date_norm -------------------------------------------------------


def test_service_date_norm_before_3am_belongs_to_prior_day():
    assert labels.service_date_norm(_et(2026, 8, 13, 2, 59)) == dt.date(2026, 8, 12)


def test_service_date_norm_at_3am_belongs_to_same_day():
    assert labels.service_date_norm(_et(2026, 8, 13, 3, 0)) == dt.date(2026, 8, 13)


def test_service_date_norm_daytime_is_the_calendar_date():
    assert labels.service_date_norm(_et(2026, 8, 13, 14, 30)) == dt.date(2026, 8, 13)


def test_service_date_norm_midnight_belongs_to_prior_day():
    assert labels.service_date_norm(_et(2026, 8, 13, 0, 5)) == dt.date(2026, 8, 12)


def test_service_date_norm_converts_from_utc_not_naive_date():
    # 2026-08-13T02:30:00Z is 2026-08-12 22:30 America/New_York (EDT) --
    # calendar dates disagree, and the correct answer is the NY date (12th),
    # which by the 3AM rule (22:30 is well past 3am) stays the 12th.
    instant = dt.datetime(2026, 8, 13, 2, 30, tzinfo=_UTC)
    assert labels.service_date_norm(instant) == dt.date(2026, 8, 12)


# -- compute_gap_intervals ---------------------------------------------------


def _run(polled_at, error=None):
    return {"polled_at": polled_at, "error": error}


def test_compute_gap_intervals_empty_when_no_errors_and_no_wide_spacing():
    t0 = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    rows = [_run(t0 + dt.timedelta(seconds=66 * i)) for i in range(5)]
    assert labels.compute_gap_intervals(rows) == []


def test_compute_gap_intervals_errored_row_is_a_point_interval():
    t0 = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    rows = [_run(t0), _run(t0 + dt.timedelta(seconds=66), error="batch failed"), _run(t0 + dt.timedelta(seconds=132))]
    intervals = labels.compute_gap_intervals(rows)
    assert intervals == [(t0 + dt.timedelta(seconds=66), t0 + dt.timedelta(seconds=66))]


def test_compute_gap_intervals_missing_cycle_wide_spacing_is_an_interval():
    t0 = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    # One cycle skipped: rows 132s apart instead of 66s.
    rows = [_run(t0), _run(t0 + dt.timedelta(seconds=132))]
    intervals = labels.compute_gap_intervals(rows, gap_threshold_seconds=100.0)
    assert intervals == [(t0, t0 + dt.timedelta(seconds=132))]


def test_compute_gap_intervals_ignores_input_order():
    t0 = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    rows = [_run(t0 + dt.timedelta(seconds=132)), _run(t0)]  # reversed
    intervals = labels.compute_gap_intervals(rows, gap_threshold_seconds=100.0)
    assert intervals == [(t0, t0 + dt.timedelta(seconds=132))]


# -- gap_abuts ---------------------------------------------------------------


def test_gap_abuts_true_when_gap_starts_right_after_last_seen():
    last_seen = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    gap = (last_seen + dt.timedelta(seconds=10), last_seen + dt.timedelta(seconds=200))
    assert labels.gap_abuts(last_seen, [gap], settle_margin_seconds=198.0) is True


def test_gap_abuts_false_when_gap_is_well_outside_the_settle_window():
    last_seen = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    gap = (last_seen + dt.timedelta(seconds=500), last_seen + dt.timedelta(seconds=600))
    assert labels.gap_abuts(last_seen, [gap], settle_margin_seconds=198.0) is False


def test_gap_abuts_true_when_gap_straddles_last_seen():
    last_seen = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    gap = (last_seen - dt.timedelta(seconds=50), last_seen + dt.timedelta(seconds=50))
    assert labels.gap_abuts(last_seen, [gap], settle_margin_seconds=198.0) is True


def test_gap_abuts_false_with_no_gap_intervals_at_all():
    last_seen = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    assert labels.gap_abuts(last_seen, [], settle_margin_seconds=198.0) is False


# -- is_settled ---------------------------------------------------------------


def test_is_settled_false_within_margin():
    last_seen = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    build_as_of = last_seen + dt.timedelta(seconds=100)
    assert labels.is_settled(last_seen, build_as_of, settle_margin_seconds=198.0) is False


def test_is_settled_true_at_exactly_the_margin():
    last_seen = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    build_as_of = last_seen + dt.timedelta(seconds=198)
    assert labels.is_settled(last_seen, build_as_of, settle_margin_seconds=198.0) is True


def test_is_settled_true_well_past_margin():
    last_seen = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=_UTC)
    build_as_of = last_seen + dt.timedelta(hours=2)
    assert labels.is_settled(last_seen, build_as_of, settle_margin_seconds=198.0) is True


# -- derive_label_row ---------------------------------------------------------


def _group(**overrides) -> dict:
    base = dt.datetime(2026, 8, 13, 12, 0, 0, tzinfo=_UTC)
    group = {
        "route_id": "1",
        "direction_id": 0,
        "stop_id": "110",
        "trip_id": "trip-1",
        "scheduled_arrival": base,
        "final_predicted_arrival": base + dt.timedelta(seconds=60),
        "observed_span_start": base - dt.timedelta(minutes=10),
        "observed_span_end": base - dt.timedelta(seconds=5),
        "n_snapshots": 10,
    }
    group.update(overrides)
    return group


def test_derive_label_row_none_when_not_settled():
    group = _group()
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=50)  # well within margin
    assert labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0) is None


def test_derive_label_row_normal_close_computes_delay_and_late():
    group = _group()
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    row = labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0)

    assert row is not None
    assert row["closed_reason"] is None
    assert row["delay_seconds"] == 60
    assert row["late"] is False  # 60s <= 180s threshold


def test_derive_label_row_late_true_above_180s():
    group = _group(final_predicted_arrival=_group()["scheduled_arrival"] + dt.timedelta(seconds=181))
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    row = labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0)
    assert row["delay_seconds"] == 181
    assert row["late"] is True


def test_derive_label_row_exactly_180s_is_not_late():
    group = _group(final_predicted_arrival=_group()["scheduled_arrival"] + dt.timedelta(seconds=180))
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    row = labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0)
    assert row["delay_seconds"] == 180
    assert row["late"] is False  # strictly > 180, not >=


def test_derive_label_row_both_null_is_no_arrival_signal():
    group = _group(scheduled_arrival=None, final_predicted_arrival=None)
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    row = labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0)

    assert row["closed_reason"] == "no_arrival_signal"
    assert row["delay_seconds"] is None
    assert row["late"] is None


def test_derive_label_row_mixed_null_is_also_no_arrival_signal():
    # Empirically unobserved in the real data (M1's clean cross-tab) but
    # guarded defensively -- see pulse.labels' module docstring.
    group = _group(scheduled_arrival=None)  # final_predicted_arrival still present
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    row = labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0)
    assert row["closed_reason"] == "no_arrival_signal"


def test_derive_label_row_gap_abutted_still_computes_delay_but_excludes():
    group = _group()
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    gap = (group["observed_span_end"] + dt.timedelta(seconds=5), group["observed_span_end"] + dt.timedelta(seconds=20))
    row = labels.derive_label_row(group, [gap], build_as_of, settle_margin_seconds=198.0)

    assert row["closed_reason"] == "gap_abutted"
    assert row["delay_seconds"] == 60  # still populated -- informational, view excludes it
    assert row["late"] is False


def test_derive_label_row_service_date_norm_anchors_on_scheduled_arrival():
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    scheduled = dt.datetime(2026, 8, 13, 2, 0, tzinfo=eastern).astimezone(_UTC)  # 2am -> prior service date
    group = _group(
        scheduled_arrival=scheduled,
        final_predicted_arrival=scheduled + dt.timedelta(seconds=30),
        observed_span_start=scheduled - dt.timedelta(hours=1),
        observed_span_end=scheduled,
    )
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    row = labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0)
    assert row["service_date_norm"] == dt.date(2026, 8, 12)


def test_derive_label_row_service_date_norm_falls_back_to_first_sighting_when_no_schedule():
    from zoneinfo import ZoneInfo

    eastern = ZoneInfo("America/New_York")
    first_sighting = dt.datetime(2026, 8, 13, 2, 0, tzinfo=eastern).astimezone(_UTC)
    group = _group(
        scheduled_arrival=None,
        final_predicted_arrival=first_sighting + dt.timedelta(minutes=5),
        observed_span_start=first_sighting,
        observed_span_end=first_sighting + dt.timedelta(minutes=5),
    )
    # scheduled_arrival is None -> this becomes a no_arrival_signal row, but
    # service_date_norm must still be computed correctly off the fallback
    # anchor (observed_span_start) even for an excluded row.
    build_as_of = group["observed_span_end"] + dt.timedelta(seconds=300)
    row = labels.derive_label_row(group, [], build_as_of, settle_margin_seconds=198.0)
    assert row["closed_reason"] == "no_arrival_signal"
    assert row["service_date_norm"] == dt.date(2026, 8, 12)
