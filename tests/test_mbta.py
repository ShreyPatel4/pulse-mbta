"""Tests for pulse.mbta: pure map_rows over a real, stripped fixture payload,
plus a request-shape check for fetch_predictions against a fake session
(no network -- fetch_predictions itself is I/O, not something map_rows
tests need to exercise)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from pulse import mbta

FIXTURE = json.loads(
    (Path(__file__).resolve().parent / "fixtures" / "predictions_sample.json").read_text()
)

# 2026-08-13T02:30:00Z is 2026-08-12 22:30 in America/New_York (EDT, UTC-4).
# Deliberately chosen so the UTC calendar date (13th) and the correct NY
# local date (12th) disagree -- proving map_rows does a real timezone
# conversion rather than taking polled_at's naive .date().
POLLED_AT = dt.datetime(2026, 8, 13, 2, 30, tzinfo=dt.timezone.utc)


def _rows() -> list[dict]:
    return mbta.map_rows(FIXTURE, POLLED_AT)


def _row_for_trip(rows: list[dict], trip_id: str) -> dict:
    matches = [r for r in rows if r["trip_id"] == trip_id]
    assert len(matches) == 1, f"expected exactly one row for trip {trip_id}, got {len(matches)}"
    return matches[0]


def test_fixture_has_8_predictions_and_7_schedules():
    # Sanity check on the fixture itself, so a future edit that breaks the
    # intended coverage (6 mapped + 2 skipped) fails loudly here first.
    assert len(FIXTURE["data"]) == 8
    assert len(FIXTURE["included"]) == 7


def test_map_rows_skips_predictions_with_no_arrival_and_no_departure():
    rows = _rows()
    # 8 predictions in, 2 are CANCELLED with arrival_time=None and
    # departure_time=None (no temporal signal at all) -> skipped.
    assert len(rows) == 6
    trip_ids = {r["trip_id"] for r in rows}
    assert "77055343" not in trip_ids  # both skipped predictions share this trip_id


def test_map_rows_joins_scheduled_arrival_from_included_schedule():
    rows = _rows()
    row = _row_for_trip(rows, "76678380")
    assert row["route_id"] == "23"
    assert row["stop_id"] == "468"
    assert row["direction_id"] == 0
    assert row["vehicle_id"] == "y1789"
    # schedule-76678380-468-18's arrival_time in the fixture is 21:38:00-04:00
    assert row["scheduled_arrival"] == dt.datetime.fromisoformat("2026-08-12T21:38:00-04:00")
    # the prediction's own arrival_time is 21:35:03-04:00
    assert row["predicted_arrival"] == dt.datetime.fromisoformat("2026-08-12T21:35:03-04:00")


def test_map_rows_scheduled_arrival_none_when_schedule_relationship_missing():
    rows = _rows()
    # prediction-77140530-OL1-23391-26-39 has relationships.schedule.data = null
    # (an ADDED trip with no scheduled counterpart) but does have arrival_time,
    # so it's not skipped -- scheduled_arrival must be None, predicted_arrival
    # must still be populated.
    row = _row_for_trip(rows, "77140530-OL1")
    assert row["route_id"] == "39"  # guard against fixture drift
    assert row["scheduled_arrival"] is None
    assert row["predicted_arrival"] == dt.datetime.fromisoformat("2026-08-12T21:37:30-04:00")


def test_map_rows_none_when_schedule_present_but_its_arrival_time_is_null():
    rows = _rows()
    # prediction-76678239-64-1-1: schedule IS present (schedule-76678239-64-1)
    # but that schedule's own arrival_time is null (origin stop, departure
    # only) -- scheduled_arrival is None, and since the prediction's own
    # arrival_time is also null, predicted_arrival is None too. vehicle
    # relationship is null on this one, so vehicle_id is None.
    row = _row_for_trip(rows, "76678239")
    assert row["scheduled_arrival"] is None
    assert row["predicted_arrival"] is None
    assert row["vehicle_id"] is None


def test_map_rows_service_date_falls_back_to_america_new_york_local_date():
    rows = _rows()
    # None of the fixture's included schedules carry a service_date field
    # (confirmed against a live fetch: include=schedule,vehicle never
    # returns one), so every row falls back to polled_at's NY local date.
    # POLLED_AT is 2026-08-13T02:30:00Z = 2026-08-12 22:30 America/New_York.
    assert {r["service_date"] for r in rows} == {dt.date(2026, 8, 12)}


def test_map_rows_carries_polled_at_through_unchanged():
    rows = _rows()
    assert all(r["polled_at"] == POLLED_AT for r in rows)


def test_map_rows_direction_id_comes_from_prediction_not_schedule():
    rows = _rows()
    row = _row_for_trip(rows, "77140530-OL1")
    assert row["direction_id"] == 1


def test_map_rows_is_pure_does_not_mutate_input_payload():
    import copy

    before = copy.deepcopy(FIXTURE)
    mbta.map_rows(FIXTURE, POLLED_AT)
    assert FIXTURE == before


def test_map_rows_empty_payload_returns_empty_list():
    assert mbta.map_rows({"data": [], "included": []}, POLLED_AT) == []


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return _FakeResponse(self._payload)


def test_fetch_predictions_builds_expected_request_without_api_key():
    session = _FakeSession({"data": [], "included": []})
    result = mbta.fetch_predictions(["1", "15", "22"], session, api_key=None)

    assert result == {"data": [], "included": []}
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://api-v3.mbta.com/predictions"
    assert call["params"]["filter[route]"] == "1,15,22"
    assert call["params"]["include"] == "schedule,vehicle"
    assert call["timeout"] == 20
    assert call["headers"] == {}


def test_fetch_predictions_sends_x_api_key_header_when_set():
    session = _FakeSession({"data": [], "included": []})
    mbta.fetch_predictions(["1"], session, api_key="secret-key")

    assert session.calls[0]["headers"] == {"x-api-key": "secret-key"}


def test_batched_splits_13_routes_into_5_5_3():
    routes = ["1", "15", "22", "23", "28", "32", "39", "57", "66", "71", "73", "77", "111"]
    batches = mbta.batched(routes, (5, 5, 3))
    assert [len(b) for b in batches] == [5, 5, 3]
    assert batches[0] == ["1", "15", "22", "23", "28"]
    assert batches[1] == ["32", "39", "57", "66", "71"]
    assert batches[2] == ["73", "77", "111"]
