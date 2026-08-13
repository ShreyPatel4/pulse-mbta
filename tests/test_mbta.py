"""Tests for pulse.mbta: pure map_rows over a real, stripped fixture payload,
plus a request-shape check for fetch_predictions against a fake session
(no network -- fetch_predictions itself is I/O, not something map_rows
tests need to exercise)."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
import requests

from pulse import mbta

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE = json.loads((FIXTURES_DIR / "predictions_sample.json").read_text())
PAGE1 = json.loads((FIXTURES_DIR / "predictions_sample_page1.json").read_text())
PAGE2 = json.loads((FIXTURES_DIR / "predictions_sample_page2.json").read_text())

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
    result, pages, hit_cap, violations = mbta.fetch_predictions(["1", "15", "22"], session, api_key=None)

    assert result == {"data": [], "included": []}
    assert pages == 1
    assert hit_cap is False
    assert violations == []
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


class _FakeMultiPageSession:
    """Returns one payload per call, in order -- models following links.next
    across successive GETs (real fetch_predictions calls, one per page)."""

    def __init__(self, payloads):
        self._payloads = list(payloads)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        return _FakeResponse(self._payloads.pop(0))


def test_fetch_predictions_follows_links_next_and_merges_pages():
    # PAGE1/PAGE2 are a real 8-prediction fixture split 5/3 with a
    # links.next chain -- derived from predictions_sample.json, modeling the
    # observed real-world case (5-route batch, page[limit]=1000, 1667 rows ->
    # 2 pages) that silently truncated data before this fix.
    session = _FakeMultiPageSession([PAGE1, PAGE2])
    result, pages, hit_cap, violations = mbta.fetch_predictions(["1", "15", "22", "23", "28"], session, api_key=None)

    assert violations == []  # both pages are real, contract-valid fixtures
    assert len(result["data"]) == len(PAGE1["data"]) + len(PAGE2["data"]) == 8
    assert len(result["included"]) == len(PAGE1["included"]) + len(PAGE2["included"]) == 7
    # Order preserved, page 1 first.
    assert result["data"][0]["id"] == PAGE1["data"][0]["id"]
    assert result["data"][-1]["id"] == PAGE2["data"][-1]["id"]

    assert pages == 2
    assert hit_cap is False
    assert len(session.calls) == 2
    first, second = session.calls
    assert first["url"] == mbta.PREDICTIONS_URL
    assert first["params"]["filter[route]"] == "1,15,22,23,28"
    # Second GET hits links.next verbatim -- it's already a complete URL
    # (query string included), so no separate params are sent.
    assert second["url"] == PAGE1["links"]["next"]
    assert second["params"] is None


def test_fetch_predictions_merged_payload_maps_the_same_as_the_unpaginated_fixture():
    # The merged 2-page payload should be indistinguishable from a single-page
    # response to map_rows -- same 6 non-skipped rows as the original fixture.
    session = _FakeMultiPageSession([PAGE1, PAGE2])
    merged, _pages, _hit_cap, _violations = mbta.fetch_predictions(["1", "15", "22", "23", "28"], session, api_key=None)

    polled_at = dt.datetime(2026, 8, 13, 2, 30, tzinfo=dt.timezone.utc)
    assert mbta.map_rows(merged, polled_at) == mbta.map_rows(FIXTURE, polled_at)


def test_fetch_predictions_stops_when_links_next_is_absent():
    # A single-page response (no links.next) should not trigger a second GET.
    session = _FakeMultiPageSession([{"data": [], "included": [], "links": {}}])
    _result, pages, hit_cap, _violations = mbta.fetch_predictions(["1"], session, api_key=None)
    assert len(session.calls) == 1
    assert pages == 1
    assert hit_cap is False


def test_fetch_predictions_drops_only_the_violating_page_and_keeps_the_rest(capsys):
    # PAGE2's first prediction gets a mutated, contract-violating
    # direction_id -- PAGE1 (5 real, valid predictions) must still be used,
    # PAGE2's 3 must be dropped entirely, and the violation must name the
    # page and the clause.
    import copy

    bad_page2 = copy.deepcopy(PAGE2)
    bad_page2["data"][0]["attributes"]["direction_id"] = None

    session = _FakeMultiPageSession([PAGE1, bad_page2])
    result, pages, hit_cap, violations = mbta.fetch_predictions(["1", "15", "22", "23", "28"], session, api_key=None)

    assert len(result["data"]) == len(PAGE1["data"])  # PAGE2's 3 rows dropped
    assert result["data"] == PAGE1["data"]
    assert pages == 2  # both pages were still fetched -- pagination wasn't aborted
    assert hit_cap is False
    assert len(violations) == 1
    assert "page 2" in violations[0]
    assert "direction_id" in violations[0]
    assert "non-nullable" in violations[0]

    captured = capsys.readouterr()
    assert "contract violation" in captured.err
    assert "page 2" in captured.err


class _FakeInfiniteSession:
    """Always returns a payload whose links.next points at itself -- models a
    pathological/never-terminating pagination chain."""

    def __init__(self, payload):
        self._payload = payload
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        return _FakeResponse(self._payload)


def test_fetch_predictions_caps_at_5_pages_and_warns_on_stderr(capsys):
    payload = {
        # A contract-valid minimal prediction -- this test exercises the
        # pagination cap, not contract validation, so the fixture must pass
        # contract.validate_page cleanly or every page would get dropped as
        # a violation before the cap logic is even reached.
        "data": [
            {
                "id": "x",
                "type": "prediction",
                "attributes": {"direction_id": 0},
                "relationships": {
                    "route": {"data": {"id": "1"}},
                    "stop": {"data": {"id": "110"}},
                    "trip": {"data": {"id": "trip-1"}},
                },
            }
        ],
        "included": [],
        "links": {"next": "https://api-v3.mbta.com/predictions?page[offset]=999"},
    }
    session = _FakeInfiniteSession(payload)
    result, pages, hit_cap, violations = mbta.fetch_predictions(["1"], session, api_key=None)
    assert violations == []

    assert len(session.calls) == mbta.MAX_PAGES_PER_BATCH == 5
    assert len(result["data"]) == 5  # one row merged per page fetched
    assert pages == mbta.MAX_PAGES_PER_BATCH == 5
    assert hit_cap is True

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "cap" in captured.err.lower()
    assert "['1']" in captured.err


class _RetryableResponse:
    """A response whose raise_for_status() raises HTTPError with the given
    status/headers attached -- exercises _get_with_retry's 429/5xx path
    without a real HTTP round-trip."""

    def __init__(self, status_code, headers=None, payload=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._payload = payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        return self._payload


class _ScriptedSession:
    """Returns responses (or raises exceptions) from a scripted list, one per
    .get() call -- models a request failing N times before succeeding."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(url)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_fetch_predictions_retries_on_429_honoring_retry_after_header():
    ok_payload = {"data": [], "included": [], "links": {}}
    session = _ScriptedSession(
        [_RetryableResponse(429, headers={"Retry-After": "2"}), _RetryableResponse(200, payload=ok_payload)]
    )
    sleeps: list[float] = []
    result, pages, hit_cap, violations = mbta.fetch_predictions(["1"], session, api_key=None, sleep=sleeps.append)

    assert result == {"data": [], "included": []}
    assert pages == 1
    assert hit_cap is False
    assert len(session.calls) == 2
    assert sleeps == [2.0]


def test_fetch_predictions_retries_on_5xx_with_default_backoff_when_no_header():
    ok_payload = {"data": [], "included": [], "links": {}}
    session = _ScriptedSession([_RetryableResponse(503), _RetryableResponse(200, payload=ok_payload)])
    sleeps: list[float] = []
    result, _pages, _hit_cap, _violations = mbta.fetch_predictions(["1"], session, api_key=None, sleep=sleeps.append)

    assert result == {"data": [], "included": []}
    assert sleeps == [mbta.DEFAULT_BACKOFF_SECONDS]


def test_fetch_predictions_clamps_retry_after_above_max():
    ok_payload = {"data": [], "included": [], "links": {}}
    session = _ScriptedSession(
        [_RetryableResponse(429, headers={"Retry-After": "3600"}), _RetryableResponse(200, payload=ok_payload)]
    )
    sleeps: list[float] = []
    mbta.fetch_predictions(["1"], session, api_key=None, sleep=sleeps.append)

    assert sleeps == [mbta.MAX_HONORED_RETRY_AFTER_SECONDS]


def test_fetch_predictions_retries_on_timeout_then_succeeds():
    ok_payload = {"data": [], "included": [], "links": {}}
    session = _ScriptedSession(
        [requests.exceptions.Timeout("read timed out (fabricated)"), _RetryableResponse(200, payload=ok_payload)]
    )
    sleeps: list[float] = []
    result, _pages, _hit_cap, _violations = mbta.fetch_predictions(["1"], session, api_key=None, sleep=sleeps.append)

    assert result == {"data": [], "included": []}
    assert sleeps == [mbta.DEFAULT_BACKOFF_SECONDS]


def test_fetch_predictions_gives_up_after_max_retries():
    session = _ScriptedSession([_RetryableResponse(503), _RetryableResponse(503), _RetryableResponse(503)])
    sleeps: list[float] = []

    with pytest.raises(requests.exceptions.HTTPError):
        mbta.fetch_predictions(["1"], session, api_key=None, sleep=sleeps.append)

    assert len(session.calls) == mbta.MAX_RETRIES_PER_REQUEST + 1 == 3
    assert len(sleeps) == mbta.MAX_RETRIES_PER_REQUEST == 2


def test_fetch_predictions_does_not_retry_non_retryable_4xx():
    session = _ScriptedSession([_RetryableResponse(404)])
    sleeps: list[float] = []

    with pytest.raises(requests.exceptions.HTTPError):
        mbta.fetch_predictions(["1"], session, api_key=None, sleep=sleeps.append)

    assert len(session.calls) == 1
    assert sleeps == []


def test_batched_splits_13_routes_into_5_5_3():
    routes = ["1", "15", "22", "23", "28", "32", "39", "57", "66", "71", "73", "77", "111"]
    batches = mbta.batched(routes, (5, 5, 3))
    assert [len(b) for b in batches] == [5, 5, 3]
    assert batches[0] == ["1", "15", "22", "23", "28"]
    assert batches[1] == ["32", "39", "57", "66", "71"]
    assert batches[2] == ["73", "77", "111"]
