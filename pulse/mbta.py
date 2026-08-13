"""MBTA V3 /predictions client + pure payload -> stop_events row mapping."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import requests

PREDICTIONS_URL = "https://api-v3.mbta.com/predictions"
_EASTERN = ZoneInfo("America/New_York")

# The five stop_events columns that are NOT NULL per migrations/001_stop_events.sql.
# Exposed so pulse/poll.py can pre-filter rows that would otherwise abort a whole
# upsert batch on a NotNullViolation.
REQUIRED_FIELDS = ("route_id", "direction_id", "stop_id", "trip_id", "service_date")


def fetch_predictions(
    route_ids: Sequence[str],
    session: requests.Session,
    api_key: str | None = None,
) -> dict:
    """GET /predictions for the given routes, with included schedule + vehicle.

    Anonymous MBTA V3 access is fine at this volume; when api_key is set,
    it's sent as the x-api-key header (raises the rate limit).
    """
    params = {
        "filter[route]": ",".join(route_ids),
        "include": "schedule,vehicle",
        "page[limit]": "1000",
    }
    headers = {"x-api-key": api_key} if api_key else {}
    resp = session.get(PREDICTIONS_URL, params=params, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def map_rows(payload: Mapping[str, Any], polled_at: dt.datetime) -> list[dict]:
    """Map a /predictions JSON:API payload to stop_events row dicts. Pure.

    - Joins each prediction to its included `schedule` resource via the
      prediction's `schedule` relationship id; scheduled_arrival is None when
      that schedule is missing (relationship null, or schedule not present in
      `included` -- e.g. schedule_relationship=ADDED trips carry no schedule).
    - direction_id comes from the prediction's own attributes (not the
      schedule's).
    - service_date: MBTA's schedule resource doesn't expose a service_date
      field under `include=schedule,vehicle` (only trip/stop/route/added_routes
      relationships + arrival/departure/stop_sequence/timepoint attributes --
      verified against a live fetch). This function still checks for one
      defensively (forward-compatible with a payload that did include it, e.g.
      via `include=schedule.service`), but as fetched today it is always
      absent, so service_date is always the America/New_York local date of
      polled_at.
    - Predictions lacking both arrival_time and departure_time carry no
      temporal signal at all (empirically: MBTA's schedule_relationship=
      CANCELLED trips) and are skipped entirely.
    - predicted_arrival and scheduled_arrival are each read strictly from
      their source's arrival_time (not backfilled from departure_time) --
      a terminal/origin stop that only has a departure_time still produces a
      row (it clears the skip rule), just with predicted_arrival/
      scheduled_arrival left None.
    """
    included = payload.get("included") or []
    schedules_by_id = {
        item["id"]: item for item in included if item.get("type") == "schedule"
    }

    fallback_service_date = polled_at.astimezone(_EASTERN).date()

    rows: list[dict] = []
    for pred in payload.get("data") or []:
        attrs = pred.get("attributes") or {}
        arrival_time = attrs.get("arrival_time")
        departure_time = attrs.get("departure_time")
        if arrival_time is None and departure_time is None:
            continue

        rel = pred.get("relationships") or {}
        schedule_id = _rel_id(rel, "schedule")
        schedule = schedules_by_id.get(schedule_id) if schedule_id else None

        service_date = fallback_service_date
        scheduled_arrival = None
        if schedule is not None:
            sched_attrs = schedule.get("attributes") or {}
            scheduled_arrival = _parse_dt(sched_attrs.get("arrival_time"))
            sched_service_date = sched_attrs.get("service_date")
            if sched_service_date:
                service_date = dt.date.fromisoformat(sched_service_date)

        rows.append(
            {
                "route_id": _rel_id(rel, "route"),
                "direction_id": attrs.get("direction_id"),
                "stop_id": _rel_id(rel, "stop"),
                "trip_id": _rel_id(rel, "trip"),
                "vehicle_id": _rel_id(rel, "vehicle"),
                "service_date": service_date,
                "scheduled_arrival": scheduled_arrival,
                "predicted_arrival": _parse_dt(arrival_time),
                "status": attrs.get("status"),
                "polled_at": polled_at,
            }
        )
    return rows


def _rel_id(relationships: Mapping[str, Any], name: str) -> str | None:
    rel = relationships.get(name) or {}
    data = rel.get("data")
    return data["id"] if data else None


def _parse_dt(value: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(value) if value else None


def batched(route_ids: Iterable[str], sizes: Sequence[int]) -> list[list[str]]:
    """Split route_ids into consecutive chunks of the given sizes."""
    route_ids = list(route_ids)
    batches = []
    start = 0
    for size in sizes:
        batches.append(route_ids[start : start + size])
        start += size
    return batches
