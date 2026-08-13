"""MBTA V3 /predictions client + pure payload -> stop_events row mapping."""

from __future__ import annotations

import datetime as dt
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import requests

from pulse import contract

PREDICTIONS_URL = "https://api-v3.mbta.com/predictions"
_EASTERN = ZoneInfo("America/New_York")

# Runaway guard for fetch_predictions' pagination follow. Observed 2026-08-12
# (Task 2 live smoke): a single 5-route batch (1,15,22,23,28) returned 1667
# predictions against page[limit]=1000 -- 2 pages. This caps how many GETs
# one batch can spend chasing links.next before giving up and logging what
# got cut off, rather than looping forever against a pathological response.
MAX_PAGES_PER_BATCH = 5

# The five stop_events columns that are NOT NULL per migrations/001_stop_events.sql.
# Exposed so pulse/poll.py can pre-filter rows that would otherwise abort a whole
# upsert batch on a NotNullViolation.
REQUIRED_FIELDS = ("route_id", "direction_id", "stop_id", "trip_id", "service_date")

# M2 must-address item 8: Retry-After-aware backoff. Scoped per HTTP GET (not
# per batch as a whole) -- a batch can span several GETs via pagination, and
# retrying each request independently means a transient failure on page 2
# doesn't force re-fetching an already-succeeded page 1. This is a deliberate
# reading of "per batch": the retry budget lives inside fetch_predictions, so
# every GET issued while fetching one batch gets its own up-to-2-retries
# allowance, still bounded by the same MAX_PAGES_PER_BATCH page cap.
MAX_RETRIES_PER_REQUEST = 2  # up to 3 total attempts per GET
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
DEFAULT_BACKOFF_SECONDS = 5.0
# Retry-After is honored but clamped: an API returning `Retry-After: 3600`
# must not be allowed to make the backoff itself blow the poll cycle's 240s
# deadline (pulse.poll.CYCLE_DEADLINE_SECONDS). 30s keeps 2 retries within a
# single batch's realistic budget (2 * 30s = 60s worst case) while still
# comfortably exceeding the unclamped 5s default for a server that's asking
# for real breathing room.
MAX_HONORED_RETRY_AFTER_SECONDS = 30.0


def _retry_after_seconds(response: requests.Response) -> float:
    """Parse Retry-After from a response, clamped to
    MAX_HONORED_RETRY_AFTER_SECONDS. Falls back to DEFAULT_BACKOFF_SECONDS
    when the header is absent or not a plain integer-seconds value (the
    HTTP-date form is legal per RFC 9110 but MBTA has never been observed to
    send it, so it isn't worth the parsing surface)."""
    header = response.headers.get("Retry-After")
    if header is None:
        return DEFAULT_BACKOFF_SECONDS
    try:
        seconds = float(header)
    except ValueError:
        return DEFAULT_BACKOFF_SECONDS
    return max(0.0, min(seconds, MAX_HONORED_RETRY_AFTER_SECONDS))


def _get_with_retry(
    session: requests.Session,
    url: str,
    params: Mapping[str, str] | None,
    headers: Mapping[str, str],
    timeout: int,
    sleep: Callable[[float], None],
) -> requests.Response:
    """One GET, retried up to MAX_RETRIES_PER_REQUEST times (3 attempts
    total) on a timeout or a retryable status (429/5xx). Non-retryable
    statuses (4xx other than 429) raise immediately via raise_for_status --
    retrying a malformed request would just fail the same way three times."""
    last_exc: Exception | None = None
    for attempt in range(MAX_RETRIES_PER_REQUEST + 1):
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.Timeout as exc:
            last_exc = exc
            backoff = DEFAULT_BACKOFF_SECONDS
        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status not in RETRYABLE_STATUS_CODES:
                raise
            last_exc = exc
            backoff = _retry_after_seconds(exc.response)

        if attempt < MAX_RETRIES_PER_REQUEST:
            print(
                f"pulse.mbta: retryable failure ({last_exc}) on attempt {attempt + 1}/"
                f"{MAX_RETRIES_PER_REQUEST + 1}, backing off {backoff:.1f}s",
                file=sys.stderr,
            )
            sleep(backoff)

    assert last_exc is not None  # loop always sets it before falling through
    raise last_exc


def fetch_predictions(
    route_ids: Sequence[str],
    session: requests.Session,
    api_key: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    contract_spec: Mapping[str, Any] | None = None,
) -> tuple[dict, int, bool, list[str]]:
    """GET /predictions for the given routes, with included schedule + vehicle.

    Follows the JSON:API `links.next` URL until it's absent (page exhausted),
    merging `data` and `included` across pages into a single payload shaped
    like a one-page response. Real batches routinely exceed page[limit]=1000
    -- 5 high-frequency routes alone returned 1667 predictions in one observed
    fetch, silently truncated to 1000 without this follow -- so a single page
    is not a safe assumption. Capped at MAX_PAGES_PER_BATCH pages; hitting the
    cap logs a warning to stderr and returns what was gathered so far rather
    than looping unbounded.

    Each GET is retried up to MAX_RETRIES_PER_REQUEST times on a timeout or a
    429/5xx response, honoring Retry-After when present (clamped -- see
    MAX_HONORED_RETRY_AFTER_SECONDS) else backing off DEFAULT_BACKOFF_SECONDS.
    `sleep` is injectable so tests don't pay real wall-clock backoff time.

    Each raw page (one HTTP response's parsed JSON, before it's merged into
    the running total) is validated against
    contracts/mbta-predictions.v1.json via pulse.contract.validate_page --
    M2 must-address item 4. A page with any violation is NOT merged into the
    returned payload (its data is dropped entirely, page granularity, not
    row granularity), but pagination still continues to the next page if
    links.next is present: the envelope fields (`links`) are a separate
    concern from the resource-shape violations this contract checks, so a
    malformed `data[3].attributes.direction_id` on page 2 shouldn't stop
    page 3 (or page 1's already-validated data) from being used. Every
    violation message found across every page is collected and returned --
    pulse.poll folds them into that batch's poll_runs error, naming the
    violated clause rather than surfacing only as a silently rising
    skipped/null count.

    Returns (merged_payload, pages_fetched, hit_page_cap,
    contract_violations). pages_fetched is the number of GETs actually
    completed (1 when links.next is absent on the first page, up to
    MAX_PAGES_PER_BATCH when the cap is hit) -- pulse.poll sums this across a
    cycle's batches into poll_runs.pages_fetched. hit_page_cap is True
    exactly when MAX_PAGES_PER_BATCH was reached with links.next still
    present (data was truncated) -- pulse.poll surfaces this as an explicit
    poll_runs.error entry rather than only the stderr line, so a cap hit
    doesn't silently understate routes_ok coverage in the ledger.
    If a request partway through pagination raises after exhausting retries
    (network error, non-2xx status), the exception propagates before any
    count is returned, so a batch that fails mid-pagination contributes 0 to
    that cycle's pages_fetched even though it may have completed one or more
    GETs -- by design, not a bug: the caller already logs and counts that
    batch as failed.

    Caveat, confirmed empirically (2026-08-13 live smoke): `/predictions` is a
    live, mutating collection, and this is offset pagination (`page[offset]`),
    not a cursor over a fixed snapshot. When the underlying set changes
    between the page-N and page-N+1 requests, items near the offset boundary
    can shift across it -- observed as ~5 predictions appearing at the tail of
    page 1 (offsets ~994-998) and again at the head of page 2 (offsets 1-5) in
    one real fetch. Downstream this is harmless (`stop_events`'s unique key
    absorbs the duplicate as an `ON CONFLICT DO NOTHING`), but the symmetric
    failure mode -- an item shifting past a boundary and being skipped by
    both pages -- is possible in principle and not distinguishable from the
    DB after the fact. Per-cycle completeness is a strong improvement over
    the unpaginated version but not a guaranteed-exact snapshot.

    Anonymous MBTA V3 access is fine at this volume; when api_key is set,
    it's sent as the x-api-key header (raises the rate limit).
    """
    params: Mapping[str, str] | None = {
        "filter[route]": ",".join(route_ids),
        "include": "schedule,vehicle",
        "page[limit]": "1000",
    }
    headers = {"x-api-key": api_key} if api_key else {}

    # Loaded once per call (not once per page) -- the contract file is tiny
    # and doesn't change mid-fetch, so re-reading it for every page in a
    # pagination chain would be pure overhead.
    loaded_contract = contract_spec if contract_spec is not None else contract.load_contract()

    all_data: list[Any] = []
    all_included: list[Any] = []
    url = PREDICTIONS_URL
    hit_page_cap = False
    contract_violations: list[str] = []

    for page in range(1, MAX_PAGES_PER_BATCH + 1):
        resp = _get_with_retry(session, url, params, headers, timeout=20, sleep=sleep)
        payload = resp.json()

        page_violations = contract.validate_page(payload, loaded_contract)
        if page_violations:
            contract_violations.extend(f"page {page}: {v}" for v in page_violations)
            print(
                f"pulse.mbta: contract violation on page {page} fetching routes "
                f"{list(route_ids)}: {'; '.join(page_violations)}",
                file=sys.stderr,
            )
        else:
            all_data.extend(payload.get("data") or [])
            all_included.extend(payload.get("included") or [])

        next_url = (payload.get("links") or {}).get("next")
        if not next_url:
            break
        if page == MAX_PAGES_PER_BATCH:
            hit_page_cap = True
            print(
                f"pulse.mbta: hit {MAX_PAGES_PER_BATCH}-page cap fetching routes "
                f"{list(route_ids)}; links.next still present, remaining data dropped",
                file=sys.stderr,
            )
            break
        # links.next from MBTA is already a complete URL (query string
        # included, e.g. ...&page[offset]=1000) -- no separate params on the
        # follow-up request.
        url = next_url
        params = None

    return {"data": all_data, "included": all_included}, page, hit_page_cap, contract_violations


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
