"""One MBTA /predictions -> stop_events polling cycle.

Usage: uv run python -m pulse.poll

Always exits 0. launchd's StartInterval fires this on a fixed schedule --
there is no KeepAlive on the plist, so launchd is not watching for crashes to
relaunch from. A failed cycle logs to stderr, records what it can in the
poll_runs ledger (see _build_run_row below), and gets out of the way rather
than raising, so the next StartInterval fire (not a crash-triggered restart)
is what picks the poller back up.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import os
import sys
import time

import psycopg
import requests

from pulse import db, mbta

ROUTE_IDS = ["1", "15", "22", "23", "28", "32", "39", "57", "66", "71", "73", "77", "111"]
BATCH_SIZES = (5, 5, 3)
BATCH_SLEEP_SECONDS = 1.5
DB_CONNECT_ATTEMPTS = 3
DB_CONNECT_RETRY_SECONDS = 2.0


def _prefilter(rows: list[dict]) -> tuple[list[dict], int]:
    """Drop rows that would violate a stop_events NOT NULL column.

    pulse.db.connect() runs autocommit=True, so a NotNullViolation partway
    through executemany would otherwise commit the rows ahead of the bad one
    and silently drop the rest of the batch. Filtering here -- paired with
    wrapping each batch's upsert in an explicit conn.transaction() below --
    means one malformed row can't take out (or partially commit) its whole
    batch: it's counted as skipped and every valid row in the batch still
    lands atomically.
    """
    valid = []
    skipped = 0
    for row in rows:
        if any(row.get(field) is None for field in mbta.REQUIRED_FIELDS):
            skipped += 1
            continue
        valid.append(row)
    return valid, skipped


@dataclasses.dataclass
class CycleResult:
    """Everything one poll cycle produced: both the human summary line and
    the fields the poll_runs ledger row needs, so main() never has to
    re-derive anything run_cycle already computed."""

    polled_at: dt.datetime
    rows: int
    inserted: int
    skipped: int
    routes_ok: int
    routes_total: int
    batches_ok: int
    batches_total: int
    pages_fetched: int
    errors: list[str]

    @property
    def summary_line(self) -> str:
        return (
            f"polled_at={self.polled_at.isoformat()} rows={self.rows} inserted={self.inserted} "
            f"routes_ok={self.routes_ok}/{self.routes_total} skipped={self.skipped}"
        )

    @property
    def error_text(self) -> str | None:
        """Joined batch-failure messages, or None when every batch in the
        cycle succeeded. This is what lands in poll_runs.error -- a cycle can
        have batches_ok < batches_total (some routes failed) without the
        whole cycle being a "total failure"; the ledger row still gets
        written either way (see main())."""
        return "; ".join(self.errors) if self.errors else None


def run_cycle(conn: psycopg.Connection, session: requests.Session, api_key: str | None) -> CycleResult:
    """Run one poll cycle across all routes in 5/5/3 batches. Never raises:
    a batch failure is caught, logged to stderr, and folded into the
    returned CycleResult so the caller can still record a poll_runs row."""
    polled_at = dt.datetime.now(dt.timezone.utc)
    batches = mbta.batched(ROUTE_IDS, BATCH_SIZES)

    total_rows = 0
    total_inserted = 0
    total_skipped = 0
    routes_ok = 0
    batches_ok = 0
    pages_fetched = 0
    errors: list[str] = []

    for i, batch in enumerate(batches):
        try:
            payload, pages = mbta.fetch_predictions(batch, session, api_key=api_key)
            pages_fetched += pages
            rows = mbta.map_rows(payload, polled_at)
            valid_rows, skipped = _prefilter(rows)
            with conn.transaction():
                inserted = db.upsert_stop_events(conn, valid_rows)
            total_rows += len(valid_rows)
            total_inserted += inserted
            total_skipped += skipped
            routes_ok += len(batch)
            batches_ok += 1
        except Exception as exc:  # noqa: BLE001 - batch failure logs and the cycle continues
            message = f"batch {batch} failed: {exc}"
            print(f"pulse.poll: {message}", file=sys.stderr)
            errors.append(message)

        if i < len(batches) - 1:
            time.sleep(BATCH_SLEEP_SECONDS)

    return CycleResult(
        polled_at=polled_at,
        rows=total_rows,
        inserted=total_inserted,
        skipped=total_skipped,
        routes_ok=routes_ok,
        routes_total=len(ROUTE_IDS),
        batches_ok=batches_ok,
        batches_total=len(batches),
        pages_fetched=pages_fetched,
        errors=errors,
    )


def _connect_with_retry(
    attempts: int = DB_CONNECT_ATTEMPTS, delay_seconds: float = DB_CONNECT_RETRY_SECONDS
) -> psycopg.Connection:
    """Connect to Postgres, retrying a few times before giving up.

    Without this, a Postgres-not-yet-up login race (e.g. right after a
    reboot, before postgresql@16 has finished starting under brew services)
    presents as an invisible hole in ingestion -- exactly the kind of gap
    poll_runs exists to make visible. Retrying turns that race into a
    recorded slow cycle (finished_at - started_at reflects the wait) instead
    of a silently lost one.
    """
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return db.connect()
        except Exception as exc:  # noqa: BLE001 - retried below, or re-raised after the last attempt
            last_exc = exc
            print(f"pulse.poll: db.connect attempt {attempt}/{attempts} failed: {exc}", file=sys.stderr)
            if attempt < attempts:
                time.sleep(delay_seconds)
    assert last_exc is not None  # attempts >= 1 guarantees at least one iteration ran
    raise last_exc


def _build_run_row(
    *,
    polled_at: dt.datetime,
    started_at: dt.datetime,
    finished_at: dt.datetime,
    batches_ok: int,
    batches_total: int,
    pages_fetched: int,
    rows: int,
    inserted: int,
    skipped: int,
    error: str | None,
) -> dict:
    """Pure construction of one poll_runs row dict. No I/O -- isolated so the
    failure path (db.connect exhausted, or a cycle where every batch raised)
    is unit-testable with fabricated exceptions instead of a live Postgres.
    """
    return {
        "polled_at": polled_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "batches_ok": batches_ok,
        "batches_total": batches_total,
        "pages_fetched": pages_fetched,
        "rows": rows,
        "inserted": inserted,
        "skipped": skipped,
        "error": error,
    }


def _record_run(conn: psycopg.Connection, row: dict) -> None:
    """Best-effort ledger write. If the write itself fails, stderr is the
    fallback record and the cycle still exits 0 -- a ledger that can crash
    the poller would defeat its own purpose."""
    try:
        db.insert_poll_run(conn, row)
    except Exception as exc:  # noqa: BLE001 - stderr is the fallback record
        print(f"pulse.poll: failed to record poll_runs row: {exc}", file=sys.stderr)


def main() -> int:
    api_key = os.environ.get("MBTA_API_KEY") or None
    started_at = dt.datetime.now(dt.timezone.utc)
    batches_total = len(BATCH_SIZES)

    try:
        conn = _connect_with_retry()
    except Exception as exc:  # noqa: BLE001 - never crash-loop launchd
        finished_at = dt.datetime.now(dt.timezone.utc)
        print(f"pulse.poll: cycle failed to run: {exc}", file=sys.stderr)
        # polled_at falls back to started_at: no cycle ever ran, so there is
        # no snapshot timestamp to tag stop_events rows with, but the ledger
        # still needs a primary key for this cycle.
        row = _build_run_row(
            polled_at=started_at,
            started_at=started_at,
            finished_at=finished_at,
            batches_ok=0,
            batches_total=batches_total,
            pages_fetched=0,
            rows=0,
            inserted=0,
            skipped=0,
            error=f"db.connect failed after {DB_CONNECT_ATTEMPTS} attempts: {exc}",
        )
        # Best-effort, single attempt (not the retrying connect above -- we
        # just exhausted 3 tries against this same Postgres). If the db
        # itself is down, this also fails and the stderr line above is the
        # only record; exit 0 still stands.
        try:
            write_conn = db.connect()
            try:
                _record_run(write_conn, row)
            finally:
                write_conn.close()
        except Exception as write_exc:  # noqa: BLE001 - stderr is the fallback record
            print(f"pulse.poll: failed to record poll_runs failure row: {write_exc}", file=sys.stderr)
        return 0

    try:
        with requests.Session() as session:
            result = run_cycle(conn, session, api_key)
        finished_at = dt.datetime.now(dt.timezone.utc)
        row = _build_run_row(
            polled_at=result.polled_at,
            started_at=started_at,
            finished_at=finished_at,
            batches_ok=result.batches_ok,
            batches_total=result.batches_total,
            pages_fetched=result.pages_fetched,
            rows=result.rows,
            inserted=result.inserted,
            skipped=result.skipped,
            error=result.error_text,
        )
        _record_run(conn, row)
        print(result.summary_line)
    except Exception as exc:  # noqa: BLE001 - never crash-loop launchd
        finished_at = dt.datetime.now(dt.timezone.utc)
        print(f"pulse.poll: cycle failed to run: {exc}", file=sys.stderr)
        row = _build_run_row(
            polled_at=started_at,
            started_at=started_at,
            finished_at=finished_at,
            batches_ok=0,
            batches_total=batches_total,
            pages_fetched=0,
            rows=0,
            inserted=0,
            skipped=0,
            error=str(exc),
        )
        _record_run(conn, row)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
