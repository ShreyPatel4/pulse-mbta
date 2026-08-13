"""Tests for the pure parts of pulse.poll: poll_runs row construction and
run_cycle's failure-path aggregation (pages_fetched, batches_ok, error text).
No real Postgres or network needed -- run_cycle only touches `conn` inside
the try block after a successful fetch/map, so a batch that fails at fetch
never reaches it, and _build_run_row does no I/O at all."""

from __future__ import annotations

import datetime as dt

from pulse import poll


def test_build_run_row_shapes_all_fields():
    started_at = dt.datetime(2026, 8, 13, 2, 0, 0, tzinfo=dt.timezone.utc)
    polled_at = dt.datetime(2026, 8, 13, 2, 0, 1, tzinfo=dt.timezone.utc)
    finished_at = dt.datetime(2026, 8, 13, 2, 0, 5, tzinfo=dt.timezone.utc)

    row = poll._build_run_row(
        polled_at=polled_at,
        started_at=started_at,
        finished_at=finished_at,
        batches_ok=3,
        batches_total=3,
        pages_fetched=4,
        rows=100,
        inserted=99,
        skipped=1,
        error=None,
    )

    assert row == {
        "polled_at": polled_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "batches_ok": 3,
        "batches_total": 3,
        "pages_fetched": 4,
        "rows": 100,
        "inserted": 99,
        "skipped": 1,
        "error": None,
    }


def test_build_run_row_on_failure_path_carries_fabricated_exception_text():
    # Mirrors main()'s db.connect-exhausted branch: no cycle ever ran, so
    # every count is zero and polled_at falls back to started_at.
    started_at = dt.datetime(2026, 8, 13, 2, 0, 0, tzinfo=dt.timezone.utc)
    finished_at = dt.datetime(2026, 8, 13, 2, 0, 6, tzinfo=dt.timezone.utc)
    exc = ConnectionError("connection refused (fabricated for this test)")

    row = poll._build_run_row(
        polled_at=started_at,
        started_at=started_at,
        finished_at=finished_at,
        batches_ok=0,
        batches_total=3,
        pages_fetched=0,
        rows=0,
        inserted=0,
        skipped=0,
        error=f"db.connect failed after {poll.DB_CONNECT_ATTEMPTS} attempts: {exc}",
    )

    assert row["polled_at"] == started_at
    assert row["batches_ok"] == 0
    assert row["rows"] == 0
    assert row["inserted"] == 0
    assert row["skipped"] == 0
    assert "connection refused (fabricated for this test)" in row["error"]
    assert "db.connect failed after 3 attempts" in row["error"]


class _RaisingSession:
    """A fake requests.Session whose .get always raises -- models every
    batch's fetch_predictions call failing (API down, DNS failure, etc)."""

    def __init__(self, exc: Exception):
        self._exc = exc

    def get(self, *args, **kwargs):
        raise self._exc


def test_run_cycle_all_batches_failing_zeroes_counts_and_collects_errors(monkeypatch, capsys):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)
    session = _RaisingSession(RuntimeError("network down (fabricated for this test)"))

    # conn is never touched on this path: run_cycle only reaches
    # conn.transaction() after a successful fetch + map, and every fetch
    # raises here -- so a real psycopg.Connection isn't needed to exercise
    # this failure path. ensure_partitions_fn is stubbed to a no-op for the
    # same reason (conn=None can't back a real pulse.partitions call).
    result = poll.run_cycle(
        conn=None, session=session, api_key=None, ensure_partitions_fn=lambda _conn: []
    )  # type: ignore[arg-type]

    assert result.batches_total == 3
    assert result.batches_ok == 0
    assert result.routes_ok == 0
    assert result.pages_fetched == 0
    assert result.rows == 0
    assert result.inserted == 0
    assert result.skipped == 0
    assert len(result.errors) == 3
    assert result.error_text is not None
    assert result.error_text.count("network down (fabricated for this test)") == 3

    captured = capsys.readouterr()
    assert captured.err.count("network down (fabricated for this test)") == 3


def test_run_cycle_success_has_no_error_text(monkeypatch):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)

    class _EmptySession:
        def get(self, *args, **kwargs):
            class _Resp:
                def raise_for_status(self):
                    pass

                def json(self):
                    return {"data": [], "included": []}

            return _Resp()

    class _NoopTransactionConn:
        def transaction(self):
            import contextlib

            return contextlib.nullcontext()

    result = poll.run_cycle(
        conn=_NoopTransactionConn(),  # type: ignore[arg-type]
        session=_EmptySession(),
        api_key=None,
        ensure_partitions_fn=lambda _conn: [],
    )

    assert result.batches_ok == 3
    assert result.batches_total == 3
    assert result.pages_fetched == 3  # one page per batch, 3 batches
    assert result.errors == []
    assert result.error_text is None


def test_summary_line_includes_pages_fetched():
    result = poll.CycleResult(
        polled_at=dt.datetime(2026, 8, 13, 2, 0, tzinfo=dt.timezone.utc),
        rows=10,
        inserted=10,
        skipped=0,
        routes_ok=13,
        routes_total=13,
        batches_ok=3,
        batches_total=3,
        pages_fetched=4,
        errors=[],
    )
    assert "pages=4" in result.summary_line


class _NoopTransactionConn:
    def transaction(self):
        import contextlib

        return contextlib.nullcontext()


class _EmptyPageSession:
    """A single-page, no-data response for every batch -- used where the
    test only cares about cycle-level control flow (deadline, cap-hit), not
    row mapping."""

    def get(self, url, params=None, headers=None, timeout=None):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [], "included": []}

        return _Resp()


def test_run_cycle_aborts_cleanly_when_deadline_exceeded_before_a_batch(monkeypatch):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)

    # 1st call: cycle_start = 0.0. 2nd call: elapsed check before batch 0 =
    # 0.0 (proceeds). 3rd call: elapsed check before batch 1 = 999.0 (well
    # past the 240s deadline) -- models real time passing while batch 0 ran.
    clock = iter([0.0, 0.0, 999.0])

    result = poll.run_cycle(
        conn=_NoopTransactionConn(),  # type: ignore[arg-type]
        session=_EmptyPageSession(),  # type: ignore[arg-type]
        api_key=None,
        deadline_seconds=240.0,
        monotonic=lambda: next(clock),
        ensure_partitions_fn=lambda _conn: [],
    )

    assert result.batches_ok == 1  # only batch 0 ran before the deadline hit
    assert result.batches_total == 3
    assert result.routes_ok == 5  # batch 0's route count (BATCH_SIZES = (5, 5, 3))
    assert result.error_text is not None
    assert "deadline" in result.error_text
    assert "240" in result.error_text


def test_run_cycle_deadline_not_exceeded_runs_all_batches(monkeypatch):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)

    result = poll.run_cycle(
        conn=_NoopTransactionConn(),  # type: ignore[arg-type]
        session=_EmptyPageSession(),  # type: ignore[arg-type]
        api_key=None,
        deadline_seconds=240.0,
        monotonic=lambda: 0.0,  # clock never advances -- deadline is never hit
        ensure_partitions_fn=lambda _conn: [],
    )

    assert result.batches_ok == 3
    assert result.error_text is None


class _PageCapSession:
    """Every GET returns a payload whose links.next is always present, so
    every batch runs the pagination loop out to MAX_PAGES_PER_BATCH and hits
    the cap -- models a batch whose result set is truncated."""

    def get(self, url, params=None, headers=None, timeout=None):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {"data": [], "included": [], "links": {"next": "https://api-v3.mbta.com/predictions?page[offset]=1"}}

        return _Resp()


def test_run_cycle_records_page_cap_hit_as_error_but_batch_still_counts_ok(monkeypatch):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)

    result = poll.run_cycle(
        conn=_NoopTransactionConn(),  # type: ignore[arg-type]
        session=_PageCapSession(),
        api_key=None,
        ensure_partitions_fn=lambda _conn: [],
    )

    assert result.batches_ok == 3  # a cap hit doesn't fail the batch
    assert result.pages_fetched == 15  # MAX_PAGES_PER_BATCH (5) * 3 batches
    assert result.error_text is not None
    assert result.error_text.count("page cap") == 3  # one explicit error per batch


class _ContractViolatingSession:
    """Every batch's single page has a prediction missing its required
    direction_id attribute -- models a live MBTA schema change surfacing as
    a contract violation, not a fetch/network failure."""

    def get(self, url, params=None, headers=None, timeout=None):
        class _Resp:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "data": [
                        {
                            "id": "x",
                            "type": "prediction",
                            "attributes": {},  # direction_id missing entirely
                            "relationships": {
                                "route": {"data": {"id": "1"}},
                                "stop": {"data": {"id": "110"}},
                                "trip": {"data": {"id": "trip-1"}},
                            },
                        }
                    ],
                    "included": [],
                }

        return _Resp()


def test_run_cycle_records_contract_violation_as_a_named_error_but_batch_still_counts_ok(monkeypatch):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)

    result = poll.run_cycle(
        conn=_NoopTransactionConn(),  # type: ignore[arg-type]
        session=_ContractViolatingSession(),
        api_key=None,
        ensure_partitions_fn=lambda _conn: [],
    )

    assert result.batches_ok == 3  # fetch itself succeeded -- a bad row, not a failed batch
    assert result.error_text is not None
    assert result.error_text.count("contract violation") == 3  # one per batch
    assert "direction_id" in result.error_text
    # The violating page's data was dropped inside fetch_predictions before
    # map_rows ever saw it, so nothing from it landed as a row.
    assert result.rows == 0


def test_run_cycle_calls_ensure_partitions_fn_exactly_once_with_the_conn(monkeypatch):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)
    sentinel_conn = object()
    calls = []

    def fake_ensure_partitions(conn):
        calls.append(conn)
        return []

    poll.run_cycle(
        conn=sentinel_conn,  # type: ignore[arg-type]
        session=_EmptyPageSession(),  # type: ignore[arg-type]
        api_key=None,
        sleep=lambda *_a, **_kw: None,
        ensure_partitions_fn=fake_ensure_partitions,
    )

    assert calls == [sentinel_conn]


def test_run_cycle_folds_ensure_partitions_failure_into_errors_but_batches_still_run(monkeypatch, capsys):
    monkeypatch.setattr(poll.time, "sleep", lambda *_a, **_kw: None)

    def failing_ensure_partitions(_conn):
        raise RuntimeError("lock_timeout exceeded (fabricated for this test)")

    result = poll.run_cycle(
        conn=_NoopTransactionConn(),  # type: ignore[arg-type]
        session=_EmptyPageSession(),  # type: ignore[arg-type]
        api_key=None,
        sleep=lambda *_a, **_kw: None,
        ensure_partitions_fn=failing_ensure_partitions,
    )

    # The partition-maintenance failure doesn't skip ingestion: every batch
    # still ran (stop_events_default is the backstop for an unprovisioned
    # month, not a reason to stop polling).
    assert result.batches_ok == 3
    assert result.error_text is not None
    assert "ensure_partitions failed" in result.error_text
    assert "lock_timeout exceeded (fabricated for this test)" in result.error_text

    captured = capsys.readouterr()
    assert "ensure_partitions failed" in captured.err
