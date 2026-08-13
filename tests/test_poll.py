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
    # this failure path.
    result = poll.run_cycle(conn=None, session=session, api_key=None)  # type: ignore[arg-type]

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

    result = poll.run_cycle(conn=_NoopTransactionConn(), session=_EmptySession(), api_key=None)  # type: ignore[arg-type]

    assert result.batches_ok == 3
    assert result.batches_total == 3
    assert result.pages_fetched == 3  # one page per batch, 3 batches
    assert result.errors == []
    assert result.error_text is None
