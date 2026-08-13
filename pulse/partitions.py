"""Keep stop_events' monthly RANGE partitions provisioned ahead of need.

Called from pulse.poll's cycle (cheap no-op on every call once the target
months already have partitions -- see ensure_partitions' docstring for why
that matters) and from scripts/ensure-partitions.py for standalone/manual
use. migrations/004_partition_stop_events.sql provisions the current month +
next 2 once, at swap time; this module is what keeps that lookahead rolling
forward every cycle afterwards so stop_events_default (the DEFAULT partition
backstop) is only ever hit by a rare straggler, not steady-state traffic
crossing an un-provisioned month boundary.
"""

from __future__ import annotations

import datetime as dt

import psycopg
from psycopg import sql

DEFAULT_MONTHS_AHEAD = 2

_PARTITION_EXISTS_SQL = "SELECT 1 FROM pg_class WHERE relname = %s AND relkind = 'r'"


def _month_bounds(year: int, month: int) -> tuple[dt.date, dt.date]:
    start = dt.date(year, month, 1)
    if month == 12:
        end = dt.date(year + 1, 1, 1)
    else:
        end = dt.date(year, month + 1, 1)
    return start, end


def _target_months(today: dt.date, months_ahead: int) -> list[tuple[int, int]]:
    """(year, month) pairs for today's month through months_ahead months
    ahead, inclusive -- e.g. months_ahead=2 from August gives Aug/Sep/Oct.
    Pure and independently tested for the December -> January rollover."""
    months = []
    year, month = today.year, today.month
    for _ in range(months_ahead + 1):
        months.append((year, month))
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months


def partition_name(year: int, month: int) -> str:
    return f"stop_events_y{year:04d}m{month:02d}"


def ensure_partitions(
    conn: psycopg.Connection, months_ahead: int = DEFAULT_MONTHS_AHEAD, today: dt.date | None = None
) -> list[str]:
    """Create any missing stop_events_yYYYYmMM partitions for the target
    window. Returns the names of partitions actually created (empty when
    everything's already provisioned -- the steady-state case, called every
    ~66s poll cycle).

    The existence check is a plain SELECT against pg_class (ACCESS SHARE,
    never contends with the poller's concurrent INSERTs) run BEFORE any DDL.
    This matters: `CREATE TABLE ... PARTITION OF parent` -- even guarded by
    IF NOT EXISTS -- still has to open and lock the parent (ACCESS EXCLUSIVE)
    to modify its partition descriptor, so calling it unconditionally every
    cycle would mean briefly locking stop_events out from under the poller
    ~1,300 times a day for what should be a no-op the vast majority of the
    time. Skipping the DDL entirely once the cheap check confirms the
    partition exists means that lock is only ever requested on the rare
    cycle that actually needs to provision a new month.

    Raises on a real DDL failure (e.g. lock contention, permission error) --
    pulse.poll is the layer responsible for catching that and folding it
    into the cycle's error reporting rather than aborting ingestion; this
    function stays honest about failing loud rather than swallowing it here.
    """
    today = today or dt.date.today()
    created: list[str] = []

    for year, month in _target_months(today, months_ahead):
        name = partition_name(year, month)
        with conn.cursor() as cur:
            cur.execute(_PARTITION_EXISTS_SQL, (name,))
            if cur.fetchone() is not None:
                continue

        start, end = _month_bounds(year, month)
        with conn.cursor() as cur:
            # FOR VALUES FROM (...) TO (...) takes constant expressions, not
            # query parameters (Postgres's DDL grammar rejects a bind
            # placeholder there) -- sql.Literal renders a safely quoted SQL
            # literal into the composed statement instead.
            cur.execute(
                sql.SQL("CREATE TABLE IF NOT EXISTS {} PARTITION OF stop_events FOR VALUES FROM ({}) TO ({})").format(
                    sql.Identifier(name), sql.Literal(start), sql.Literal(end)
                )
            )
        created.append(name)

    return created
