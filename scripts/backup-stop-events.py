"""pg_dump a stop_events backup to data-backups/ (gitignored) before a
live migration touches the table -- see migrations/004_partition_stop_events.sql's
header for the operational sequence this is step 1 of.

Usage: uv run python scripts/backup-stop-events.py [--dsn DSN]

Records the pre-backup row count alongside the dump so a later count check
(scripts/backup-stop-events.py --verify, or a manual `select count(*) from
stop_events`) has something concrete to diff against. Exits non-zero if
pg_dump fails or the dump file ends up empty -- a backup that silently
didn't happen is worse than no backup, because it would be trusted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path

from psycopg.conninfo import conninfo_to_dict

from pulse import db

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKUP_DIR = REPO_ROOT / "data-backups"


def _dbname_and_args(dsn: str) -> tuple[str, list[str]]:
    """pg_dump takes connection pieces as flags, not a single DSN string, so
    this pulls dbname/host/port/user out of whatever DSN pulse.db would
    otherwise hand to psycopg.connect."""
    params = conninfo_to_dict(dsn)
    dbname = params.get("dbname")
    if not dbname:
        raise ValueError(f"dsn has no dbname: {dsn!r}")
    args = []
    if params.get("host"):
        args += ["--host", params["host"]]
    if params.get("port"):
        args += ["--port", str(params["port"])]
    if params.get("user"):
        args += ["--username", params["user"]]
    return dbname, args


def backup(dsn: str) -> Path:
    """Run pg_dump for stop_events only, custom format, to
    data-backups/stop_events-<UTC timestamp>.dump. Returns the output path.
    Raises CalledProcessError on a pg_dump failure, or RuntimeError if the
    dump file is empty (0 bytes) -- both are treated as "the backup did not
    happen", not swallowed."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    dbname, conn_args = _dbname_and_args(dsn)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BACKUP_DIR / f"stop_events-{timestamp}.dump"

    cmd = [
        "pg_dump",
        *conn_args,
        "--dbname", dbname,
        "--table", "stop_events",
        "--format", "custom",
        "--file", str(out_path),
    ]
    subprocess.run(cmd, check=True)

    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"pg_dump produced an empty or missing file: {out_path}")

    return out_path


def row_count(dsn: str) -> int:
    conn = db.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM stop_events")
            (count,) = cur.fetchone()
        return count
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None, help="Postgres DSN (default: PULSE_DSN env or local pulse db)")
    args = parser.parse_args(argv)
    dsn = args.dsn or os.environ.get("PULSE_DSN", db.DEFAULT_DSN)

    count_before = row_count(dsn)
    out_path = backup(dsn)
    manifest_path = out_path.with_suffix(".dump.count.txt")
    manifest_path.write_text(f"row_count_at_backup_time={count_before}\ndsn={dsn}\n")

    print(f"backup written: {out_path}")
    print(f"row_count_at_backup_time={count_before}")
    print(f"manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
