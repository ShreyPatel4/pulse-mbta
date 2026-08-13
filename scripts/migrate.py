"""Create the pulse database (if absent) and apply migrations in order.

Usage: uv run python scripts/migrate.py [--dsn DSN]
Defaults to the PULSE_DSN env var, falling back to pulse.db.DEFAULT_DSN.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from pulse.db import DEFAULT_DSN

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATIONS_DIR = REPO_ROOT / "migrations"

_SCHEMA_MIGRATIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name text PRIMARY KEY
)
"""


def _admin_dsn(dsn: str) -> tuple[str, str]:
    """Return (admin_dsn, target_dbname): admin_dsn points at the `postgres`
    maintenance database so CREATE DATABASE can run outside the target db.
    """
    params = conninfo_to_dict(dsn)
    target_db = params.get("dbname")
    if not target_db:
        raise ValueError(f"dsn has no dbname: {dsn!r}")
    admin_params = dict(params)
    admin_params["dbname"] = "postgres"
    return make_conninfo(**admin_params), target_db


def ensure_database(dsn: str) -> bool:
    """Create the target database if it doesn't exist. Returns True if created."""
    admin_dsn, target_db = _admin_dsn(dsn)
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s", (target_db,)
        ).fetchone()
        if exists:
            return False
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(target_db)))
        return True


def apply_migrations(dsn: str) -> list[str]:
    """Apply any migrations in migrations/*.sql not yet recorded. Returns applied names."""
    applied: list[str] = []
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(_SCHEMA_MIGRATIONS_TABLE_SQL)
        already = {row[0] for row in conn.execute("SELECT name FROM schema_migrations").fetchall()}
        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in already:
                continue
            conn.execute(path.read_text())
            conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (path.name,))
            applied.append(path.name)
    return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=None, help="Postgres DSN (default: PULSE_DSN env or local pulse db)")
    args = parser.parse_args(argv)

    dsn = args.dsn or os.environ.get("PULSE_DSN", DEFAULT_DSN)

    created = ensure_database(dsn)
    if created:
        print(f"created database (dsn={dsn})")

    applied = apply_migrations(dsn)
    if applied:
        print(f"applied: {', '.join(applied)}")
    else:
        print("up to date")

    return 0


if __name__ == "__main__":
    sys.exit(main())
