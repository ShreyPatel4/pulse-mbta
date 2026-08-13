"""CLI wrapper for pulse.features.build_features.

Usage: uv run python scripts/build-features.py [--since ISO] [--until ISO]

Idempotent and backfillable over any window on scheduled_arrival (see
pulse.features' module docstring). Prints one summary line, PASS/FAIL for
each quality gate, appends one row to transform_runs, exits non-zero if any
gate failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys

from pulse import db, features


def _parse_iso(value: str | None) -> dt.datetime | None:
    if value is None:
        return None
    parsed = dt.datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - informational only
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=None, help="ISO timestamp, lower bound on scheduled_arrival (default: all history)")
    parser.add_argument("--until", default=None, help="ISO timestamp, upper bound on scheduled_arrival (default: now)")
    args = parser.parse_args(argv)

    conn = db.connect()
    try:
        summary = features.build_features(conn, since=_parse_iso(args.since), until=_parse_iso(args.until))
        features.record_transform_run(conn, summary, git_sha=_git_sha())
    finally:
        conn.close()

    print(
        f"build-features: window=[{summary['window_since']}, {summary['window_until']}) "
        f"candidate_rows={summary['candidate_rows']} rows_written={summary['rows_written']}"
    )
    print()
    print("-- feature quality gates --")
    for name, result in summary["gate_results"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {name}: {result['detail']}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
