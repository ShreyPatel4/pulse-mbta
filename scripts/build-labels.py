"""CLI wrapper for pulse.labels.run_build -- the label-derivation job.

Usage: uv run python scripts/build-labels.py [--since ISO] [--until ISO] [--as-of ISO]

Idempotent and backfillable over any window: rerunning over the same or an
overlapping window UPDATEs existing trip_stop_labels rows in place (never
duplicates -- see pulse.labels' module docstring for why the aggregation
itself, not just the upsert, is what makes this safe). Prints one summary
line, then PASS/FAIL for each quality gate (volume, null_rate, label_rate --
see pulse.labels.compute_quality_gates), appends one row to transform_runs
regardless of outcome, and exits non-zero if any gate failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys

from pulse import db, labels


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
    except Exception:  # noqa: BLE001 - git sha is informational, never worth failing the build over
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=None, help="ISO timestamp, lower bound on stop_events.polled_at (default: all history)")
    parser.add_argument("--until", default=None, help="ISO timestamp, upper bound on stop_events.polled_at (default: now)")
    parser.add_argument("--as-of", default=None, help="ISO timestamp for settle-margin evaluation (default: latest poll_runs row)")
    args = parser.parse_args(argv)

    since = _parse_iso(args.since)
    until = _parse_iso(args.until)
    as_of = _parse_iso(args.as_of)

    conn = db.connect()
    try:
        summary = labels.run_build(conn, since=since, until=until, as_of=as_of)
        labels.record_transform_run(conn, summary, git_sha=_git_sha())
    finally:
        conn.close()

    print(
        f"build-labels: window=[{summary['window_since']}, {summary['window_until']}) "
        f"as_of={summary['as_of']} groups_considered={summary['groups_considered']} "
        f"rows_written={summary['rows_written']}"
    )
    print()
    print("-- label quality gates --")
    for name, result in summary["gate_results"].items():
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {name}: {result['detail']}")

    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
