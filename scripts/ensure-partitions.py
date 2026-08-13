"""CLI wrapper for pulse.partitions.ensure_partitions -- standalone/manual use.

Usage: uv run python scripts/ensure-partitions.py [--months-ahead N]

The reusable logic lives in pulse/partitions.py (importable -- this file's
own name has a hyphen, so it can't be imported as a module) so pulse.poll's
cycle can call it directly every ~66s. This script is a thin entry point for
running the same check by hand.
"""

from __future__ import annotations

import argparse
import sys

from pulse import db
from pulse.partitions import DEFAULT_MONTHS_AHEAD, ensure_partitions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months-ahead", type=int, default=DEFAULT_MONTHS_AHEAD)
    args = parser.parse_args(argv)

    conn = db.connect()
    try:
        created = ensure_partitions(conn, months_ahead=args.months_ahead)
    finally:
        conn.close()

    if created:
        print(f"ensure-partitions: created {', '.join(created)}")
    else:
        print("ensure-partitions: up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
