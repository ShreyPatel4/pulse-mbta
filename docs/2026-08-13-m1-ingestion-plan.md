# Pulse M1 (ingestion) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox syntax.

**Goal:** MBTA prediction snapshots flowing into local Postgres every 60s, unattended, verified, by tonight.

**Architecture:** One Python package (`pulse/`), uv-managed. A poller module maps MBTA V3 /predictions (with included schedule) to `stop_events` rows and upserts idempotently. launchd runs it every 60s. A check script proves the flow.

**Tech Stack:** Python 3.13, uv, requests, psycopg[binary], Postgres 16 (brew), launchd.

## Global Constraints

- $0 standing cost. Keyless MBTA to start; honor `MBTA_API_KEY` env when set. Route set exactly: 1,15,22,23,28,32,39,57,66,71,73,77,111.
- Idempotent upserts on UNIQUE (trip_id, stop_id, polled_at), ON CONFLICT DO NOTHING.
- Never fabricate data; API failures log and skip the cycle, exit 0 (launchd re-fires).
- Commit identity Shrey Patel <patelshrey77@gmail.com>, no co-author lines, stage by explicit path.

### Task 1: Package scaffold + schema migration

**Files:** Create `pyproject.toml`, `.gitignore`, `README.md`, `pulse/__init__.py`, `pulse/db.py`, `migrations/001_stop_events.sql`, `scripts/migrate.py`, `tests/test_db.py`

- [ ] `uv init --no-workspace`-style pyproject (name pulse-mbta, requires-python >=3.13, deps: requests, psycopg[binary]; dev: pytest). `.gitignore`: `.venv/`, `__pycache__/`, `*.log`, `.DS_Store`, `data/`.
- [ ] `migrations/001_stop_events.sql` with the exact table from the design doc plus indexes: `(route_id, service_date)`, `(trip_id, stop_id)`.
- [ ] `pulse/db.py`: `connect(dsn=env PULSE_DSN default "postgresql://localhost/pulse")`, `upsert_stop_events(conn, rows) -> int` (executemany INSERT ... ON CONFLICT DO NOTHING, returns inserted count via rowcount sum).
- [ ] `scripts/migrate.py`: creates database `pulse` if absent (connect to postgres db, CREATE DATABASE), applies migrations in order, records in `schema_migrations(name text primary key)`.
- [ ] Test (pytest, against real local Postgres, database `pulse_test` created/dropped by fixture): upsert twice with the same (trip_id, stop_id, polled_at) → second call inserts 0; distinct polled_at → inserts. Run `uv run pytest` → PASS. Commit `scaffold: schema, db layer, migrations`.

### Task 2: Poller

**Files:** Create `pulse/mbta.py`, `pulse/poll.py`, `tests/test_mbta.py`, fixture `tests/fixtures/predictions_sample.json`

- [ ] `pulse/mbta.py`: `fetch_predictions(route_ids, session, api_key=None) -> dict` (GET https://api-v3.mbta.com/predictions?filter[route]=<comma-joined>&include=schedule,vehicle&page[limit]=1000, header x-api-key when key set, timeout 20s); `map_rows(payload, polled_at) -> list[dict]` pure: joins included schedules by relationship id, emits stop_events dicts; missing schedule → scheduled_arrival None; skips predictions with neither arrival nor departure time.
- [ ] Fixture: one real anonymized /predictions payload (fetch once during implementation, strip to ~8 predictions + their included schedules, commit).
- [ ] Tests on `map_rows` (pure): row count, schedule join correctness, None handling, skip rule. `uv run pytest` PASS.
- [ ] `pulse/poll.py`: one cycle = fetch all 13 routes in 3 batches (5/5/3, 1.5s sleep between batches), map, upsert, print one summary line `polled_at=... rows=N inserted=M routes_ok=K/13`; any batch failure logged to stderr, cycle continues; process exits 0 always (KeepAlive re-fires on schedule, not on crash loops).
- [ ] Live smoke (real API, real db): `uv run python -m pulse.poll` twice; second run inserts >0 (new polled_at) and duplicates 0. Paste both summary lines into the task report. Commit `poller: mbta v3 to stop_events, idempotent`.

### Task 3: launchd + first-night verification

**Files:** Create `ops/org.coconutlabs.pulse-ingest.plist`, `scripts/install-launchd.sh`, `scripts/check.py`; Modify `README.md`

- [ ] Plist: StartInterval 60, ProgramArguments = [`/bin/bash`, `-lc`, `cd "/Users/shrey/Personal Projects/pulse-mbta" && /opt/homebrew/bin/uv run python -m pulse.poll >> /tmp/pulse-ingest.log 2>&1`], RunAtLoad true, Label org.coconutlabs.pulse-ingest.
- [ ] `scripts/install-launchd.sh`: ensures postgres service running (`brew services start postgresql@16` idempotent), runs migrate, copies plist to ~/Library/LaunchAgents, `launchctl bootstrap gui/$(id -u)` (bootout first if loaded). Run it. Verify with `launchctl list | grep pulse` and two summary lines appearing in /tmp/pulse-ingest.log within 3 minutes.
- [ ] `scripts/check.py`: prints rows total, rows last hour, distinct trips/routes/stops, min/max polled_at, null-rate of scheduled_arrival and predicted_arrival. Run after >=10 minutes of ingestion; paste output in the task report and README ("first data" section).
- [ ] README: what this is (PRD line), the ML spec summary, how to run (migrate, poll once, install launchd, check), honest label-proxy note. Commit `ops: launchd ingestion + verification`, push all commits.
