-- Cloudflare D1 database `pulse-status` (uuid 969a7193-e4fd-4acd-85cc-04536f29790d).
-- Applied once via:
--   npx wrangler d1 execute pulse-status --remote --file d1/schema.sql
--
-- This is the ONLY thing a public surface for the pulse-mbta uptime clock is
-- allowed to read. The Mac never serves a visitor -- scripts/publish-status.py
-- pushes rows here every 5 minutes (org.coconutlabs.pulse-publish). If the
-- laptop sleeps or Postgres is briefly unreachable, publish-status.py simply
-- doesn't write -- these tables keep whatever they last held, so a public
-- page reading them degrades to a stale-but-honest last-publish timestamp
-- instead of a connection error.

-- One row per publish cycle. INSERT OR REPLACE on published_at (truncated to
-- the minute) keeps a re-run within the same minute from duplicating.
CREATE TABLE IF NOT EXISTS snapshots (
    published_at        TEXT PRIMARY KEY,
    first_cycle_at       TEXT,
    days_running          REAL,
    cycles                INTEGER,
    rows_total            INTEGER,
    rows_last_hour        INTEGER,
    cadence_seconds       REAL,
    distinct_routes       INTEGER,
    distinct_trips        INTEGER,
    distinct_stops        INTEGER,
    labels_total          INTEGER,
    labels_gap_abutted    INTEGER,
    labels_no_arrival     INTEGER,
    disk_free_gib         REAL,
    gates_pass            INTEGER,
    gates_total           INTEGER,
    gate_detail           TEXT
);

-- The gap ledger. One row per outage derived from Postgres's poll_runs table:
-- either a span between consecutive poll_runs rows longer than 3x the median
-- cadence ("no poll recorded" unless the row that closes the gap also has an
-- error, in which case that error is the reason), or a poll_runs row itself
-- carrying a non-null error. Every row must have a reason -- a perfect
-- uptime page is suspicious, a page that names its own outages is not.
--
-- `reason` is a short, classified, human sentence (string-matched off the
-- raw poll_runs.error text -- see scripts/publish-status.py's
-- _classify_reason) because this ledger renders on a public surface and a
-- raw Python traceback there reads as sloppy, not transparent. `detail`
-- keeps the original raw text (NULL when there was nothing raw to keep,
-- i.e. a true "no poll recorded" gap) so full fidelity is always one query
-- away.
CREATE TABLE IF NOT EXISTS gaps (
    started_at       TEXT PRIMARY KEY,
    ended_at         TEXT,
    duration_seconds INTEGER,
    reason           TEXT,
    cycles_missed    INTEGER,
    detail           TEXT
);
