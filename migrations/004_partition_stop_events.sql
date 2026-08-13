-- M2 must-address item 2: convert stop_events to a monthly RANGE partition
-- on polled_at. Two reasons, both from the M1 review: (1) retention posture
-- -- we keep ALL data (no deletion; the disk gate in scripts/check.py is the
-- alarm, not a TTL), so the table only grows, and partitioning is how a
-- growing table stays operable without ever throwing rows away; (2) cycle
-- time -- the UNIQUE (trip_id, stop_id, polled_at) index is a random-insert
-- btree on an ever-growing single relation; monthly partitions keep each
-- partition's working set (and its indexes) bounded, which keeps per-cycle
-- insert time flat as volume grows instead of degrading with total table
-- size.
--
-- OPERATIONAL SEQUENCE (this is a live table -- launchd's
-- org.coconutlabs.pulse-ingest inserts into it roughly every 66s and this
-- migration is written to survive that):
--   1. BEFORE running this (via scripts/migrate.py), take a pg_dump backup:
--      `uv run python scripts/backup-stop-events.py`. Writes to
--      data-backups/ (gitignored) and records the pre-migration row count.
--      Keep this and stop_events_old (below) until the operator has
--      confirmed the swap; both are cheap insurance, not meant to be
--      dropped as part of this migration.
--   2. Phase A (below): create the new partitioned table + this month's and
--      the next two months' partitions, plus a DEFAULT partition as a
--      backstop (see "no partition of relation found" note below), then
--      bulk-copy every existing row across. No lock beyond the ordinary
--      ACCESS SHARE a plain SELECT takes -- the poller's concurrent INSERTs
--      into the OLD table are never blocked by this step. Verified against
--      a >=  count check (>= because more rows may have landed via the
--      poller since the SELECT's snapshot was taken -- under Postgres's
--      default READ COMMITTED isolation, that's expected, not a bug: the
--      catch-up step below re-reads the live table with a fresh statement
--      snapshot).
--   3. Phase B (below): a short, explicit transaction. SET LOCAL
--      lock_timeout='5s' bounds how long this can make the poller's next
--      INSERT queue behind a pending ACCESS EXCLUSIVE request -- if some
--      other session is holding a conflicting lock for more than 5s, this
--      aborts cleanly (no partial state) rather than stalling ingestion
--      indefinitely. Once the lock is held: catch-up copy (rows with
--      id > the new table's current max(id) -- self-referencing the
--      watermark this way needs no session variable and is safe under
--      ON CONFLICT DO NOTHING even if the catch-up query is ever re-run),
--      an exact-count assertion (raises and rolls back the whole
--      transaction on any mismatch -- zero row loss is enforced, not just
--      hoped for), then the two RENAMEs and the sequence-ownership
--      reassignment, then this migration's own schema_migrations bookkeeping
--      row (see IDEMPOTENCY below), all in the one commit.
--   4. AFTER: re-run scripts/backup-stop-events.py's count check (or just
--      `select count(*) from stop_events`) against the post-swap table and
--      diff against the pre-migration count from step 1. Confirm
--      poll_runs is still landing rows (the M2-REPORT.md evidence for this
--      run is three consecutive post-swap poll_runs rows). Do NOT drop
--      stop_events_old in this migration -- that is a separate, later,
--      deliberate operator decision once the swap is trusted.
--
-- SCHEMA CHOICE, documented: Postgres requires every UNIQUE/PK constraint on
-- a partitioned table to include all partition-key columns. polled_at is the
-- partition key. UNIQUE (trip_id, stop_id, polled_at) already includes it --
-- that's also the exact key pulse.db's ON CONFLICT targets, unchanged. `id`
-- alone can no longer be a bare PRIMARY KEY (it doesn't contain polled_at),
-- so the new table uses PRIMARY KEY (id, polled_at) instead of dropping a
-- primary key entirely -- id stays NOT NULL and globally monotonic (it's
-- still fed by the one shared stop_events_id_seq sequence, reused explicitly
-- via `DEFAULT nextval('stop_events_id_seq')` rather than let a fresh
-- `bigserial` mint a second sequence starting back at 1, which would have
-- produced silently duplicate ids and broken the catch-up watermark), just
-- not independently unique-enforced by Postgres across partitions the way a
-- lone-column PK would be on an unpartitioned table. Nothing in this repo
-- foreign-keys against stop_events.id today.
--
-- NEW FAILURE MODE THIS INTRODUCES: an INSERT whose polled_at doesn't match
-- any partition's range fails the whole statement ("no partition of
-- relation ... found for row") -- a single point of failure the
-- unpartitioned table didn't have. stop_events_default (a DEFAULT partition)
-- is the backstop: any row outside the explicitly created monthy ranges
-- lands there instead of failing the insert. scripts/ensure-partitions.py is
-- the ongoing mitigation -- called every poll cycle, it keeps a 2-month
-- lookahead of real partitions provisioned (cheap no-op check before any
-- DDL: CREATE TABLE ... PARTITION OF takes an ACCESS EXCLUSIVE lock on the
-- parent, so it must not run blind every ~66s) so the DEFAULT partition
-- should only ever catch a rare clock-skew straggler, not steady-state
-- traffic.
--
-- IDEMPOTENCY: Phase A is safe to re-run (CREATE TABLE IF NOT EXISTS +
-- ON CONFLICT DO NOTHING on the copy). Phase B is all-or-nothing in one
-- transaction, so a failure there leaves nothing partially applied. The
-- schema_migrations insert happens inside Phase B's own transaction (not as
-- a separate statement after this file returns, the way
-- scripts/migrate.py's apply_migrations() normally records a migration) so
-- the bookkeeping commits atomically with the rename -- migrate.py's own
-- follow-up insert is a redundant, harmless ON CONFLICT DO NOTHING once this
-- file's copy has already landed it. Empirically rehearsed (scratch DB,
-- 2026-08-13): a process crash between Phase A committing and Phase B ever
-- starting recovers cleanly on re-run (Phase A no-ops, Phase B completes).
-- The one un-guarded case: running this .sql file directly via psql (NOT
-- through scripts/migrate.py) after it has already succeeded once. Because
-- the swap already renamed stop_events_new away, a direct re-run creates a
-- *fresh*, empty stop_events_new with no partitions attached (the target
-- partition names are already taken by the real table's partitions, so
-- `CREATE TABLE IF NOT EXISTS ... PARTITION OF stop_events_new` silently
-- skips attaching them) -- the very next bulk-copy INSERT then fails loud
-- with "no partition of relation found", leaving an empty orphaned
-- stop_events_new to clean up (`DROP TABLE stop_events_new;`). The live
-- stop_events table is never touched by this failure mode: rehearsed and
-- confirmed. Always apply migrations through scripts/migrate.py, which
-- checks schema_migrations before ever reading this file's text.
--
-- COSMETIC WART, documented rather than fixed: Postgres index/constraint
-- names are schema-scoped, not table-scoped, and RENAME TABLE does not
-- rename a table's indexes or constraints. Since the original names
-- (stop_events_pkey, stop_events_route_id_service_date_idx,
-- stop_events_trip_id_stop_id_polled_at_key) are still held by
-- stop_events_old after the swap, the new table's equivalents keep the
-- "_new" names they were created with (stop_events_new_pkey, etc.) even
-- after the table itself is renamed to stop_events. Functionally identical;
-- cosmetic only.

-- Phase A: create the partitioned skeleton + bulk-copy, no exclusive lock.
CREATE TABLE IF NOT EXISTS stop_events_new (
    id bigint NOT NULL DEFAULT nextval('stop_events_id_seq'),
    route_id text NOT NULL,
    direction_id int NOT NULL,
    stop_id text NOT NULL,
    trip_id text NOT NULL,
    vehicle_id text,
    service_date date NOT NULL,
    scheduled_arrival timestamptz,
    predicted_arrival timestamptz,
    status text,
    polled_at timestamptz NOT NULL,
    PRIMARY KEY (id, polled_at),
    UNIQUE (trip_id, stop_id, polled_at)
) PARTITION BY RANGE (polled_at);

-- Current month + next 2 (authored 2026-08-13: Aug/Sep/Oct 2026), plus a
-- DEFAULT backstop partition. scripts/ensure-partitions.py maintains the
-- rolling lookahead from here on; this migration only needs to get the
-- table into a working partitioned state once.
CREATE TABLE IF NOT EXISTS stop_events_y2026m08 PARTITION OF stop_events_new
    FOR VALUES FROM ('2026-08-01') TO ('2026-09-01');
CREATE TABLE IF NOT EXISTS stop_events_y2026m09 PARTITION OF stop_events_new
    FOR VALUES FROM ('2026-09-01') TO ('2026-10-01');
CREATE TABLE IF NOT EXISTS stop_events_y2026m10 PARTITION OF stop_events_new
    FOR VALUES FROM ('2026-10-01') TO ('2026-11-01');
CREATE TABLE IF NOT EXISTS stop_events_default PARTITION OF stop_events_new DEFAULT;

-- CREATE INDEX on a partitioned parent automatically creates (and, for
-- future partitions, keeps creating) the matching index on every partition.
CREATE INDEX IF NOT EXISTS stop_events_new_route_id_service_date_idx
    ON stop_events_new (route_id, service_date);

INSERT INTO stop_events_new (
    id, route_id, direction_id, stop_id, trip_id, vehicle_id,
    service_date, scheduled_arrival, predicted_arrival, status, polled_at
)
SELECT
    id, route_id, direction_id, stop_id, trip_id, vehicle_id,
    service_date, scheduled_arrival, predicted_arrival, status, polled_at
FROM stop_events
ON CONFLICT (trip_id, stop_id, polled_at) DO NOTHING;

DO $$
DECLARE
    old_count bigint;
    new_count bigint;
BEGIN
    SELECT count(*) INTO old_count FROM stop_events;
    SELECT count(*) INTO new_count FROM stop_events_new;
    IF new_count < old_count THEN
        RAISE EXCEPTION
            'partition swap: row count regression after bulk copy (stop_events=% stop_events_new=%)',
            old_count, new_count;
    END IF;
    RAISE NOTICE 'stop_events partition swap, phase A ok: stop_events=% stop_events_new=%', old_count, new_count;
END $$;

-- Phase B: short transaction -- catch-up copy + verified swap under an
-- exclusive lock, bounded by lock_timeout so a contended lock aborts
-- cleanly instead of stalling the poller.
BEGIN;
SET LOCAL lock_timeout = '5s';
LOCK TABLE stop_events IN ACCESS EXCLUSIVE MODE;

INSERT INTO stop_events_new (
    id, route_id, direction_id, stop_id, trip_id, vehicle_id,
    service_date, scheduled_arrival, predicted_arrival, status, polled_at
)
SELECT
    id, route_id, direction_id, stop_id, trip_id, vehicle_id,
    service_date, scheduled_arrival, predicted_arrival, status, polled_at
FROM stop_events
WHERE id > (SELECT COALESCE(max(id), 0) FROM stop_events_new)
ON CONFLICT (trip_id, stop_id, polled_at) DO NOTHING;

DO $$
DECLARE
    old_count bigint;
    new_count bigint;
BEGIN
    SELECT count(*) INTO old_count FROM stop_events;
    SELECT count(*) INTO new_count FROM stop_events_new;
    IF new_count <> old_count THEN
        RAISE EXCEPTION
            'partition swap: final row count mismatch, aborting (stop_events=% stop_events_new=%)',
            old_count, new_count;
    END IF;
    RAISE NOTICE 'stop_events partition swap, phase B ok: stop_events=% stop_events_new=%', old_count, new_count;
END $$;

ALTER TABLE stop_events RENAME TO stop_events_old;
ALTER TABLE stop_events_new RENAME TO stop_events;
-- Reassign sequence ownership to the new table so a future DROP of
-- stop_events_old doesn't cascade-drop the sequence the live table's id
-- DEFAULT still depends on.
ALTER SEQUENCE stop_events_id_seq OWNED BY stop_events.id;

-- Self-contained bookkeeping (see IDEMPOTENCY above): create
-- schema_migrations if this file is ever applied before scripts/migrate.py
-- has bootstrapped it (e.g. the tests/test_db.py fixture, which applies
-- migrations/*.sql directly), and record this migration atomically with the
-- rename rather than relying solely on apply_migrations()'s separate,
-- later INSERT.
CREATE TABLE IF NOT EXISTS schema_migrations (name text PRIMARY KEY);
INSERT INTO schema_migrations (name) VALUES ('004_partition_stop_events.sql') ON CONFLICT DO NOTHING;

COMMIT;
