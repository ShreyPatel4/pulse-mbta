-- Milestone 1 fix wave (B3). stop_events_trip_id_stop_id_idx (trip_id,
-- stop_id) is a strict prefix of the UNIQUE (trip_id, stop_id, polled_at)
-- index Postgres already maintains for the table's unique constraint, so
-- every query the redundant index could serve, the unique index serves too.
-- Confirmed via pg_stat_user_indexes before drafting this migration: 3
-- lifetime scans against real, growing disk cost. Drop it.
DROP INDEX IF EXISTS stop_events_trip_id_stop_id_idx;
