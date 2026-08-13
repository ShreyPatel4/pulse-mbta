-- Milestone 1 fix wave (B2). Records one row per poll cycle, success or
-- total failure, so a gap in stop_events is distinguishable from "nothing
-- was arriving": an absence of poll_runs rows (or a row with error set)
-- means the poller didn't run or failed, not that buses stopped moving.
CREATE TABLE IF NOT EXISTS poll_runs (
    polled_at timestamptz PRIMARY KEY,
    started_at timestamptz NOT NULL,
    finished_at timestamptz NOT NULL,
    batches_ok int NOT NULL,
    batches_total int NOT NULL,
    pages_fetched int NOT NULL,
    rows int NOT NULL,
    inserted int NOT NULL,
    skipped int NOT NULL,
    error text
);
