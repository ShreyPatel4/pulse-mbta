-- M2 must-address items 1, 5, 6 (the core of M2): the staged label layer.
-- One row per (service_date_norm, route_id, direction_id, stop_id, trip_id)
-- -- pulse/labels.py derives these from stop_events + poll_runs, and
-- scripts/build-labels.py is the idempotent, backfillable job that runs it.
-- See pulse/labels.py's module docstring for the full closing rule (gap
-- exclusion, service_date_norm's GTFS 3AM anchor, no_arrival_signal
-- filtering) -- this migration only states the shape.
CREATE TABLE IF NOT EXISTS trip_stop_labels (
    service_date_norm date NOT NULL,
    route_id text NOT NULL,
    direction_id int NOT NULL,
    stop_id text NOT NULL,
    trip_id text NOT NULL,
    scheduled_arrival timestamptz,
    final_predicted_arrival timestamptz,
    delay_seconds integer,
    late boolean,
    observed_span_start timestamptz NOT NULL,
    observed_span_end timestamptz NOT NULL,
    n_snapshots integer NOT NULL,
    -- NULL = a normal, valid closed label (delay_seconds/late populated).
    -- 'no_arrival_signal': scheduled_arrival or final_predicted_arrival is
    --   null (delay isn't computable) -- includes the ~3-4% origin/terminus
    --   rows the M1 review measured, plus the (empirically unobserved but
    --   logically possible) mixed-null case, e.g. an ADDED trip with a live
    --   prediction but no GTFS schedule counterpart.
    -- 'gap_abutted': the prediction's disappearance from the feed is
    --   ambiguous because a recorded polling gap (missing or errored
    --   poll_runs cycle) abuts it -- delay_seconds is still populated
    --   (informational) but this row must be excluded from training.
    -- Both exclusion reasons are filtered out of trip_stop_labels_training
    -- below -- query that view, not this table directly, for anything
    -- feeding a model.
    closed_reason text,
    built_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (service_date_norm, route_id, direction_id, stop_id, trip_id)
);

CREATE INDEX IF NOT EXISTS trip_stop_labels_route_service_date_idx
    ON trip_stop_labels (route_id, service_date_norm);

-- M2 must-address item 3 (quality gates wired into the run record) applied
-- to the transform layer generically -- scripts/build-labels.py appends one
-- row per run; later transform stages (e.g. a features build) can reuse the
-- same table rather than each growing its own bespoke run-ledger shape.
CREATE TABLE IF NOT EXISTS transform_runs (
    id bigserial PRIMARY KEY,
    transform_name text NOT NULL,
    started_at timestamptz NOT NULL,
    finished_at timestamptz NOT NULL,
    window_since timestamptz,
    window_until timestamptz,
    as_of timestamptz NOT NULL,
    groups_considered integer NOT NULL,
    rows_written integer NOT NULL,
    -- Per-gate {name: {passed, detail}} -- jsonb rather than one column per
    -- gate so a later transform (different gate set) doesn't need its own
    -- migration just to add columns.
    gate_results jsonb NOT NULL,
    passed boolean NOT NULL,
    git_sha text
);

CREATE INDEX IF NOT EXISTS transform_runs_transform_name_started_at_idx
    ON transform_runs (transform_name, started_at);

-- The training-safe view: every downstream feature/training query should
-- read from here, not trip_stop_labels directly, so an exclusion reason
-- added in pulse/labels.py in the future is automatically honored
-- everywhere without hunting down every WHERE clause.
CREATE OR REPLACE VIEW trip_stop_labels_training AS
SELECT *
FROM trip_stop_labels
WHERE closed_reason IS NULL;
