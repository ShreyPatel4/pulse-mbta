-- M2 deliverable 5 (features + training scaffold, DE-shaped): one row per
-- trip-stop that has a training-usable label (features_trip_stop is built
-- FROM trip_stop_labels_training, see pulse/features.py), holding only
-- engineered features -- never the label itself. scripts/train.py joins
-- this back to trip_stop_labels_training on the shared natural key to build
-- X/y; keeping features and labels in separate tables is deliberate so
-- there's exactly one place ground truth lives (docs/lineage.md).
CREATE TABLE IF NOT EXISTS features_trip_stop (
    service_date_norm date NOT NULL,
    route_id text NOT NULL,
    direction_id int NOT NULL,
    stop_id text NOT NULL,
    trip_id text NOT NULL,
    -- P(late) for this (route, hour-of-day), computed ONLY over
    -- trip_stop_labels_training rows with scheduled_arrival STRICTLY
    -- EARLIER than this row's own scheduled_arrival (point-in-time /
    -- as-of correctness -- see pulse/features.py's module docstring). NULL
    -- when no such history exists yet (early in the observed window --
    -- expected and common given M2's ~1-day data volume; never
    -- backfilled/imputed at this layer).
    route_hour_historical_late_rate double precision,
    -- Seconds since the previous scheduled_arrival at the same
    -- (route_id, direction_id, stop_id), schedule-only (no leakage risk --
    -- doesn't depend on any observed outcome). NULL when this is the
    -- earliest observed scheduled trip at that stop within the data.
    headway_seconds integer,
    hour_of_day int NOT NULL,
    day_of_week int NOT NULL,
    -- delay_seconds of this trip's most recent EARLIER scheduled stop (same
    -- trip_id, by scheduled_arrival) -- the "current delay carries forward"
    -- persistence baseline's underlying signal. NULL for a trip's first
    -- labeled stop, or when the prior stop's own label isn't
    -- training-usable.
    current_delay_persistence_seconds integer,
    built_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (service_date_norm, route_id, direction_id, stop_id, trip_id)
);

CREATE INDEX IF NOT EXISTS features_trip_stop_route_service_date_idx
    ON features_trip_stop (route_id, service_date_norm);
