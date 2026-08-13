-- Milestone 1 ingestion table. One row per (trip, stop) prediction snapshot.
-- Snapshots are immutable facts: upserts on the unique key are
-- ON CONFLICT DO NOTHING (never updated in place).
CREATE TABLE IF NOT EXISTS stop_events (
    id bigserial PRIMARY KEY,
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
    UNIQUE (trip_id, stop_id, polled_at)
);

CREATE INDEX IF NOT EXISTS stop_events_route_id_service_date_idx
    ON stop_events (route_id, service_date);

CREATE INDEX IF NOT EXISTS stop_events_trip_id_stop_id_idx
    ON stop_events (trip_id, stop_id);
