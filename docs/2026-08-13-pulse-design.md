# Pulse — design

Project 1 of the five-project program. The one-line PRD: riders want to
know if their bus will be more than 3 minutes late. This repo turns that
line into an ML spec, a round-the-clock ingestion pipeline, and models
that provably beat guessing. Standing cost $0: local Postgres 16, local
training, free MBTA V3 API.

## ML spec (PRD → spec, the course's first deliverable)

- Unit of prediction: (route, direction, stop, trip) at a horizon of
  10 minutes before scheduled arrival.
- Task A (classification, the product question): P(arrival delay > 180s).
  Metric: PR-AUC and recall at precision >= 0.80 (riders punished more by
  false "on time" than false "late"). Baselines to beat, in order:
  (1) always-on-time, (2) route-hour historical delay rate, (3) "current
  delay carries forward" persistence.
- Task B (regression, supporting): expected delay seconds; MAE vs the
  persistence baseline.
- Label: final observed delay = last predicted arrival time seen for the
  trip-stop before the prediction disappears (proxy for actual arrival;
  documented as such honestly) minus scheduled arrival.
- Data: MBTA V3 API, 13 high-frequency bus routes
  (1, 15, 22, 23, 28, 32, 39, 57, 66, 71, 73, 77, 111), both directions,
  all stops. Snapshots every 60s around the clock.

## Ingestion (Milestone 1, running the first night)

- Postgres 16 (brew, local), database `pulse`. Table `stop_events`:
  (id bigserial PK, route_id text, direction_id int, stop_id text,
  trip_id text, vehicle_id text|null, service_date date,
  scheduled_arrival timestamptz|null, predicted_arrival timestamptz|null,
  status text|null, polled_at timestamptz NOT NULL,
  UNIQUE (trip_id, stop_id, polled_at)) — idempotent upserts on the
  unique key (ON CONFLICT DO NOTHING; snapshots are immutable facts).
- Poller: Python 3.13 + requests + psycopg (uv-managed venv). One cycle:
  GET /predictions?filter[route]=...&include=schedule per route batch,
  map to rows, upsert, log one summary line. Keyless to start (respects
  the anonymous rate limit with per-route pacing); MBTA_API_KEY env var
  used when present (operator bend: free key raises limits).
- Scheduling: launchd agent org.coconutlabs.pulse-ingest, 60s interval,
  KeepAlive, logs to /tmp/pulse-ingest.log (house precedent: waterline
  agents).
- Verification: after two hours, a check script reports rows ingested,
  distinct trips, routes covered, null rates. The first-night report is
  committed.

## Later milestones (planned after data flows)

- M2: label derivation + feature build (route-hour aggregates, headway,
  weather optional-cut) as pure SQL/Python transforms, DVC-tracked.
- M3: sklearn baselines + gradient boosting; PyTorch tabular model;
  MLflow local tracking; the beats-guessing report with the three
  baselines above.
- M4: registered model + report defending metrics (the course
  deliverable), README walkthrough.

## Out of scope for this repo

Serving (that is Project 2), paid infra, dashboards.
