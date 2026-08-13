# Lineage: raw -> staged -> features -> training set

M2 must-address item 7. Four layers, each a named table (or, for the
training set, a named query) with its producing code, its contract, and the
gates that run on it. Program-flavor's binding rule ("lineage is written
down: raw -> staged -> features -> training set, each layer a named table
or artifact with its producing code linked") applies directly here.

Dependency choice, stated once for the whole pipeline: `pulse/db.py`,
`pulse/labels.py`, and `pulse/features.py` are plain SQL executed via
psycopg -- every transform is a set-based aggregate or upsert Postgres
expresses directly, and staying in SQL keeps each stage backfillable by
window (rerun any [since, until) and it updates in place, never
duplicates). `pandas` is used only at `scripts/train.py`'s boundary, where
sklearn needs an in-memory matrix. `mlflow`/`joblib`/`scikit-learn` are
scoped to that same boundary.

## 1. `stop_events` (raw)

One row per (trip, stop) prediction snapshot, ingested every ~66s.

- **Producing code:** `pulse/mbta.py` (`fetch_predictions` — pagination,
  retry/backoff, per-page contract validation; `map_rows` — pure
  payload-to-row mapping) called from `pulse/poll.py` (`run_cycle` — batches
  routes, enforces the 240s cycle deadline, calls
  `pulse/partitions.py:ensure_partitions` each cycle) via
  `pulse/db.py:upsert_stop_events`. Schema:
  `migrations/001_stop_events.sql`, `003_drop_redundant_index.sql`,
  `004_partition_stop_events.sql` (monthly RANGE partitions on
  `polled_at`, current month + 2 ahead maintained by
  `pulse/partitions.py`).
- **Contract:** `contracts/mbta-predictions.v1.json`, enforced per-page (not
  per-batch, not per-cycle) inside `fetch_predictions` via
  `pulse/contract.py:validate_page`. A violating page's data is dropped
  before merging; the violation is named and recorded as a `poll_runs`
  error rather than silently lowering `inserted` with no explanation.
- **Gates:** `scripts/check.py` — disk, freshness, volume, null-rate,
  PASS/FAIL, exit 1 on breach. `poll_runs` (`migrations/002_poll_runs.sql`)
  is the run ledger every layer downstream ultimately depends on: it's the
  only record of *when the feed wasn't being watched*, which is what
  `pulse/labels.py`'s gap-exclusion rule reads.
- **Known, accepted incompleteness:** offset pagination over a live,
  mutating collection (README's Pagination section) and the poller's
  per-batch retry/deadline logic mean a cycle's coverage is a strong
  improvement over the unpaginated version, not a guaranteed-exact
  snapshot. This is why the label layer treats a *recorded gap*, not just
  "no snapshot", as the trustworthy signal.

### The `status` column: kept, documented as structurally null

`status` (from the prediction resource's own `status` attribute) has been
**0/633,774 rows non-null** in every sample taken across M1 and M2,
including the fixture in `tests/fixtures/predictions_sample.json` (every
prediction: `"status": null`). This is not a mapping bug — `pulse/mbta.py`
reads it straight off `attributes.status` — MBTA's V3 `/predictions`
endpoint simply doesn't populate this attribute in this integration's
observed traffic (13 bus routes, `include=schedule,vehicle`). Decision:
**keep the column, don't drop it.** Dropping a column that's a stable part
of MBTA's documented schema on the strength of "always null so far" trades
a small amount of storage for real risk (a future MBTA API change that
starts populating it would need a new migration to get it back, and in the
meantime any consumer expecting it to exist would break). `contracts/mbta-predictions.v1.json`
marks `status` explicitly `"nullable": true` with this exact rationale in
its `semantics` field, so the contract validator never flags the observed
100%-null rate as a violation.

## 2. `trip_stop_labels` (staged)

One row per (`service_date_norm`, `route_id`, `direction_id`, `stop_id`,
`trip_id`) — a *closed* determination about whether that trip-stop ended up
late, once settled.

- **Producing code:** `pulse/labels.py` (`run_build`,
  `fetch_touched_groups`, `derive_label_row`, `compute_gap_intervals`,
  `service_date_norm`) via `scripts/build-labels.py`. Schema:
  `migrations/005_trip_stop_labels.sql` (also creates `transform_runs` and
  the `trip_stop_labels_training` view).
- **Contract (the closing rule, not a JSON file — this layer's "schema" is
  behavioral):**
  - A trip-stop closes only once *settled*: last seen
    `>= SETTLE_MARGIN_SECONDS` (~3 poll cycles, ~200s) in the past relative
    to `as_of` (defaults to the latest `poll_runs` row).
  - `closed_reason='gap_abutted'` when a recorded polling gap (an errored
    `poll_runs` row, or a stretch with none at all —
    `compute_gap_intervals`) overlaps the window right after the last
    sighting. A missing prediction is indistinguishable from a missing
    observation there.
  - `closed_reason='no_arrival_signal'` when `scheduled_arrival` or
    `final_predicted_arrival` is null — never imputed as on-time.
  - `service_date_norm` applies the GTFS 3AM rule (America/New_York),
    anchored on `scheduled_arrival` when present, else the trip-stop's
    first sighting over its **full** `stop_events` history (not the current
    build's window) — the anchor most builds' idempotent upsert depends on;
    see the module docstring for the rerun-stability argument in full.
- **Gates:** `pulse/labels.py:compute_quality_gates` — volume (`>= 1` row
  written), null_rate (`0%` of normal-closed rows may have a null
  `delay_seconds` — a construction invariant, gated as a safety net), and
  label_rate (`>= 50%` of written rows must be training-usable). PASS/FAIL
  printed by `scripts/build-labels.py`, non-zero exit on breach, one row
  appended to `transform_runs` every run regardless of outcome.
- **Training-safe read path:** `trip_stop_labels_training` (a view —
  `WHERE closed_reason IS NULL`). Every downstream layer reads this, never
  `trip_stop_labels` directly, so a future exclusion reason added to
  `pulse/labels.py` is automatically honored everywhere.

## 3. `features_trip_stop`

One row per training-usable label, engineered features only — never the
label itself (kept in a separate table on purpose: exactly one place ground
truth lives).

- **Producing code:** `pulse/features.py` (`build_features`) via
  `scripts/build-features.py`. Schema: `migrations/006_features_trip_stop.sql`.
- **Contract:** point-in-time correctness, enforced structurally by the SQL
  itself, not a separate validation pass — `route_hour_historical_late_rate`
  and `headway_seconds` are both correlated subqueries filtered to
  `scheduled_arrival` **strictly earlier** than the row being featured
  (see `pulse/features.py`'s module docstring). NULL when no such history
  exists yet rather than a fabricated default. Stated tradeoff: these are
  O(n²) correlated subqueries — correct and simple at M2's volume (~13s for
  9,352 rows); a real-scale system would materialize a rolling aggregate
  instead of recomputing full history inline per row.
- **Gates:** volume (`>= 1` row written) and `historical_rate_coverage`
  (fraction of written rows with a non-null `route_hour_historical_late_rate`
  — informational at M2's volume, `>= 0%`, exists to catch a future
  regression like the join predicate silently matching nothing). Recorded
  to `transform_runs` (`transform_name='build_features'`) via
  `pulse/features.py:record_transform_run`.

## 4. Training set

Not a persisted table — the join `scripts/train.py` builds at run time.

- **Producing code:** `scripts/train.py`'s `_TRAINING_SET_SQL`
  (`features_trip_stop JOIN trip_stop_labels_training ... ORDER BY
  scheduled_arrival`) plus `pulse/train.py` (`temporal_split`,
  `baseline_scores`, `build_models`) and `pulse/metrics.py` (`pr_auc`,
  `recall_at_precision`).
- **Contract:** the temporal split itself is the load-bearing guarantee —
  `pulse/train.py:temporal_split` takes the first `TRAIN_FRACTION` of rows
  by `scheduled_arrival` order, never a random split, so evaluation never
  happens on data chronologically interleaved with its own training data.
  `baseline_route_hour_rate`'s NaN fallback is computed from the **train**
  split's late rate only, never test — the one place this layer could leak
  test-set outcomes into a "baseline" if it weren't deliberate.
- **Gates:** the `PRELIMINARY - insufficient data for the report
  deliverable` banner (`window_days < 7`) is this layer's quality gate — it
  runs every time, prints loudly, and is written into
  `models/REGISTRY.md`'s entry rather than only stdout. There's no
  automated PASS/FAIL threshold on the metrics themselves (the design
  doc's bar — "provably beats guessing" — is comparative against the 3
  baselines, judged by a human reading `models/REGISTRY.md`, not a fixed
  number a script can gate on alone).
- **Honest caveat carried into every PRELIMINARY run:**
  `route_hour_historical_late_rate` is point-in-time correct by
  construction, but with only a few hours of real spread, "strictly
  earlier" is nearly the same window the model is then scored against. A
  high PR-AUC this early is evidence of that adjacency, not of model skill
  — `scripts/train.py` prints this explicitly and `models/REGISTRY.md`
  carries the same note on every entry where `window_days < 7`.

## What's out of scope here

Task B (regression: expected delay seconds, MAE vs. the persistence
baseline) is named in `docs/2026-08-13-pulse-design.md`'s ML spec as
"supporting" but isn't part of M2's explicit deliverable 5 ask (PR-AUC +
recall@precision>=0.80 only) — `delay_seconds` is already carried on
`trip_stop_labels`/the training-set join, so a Task B pass is additive, not
a schema change, when M3/M4 picks it up.
