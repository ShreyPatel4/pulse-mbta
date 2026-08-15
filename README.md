# pulse-mbta

Riders want to know if their bus will be more than 3 minutes late. This repo
turns that line into an ML spec, a round-the-clock MBTA ingestion pipeline,
and (later milestones) models that provably beat guessing. Standing cost
$0: local Postgres 16, local training, free MBTA V3 API. Full design:
`docs/2026-08-13-pulse-design.md`.

## ML spec

- Unit of prediction: (route, direction, stop, trip) at a horizon of 10
  minutes before scheduled arrival.
- Task A (classification, the product question): P(arrival delay > 180s).
  Metric: PR-AUC and recall at precision >= 0.80 (riders punished more by a
  false "on time" than a false "late"). Baselines to beat, in order: (1)
  always-on-time, (2) route-hour historical delay rate, (3) "current delay
  carries forward" persistence.
- Task B (regression, supporting): expected delay seconds; MAE vs the
  persistence baseline.
- **Label is a proxy, documented honestly:** MBTA's API never tells you when
  a bus actually arrived. The label used is the *final observed delay* =
  the last predicted arrival time seen for a trip-stop before that
  prediction disappears from the feed (proxy for actual arrival) minus
  scheduled arrival. This is the best signal available from a free,
  keyless, polling-only integration, but it is not ground truth arrival
  time and later milestones should not present it as one.
- Data: MBTA V3 API, 13 high-frequency bus routes (1, 15, 22, 23, 28, 32,
  39, 57, 66, 71, 73, 77, 111), both directions, all stops. launchd's
  StartInterval is configured at 60s, but the batch-fetch work (3 batches,
  2x1.5s inter-batch sleep, plus request time) pushes the measured cadence
  to ~66s in practice -- about 1,309 cycles/day, not 1,440.

## How to run

```bash
# 1. Create the database + apply migrations (idempotent, safe to re-run)
uv run python scripts/migrate.py

# 2. One poll cycle by hand (prints a summary line, exits 0)
uv run python -m pulse.poll

# 3. Install the launchd agents: the poller (StartInterval 60s configured,
#    ~66s measured cadence, RunAtLoad, logs to /tmp/pulse-ingest.log) and a
#    caffeinate agent that keeps the Mac awake on AC power so ingestion
#    doesn't get gapped by sleep. Idempotent -- bootout + re-bootstrap on
#    re-run.
scripts/install-launchd.sh
launchctl list | grep pulse          # verify both are loaded
tail -f /tmp/pulse-ingest.log        # watch cycles land

# 4. Check ingestion health: totals, distinct trips/routes/stops,
#    min/max polled_at, null rates -- then four PASS/FAIL M1 quality gates
#    (disk, freshness, volume, null-rate). Exit code 1 if any gate fails.
uv run python scripts/check.py

# 5. The transform + training pipeline. Pin --until so the numbers stay
#    stable while the poller keeps running.
uv run python scripts/build-labels.py \
  --until 2026-08-15T20:11:17Z --as-of 2026-08-15T20:11:17Z
uv run python scripts/build-features.py --until 2026-08-15T20:11:17Z
uv run python scripts/verify-point-in-time.py   # 3 checks, exits 1 on FAIL
uv run python scripts/train.py                  # metrics, MLflow, register
```

Optional: set `MBTA_API_KEY` in the environment before running the poller
or installing the launchd agent to use a free MBTA key instead of
anonymous access (raises the rate limit; anonymous is fine at this
volume).

## Results

Full write-up, including the metrics defense and the limitations, is
`docs/report.md`. The short version.

The regime: 2.77 days of ingestion (2026-08-12 21:45 to 2026-08-15 16:07 ET),
10,532,354 `stop_events` rows, 65.0% uptime because the laptop sleeps. Those
closed into 185,247 labels, of which 142,258 are training-usable, 32,047 were
excluded as `gap_abutted` and 10,942 as `no_arrival_signal`. 142,239 rows
reach the training set at a 37.44% late rate. The label is a proxy: the last
predicted arrival seen before the prediction leaves the feed, minus scheduled
arrival. Split by time, never randomly: train is Wednesday night through
Friday (99,590 rows, 38.69% late), test is Saturday (42,649 rows, 34.54%).

| candidate | PR-AUC | recall@P>=0.80 (oracle) | test precision at train-chosen threshold |
|---|---|---|---|
| baseline: always-on-time | 0.3454 | n/a | n/a |
| baseline: route-hour historical rate | 0.4913 | n/a | 0.6514 |
| baseline: persistence | 0.8636 | 0.8758 | 0.9371 |
| LogisticRegression | 0.9678 | 0.9373 | 0.7999 |
| HistGradientBoosting | **0.9711** | **0.9557** | 0.7782 |

Both models beat all three baselines on PR-AUC and on recall at the precision
floor. Two results cut against that, and both are in the report rather than
buried:

- **Neither model holds precision >= 0.80 on test at a threshold chosen from
  training data.** LogisticRegression lands at 0.7999, missing by 0.0001.
  HistGradientBoosting lands at 0.7782. The ranking transfers to Saturday.
  The calibration does not.
- **The features do not honor the design doc's 10-minute horizon.** They are
  built from settled labels, and a label settles roughly when the bus
  arrives. Only 0.63% of rows had the persistence input settled 10 minutes
  before scheduled arrival, and 0.98% for the route-hour aggregate. Every
  PR-AUC above is an upper bound on what a horizon-honoring model would
  score. The fix is specified in `docs/report.md`.

The pipeline's own seven-day gate fires at 2.77 days and was left where it
is. `docs/report.md` has the full "what this cannot say" list: no weekly
seasonality (4 of 7 day types), one weather regime, 45 of 96 (service date x
hour) cells empty because of sleep gaps, no spatial holdout, and a named
experiment for each.

## Ops: staying alive, and gaps when it doesn't

`scripts/install-launchd.sh` installs two agents. `org.coconutlabs.pulse-ingest`
is the poller. `org.coconutlabs.pulse-caffeinate` runs `caffeinate -s -i`
with `KeepAlive`, which prevents the Mac from sleeping while ingestion needs
to run -- but `-s` only holds off sleep on AC power; on battery the machine
can still sleep, and ingestion stops for however long it's asleep.

That gap is not silently lost. Every poll cycle, success or total failure,
writes one row to `poll_runs` (`migrations/002_poll_runs.sql`): started_at,
finished_at, how many batches/pages succeeded, rows/inserted/skipped, and an
error column for failures. A stretch of missing `stop_events` with no
matching `poll_runs` row (or a `poll_runs` row with `error` set) means the
poller didn't run or failed -- not that no buses were arriving.

Backfill for that gap is impossible by nature, not a shortcut being
deferred: MBTA's `/predictions` feed is realtime-only, with no history
endpoint, so a missed polling window's data is gone forever. `poll_runs` is
the honest ledger of exactly where those windows are. M2's label-building
rule follows from this directly: labels must not close across a recorded
gap.

## Data quality note: no-arrival-signal rows

Roughly 4% of ingested rows (228/5408, 4.2%, in the first measured sample;
3.66% in a later, larger sample -- see `scripts/check.py` output below for
the current rate) have both `scheduled_arrival` and `predicted_arrival`
null. These are
real origin/terminus stops: MBTA's API gives only a `departure_time` for a
trip's first stop, never an `arrival_time`, so there is no temporal signal
at that stop to record. The row still gets ingested (it isn't a malformed
or dropped prediction -- `departure_time` is present, which is what clears
the skip rule in `pulse/mbta.py:map_rows`), but it is structurally
unlabelable for Task A/B above: there's no arrival to be early, on-time, or
late relative to. Downstream feature/label building (M2) should filter
these out rather than treat the null as "on time."

## Pagination

MBTA's `/predictions` endpoint returns one row per (trip, stop) pair
currently active on a route; `page[limit]=1000` is regularly exceeded by
the 13-route route set (observed 2026-08-12: a single 5-route batch alone
returned 1667 predictions across 2 pages). `pulse/mbta.py:fetch_predictions`
follows `links.next` until it's absent, capped at 5 pages per batch as a
runaway guard (logs a warning to stderr if the cap is ever hit).

Honest caveat, confirmed empirically: this is offset pagination
(`page[offset]`) over a live, mutating collection, not a cursor over a
fixed snapshot. A real fetch showed ~5 predictions duplicated across the
page 1 / page 2 boundary (present at both the tail of page 1 and the head
of page 2) -- harmless downstream, since `stop_events`'s unique key on
`(trip_id, stop_id, polled_at)` absorbs it via `ON CONFLICT DO NOTHING`
(this is exactly why some cycles show `rows > inserted` even within a
single batch). The symmetric case -- an item shifting past a page boundary
and being skipped by both requests -- is possible in principle and can't be
distinguished from the ingested data after the fact. Pagination is a large
net improvement in completeness over the unpaginated version, not a
guarantee of an exact per-cycle snapshot.

## First data

`scripts/check.py` output after the launchd agent had been driving
ingestion for a first stretch (launchd's own cycles span 2026-08-12
21:59:16 -> 22:10:22 ET, about 11 minutes real elapsed). `min_polled_at`
below predates that: this database also holds two pre-launchd manual
smoke cycles from Task 2 (the 2,704-row baseline) plus one cycle that
returned zero rows (`routes_ok=0/13`) after a self-inflicted MBTA
rate-limit hit during this task's own diagnostic probing -- both are
counted in `total_rows` but neither is launchd's doing:

```
total_rows             = 50913
last_hour_rows         = 50913
distinct_trips         = 223
distinct_routes        = 13
distinct_stops         = 630
min_polled_at          = 2026-08-12 21:45:01.657904-04:00
max_polled_at          = 2026-08-12 22:10:22.269600-04:00
scheduled_arrival_null = 1522 (2.99%)
predicted_arrival_null = 1522 (2.99%)
```

## Out of scope for this repo

Serving (a later, separate project), paid infra, dashboards. See
`docs/2026-08-13-pulse-design.md` for the full milestone roadmap (M2 label
+ feature build, M3 baselines + models, M4 registered model + report).
