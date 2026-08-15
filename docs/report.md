# Pulse model report: what 2.77 days of ingestion can and cannot say

M4 deliverable. Task A from `docs/2026-08-13-pulse-design.md`: classify
P(arrival delay > 180s) for a (route, direction, stop, trip), scored by
PR-AUC and recall at precision >= 0.80, against three baselines.

## The regime

Everything below rests on 2.77 days of MBTA ingestion, from 2026-08-12
21:45:01 to 2026-08-15 16:07:41 America/New_York. That is 10,532,354
`stop_events` prediction snapshots across 13 bus routes, 719 stops and 5,043
trips. Ingestion uptime over that span was 65.0%. The laptop running the
poller sleeps: 2,346 poll cycles landed in a 65.6 hour ledger span, with 27
recorded gaps longer than 100 seconds totalling 23.0 hours, the longest a
single 12.22 hour stretch on 2026-08-13. The gap ledger is `poll_runs` and it
is published at coconutlabs.org/live/pulse.

Those snapshots closed into 185,247 labels. 142,258 are training-usable
(76.79%). 32,047 were excluded as `gap_abutted` (17.30%) and 10,942 as
`no_arrival_signal` (5.91%). A further 19 usable labels were dropped from the
feature table because their scheduled arrival falls after the pinned build
boundary, which leaves 142,239 rows in the training set. 53,258 of them are
late, a 37.44% positive rate.

The label is a proxy and the model is only ever as good as the proxy. MBTA's
API never reports when a bus actually arrived. What it reports is a
prediction, which keeps updating until the prediction disappears from the
feed. The label here is the last predicted arrival seen before that
disappearance, minus the scheduled arrival. Late means that difference
exceeds 180 seconds. This is the best signal a free, keyless, polling-only
integration can get, and it is not ground truth.

The proxy at least behaves the way the design assumes. Across the 142,258
usable labels, the median gap between a trip-stop's last sighting and its
final predicted arrival is 2.6 seconds, and the mean is 14 seconds. The
prediction vanishes essentially at the moment it last predicted. That is
consistent with "the bus arrived and the prediction retired". It does not
prove it. 414 trip-stops (0.29%) vanished more than 30 minutes before their
scheduled arrival, and those are kept in the training set with whatever delay
was last predicted.

## Why 2.77 days and not seven

The design doc's own gate wants seven days before a "beats guessing" claim.
`scripts/train.py` still enforces it and still prints the banner. It fired on
this run:

```
PRELIMINARY - insufficient data for the report deliverable
Data window is 2.79 days; the report deliverable (M3/M4) needs >= 7.
```

The gate was not lowered to make the banner go away. The operator's call was
to ship at the real window rather than wait, so the window is stated plainly
everywhere and the gate stays where it is. Read the results as measurements
of this window, not as claims about MBTA bus service.

## A bug this window found

M2's label layer grouped `stop_events` by `(trip_id, stop_id)`. MBTA reuses
`trip_id` across service dates. On M2's four-hour dataset a `trip_id` cannot
recur, so the grain looked fine. On 2.77 days it is wrong: 17,968 of 172,297
`(trip_id, stop_id)` pairs had snapshots more than six hours apart, and
sampling them showed one `trip_id` carrying two distinct `scheduled_arrival`
values. Two days of the same scheduled run were collapsing into one label
whose `final_predicted_arrival` came from whichever day was later.

The group key now carries the GTFS service date. After the fix, no group
spans more than 5 hours 56 minutes. Two regression tests in
`tests/test_labels.py` cover it. The same reuse also meant the "previous stop
on this trip" feature could inherit its delay from yesterday's last stop, so
that subquery is now scoped to the same service date too.

This is worth naming because it is the whole argument for growing the window.
The bug was invisible at four hours and obvious at three days.

## The split

Split by time, never randomly. A random split on a 2.77 day window would put
a route-hour's rows on both sides and score the model on data chronologically
interleaved with its own training data.

| split | rows | first scheduled_arrival | last scheduled_arrival | late | late rate |
|---|---|---|---|---|---|
| train | 99,590 | 2026-08-12 21:17:00-04 | 2026-08-15 00:04:00-04 | 38,529 | 38.69% |
| test | 42,649 | 2026-08-15 00:05:00-04 | 2026-08-15 16:08:00-04 | 14,729 | 34.54% |

The cut is placed at the first 70% of rows and then pushed forward to the
next change of `scheduled_arrival`, so no single minute lands on both sides.
Without that, two runs against the same table produced train late rates of
38.69% and 38.68%, because Postgres ordered the boundary minute's rows
differently each time. The SQL now carries a full natural-key tiebreak and
two consecutive runs produce byte-identical output.

What the boundary actually means, in days: train is Wednesday night,
Thursday and Friday. Test is Saturday, all of it, and nothing else. The
model has never seen a Saturday daytime. The split's 4.15 point drop in late
rate is a weekend effect being measured for the first time at test.

| split | service date | wall-clock day | rows | late rate |
|---|---|---|---|---|
| train | 2026-08-12 | Wed | 10,853 | 34.50% |
| train | 2026-08-13 | Thu | 32,909 | 36.11% |
| train | 2026-08-14 | Fri | 55,828 | 41.02% |
| test | 2026-08-14 | Sat (late-night service) | 4,785 | 34.86% |
| test | 2026-08-15 | Sat | 37,864 | 34.49% |

## Point-in-time correctness, and how it was verified

`route_hour_historical_late_rate` is a per-row as-of aggregate: for the row
being featured, the late rate over trip-stops on the same route, in the same
local hour, with a strictly earlier `scheduled_arrival`. Never the whole
table. NULL when no such history exists, never a fabricated default.

Prose restating a `WHERE` clause is not verification. Three checks run
instead, all in `scripts/verify-point-in-time.py`, against the live table:

```
[PASS] independent_recomputation: 142239 rows compared against a GROUPS-framed
       window function, 0 disagree (threshold: 0)
[PASS] null_arithmetic: 1093 rows have a NULL route_hour_historical_late_rate,
       1093 rows are tied-earliest in their (route, hour) bucket
[PASS] no_future_in_the_past: 142239 rows checked, 0 have a contributing label
       scheduled at or after the row itself (threshold: 0)
```

The first recomputes every row's aggregate a second, structurally different
way, as a window function with a `GROUPS BETWEEN UNBOUNDED PRECEDING AND 1
PRECEDING` frame, which covers exactly the rows with a strictly smaller
`scheduled_arrival`. Two independent implementations agreeing on all 142,239
rows is stronger evidence than one implementation read carefully. The second
pins down what a NULL means: exactly the rows tied for earliest in their
bucket, so NULLs cannot be coming from a failed join. The third states the
property directly over the whole table.

The durable version is a mutation test in `tests/test_features.py`. It builds
a row's features, then inserts 20 same-route same-hour labels scheduled after
it, all late, enough to drag a leaky aggregate from 0.50 toward 0.95,
rebuilds, and asserts the earlier row did not move. Changing the future and
checking that the past holds still is the test with teeth.

There is a second, stricter reading of "training window only": compute the
route-hour rate once from the train split and freeze it. That removes any
overlap between a test row's history and the test period itself, which is the
adjacency the M2 report flagged as a worry. Both were run. The frozen version
costs 0.0007 PR-AUC for both models. The worry is real and it is worth 0.0007
on this window.

## The horizon this actually honors

This is the largest caveat in the report and it belongs next to the results,
not after them.

The design doc's unit of prediction is a trip-stop **10 minutes before
scheduled arrival**. The features do not honor that. They are computed from
`trip_stop_labels`, which by construction only contains *settled*
determinations, and a determination settles when the prediction disappears
from the feed, which is roughly when the bus arrives. Measured on the exact
142,239-row training set:

| feature | rows with the input available | of those, settled 10+ min before scheduled arrival |
|---|---|---|
| `current_delay_persistence_seconds` | 136,733 | 862 (0.63%) |
| `route_hour_historical_late_rate` | 141,146 | 1,386 (0.98%) |

So for 99% of rows, at least one input carries information that did not exist
at the stated prediction horizon. This is not a leak in the "future data"
sense. Every input is still strictly earlier than the row it describes, which
is what the three checks above prove. It is a leak in the "not yet knowable
at T minus 10 minutes" sense, and it is a different property.

Two consequences, both of which change how the results table reads:

1. **The persistence baseline is stronger than the spec's persistence
   baseline.** The spec says "current delay carries forward". This one
   carries the previous stop's *final settled* delay, which for a city bus
   usually settled two to five minutes before the next stop's scheduled
   arrival, i.e. inside the horizon. Beating it is a harder win than the
   spec asked for. Losing to it would be worse than it looks.
2. **Every model's PR-AUC here is an upper bound**, not an estimate, on what
   a horizon-honoring model would score.

The fix is known and is not a redesign. The as-of predicted delay is
available directly from `stop_events`: take the last snapshot with
`polled_at <= scheduled_arrival - 10 minutes` and read `predicted_arrival`.
That is a one-pass aggregate over `stop_events` on the same pattern as
`pulse/labels.py`'s group query, which runs in 13.5 seconds over 10.5M rows.
It needs a migration, a new feature column, tests, and a re-derivation of
every number here. It is the first thing M5 should do.

## Results

All five candidates, on the held-out test split. Threshold chosen on train,
then applied fixed to test.

| candidate | PR-AUC | ROC-AUC | recall@P>=0.80 (oracle) | train threshold | test precision | test recall | test accuracy | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline: always-on-time | 0.3454 | 0.5000 | n/a | n/a | n/a | n/a | n/a | - | - | - | - |
| baseline: route-hour historical rate | 0.4913 | 0.6711 | n/a | 0.7660 | 0.6514 | 0.0222 | 0.6582 | 27,745 | 175 | 14,402 | 327 |
| baseline: persistence | 0.8636 | 0.9224 | 0.8758 | 1.0000 | 0.9371 | 0.8758 | 0.9368 | 27,054 | 866 | 1,830 | 12,899 |
| LogisticRegression | 0.9678 | 0.9799 | 0.9373 | 0.2208 | 0.7999 | 0.9374 | 0.8974 | 24,466 | 3,454 | 922 | 13,807 |
| HistGradientBoosting | **0.9711** | 0.9819 | **0.9557** | 0.1762 | 0.7782 | 0.9645 | 0.8928 | 23,871 | 4,049 | 523 | 14,206 |

Reference point: predicting on-time for every test row scores 0.6546 accuracy
and 0.3454 PR-AUC.

Sensitivity, route-hour aggregate frozen at the train/test boundary:

| candidate | PR-AUC | delta vs per-row as-of | recall@P>=0.80 (oracle) |
|---|---|---|---|
| LogisticRegression | 0.9672 | -0.0007 | 0.9386 |
| HistGradientBoosting | 0.9703 | -0.0007 | 0.9545 |

Three things this table says, in order of how much they matter.

**Both models beat all three baselines on PR-AUC and on recall at 80%
precision.** HistGradientBoosting reaches 0.9711 PR-AUC against persistence's
0.8636, and 0.9557 recall at the precision floor against persistence's
0.8758. Given the horizon caveat above, the honest phrasing of the win is
narrow: on 2.77 days, with no Saturday in training, both models outrank a
persistence baseline that is itself using post-horizon information.

**Neither model delivers the 80% precision guarantee at a threshold chosen
without seeing the test set.** This is the finding that matters most for
shipping. LogisticRegression's train-chosen cutoff of 0.2208 produces 13,807
true positives against 3,454 false positives on test, which is 0.7999
precision. That misses the 0.80 floor by 0.0001, which is a hair, but it is
on the wrong side of the line. HistGradientBoosting's cutoff of 0.1762 lands
at 0.7782, missing by 0.022. Both models *can* hit 80% precision on test at
some threshold, which is what the oracle column reports. Neither can be
*aimed* there from training data alone. The ranking transfers from train to
Saturday. The calibration does not.

**The route-hour historical rate baseline is barely better than a coin flip
on this window, and worse than useless as a classifier.** 0.4913 PR-AUC
against a 0.3454 prevalence floor. At its train-chosen threshold it fires on
502 test rows and gets 327 right, catching 2.2% of the late trip-stops. Three
days is not enough history for a route-hour rate to mean anything, which the
1,093 rows with no history at all also say.

The registered model is `gradient_boosting`, selected by test PR-AUC, the
design doc's primary metric. The selection rule was fixed before the run.
Worth stating plainly: if the deployment constraint is precision >= 0.80 at a
threshold you have to pick in advance, neither model qualifies today, and the
persistence baseline is the only candidate that holds the floor out of sample
(0.9371 precision at recall 0.8758). Registering by PR-AUC and saying that
out loud is more useful than swapping the selection rule after seeing the
test numbers.

## Defending the metrics

The assignment is to defend PR-AUC and recall at precision, not to report
them. The argument is about a 34.54% positive rate and an asymmetric cost.

**Accuracy is disqualified by the base rate.** 65.46% of test trip-stops are
on time. A model that predicts "on time" for every single one scores 0.6546
accuracy while being worth nothing to a rider. HistGradientBoosting scores
0.8928. The gap between a useless model and a useful one is 24 accuracy
points, which is not enough resolution to make decisions with. Worse,
accuracy moves the wrong way when you tune for the thing riders care about:
HistGradientBoosting has *lower* accuracy than the persistence baseline
(0.8928 against 0.9368) and higher PR-AUC (0.9711 against 0.8636), because it
accepts 4,049 false positives to cut false negatives from 1,830 to 523.
Ranked by accuracy you would ship the worse model.

**ROC-AUC is disqualified by what it averages over.** ROC-AUC is the average
true-positive rate across all false-positive rates, and the false-positive
rate has 27,920 on-time trip-stops in its denominator. That denominator
dominates. On this test split ROC-AUC compresses everything into a narrow,
flattering band: 0.9224 for the persistence baseline, 0.9799 and 0.9819 for
the two models. It rates the route-hour baseline at 0.6711, which sounds like
a weak-but-real signal, while PR-AUC rates it 0.4913 and its actual behaviour
at an operating point is 327 correct alerts out of 14,729 late trip-stops.
PR-AUC has the right denominator, because precision only counts the rows the
model actually flagged.

**PR-AUC is the summary; recall at precision >= 0.80 is the operating
point.** PR-AUC integrates over every threshold and answers "how good is the
ranking". That is the right summary when you have not yet chosen where to
fire. Recall at a precision floor answers the product question: hold false
alarms to one in five, and how many late buses do you still catch. The floor
is 0.80 because the cost is asymmetric in a specific direction. A false
"late" costs a rider a few minutes of standing at a stop early. A false "on
time" costs them the bus. So recall on the late class is what we want to
maximize, and precision is the constraint we hold while doing it, not the
other way around.

## The threshold and the confusion matrix

The threshold is chosen on the train split, as the lowest cutoff whose train
precision reaches 0.80. Lowest, because among cutoffs that clear the floor,
the lowest keeps the most recall. It is then frozen and applied to test.

Choosing it on test instead would produce a better-looking confusion matrix
and a meaningless one: searching thresholds on the data you are scoring is
tuning on the held-out split, and the "recall@P>=0.80" column reports exactly
that oracle number for comparison. The gap between the oracle column and the
realized test recall is how much of the oracle number was hindsight.

Registered model, `gradient_boosting`, at its train-chosen threshold of
0.1762, on 42,649 held-out test rows:

|  | predicted on-time | predicted late |
|---|---|---|
| **actually on-time** (27,920) | 23,871 | 4,049 |
| **actually late** (14,729) | 523 | 14,206 |

Precision 0.7782. Recall 0.9645. Accuracy 0.8928.

What the threshold was chosen to satisfy: precision >= 0.80 on the training
split, i.e. no more than one false "late" in five alerts. What it actually
delivered on Saturday: 0.7782, roughly one false alert in 4.5. It missed.

The 523 false negatives are the number that matters. Those are riders told
their bus was on time when it ran more than three minutes late. 523 of 14,729
late trip-stops, 3.55%, is the miss rate this operating point buys, and it
buys it with 4,049 false alarms. LogisticRegression at its own train-chosen
threshold sits at the other end of the same tradeoff: 922 false negatives and
3,454 false alarms.

## What this cannot say

Exhaustively, with the experiment that would settle each one.

**No weekly seasonality.** The window covers four wall-clock day types:
Wednesday (7,233 rows), Thursday (35,698), Friday (56,385), Saturday
(42,923). Monday, Tuesday and Sunday are absent entirely, and there is
exactly one weekend day. `day_of_week` is a model feature and it has four of
seven values, one of which appears only in test. *Settle it:* run ingestion
for three full weeks and refit, then report metrics per day type rather than
pooled.

**Train has no Saturday and test is entirely Saturday.** The 70/30 temporal
cut landed at 2026-08-15 00:04. Every metric in this report is a
Wednesday-to-Friday model scored on a Saturday. The 4.15 point late-rate drop
between the splits is a real regime change the model was not trained through,
and it is confounded with everything else the split measures. *Settle it:*
rolling-origin evaluation, refit each day and score the next, and report the
distribution of metrics rather than one draw.

**One weather regime.** Three days of August in Boston. No precipitation
variable exists in the feature set and no weather variation exists in the
window to learn one from. *Settle it:* join hourly KBOS observations to the
label table by hour and test whether a precipitation feature moves PR-AUC.

**One segment of the service calendar.** August is summer schedule. No
school-year service, no holiday, no planned diversion. *Settle it:* hold out
a September week and score the August-trained model on it without refitting.

**The label set is biased toward hours the laptop was awake.** This is not a
hedge, it is measurable. 45 of the 96 (service date x hour) cells in the
training set are empty, 46.9%. 2026-08-12 has only 21:00 through 01:00.
2026-08-13 lost 01:52 through 14:05 to a single 12.22 hour sleep gap.
2026-08-15 has nothing after 16:00. Only 2026-08-14 has both a morning and an
evening peak. On top of that, 32,047 labels (17.30% of everything closed)
were dropped as `gap_abutted`, and those exclusions are not random: they
cluster at the edges of sleep windows, which are overnight and early morning.
The model's picture of "hour of day" is built from whichever hours the
machine happened to be up. *Settle it:* run ingestion on hardware that does
not sleep for a full week and compare the hour-of-day distribution of labels
before and after.

**The label is a proxy for arrival, not an arrival.** Covered above. The
specific failure it cannot distinguish is a trip cancelled or dropped from
the feed near its predicted arrival, which looks identical to a bus arriving.
414 trip-stops vanished more than 30 minutes before their scheduled arrival
and were kept. *Settle it:* compare against MBTA's published historical bus
performance dataset for overlapping trips and quantify the proxy-vs-actual
delta per route.

**The features do not honor the 10-minute horizon.** 0.63% and 0.98%
availability, as measured above. Every PR-AUC here is an upper bound.
*Settle it:* build the as-of feature from `stop_events` snapshots, described
in that section, and re-run the whole table.

**Beating persistence on this window may not hold on a full week.** The
margin is 0.9711 against 0.8636 PR-AUC, which looks comfortable, but it is
one draw from one test block on one day type, and the baseline it beats is
itself using post-horizon information. *Settle it:* the rolling-origin
evaluation above, reporting the model-minus-persistence PR-AUC delta per fold
with its spread.

**No spatial holdout.** Train and test share all 13 routes and all 719 stops.
The model may be memorizing stop-level base rates rather than learning
anything transferable. Nothing here tests generalization to a route it has
not seen. *Settle it:* hold out entire routes, train on the rest, and measure
the drop.

**Threshold stability is unknown and already looks shaky.** One threshold,
chosen on one train split, missed the precision floor on the very next day.
That is a single observation, not a trend. *Settle it:* prequential
evaluation, fix the threshold from day N and report the precision actually
realized on day N+1 across a full week.

**The rerun is not bit-for-bit pinned, only snapshot-pinned.** `--until`
bounds the snapshots considered, but the gap ledger is read unwindowed, so
rerunning these commands after the laptop sleeps again can reclassify a
small number of labels near the boundary and shift the counts. This is a
reproducibility limit, not a data limit, and it is the one place the numbers
here could fail to reproduce from the stated commands. *Settle it:* bound
`fetch_poll_runs` at `until`, rebuild, and confirm the counts land where
they did. Details under Reproduce.

**Task B is not done.** The design doc names a supporting regression task
(expected delay in seconds, MAE against persistence). `delay_seconds` is
already carried on `trip_stop_labels` and in the training-set join, so it is
additive rather than a schema change, but it has not been run.

## Reproduce

Requires local Postgres 16 with database `pulse` and the migrations applied.
The `--until` boundary pins which snapshots the label build considers, which
is what keeps the numbers above stable while the poller keeps running. It is
not a complete pin, and the gap is named below the commands.

```bash
# 0. Schema (idempotent).
uv run python scripts/migrate.py

# 1. Labels over the full window, pinned. ~70s at 10.5M stop_events rows.
#    Prints volume / null_rate / label_rate gates, exits non-zero on breach.
uv run python scripts/build-labels.py \
  --until 2026-08-15T20:11:17Z --as-of 2026-08-15T20:11:17Z

# 2. Features. Slow by design at this volume: pulse/features.py uses
#    correlated subqueries, which are O(n^2) and took 30 minutes for 142,239
#    rows. Correct, and the next thing to rewrite.
uv run python scripts/build-features.py --until 2026-08-15T20:11:17Z

# 3. Point-in-time correctness, three checks, ~1s. Exits non-zero on any FAIL.
uv run python scripts/verify-point-in-time.py

# 4. Train, evaluate, log to local MLflow, register the best model.
#    ~13s. Two consecutive runs produce identical output.
uv run python scripts/train.py

# 5. Tests.
uv run pytest -q          # 148 passed

# Optional: browse the runs, including baselines and the frozen-aggregate
# sensitivity runs, tagged kind=baseline and kind=sensitivity.
uv run mlflow ui --backend-store-uri file:./mlruns
```

**Where the pin leaks.** `--until` bounds which `stop_events` snapshots the
label build considers. It does not bound the gap ledger:
`pulse/labels.py:fetch_poll_runs` reads all of `poll_runs`, unwindowed,
because gap intervals are derived from the ledger as it stands at run time.
So a rerun tomorrow, after the laptop has slept again, sees a gap interval
opening just past the boundary, and any trip-stop whose trailing settle
window overlaps it flips to `gap_abutted`. A small number of labels near the
boundary can move, and the counts downstream with them. The change that
closes this is bounding the ledger read at `--until` too. It was not made
here because `poll_runs` already had rows past `20:11:17Z` when the label
build ran, so bounding it now would reclassify gaps and force a label rebuild
plus a 30 minute feature rebuild to re-verify.

Rebuilding from scratch rather than incrementally is deliberate here: the
label table had to be truncated and rebuilt after the service-date grain fix,
because every row built under the old grain was keyed wrong. Both transforms
are idempotent and backfillable by window otherwise (`docs/lineage.md`).

`mlruns/` and `models/*.joblib` are gitignored. `models/REGISTRY.md` is the
durable record and carries the entry for this run, including the commit HEAD
pointed at while it ran.
