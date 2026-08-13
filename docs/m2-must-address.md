# M2 must address (from the M1 final review, 2026-08-13)

Carried obligations. M2's label/feature work does not start clean without
these; each traces to a measured finding in the M1 review.

1. Label derivation consumes the poll_runs gap ledger: no label closes
   across a recorded gap (a polling gap is indistinguishable from the
   prediction-disappearance event the label is defined by).
2. Retention or monthly partitioning on polled_at, plus keeping the disk
   gate honest as volume grows (204 B/row measured; volume was at 89%
   when the gate landed). Partitioning also keeps cycle time flat: the
   unique index is a random-insert btree.
3. Quality gates run in the pipeline with pass/fail persisted to the run
   record (check.py gates exist; M2 wires them into the transform runs).
4. Versioned inbound data contract + fail-loud payload validation: a
   silent MBTA schema change must surface as an error, not as a rising
   skipped count.
5. Service-date normalization at the staged layer: GTFS service day runs
   to ~3 AM; raw service_date is the polled_at calendar date and splits
   late-night trips at midnight. The (route_id, service_date) index is on
   the un-normalized column.
6. Filter the ~3% origin-stop rows (both arrival fields null) at label
   build. Never impute as on-time. Null cross-tab verified clean: zero
   mixed-null rows.
7. Lineage doc: stop_events (raw) -> labeled trip-stops (staged) ->
   features -> training set, each a named table with producing code.
8. Smaller carried items: status column is 100% null (drop or document),
   cycle deadline (worst case ~300s hangs eat samples), routes_ok
   overstates coverage at the pagination page cap (surface pages in the
   summary), connect_timeout in the DSN, test_db fixture should apply
   migrations/*.sql not just 001, HTTP Retry-After-aware backoff.
