# The program, in a data engineer's life

Operator directive 2026-08-13: the five projects run in the flavour of a
data engineer living inside an AI/ML program. The syllabus says what
ships; this document says who ships it. Every project gets a DE artifact
standing next to the syllabus's ML deliverable, and the story each repo
tells is the data engineer's story: the model is a downstream consumer,
the pipeline is the product.

Binding across all five projects:

- Data contracts at every boundary: ingestion validates against a stated
  schema and the contract lives in the repo, versioned.
- Idempotency and backfill are designed, not hoped for: every pipeline
  can rerun any window without duplicating or losing facts.
- Quality gates run in the pipeline, not in the notebook: null-rate,
  freshness, volume, and distribution checks with stated thresholds, and
  the run record shows them passing and failing.
- Lineage is written down: raw -> staged -> features -> training set,
  each layer a named table or artifact with its producing code linked.
- The pipeline is observable: one summary line per run, a check script,
  and an honest account of what the data cannot support (Pulse: the
  ~4.2% origin-stop rows with no arrival signal).

Per project:

1. Pulse — the star DE artifact is the ingestion + transform stack:
   MBTA feed -> stop_events (raw facts) -> labeled trips (staged) ->
   feature tables, with quality gates between layers and a backfillable
   label-derivation job. The model report cites the pipeline's guarantees
   as part of its defense.
2. Pulse on AWS — the DE artifact is the data path under the service:
   API -> queue -> worker -> DynamoDB with delivery semantics stated
   (at-least-once, dedup key), plus cost-per-thousand-predictions in the
   SLO doc.
3. CareScribe — the DE artifact is the curation pipeline: de-identification
   as a staged, audited transform with a re-identification risk note and
   a data card for the training set.
4. Resolve — the DE artifact is the knowledge-base ETL: document
   ingestion, chunking as a versioned transform, index freshness, and an
   eval set built like a dataset (provenance per item).
5. Atlas capstone — the DE artifact is the product's data platform: one
   diagram and one contract per feature showing where every number on the
   dashboard comes from.

Voice: same house rules as everything else. Plain sentences, numbers with
their regime, failures published.
