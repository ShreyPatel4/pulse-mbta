"""Metrics for Task A (docs/2026-08-13-pulse-design.md): P(arrival delay >
180s). PR-AUC and recall at precision >= 0.80 -- riders are punished more by
a false "on time" than a false "late", so precision on the "late" class is
the operating constraint the design doc names, not accuracy or ROC-AUC.

Split out from scripts/train.py so recall_at_precision's edge case (no
threshold reaches the target precision at all) is unit-testable without
fitting a real sklearn model.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import average_precision_score, precision_recall_curve


def pr_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """Average precision (PR-AUC). Thin wrapper -- named to match the design
    doc's vocabulary and to keep scripts/train.py's metric calls symmetric
    with recall_at_precision below."""
    return float(average_precision_score(y_true, y_score))


def recall_at_precision(
    y_true: Sequence[int], y_score: Sequence[float], min_precision: float = 0.80
) -> float | None:
    """Max recall among REAL thresholds where precision >= min_precision.

    Returns None when no real threshold reaches min_precision at all -- a
    genuine, reportable outcome ("this model/baseline cannot hit 80%
    precision on this data").

    sklearn.metrics.precision_recall_curve appends one trivial trailing
    point, (precision=1.0, recall=0.0), that has NO corresponding threshold
    -- `thresholds` is one element shorter than `precision`/`recall` by
    construction. That point trivially "reaches" any min_precision <= 1.0
    for any classifier (predict nothing positive -> zero false positives ->
    precision defined as 1.0), so if it isn't excluded, this function could
    never actually return None: it would silently report the useless
    "predict nothing" operating point as if 80% precision were achievable.
    Slicing to precision[:-1]/recall[:-1] restricts the search to points
    that correspond to a real decision threshold.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    real_precision, real_recall = precision[:-1], recall[:-1]
    eligible_recall = real_recall[real_precision >= min_precision]
    if eligible_recall.size == 0:
        return None
    return float(eligible_recall.max())
