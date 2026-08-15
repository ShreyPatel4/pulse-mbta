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
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


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


def roc_auc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    """ROC-AUC. This project does NOT rank models by it -- it is computed and
    printed only so docs/report.md's argument against it is made with this
    dataset's own numbers instead of a general claim. ROC-AUC averages over
    the true-negative rate, which on a 34%-positive problem is dominated by
    the on-time class nobody is asking about, so it stays flattering while
    precision on the late class falls apart."""
    return float(roc_auc_score(y_true, y_score))


def threshold_for_precision(
    y_true: Sequence[int], y_score: Sequence[float], min_precision: float = 0.80
) -> float | None:
    """The score cutoff that recall_at_precision's answer corresponds to:
    among thresholds whose precision >= min_precision, the LOWEST one, which
    is the one that keeps the most recall.

    This exists so a threshold can be chosen on the TRAIN split and then
    applied, fixed, to test. recall_at_precision alone is an oracle number --
    it searches thresholds on whatever data it is scoring, so calling it on
    test reports the best operating point in hindsight, which nothing
    deployable can reach. Choosing here on train and reporting the confusion
    matrix that threshold actually produces on test is the honest version of
    the same question, and the two numbers are worth printing side by side:
    the gap between them is how much of the oracle number was hindsight.

    Same trailing-point exclusion as recall_at_precision (see its docstring):
    `thresholds` is already one shorter than `precision`, so precision[:-1]
    lines them up and drops the no-threshold "predict nothing" point.

    Returns None when nothing reaches min_precision. sklearn's convention is
    that a sample is predicted positive when score >= threshold; callers must
    use >= to match.
    """
    precision, _recall, thresholds = precision_recall_curve(y_true, y_score)
    eligible = thresholds[precision[:-1] >= min_precision]
    if eligible.size == 0:
        return None
    return float(eligible.min())


def confusion_at_threshold(
    y_true: Sequence[int], y_score: Sequence[float], threshold: float
) -> dict[str, Any]:
    """Counts and rates at a FIXED threshold (predict late when score >=
    threshold). Returns tn/fp/fn/tp plus the precision and recall those
    counts imply, with precision None when nothing was predicted late at all
    (0/0 is undefined, not zero -- reporting it as 0.0 would read as "the
    model was wrong every time" instead of "the model never fired").

    fn is the count that matters most for this product: a trip-stop the model
    called on-time that was late. See docs/report.md on asymmetric cost.
    """
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(np.asarray(y_true, dtype=int), y_pred, labels=[0, 1]).ravel()
    predicted_late = int(tp + fp)
    actual_late = int(tp + fn)
    total = int(tn + fp + fn + tp)
    return {
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": (float(tp) / predicted_late) if predicted_late else None,
        "recall": (float(tp) / actual_late) if actual_late else None,
        # Reported, never optimized. See docs/report.md: on this dataset the
        # do-nothing classifier scores 65% accuracy, so accuracy cannot tell
        # a useful model from a useless one.
        "accuracy": (float(tn + tp) / total) if total else None,
    }
