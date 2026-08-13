"""Tests for pulse.metrics: pure metric functions over synthetic
(y_true, y_score) arrays -- no model fitting needed."""

from __future__ import annotations

from pulse import metrics


def test_pr_auc_perfect_separation_is_1():
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.0, 0.1, 0.2, 0.8, 0.9, 1.0]
    assert metrics.pr_auc(y_true, y_score) == 1.0


def test_pr_auc_random_scores_equal_prevalence_is_close_to_base_rate():
    # Constant score for every row -- PR-AUC degenerates to roughly the
    # positive-class prevalence (no ranking information at all).
    y_true = [0, 0, 0, 1]
    y_score = [0.5, 0.5, 0.5, 0.5]
    assert abs(metrics.pr_auc(y_true, y_score) - 0.25) < 1e-9


def test_recall_at_precision_perfect_classifier_recalls_everything():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 1.0]
    assert metrics.recall_at_precision(y_true, y_score, min_precision=0.80) == 1.0


def test_recall_at_precision_none_when_target_never_reached():
    # Every positive is outranked by a negative -- worst-case ordering, no
    # threshold can reach 80% precision.
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert metrics.recall_at_precision(y_true, y_score, min_precision=0.80) is None


def test_recall_at_precision_picks_the_best_eligible_threshold():
    # Scores rank: 1.0(pos), 0.9(pos), 0.8(neg), 0.1(neg). Thresholding at
    # the top 2 gives precision=1.0 (2 pos, 0 neg) at recall=1.0 (both
    # positives captured) -- eligible for the 0.80 target, and the best
    # available point.
    y_true = [1, 1, 0, 0]
    y_score = [1.0, 0.9, 0.8, 0.1]
    assert metrics.recall_at_precision(y_true, y_score, min_precision=0.80) == 1.0


def test_recall_at_precision_constant_score_with_no_real_threshold_above_target_is_none():
    # "Always on-time" baseline: constant score of 0 for every row -- only
    # one real threshold exists (predict everything positive), with
    # precision = prevalence (2/5 = 0.4), below the 0.80 target. The trivial
    # "predict nothing" trailing point is excluded (see recall_at_precision's
    # docstring), so this correctly returns None, not a misleading 0.0.
    y_true = [0, 0, 1, 0, 1]
    y_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert metrics.recall_at_precision(y_true, y_score, min_precision=0.80) is None
