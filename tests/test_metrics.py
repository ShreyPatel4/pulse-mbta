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


def test_threshold_for_precision_agrees_with_recall_at_precision_on_the_same_data():
    # The two functions answer the same question from opposite ends: one
    # returns the recall, the other returns the cutoff that recall sits at.
    # Applying the cutoff must reproduce the recall exactly, otherwise
    # "recall@P>=0.80" and the confusion matrix printed beneath it would
    # describe two different operating points.
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    y_score = [0.95, 0.9, 0.85, 0.8, 0.4, 0.35, 0.3, 0.2, 0.15, 0.05]

    recall = metrics.recall_at_precision(y_true, y_score, min_precision=0.80)
    threshold = metrics.threshold_for_precision(y_true, y_score, min_precision=0.80)
    assert threshold is not None

    confusion = metrics.confusion_at_threshold(y_true, y_score, threshold)
    assert confusion["recall"] == recall
    assert confusion["precision"] >= 0.80


def test_threshold_for_precision_is_none_when_target_never_reached():
    y_true = [1, 1, 0, 0]
    y_score = [0.1, 0.2, 0.8, 0.9]
    assert metrics.threshold_for_precision(y_true, y_score, min_precision=0.80) is None


def test_threshold_for_precision_picks_the_lowest_eligible_cutoff():
    # Two eligible cutoffs exist (top-1 and top-2 both give precision 1.0).
    # The lower one must win: among cutoffs that clear the precision floor,
    # the lowest keeps the most recall.
    y_true = [1, 1, 0, 0]
    y_score = [1.0, 0.9, 0.8, 0.1]
    assert metrics.threshold_for_precision(y_true, y_score, min_precision=0.80) == 0.9


def test_confusion_at_threshold_counts_and_uses_greater_or_equal():
    # A row scoring exactly at the threshold is predicted late (sklearn's
    # convention, and threshold_for_precision's thresholds come from the same
    # curve, so anything else would shift every count by the boundary rows).
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.5, 0.5, 0.1]
    c = metrics.confusion_at_threshold(y_true, y_score, 0.5)
    assert (c["tp"], c["fp"], c["fn"], c["tn"]) == (2, 1, 0, 1)
    assert c["precision"] == 2 / 3
    assert c["recall"] == 1.0


def test_roc_auc_stays_high_where_precision_collapses():
    """The empirical case for not ranking on ROC-AUC, in miniature. 30
    negatives, 2 positives. Both positives outrank 28 of the 30 negatives, so
    ROC-AUC reads 0.93 and looks like a good model. But the two highest-scored
    rows of all are negatives, so a rider acting on the top of this ranking is
    wrong twice before being right once. PR-AUC says 0.42 and
    recall@precision>=0.80 says the operating point does not exist.
    docs/report.md makes this argument against the real test split; this is
    the version that runs in CI."""
    y_true = [0, 0, 1, 1] + [0] * 28
    y_score = [1.0, 0.99] + [0.98, 0.97] + [0.5 - 0.01 * i for i in range(28)]
    assert metrics.roc_auc(y_true, y_score) > 0.92
    assert metrics.pr_auc(y_true, y_score) < 0.5
    assert metrics.recall_at_precision(y_true, y_score, min_precision=0.80) is None


def test_confusion_at_threshold_accuracy_counts_both_correct_classes():
    y_true = [1, 1, 0, 0]
    y_score = [0.9, 0.5, 0.5, 0.1]
    c = metrics.confusion_at_threshold(y_true, y_score, 0.5)
    assert c["accuracy"] == 3 / 4  # tp=2, tn=1, of 4


def test_confusion_at_threshold_precision_is_none_when_nothing_predicted_late():
    # 0/0 is undefined, not zero. Reporting 0.0 would read as "wrong every
    # time" rather than "never fired".
    y_true = [1, 0, 1]
    y_score = [0.1, 0.2, 0.3]
    c = metrics.confusion_at_threshold(y_true, y_score, 0.99)
    assert (c["tp"], c["fp"]) == (0, 0)
    assert c["precision"] is None
    assert c["recall"] == 0.0


def test_recall_at_precision_constant_score_with_no_real_threshold_above_target_is_none():
    # "Always on-time" baseline: constant score of 0 for every row -- only
    # one real threshold exists (predict everything positive), with
    # precision = prevalence (2/5 = 0.4), below the 0.80 target. The trivial
    # "predict nothing" trailing point is excluded (see recall_at_precision's
    # docstring), so this correctly returns None, not a misleading 0.0.
    y_true = [0, 0, 1, 0, 1]
    y_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert metrics.recall_at_precision(y_true, y_score, min_precision=0.80) is None
