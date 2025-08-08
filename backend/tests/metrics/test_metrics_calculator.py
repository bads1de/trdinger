import numpy as np
import pytest
from sklearn.metrics import (
    confusion_matrix,
    multilabel_confusion_matrix,
    average_precision_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

from app.services.ml.evaluation.enhanced_metrics import (
    EnhancedMetricsCalculator,
    MetricsConfig,
)


def test_binary_specificity_sensitivity_ppv_npv():
    # y_true / y_pred から期待値を明示的に算出
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 1])

    cm = confusion_matrix(y_true, y_pred)
    assert cm.shape == (2, 2)
    tn, fp, fn, tp = cm.ravel()

    expected_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    expected_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    expected_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    expected_ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    calc = EnhancedMetricsCalculator(MetricsConfig(include_confusion_matrix=True))
    metrics = calc.calculate_comprehensive_metrics(y_true, y_pred)

    assert (
        pytest.approx(metrics["specificity"], rel=1e-9, abs=1e-9)
        == expected_specificity
    )
    assert (
        pytest.approx(metrics["sensitivity"], rel=1e-9, abs=1e-9)
        == expected_sensitivity
    )
    assert pytest.approx(metrics["npv"], rel=1e-9, abs=1e-9) == expected_npv
    assert pytest.approx(metrics["ppv"], rel=1e-9, abs=1e-9) == expected_ppv


def test_multiclass_specificity_weighted_ppv_npv():
    # 3クラスの例
    y_true = np.array([0, 1, 2, 2, 1, 0, 2, 1, 0])
    y_pred = np.array([0, 2, 2, 1, 1, 0, 2, 0, 0])

    labels = np.unique(y_true)
    mcm = multilabel_confusion_matrix(y_true, y_pred, labels=labels)
    tn = mcm[:, 0, 0]
    fp = mcm[:, 0, 1]
    fn = mcm[:, 1, 0]
    tp = mcm[:, 1, 1]

    # クラス頻度に基づく重み
    counts = np.array([(y_true == lab).sum() for lab in labels], dtype=float)
    weights = counts / counts.sum()

    eps = 1e-12
    specificity_vec = tn / (tn + fp + eps)
    sensitivity_vec = tp / (tp + fn + eps)
    npv_vec = tn / (tn + fn + eps)
    ppv_vec = tp / (tp + fp + eps)

    expected_specificity = float(np.average(specificity_vec, weights=weights))
    expected_sensitivity = float(np.average(sensitivity_vec, weights=weights))
    expected_npv = float(np.average(npv_vec, weights=weights))
    expected_ppv = float(np.average(ppv_vec, weights=weights))

    calc = EnhancedMetricsCalculator(MetricsConfig(include_confusion_matrix=True))
    metrics = calc.calculate_comprehensive_metrics(y_true, y_pred)

    assert (
        pytest.approx(metrics["specificity"], rel=1e-9, abs=1e-9)
        == expected_specificity
    )
    assert (
        pytest.approx(metrics["sensitivity"], rel=1e-9, abs=1e-9)
        == expected_sensitivity
    )
    assert pytest.approx(metrics["npv"], rel=1e-9, abs=1e-9) == expected_npv
    assert pytest.approx(metrics["ppv"], rel=1e-9, abs=1e-9) == expected_ppv


def test_probability_metrics_binary():
    # バイナリ確率系の検証
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8, 0.6])

    expected_auc_roc = roc_auc_score(y_true, y_proba)
    expected_auc_pr = average_precision_score(y_true, y_proba)
    expected_log_loss = log_loss(y_true, y_proba)
    expected_brier = brier_score_loss(y_true, y_proba)

    calc = EnhancedMetricsCalculator(
        MetricsConfig(include_pr_auc=True, include_roc_auc=True)
    )
    metrics = calc.calculate_comprehensive_metrics(y_true, y_pred, y_proba)

    assert pytest.approx(metrics["roc_auc"], rel=1e-9, abs=1e-9) == expected_auc_roc
    assert pytest.approx(metrics["pr_auc"], rel=1e-9, abs=1e-9) == expected_auc_pr
    assert pytest.approx(metrics["log_loss"], rel=1e-9, abs=1e-9) == expected_log_loss
    assert pytest.approx(metrics["brier_score"], rel=1e-9, abs=1e-9) == expected_brier
