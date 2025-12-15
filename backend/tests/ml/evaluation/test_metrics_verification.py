import sys
import os
import threading
import numpy as np
import pytest

# パス設定
sys.path.append("c:/Users/buti3/trading/backend")

from app.services.ml.evaluation.metrics import MetricsCalculator, MetricsConfig


def test_metrics_calculator_lock_reentrancy():
    """MetricsCalculatorのロックが再入可能であることを確認"""
    calculator = MetricsCalculator()

    # 外部でロックを取得
    with calculator._lock:
        try:
            # 内部でさらにロックを取得するメソッドを呼び出し
            # RLockであればデッドロックせずに成功する
            calculator.record_metric("test_metric", 1.0)
            success = True
        except RuntimeError:
            success = False

    assert success, "再入可能ロック(RLock)が適切に機能していません（デッドロック発生）"


def test_metrics_calculator_basic_mode():
    """MetricsCalculatorの軽量モード(basic)を確認"""
    config = MetricsConfig(
        include_balanced_accuracy=True,
        include_roc_auc=True,
        include_confusion_matrix=True,
        include_classification_report=True,
    )
    calculator = MetricsCalculator(config)

    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0])
    y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.55, 0.45]])
    class_names = ["class_0", "class_1"]

    # Basicモード実行
    metrics_basic = calculator.calculate_comprehensive_metrics(
        y_true, y_pred, y_proba, class_names, level="basic"
    )

    # Basicモードに含まれるべき指標
    assert "accuracy" in metrics_basic
    assert "f1_score" in metrics_basic
    assert "balanced_accuracy" in metrics_basic  # コードに追加した通り

    # Basicモードに含まれてはいけない指標
    assert "roc_auc" not in metrics_basic
    assert "confusion_matrix" not in metrics_basic

    # Fullモード実行
    metrics_full = calculator.calculate_comprehensive_metrics(
        y_true, y_pred, y_proba, class_names, level="full"
    )

    # Fullモードに含まれるべき指標
    assert "roc_auc" in metrics_full
    assert "confusion_matrix" in metrics_full


if __name__ == "__main__":
    # 手動実行用
    try:
        test_metrics_calculator_lock_reentrancy()
        print("✅ Lock reentrancy test passed")
        test_metrics_calculator_basic_mode()
        print("✅ Basic mode test passed")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1)


