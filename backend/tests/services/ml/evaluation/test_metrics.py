"""
metrics.py のテスト

app/services/ml/evaluation/metrics.py のテストモジュール
"""

import pytest
import numpy as np
import pandas as pd

from app.services.ml.evaluation.metrics import (
    MetricsConfig,
    MetricsCalculator,
    get_default_metrics,
    metrics_collector,
)


class TestMetricsConfig:
    """MetricsConfig クラスのテスト"""

    def test_default_config(self):
        """デフォルト設定"""
        config = MetricsConfig()
        assert config.include_balanced_accuracy is True
        assert config.include_pr_auc is True
        assert config.include_roc_auc is True
        assert config.include_confusion_matrix is True
        assert config.include_classification_report is True
        assert config.average_method == "weighted"
        assert config.zero_division == "warn"

    def test_custom_config(self):
        """カスタム設定"""
        config = MetricsConfig(
            include_balanced_accuracy=False,
            average_method="macro",
            zero_division=0,
        )
        assert config.include_balanced_accuracy is False
        assert config.average_method == "macro"
        assert config.zero_division == 0


class TestMetricsCalculator:
    """MetricsCalculator クラスのテスト"""

    def test_initialization_default_config(self):
        """デフォルト設定で初期化"""
        calc = MetricsCalculator()
        assert calc.config is not None
        assert calc.config.include_balanced_accuracy is True

    def test_initialization_custom_config(self):
        """カスタム設定で初期化"""
        config = MetricsConfig(include_balanced_accuracy=False)
        calc = MetricsCalculator(config)
        assert calc.config.include_balanced_accuracy is False

    def test_calculate_comprehensive_metrics_basic(self):
        """基本モードの包括的指標計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        result = calc.calculate_comprehensive_metrics(y_true, y_pred, level="basic")

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result

    def test_calculate_comprehensive_metrics_full(self):
        """フルモードの包括的指標計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])

        result = calc.calculate_comprehensive_metrics(
            y_true, y_pred, y_proba, level="full"
        )

        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "balanced_accuracy" in result

    def test_calculate_comprehensive_metrics_with_proba(self):
        """確率付きの包括的指標計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])

        result = calc.calculate_comprehensive_metrics(y_true, y_pred, y_proba)

        assert "roc_auc" in result
        assert "pr_auc" in result
        assert "log_loss" in result

    def test_calculate_comprehensive_metrics_with_class_names(self):
        """クラス名付きの包括的指標計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        class_names = ["negative", "positive"]

        result = calc.calculate_comprehensive_metrics(
            y_true, y_pred, class_names=class_names
        )

        assert "class_names" in result
        assert result["class_names"] == class_names

    def test_calculate_comprehensive_metrics_pandas_series(self):
        """pandas Seriesの入力"""
        calc = MetricsCalculator()
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc.calculate_comprehensive_metrics(y_true, y_pred)

        assert isinstance(result, dict)
        assert "accuracy" in result

    def test_calculate_basic_metrics(self):
        """基本指標の計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc._calculate_basic_metrics(y_true, y_pred)

        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "precision_macro" in result
        assert "recall_macro" in result
        assert "f1_score_macro" in result
        assert "matthews_corrcoef" in result
        assert "cohen_kappa" in result

    def test_calculate_balanced_metrics(self):
        """バランス指標の計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc._calculate_balanced_metrics(y_true, y_pred)

        assert "balanced_accuracy" in result
        assert "weighted_accuracy" in result

    def test_calculate_probability_metrics(self):
        """確率指標の計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])

        result = calc._calculate_probability_metrics(y_true, y_proba)

        assert "roc_auc" in result
        assert "pr_auc" in result
        assert "log_loss" in result
        assert "brier_score" in result

    def test_calculate_probability_metrics_multiclass(self):
        """多クラスの確率指標計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 2])
        y_proba = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])

        result = calc._calculate_probability_metrics(y_true, y_proba)

        assert "roc_auc" in result
        assert "pr_auc" in result

    def test_calculate_confusion_matrix_metrics(self):
        """混同行列指標の計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc._calculate_confusion_matrix_metrics(y_true, y_pred)

        assert "confusion_matrix" in result
        assert "confusion_matrix_normalized" in result
        assert "true_positives" in result
        assert "true_negatives" in result
        assert "false_positives" in result
        assert "false_negatives" in result

    def test_calculate_confusion_matrix_metrics_with_class_names(self):
        """クラス名付きの混同行列指標"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        class_names = ["negative", "positive"]

        result = calc._calculate_confusion_matrix_metrics(
            y_true, y_pred, class_names
        )

        assert "class_names" in result

    def test_calculate_classification_report(self):
        """分類レポートの計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc._calculate_classification_report(y_true, y_pred)

        assert "classification_report" in result

    def test_calculate_per_class_metrics(self):
        """クラス別指標の計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc._calculate_per_class_metrics(y_true, y_pred)

        assert "per_class_metrics" in result
        assert isinstance(result["per_class_metrics"], dict)

    def test_calculate_per_class_metrics_with_class_names(self):
        """クラス名付きのクラス別指標"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        class_names = ["negative", "positive"]

        result = calc._calculate_per_class_metrics(y_true, y_pred, class_names)

        assert "per_class_metrics" in result
        # クラス名が使用されていることを確認
        assert any("negative" in str(k) for k in result["per_class_metrics"].keys())

    def test_calculate_distribution_metrics(self):
        """分布指標の計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc._calculate_distribution_metrics(y_true, y_pred)

        assert "true_label_distribution" in result
        assert "predicted_label_distribution" in result
        assert "class_imbalance_ratio" in result
        assert "total_samples" in result
        assert "n_classes" in result

    def test_calculate_class_weights(self):
        """クラス重みの計算"""
        calc = MetricsCalculator()
        y_true = np.array([0, 1, 0, 1])

        weights = calc._calculate_class_weights(y_true)

        assert len(weights) == len(y_true)
        assert all(w > 0 for w in weights)

    def test_calculate_volatility_regression_metrics(self):
        """ボラティリティ回帰指標の計算"""
        calc = MetricsCalculator()
        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.15, 0.18, 0.28])

        result = calc.calculate_volatility_regression_metrics(y_true, y_pred)

        assert "qlike" in result
        assert "rmse_log_rv" in result
        assert "mae_log_rv" in result

    def test_calculate_volatility_regression_metrics_empty(self):
        """空の入力でのボラティリティ回帰指標"""
        calc = MetricsCalculator()
        y_true = np.array([])
        y_pred = np.array([])

        result = calc.calculate_volatility_regression_metrics(y_true, y_pred)

        assert result["qlike"] == 0.0
        assert result["rmse_log_rv"] == 0.0
        assert result["mae_log_rv"] == 0.0

    def test_config_include_balanced_accuracy_false(self):
        """バランス精度無効時の挙動"""
        config = MetricsConfig(include_balanced_accuracy=False)
        calc = MetricsCalculator(config)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc.calculate_comprehensive_metrics(y_true, y_pred, level="basic")

        assert "balanced_accuracy" not in result

    def test_config_include_roc_auc_false(self):
        """ROC AUC無効時の挙動"""
        config = MetricsConfig(include_roc_auc=False)
        calc = MetricsCalculator(config)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_proba = np.array([0.1, 0.9, 0.2, 0.8])

        result = calc.calculate_comprehensive_metrics(y_true, y_pred, y_proba)

        assert "roc_auc" not in result

    def test_config_include_confusion_matrix_false(self):
        """混同行列無効時の挙動"""
        config = MetricsConfig(include_confusion_matrix=False)
        calc = MetricsCalculator(config)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = calc.calculate_comprehensive_metrics(y_true, y_pred)

        assert "confusion_matrix" not in result


class TestGetDefaultMetrics:
    """get_default_metrics 関数のテスト"""

    def test_get_default_metrics(self):
        """デフォルトメトリクスの取得"""
        metrics = get_default_metrics()

        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert all(v == 0.0 for v in metrics.values())


class TestGlobalMetricsCollector:
    """グローバルmetrics_collectorのテスト"""

    def test_global_metrics_collector_exists(self):
        """グローバルインスタンスの存在確認"""
        assert metrics_collector is not None
        assert isinstance(metrics_collector, MetricsCalculator)

    def test_global_metrics_collector_calculate(self):
        """グローバルインスタンスでの計算"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        result = metrics_collector.calculate_comprehensive_metrics(y_true, y_pred)

        assert isinstance(result, dict)
        assert "accuracy" in result
