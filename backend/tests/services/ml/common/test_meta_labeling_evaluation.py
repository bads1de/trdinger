"""
Meta-Labeling評価ユーティリティのテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.ml.common.evaluation import (
    evaluate_meta_labeling,
    print_meta_labeling_report,
    find_optimal_threshold,
)


class TestMetaLabelingEvaluation:
    """Meta-Labeling評価関数のテスト"""

    @pytest.fixture
    def perfect_predictions(self):
        """完璧な予測データ"""
        y_true = pd.Series([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        return y_true, y_pred

    @pytest.fixture
    def imperfect_predictions(self):
        """不完全な予測データ（Precision=0.67, Recall=0.67）"""
        y_true = pd.Series([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 1, 0, 0])
        return y_true, y_pred

    @pytest.fixture
    def probability_predictions(self):
        """確率付き予測データ"""
        y_true = pd.Series([1, 1, 1, 0, 0, 0])
        y_pred_proba = np.array(
            [
                [0.1, 0.9],  # 1
                [0.2, 0.8],  # 1
                [0.4, 0.6],  # 1
                [0.7, 0.3],  # 0
                [0.8, 0.2],  # 0
                [0.9, 0.1],  # 0
            ]
        )
        return y_true, y_pred_proba

    def test_perfect_predictions(self, perfect_predictions):
        """完璧な予測の評価"""
        y_true, y_pred = perfect_predictions
        metrics = evaluate_meta_labeling(y_true, y_pred)

        # Precision, Recall, F1 は全て1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1_score"] == 1.0
        assert metrics["accuracy"] == 1.0
        assert metrics["win_rate"] == 1.0

    def test_imperfect_predictions(self, imperfect_predictions):
        """不完全な予測の評価"""
        y_true, y_pred = imperfect_predictions
        metrics = evaluate_meta_labeling(y_true, y_pred)

        # TP=2, FP=1, FN=1, TN=2
        # Precision = 2/(2+1) = 0.67
        # Recall = 2/(2+1) = 0.67
        assert metrics["precision"] == pytest.approx(0.666, abs=0.01)
        assert metrics["recall"] == pytest.approx(0.666, abs=0.01)
        assert metrics["true_positives"] == 2
        assert metrics["false_positives"] == 1
        assert metrics["false_negatives"] == 1
        assert metrics["true_negatives"] == 2

    def test_signal_adoption_rate(self, imperfect_predictions):
        """シグナル採択率の計算"""
        y_true, y_pred = imperfect_predictions
        metrics = evaluate_meta_labeling(y_true, y_pred)

        # 6サンプル中3つがPositive予測
        assert metrics["signal_adoption_rate"] == pytest.approx(0.5, abs=0.01)

    def test_expected_value(self, imperfect_predictions):
        """期待値の計算"""
        y_true, y_pred = imperfect_predictions
        metrics = evaluate_meta_labeling(y_true, y_pred)

        # Precision=0.67なので、EV = 0.67*1 + 0.33*(-1) = 0.34
        assert metrics["expected_value"] == pytest.approx(0.34, abs=0.01)

    def test_probability_based_metrics(self, probability_predictions):
        """確率ベースの評価指標"""
        y_true, y_pred_proba = probability_predictions
        y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

        metrics = evaluate_meta_labeling(y_true, y_pred, y_pred_proba)

        # ROC-AUCとPR-AUCが計算されることを確認
        assert "roc_auc" in metrics
        assert "pr_auc" in metrics
        assert metrics["roc_auc"] > 0.0
        assert metrics["pr_auc"] > 0.0

    def test_zero_division_handling(self):
        """ゼロ除算のハンドリング"""
        # 全てNegative予測の場合
        y_true = pd.Series([1, 1, 1])
        y_pred = np.array([0, 0, 0])

        metrics = evaluate_meta_labeling(y_true, y_pred)

        # Precisionは0.0になる（ゼロ除算エラーにならない）
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0

    def test_custom_threshold(self, probability_predictions):
        """カスタム閾値の適用"""
        y_true, y_pred_proba = probability_predictions

        # 閾値0.7で評価
        metrics_high = evaluate_meta_labeling(
            y_true, (y_pred_proba[:, 1] >= 0.7).astype(int), y_pred_proba, threshold=0.7
        )

        # 閾値0.3で評価
        metrics_low = evaluate_meta_labeling(
            y_true, (y_pred_proba[:, 1] >= 0.3).astype(int), y_pred_proba, threshold=0.3
        )

        # 高い閾値の方がPrecisionが高く、Recallが低い
        assert metrics_high["precision"] >= metrics_low["precision"]
        assert metrics_high["recall"] <= metrics_low["recall"]


class TestOptimalThresholdFinding:
    """最適閾値探索のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用データ"""
        np.random.seed(42)
        n_samples = 100

        y_true = np.random.randint(0, 2, n_samples)
        y_pred_proba = np.random.rand(n_samples, 2)
        y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

        return pd.Series(y_true), y_pred_proba

    def test_find_optimal_threshold_precision(self, sample_data):
        """Precision最大化の閾値探索"""
        y_true, y_pred_proba = sample_data

        result = find_optimal_threshold(
            y_true, y_pred_proba, metric="precision", min_recall=0.2
        )

        # 結果に必要なキーが含まれることを確認
        assert "optimal_threshold" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result

        # Recall制約を満たしていることを確認
        assert result["recall"] >= 0.2

    def test_find_optimal_threshold_f1(self, sample_data):
        """F1-Score最大化の閾値探索"""
        y_true, y_pred_proba = sample_data

        result = find_optimal_threshold(
            y_true, y_pred_proba, metric="f1", min_recall=0.2
        )

        # F1-Scoreが計算されることを確認
        assert result["f1_score"] > 0.0

    def test_threshold_range(self, sample_data):
        """閾値が0-1の範囲内にあることを確認"""
        y_true, y_pred_proba = sample_data

        result = find_optimal_threshold(y_true, y_pred_proba)

        assert 0.0 <= result["optimal_threshold"] <= 1.0




