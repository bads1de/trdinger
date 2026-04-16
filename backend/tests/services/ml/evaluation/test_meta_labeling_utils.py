"""
meta_labeling_utils.py のテスト

app/services/ml/evaluation/meta_labeling_utils.py のテストモジュール
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from app.services.ml.evaluation.meta_labeling_utils import (
    evaluate_meta_labeling,
    find_optimal_threshold,
    print_meta_labeling_report,
)


class TestEvaluateMetaLabeling:
    """evaluate_meta_labeling 関数のテスト"""

    def test_evaluate_meta_labeling_basic(self):
        """基本的な評価"""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        result = evaluate_meta_labeling(y_true, y_pred)

        assert isinstance(result, dict)
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "win_rate" in result
        assert "signal_adoption_rate" in result
        assert "expected_value" in result

    def test_evaluate_meta_labeling_with_proba(self):
        """確率付きの評価"""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])
        result = evaluate_meta_labeling(y_true, y_pred, y_pred_proba)

        assert isinstance(result, dict)
        assert "precision" in result

    def test_evaluate_meta_labeling_default_keys(self):
        """デフォルトキーの保証"""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate_meta_labeling(y_true, y_pred)

        required_keys = [
            "precision",
            "recall",
            "f1_score",
            "accuracy",
            "specificity",
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        ]
        for key in required_keys:
            assert key in result

    def test_evaluate_meta_labeling_signal_adoption_rate(self):
        """シグナル採択率の計算"""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred = np.array([1, 1, 0, 1, 0, 1])  # 4/6 = 0.666...
        result = evaluate_meta_labeling(y_true, y_pred)

        assert result["signal_adoption_rate"] == pytest.approx(4 / 6, rel=1e-6)

    def test_evaluate_meta_labeling_expected_value(self):
        """期待値の計算"""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate_meta_labeling(y_true, y_pred)

        # precision = 1.0, expected_value = (1.0 * 1.0) + (0.0 * -1.0) = 1.0
        assert result["expected_value"] == pytest.approx(1.0, rel=1e-6)

    def test_evaluate_meta_labeling_empty_predictions(self):
        """空の予測"""
        y_true = pd.Series([])
        y_pred = np.array([])
        result = evaluate_meta_labeling(y_true, y_pred)

        assert result["signal_adoption_rate"] == 0.0

    def test_evaluate_meta_labeling_numpy_array(self):
        """numpy配列の入力"""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        result = evaluate_meta_labeling(y_true, y_pred)

        assert isinstance(result, dict)
        assert "precision" in result


class TestPrintMetaLabelingReport:
    """print_meta_labeling_report 関数のテスト"""

    def test_print_meta_labeling_report_basic(self, capsys):
        """基本的なレポート出力"""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        print_meta_labeling_report(y_true, y_pred)

        captured = capsys.readouterr()
        assert "Meta-Labeling Evaluation Report" in captured.out
        assert "Precision" in captured.out

    def test_print_meta_labeling_report_with_proba(self, capsys):
        """確率付きのレポート出力"""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        print_meta_labeling_report(y_true, y_pred, y_pred_proba)

        captured = capsys.readouterr()
        assert "Meta-Labeling Evaluation Report" in captured.out

    def test_print_meta_labeling_report_high_precision(self, capsys):
        """高精度モデルのレポート"""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])

        print_meta_labeling_report(y_true, y_pred)

        captured = capsys.readouterr()
        assert "優秀なモデルです" in captured.out

    def test_print_meta_labeling_report_low_precision(self, capsys):
        """低精度モデルのレポート"""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred = np.array([1, 1, 1, 1])  # 低精度

        print_meta_labeling_report(y_true, y_pred)

        captured = capsys.readouterr()
        assert "モデルの改善が必要です" in captured.out


class TestFindOptimalThreshold:
    """find_optimal_threshold 関数のテスト"""

    def test_find_optimal_threshold_precision(self):
        """precision最適化"""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])

        result = find_optimal_threshold(y_true, y_pred_proba, metric="precision")

        assert "optimal_threshold" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert 0.0 <= result["optimal_threshold"] <= 1.0

    def test_find_optimal_threshold_f1(self):
        """F1スコア最適化"""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])

        result = find_optimal_threshold(y_true, y_pred_proba, metric="f1")

        assert "optimal_threshold" in result
        assert "f1_score" in result

    def test_find_optimal_threshold_min_recall(self):
        """最小recall制約"""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9])

        result = find_optimal_threshold(y_true, y_pred_proba, min_recall=0.5)

        assert "optimal_threshold" in result
        assert result["recall"] >= 0.5

    def test_find_optimal_threshold_invalid_metric(self):
        """無効なメトリック"""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        with pytest.raises(ValueError, match="Unknown metric"):
            find_optimal_threshold(y_true, y_pred_proba, metric="invalid")

    def test_find_optimal_threshold_no_valid_threshold(self):
        """有効な閾値が見つからない場合"""
        y_true = pd.Series([0, 0, 0, 0])
        y_pred_proba = np.array([0.1, 0.2, 0.3, 0.4])

        result = find_optimal_threshold(y_true, y_pred_proba, min_recall=0.9)

        # 有効な閾値が見つからない場合、デフォルト値が返される
        assert "optimal_threshold" in result
        # 実際にはprecision_recall_curveが返す閾値の最大値が使用される
        assert 0.0 <= result["optimal_threshold"] <= 1.0

    def test_find_optimal_threshold_multidim_proba(self):
        """多次元確率配列"""
        y_true = pd.Series([0, 1, 0, 1])
        y_pred_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]])

        result = find_optimal_threshold(y_true, y_pred_proba, metric="precision")

        assert "optimal_threshold" in result
        assert 0.0 <= result["optimal_threshold"] <= 1.0

    def test_find_optimal_threshold_numpy_array(self):
        """numpy配列の入力"""
        y_true = np.array([0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.8])

        result = find_optimal_threshold(y_true, y_pred_proba, metric="precision")

        assert "optimal_threshold" in result
