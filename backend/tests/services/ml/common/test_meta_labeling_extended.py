import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from app.services.ml.common.evaluation import (
    evaluate_meta_labeling,
    find_optimal_threshold,
    print_meta_labeling_report,
)

class TestMetaLabelingExtended:
    @pytest.fixture
    def full_mock_metrics(self):
        # evaluate_meta_labeling が追加する独自メトリクスも含めた完全な辞書
        return {
            "precision": 0.7,
            "recall": 0.5,
            "f1_score": 0.58,
            "accuracy": 0.8,
            "specificity": 0.9,
            "true_positives": 7,
            "true_negatives": 9,
            "false_positives": 3,
            "false_negatives": 5,
            "roc_auc": 0.75,
            "pr_auc": 0.7,
            "win_rate": 0.7,
            "signal_adoption_rate": 0.5,
            "expected_value": 0.4,
            "total_samples": 20,
            "positive_samples": 10,
            "negative_samples": 10
        }

    def test_evaluate_meta_labeling_logic(self):
        y_true = pd.Series([1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0])
        
        # metrics_collector の戻り値を Mock
        base_metrics = {"precision": 0.5, "f1_score": 0.5, "recall": 0.5}
        with patch("app.services.ml.evaluation.metrics.metrics_collector.calculate_comprehensive_metrics", return_value=base_metrics):
            res = evaluate_meta_labeling(y_true, y_pred)
            assert res["win_rate"] == 0.5
            assert res["signal_adoption_rate"] == 0.5
            assert res["expected_value"] == 0.0
            assert res["total_samples"] == 4

    def test_print_meta_labeling_report_full_coverage(self, capsys, full_mock_metrics):
        y_true = pd.Series([1, 0])
        y_pred = np.array([1, 0])
        
        # evaluate_meta_labeling そのものを Mock
        with patch("app.services.ml.common.evaluation.evaluate_meta_labeling", return_value=full_mock_metrics) as mock_eval:
            # 1. 優秀なモデル
            print_meta_labeling_report(y_true, y_pred)
            out = capsys.readouterr().out
            assert "優秀なモデルです" in out
            
            # 2. 改善が必要なモデル
            m = full_mock_metrics.copy()
            m["precision"] = 0.5
            with patch("app.services.ml.common.evaluation.evaluate_meta_labeling", return_value=m):
                print_meta_labeling_report(y_true, y_pred)
                assert "改善が必要です" in capsys.readouterr().out

            # 3. 採択率が高い
            m["precision"] = 0.7
            m["signal_adoption_rate"] = 0.6
            with patch("app.services.ml.common.evaluation.evaluate_meta_labeling", return_value=m):
                print_meta_labeling_report(y_true, y_pred)
                assert "採択率が高い" in capsys.readouterr().out

    def test_find_optimal_threshold_scenarios(self, full_mock_metrics):
        y_true = pd.Series([1, 1, 0, 0])
        y_pred_proba = np.array([0.9, 0.8, 0.2, 0.1])
        
        with patch("app.services.ml.common.evaluation.evaluate_meta_labeling", return_value=full_mock_metrics) as mock_eval:
            # Precision最大化パス
            res_p = find_optimal_threshold(y_true, y_pred_proba, metric="precision")
            assert res_p["optimal_threshold"] > 0
            
            # F1最大化パス
            res_f = find_optimal_threshold(y_true, y_pred_proba, metric="f1")
            assert res_f["f1_score"] == 0.58

    def test_find_optimal_threshold_empty_case(self):
        # 制約を満たさない場合の分岐
        y_true = pd.Series([1, 0, 0, 0, 0])
        y_pred_proba = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        # Recall >= 0.99 は達成不可能
        res = find_optimal_threshold(y_true, y_pred_proba, min_recall=0.99)
        # 異常系でも辞書が返り、処理がクラッシュしないことを確認
        assert "optimal_threshold" in res
        # 値は環境により微動するため、小さな値であることを確認
        assert res["precision"] < 0.1



