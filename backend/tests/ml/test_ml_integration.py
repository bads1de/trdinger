"""
ML統合テスト

MLOrchestratorとMLIndicatorServiceの統合テスト
"""

import unittest
import warnings
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.core.services.auto_strategy.services.ml_indicator_service import (
    MLIndicatorService,
)


class TestMLIntegration(unittest.TestCase):
    """ML統合テスト"""

    def setUp(self):
        """テスト前の準備"""
        # 非推奨警告を無視
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # テストデータ作成
        self.test_data = pd.DataFrame(
            {
                "open": np.random.rand(100) * 100,
                "high": np.random.rand(100) * 100,
                "low": np.random.rand(100) * 100,
                "close": np.random.rand(100) * 100,
                "volume": np.random.rand(100) * 1000,
            }
        )

    def test_orchestrator_indicator_service_compatibility(self):
        """MLOrchestratorとMLIndicatorServiceの互換性テスト"""

        # インスタンス作成
        orchestrator = MLOrchestrator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # ML指標計算の結果比較
        result1 = orchestrator.calculate_ml_indicators(self.test_data)
        result2 = indicator_service.calculate_ml_indicators(self.test_data)

        # 結果の形式確認
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)

        # 必要なキーが存在することを確認
        required_keys = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
        for key in required_keys:
            self.assertIn(key, result1)
            self.assertIn(key, result2)

        # 結果の一致確認
        for key in required_keys:
            np.testing.assert_array_equal(result1[key], result2[key])

    def test_single_indicator_compatibility(self):
        """単一指標計算の互換性テスト"""

        orchestrator = MLOrchestrator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # 各指標タイプでテスト
        indicator_types = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]

        for indicator_type in indicator_types:
            result1 = orchestrator.calculate_single_ml_indicator(
                indicator_type, self.test_data
            )
            result2 = indicator_service.calculate_single_ml_indicator(
                indicator_type, self.test_data
            )

            # 結果の一致確認
            np.testing.assert_array_equal(result1, result2)

    def test_model_status_compatibility(self):
        """モデル状態の互換性テスト"""

        orchestrator = MLOrchestrator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # モデル状態取得
        status1 = orchestrator.get_model_status()
        status2 = indicator_service.get_model_status()

        # 基本的なキーが存在することを確認
        self.assertIn("is_model_loaded", status1)
        self.assertIn("is_model_loaded", status2)

        # モデル読み込み状態が一致することを確認
        self.assertEqual(status1["is_model_loaded"], status2["is_model_loaded"])

    def test_deprecation_warnings(self):
        """非推奨警告のテスト"""

        # 警告をキャッチ
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # MLIndicatorServiceのインスタンス化で警告が発生することを確認
            indicator_service = MLIndicatorService()

            # 警告が発生していることを確認
            self.assertTrue(len(w) > 0)
            self.assertTrue(
                any(issubclass(warning.category, DeprecationWarning) for warning in w)
            )

    def test_proxy_behavior(self):
        """プロキシ動作のテスト"""

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # プロキシが正しく設定されていることを確認
        self.assertIsNotNone(indicator_service._orchestrator)
        self.assertIsInstance(indicator_service._orchestrator, MLOrchestrator)

        # プロパティが正しく同期されていることを確認
        self.assertEqual(
            indicator_service.is_model_loaded,
            indicator_service._orchestrator.is_model_loaded,
        )

    def test_error_handling_consistency(self):
        """エラーハンドリングの一貫性テスト"""

        # 空のデータフレームでテスト
        empty_df = pd.DataFrame()

        orchestrator = MLOrchestrator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # 両方とも同じようにエラーハンドリングされることを確認
        result1 = orchestrator.calculate_ml_indicators(empty_df)
        result2 = indicator_service.calculate_ml_indicators(empty_df)

        # デフォルト値が返されることを確認
        self.assertIsInstance(result1, dict)
        self.assertIsInstance(result2, dict)

        # 結果が一致することを確認
        for key in result1.keys():
            np.testing.assert_array_equal(result1[key], result2[key])

    def test_performance_consistency(self):
        """パフォーマンスの一貫性テスト"""

        import time

        orchestrator = MLOrchestrator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # MLOrchestratorの処理時間測定
        start_time = time.time()
        result1 = orchestrator.calculate_ml_indicators(self.test_data)
        time1 = time.time() - start_time

        # MLIndicatorServiceの処理時間測定
        start_time = time.time()
        result2 = indicator_service.calculate_ml_indicators(self.test_data)
        time2 = time.time() - start_time

        # プロキシのオーバーヘッドが大きくないことを確認（3倍以内、処理時間のばらつきを考慮）
        self.assertLess(time2, time1 * 3)

    def test_feature_importance_compatibility(self):
        """特徴量重要度の互換性テスト"""

        orchestrator = MLOrchestrator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # 特徴量重要度取得
        importance1 = orchestrator.get_feature_importance()
        importance2 = indicator_service.get_feature_importance()

        # 結果の形式確認
        self.assertIsInstance(importance1, dict)
        self.assertIsInstance(importance2, dict)

        # 結果が一致することを確認
        self.assertEqual(importance1, importance2)

    def test_prediction_update_compatibility(self):
        """予測値更新の互換性テスト"""

        orchestrator = MLOrchestrator()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            indicator_service = MLIndicatorService()

        # テスト用予測値
        test_predictions = {"up": 0.4, "down": 0.3, "range": 0.3}

        # 予測値更新
        orchestrator.update_predictions(test_predictions)
        indicator_service.update_predictions(test_predictions)

        # 更新が反映されていることを確認
        status1 = orchestrator.get_model_status()
        status2 = indicator_service.get_model_status()

        self.assertEqual(status1["last_predictions"], test_predictions)
        self.assertEqual(status2["last_predictions"], test_predictions)


if __name__ == "__main__":
    unittest.main()
