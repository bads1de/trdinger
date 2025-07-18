"""
ML統合テスト

MLOrchestratorの統合テスト（MLIndicatorServiceは廃止済み）
"""

import unittest
import warnings
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from app.core.services.auto_strategy.services.ml_orchestrator import MLOrchestrator


class TestMLIntegration(unittest.TestCase):
    """ML統合テスト"""

    def setUp(self):
        """テスト前の準備"""
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

    def test_orchestrator_ml_indicators(self):
        """MLOrchestratorのML指標計算テスト"""

        # インスタンス作成
        orchestrator = MLOrchestrator()

        # ML指標計算
        result = orchestrator.calculate_ml_indicators(self.test_data)

        # 結果の形式確認
        self.assertIsInstance(result, dict)

        # 必要なキーが存在することを確認
        required_keys = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
        for key in required_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], np.ndarray)
            self.assertEqual(len(result[key]), len(self.test_data))

    def test_single_indicator_calculation(self):
        """単一指標計算テスト"""

        orchestrator = MLOrchestrator()

        # 各指標タイプでテスト
        indicator_types = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]

        for indicator_type in indicator_types:
            result = orchestrator.calculate_single_ml_indicator(
                indicator_type, self.test_data
            )

            # 結果の形式確認
            self.assertIsInstance(result, np.ndarray)
            self.assertEqual(len(result), len(self.test_data))

    # test_model_status_compatibility は削除（MLIndicatorService廃止のため）

    def test_model_status(self):
        """モデル状態取得テスト"""

        orchestrator = MLOrchestrator()

        # モデル状態を取得
        status = orchestrator.get_model_status()

        # 結果の形式確認
        self.assertIsInstance(status, dict)
        self.assertIn("is_model_loaded", status)
        self.assertIn("last_predictions", status)

    def test_error_handling(self):
        """エラーハンドリングテスト"""

        # 空のデータフレームでテスト
        empty_df = pd.DataFrame()

        orchestrator = MLOrchestrator()

        # エラーハンドリングが適切に行われることを確認
        result = orchestrator.calculate_ml_indicators(empty_df)

        # デフォルト値が返されることを確認
        self.assertIsInstance(result, dict)
        required_keys = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
        for key in required_keys:
            self.assertIn(key, result)

    def test_performance(self):
        """パフォーマンステスト"""

        import time

        orchestrator = MLOrchestrator()

        # MLOrchestratorの処理時間測定
        start_time = time.time()
        result = orchestrator.calculate_ml_indicators(self.test_data)
        execution_time = time.time() - start_time

        # 処理が合理的な時間内で完了することを確認（10秒以内）
        self.assertLess(execution_time, 10.0)

        # 結果が正しく返されることを確認
        self.assertIsInstance(result, dict)
        required_keys = ["ML_UP_PROB", "ML_DOWN_PROB", "ML_RANGE_PROB"]
        for key in required_keys:
            self.assertIn(key, result)

    def test_feature_importance(self):
        """特徴量重要度テスト"""

        orchestrator = MLOrchestrator()

        # 特徴量重要度取得
        importance = orchestrator.get_feature_importance()

        # 結果の形式確認
        self.assertIsInstance(importance, dict)

    def test_prediction_update(self):
        """予測値更新テスト"""

        orchestrator = MLOrchestrator()

        # テスト用予測値
        test_predictions = {"up": 0.4, "down": 0.3, "range": 0.3}

        # 予測値更新
        orchestrator.update_predictions(test_predictions)

        # 更新が反映されていることを確認
        status = orchestrator.get_model_status()
        self.assertEqual(status["last_predictions"], test_predictions)


if __name__ == "__main__":
    unittest.main()
