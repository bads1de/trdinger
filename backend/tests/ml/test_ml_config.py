"""
ML設定テスト

ML設定クラスの機能をテストします。
"""

import unittest
import numpy as np
from app.core.services.ml.config import ml_config
from app.core.services.ml.config.ml_config import MLConfig, PredictionConfig


class TestMLConfig(unittest.TestCase):
    """ML設定テスト"""

    def setUp(self):
        """テスト前の準備"""
        # 設定を初期化
        self.config = MLConfig()

    def test_default_values(self):
        """デフォルト値のテスト"""
        # デフォルト予測値
        self.assertEqual(self.config.prediction.DEFAULT_UP_PROB, 0.33)
        self.assertEqual(self.config.prediction.DEFAULT_DOWN_PROB, 0.33)
        self.assertEqual(self.config.prediction.DEFAULT_RANGE_PROB, 0.34)

        # 合計が1.0に近いことを確認
        predictions = self.config.prediction.get_default_predictions()
        total = sum(predictions.values())
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_env_override(self):
        """環境変数オーバーライドのテスト（削除済み機能）"""
        # 環境変数機能は削除されたため、デフォルト値のテストに変更
        config = MLConfig()

        # デフォルト値が設定されていることを確認
        self.assertEqual(config.prediction.DEFAULT_UP_PROB, 0.33)
        self.assertEqual(config.prediction.DEFAULT_DOWN_PROB, 0.33)
        self.assertEqual(config.prediction.DEFAULT_RANGE_PROB, 0.34)
        self.assertEqual(config.data_processing.MAX_OHLCV_ROWS, 10000)
        self.assertFalse(config.data_processing.DEBUG_MODE)

    def test_invalid_env_values(self):
        """無効な環境変数値のテスト（削除済み機能）"""
        # 環境変数機能は削除されたため、デフォルト値のテストに変更
        config = MLConfig()

        # デフォルト値が使用されていることを確認
        self.assertEqual(config.prediction.DEFAULT_UP_PROB, 0.33)
        self.assertEqual(config.data_processing.MAX_OHLCV_ROWS, 10000)

    def test_prediction_validation(self):
        """予測値バリデーションのテスト"""
        # 有効な予測値
        valid_predictions = {"up": 0.4, "down": 0.3, "range": 0.3}
        self.assertTrue(self.config.prediction.validate_predictions(valid_predictions))

        # 無効な予測値（キー不足）
        invalid_keys = {"up": 0.5, "down": 0.5}
        self.assertFalse(self.config.prediction.validate_predictions(invalid_keys))

        # 無効な予測値（範囲外）
        invalid_range = {"up": 1.5, "down": 0.3, "range": 0.3}
        self.assertFalse(self.config.prediction.validate_predictions(invalid_range))

        # 無効な予測値（合計が範囲外）
        invalid_sum = {"up": 0.9, "down": 0.8, "range": 0.7}
        self.assertFalse(self.config.prediction.validate_predictions(invalid_sum))

    def test_default_indicators(self):
        """デフォルト指標のテスト"""
        # デフォルト指標を取得
        data_length = 10
        indicators = self.config.prediction.get_default_indicators(data_length)

        # 指標の形式を確認
        self.assertIn("ML_UP_PROB", indicators)
        self.assertIn("ML_DOWN_PROB", indicators)
        self.assertIn("ML_RANGE_PROB", indicators)

        # 配列の長さを確認
        self.assertEqual(len(indicators["ML_UP_PROB"]), data_length)

        # 値を確認
        np.testing.assert_array_equal(
            indicators["ML_UP_PROB"],
            np.full(data_length, self.config.prediction.DEFAULT_UP_PROB),
        )

    def test_config_validation(self):
        """設定バリデーションのテスト"""
        # 有効な設定
        self.assertTrue(self.config.prediction.validate_config())

        # 無効な設定をシミュレート（直接値を変更）
        original_value = self.config.prediction.DEFAULT_UP_PROB
        self.config.prediction.DEFAULT_UP_PROB = 2.0
        try:
            self.assertFalse(self.config.prediction.validate_config())
            self.assertTrue(len(self.config.prediction.get_validation_errors()) > 0)
        finally:
            # 元の値に戻す
            self.config.prediction.DEFAULT_UP_PROB = original_value

    def test_environment_info(self):
        """環境情報のテスト"""
        # 環境情報を取得
        env_info = self.config.get_environment_info()

        # 情報の形式を確認
        self.assertIn("debug_mode", env_info)
        self.assertIn("log_level", env_info)
        self.assertIn("max_ohlcv_rows", env_info)
        self.assertIn("default_predictions", env_info)


if __name__ == "__main__":
    unittest.main()
