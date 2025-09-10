"""
ML Orchestrator テスト

MLOrchestratorの機能をテストします。
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import numpy as np

from backend.app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from backend.app.services.ml.exceptions import MLDataError, MLValidationError


class TestMLOrchestrator(unittest.TestCase):
    """MLOrchestratorテスト"""

    def setUp(self):
        """セットアップ"""
        with patch('backend.app.services.auto_strategy.services.ml_orchestrator.AutoMLConfig'), \
             patch('backend.app.services.auto_strategy.services.ml_orchestrator.FeatureEngineeringService'), \
             patch('backend.app.services.auto_strategy.services.ml_orchestrator.MLTrainingService'):
            self.orchestrator = MLOrchestrator(enable_automl=False)

            # Mock設定
            self.orchestrator.ml_training_service = Mock()
            self.orchestrator.feature_service = Mock()

    def test_calculate_ml_indicators_valid_data(self):
        """有効なデータでのML指標計算テスト"""
        # Mockデータ作成
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })
        df.index = pd.date_range('2023-01-01', periods=3, freq='H')

        # Mock設定
        mock_features = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        mock_predictions = {'up': 0.6, 'down': 0.3, 'range': 0.1}

        self.orchestrator.feature_service.calculate_advanced_features.return_value = mock_features
        self.orchestrator.ml_training_service.generate_signals.return_value = mock_predictions

        result = self.orchestrator.calculate_ml_indicators(df)

        self.assertIsInstance(result, dict)
        self.assertIn('ML_UP_PROB', result)
        self.assertIn('ML_DOWN_PROB', result)
        self.assertIn('ML_RANGE_PROB', result)
        self.assertEqual(len(result['ML_UP_PROB']), 3)

    def test_calculate_ml_indicators_empty_data(self):
        """空データでのML指標計算テスト - エラーが発生するか確認"""
        df = pd.DataFrame()

        with self.assertRaises(MLDataError) as context:
            self.orchestrator.calculate_ml_indicators(df)

        self.assertIn('入力データが空', str(context.exception))

    def test_calculate_ml_indicators_none_data(self):
        """NoneデータでのML指標計算テスト - エラーが発生するか確認"""
        with self.assertRaises(MLDataError) as context:
            self.orchestrator.calculate_ml_indicators(None)

        self.assertIn('入力データが空', str(context.exception))

    def test_calculate_ml_indicators_invalid_columns(self):
        """無効なカラムのデータでのML指標計算テスト - エラーが発生するか確認"""
        df = pd.DataFrame({
            'invalid_col': [1, 2, 3]
        })
        df.index = pd.date_range('2023-01-01', periods=3, freq='H')

        with self.assertRaises(MLDataError) as context:
            self.orchestrator.calculate_ml_indicators(df)

        self.assertIn('必要なカラムが不足', str(context.exception))

    def test_calculate_ml_indicators_with_automl(self):
        """AutoML有効でのML指標計算テスト"""
        with patch('backend.app.services.auto_strategy.services.ml_orchestrator.AutoMLConfig'), \
             patch('backend.app.services.auto_strategy.services.ml_orchestrator.MLOrchestrator.calculate_enhanced_features'):
            orchestrator = MLOrchestrator(enable_automl=True)
            orchestrator.feature_service = Mock()
            orchestrator.ml_training_service = Mock()

            # Mockデータ作成
            df = pd.DataFrame({
                'Open': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [95, 96, 97],
                'Close': [103, 104, 105],
                'Volume': [1000, 1100, 1200]
            })
            df.index = pd.date_range('2023-01-01', periods=3, freq='H')

            # Mock設定
            mock_features = pd.DataFrame({'feature1': [1, 2, 3]})
            mock_predictions = {'up': 0.6, 'down': 0.3, 'range': 0.1}

            orchestrator.feature_service.calculate_enhanced_features.return_value = mock_features
            orchestrator.ml_training_service.generate_signals.return_value = mock_predictions

            result = orchestrator.calculate_ml_indicators(df)

            self.assertIsInstance(result, dict)
            orchestrator.feature_service.calculate_enhanced_features.assert_called_once()

    def test_calculate_single_ml_indicator_valid_type(self):
        """有効なタイプでの単一ML指標計算テスト"""
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [105, 106],
            'low': [95, 96],
            'close': [103, 104],
            'volume': [1000, 1100]
        })
        df.index = pd.date_range('2023-01-01', periods=2, freq='H')

        # Mock設定
        mock_features = pd.DataFrame({'feature1': [1, 2]})
        mock_predictions = {'up': 0.5, 'down': 0.4, 'range': 0.1}

        self.orchestrator.feature_service.calculate_advanced_features.return_value = mock_features
        self.orchestrator.ml_training_service.generate_signals.return_value = mock_predictions

        result = self.orchestrator.calculate_single_ml_indicator('ML_UP_PROB', df)

        self.assertEqual(result, 0.5)
        self.assertIsInstance(result, np.ndarray)

    def test_calculate_single_ml_indicator_invalid_type(self):
        """無効なタイプでの単一ML指標計算テスト - エラーが発生するか確認"""
        df = pd.DataFrame({
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [103],
            'volume': [1000]
        })

        with self.assertRaises(MLValidationError) as context:
            self.orchestrator.calculate_single_ml_indicator('INVALID_TYPE', df)

        self.assertIn('未知のML指標タイプ', str(context.exception))

    def test_calculate_single_ml_indicator_none_df(self):
        """Noneデータフレームでの単一ML指標計算テスト - エラーが発生するか確認"""
        with self.assertRaises(MLDataError) as context:
            self.orchestrator.calculate_single_ml_indicator('ML_UP_PROB', None)

        self.assertIn('空のデータフレーム', str(context.exception))

    def test_load_model_success(self):
        """モデル読み込み成功テスト"""
        model_path = "/path/to/model.pkl"

        self.orchestrator.ml_training_service.load_model.return_value = True

        result = self.orchestrator.load_model(model_path)

        self.assertTrue(result)
        self.orchestrator.ml_training_service.load_model.assert_called_once_with(model_path)
        self.assertTrue(self.orchestrator.is_model_loaded)

    def test_load_model_failure(self):
        """モデル読み込み失敗テスト"""
        model_path = "/path/to/model.pkl"

        self.orchestrator.ml_training_service.load_model.return_value = False

        result = self.orchestrator.load_model(model_path)

        self.assertFalse(result)
        self.assertFalse(self.orchestrator.is_model_loaded)

    def test_get_model_status(self):
        """モデルステータス取得テスト"""
        # Mock設定
        self.orchestrator.ml_training_service.trainer = Mock()
        self.orchestrator.ml_training_service.trainer.is_trained = True
        self.orchestrator.ml_training_service.trainer.feature_columns = ['col1', 'col2']

        with patch('backend.app.services.auto_strategy.services.ml_orchestrator.model_manager') as mock_manager:
            mock_manager.get_latest_model.return_value = "latest_model.pkl"
            mock_manager.load_model.return_value = {
                "metadata": {
                    "accuracy": 0.85,
                    "f1_score": 0.80
                }
            }

            result = self.orchestrator.get_model_status()

            self.assertIsInstance(result, dict)
            self.assertTrue(result['is_trained'])
            self.assertEqual(result['feature_count'], 2)

    def test_get_feature_importance_from_current_model(self):
        """現在のモデルからの特徴量重要度取得テスト"""
        self.orchestrator.is_model_loaded = True
        self.orchestrator.ml_training_service.trainer.is_trained = True
        mock_importance = {'feature1': 0.8, 'feature2': 0.6, 'feature3': 0.4}

        self.orchestrator.ml_training_service.get_feature_importance.return_value = mock_importance

        result = self.orchestrator.get_feature_importance(top_n=2)

        self.assertEqual(list(result.keys()), ['feature1', 'feature2'])
        self.assertEqual(len(result), 2)

    def test_get_feature_importance_from_metadata(self):
        """メタデータからの特徴量重要度取得テスト"""
        self.orchestrator.is_model_loaded = False

        with patch('backend.app.services.auto_strategy.services.ml_orchestrator.model_manager') as mock_manager:
            mock_manager.get_latest_model.return_value = "latest_model.pkl"
            mock_manager.load_model.return_value = {
                "metadata": {
                    "feature_importance": {
                        'feature1': 0.8,
                        'feature2': 0.6,
                        'feature3': 0.4
                    }
                }
            }

            result = self.orchestrator.get_feature_importance(top_n=2)

            self.assertEqual(list(result.keys()), ['feature1', 'feature2'])

    def test_get_feature_importance_no_data(self):
        """特徴量重要度がない場合のテスト"""
        self.orchestrator.is_model_loaded = False
        self.orchestrator.ml_training_service.get_feature_importance.return_value = {}

        with patch('backend.app.services.auto_strategy.services.ml_orchestrator.model_manager') as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = self.orchestrator.get_feature_importance()

            self.assertEqual(result, {})

    def test_infer_symbol_from_data_btc(self):
        """BTCデータからのシンボル推定テスト"""
        df = pd.DataFrame({
            'Close': [50000, 51000, 52000]
        })
        df.index = pd.date_range('2023-01-01', periods=3, freq='H')

        result = self.orchestrator._infer_symbol_from_data(df)

        self.assertEqual(result, "BTC/USDT:USDT")

    def test_infer_symbol_from_data_eth(self):
        """ETHデータからのシンボル推定テスト"""
        df = pd.DataFrame({
            'Close': [2000, 2100, 2200]
        })
        df.index = pd.date_range('2023-01-01', periods=3, freq='H')

        result = self.orchestrator._infer_symbol_from_data(df)

        self.assertEqual(result, "ETH/USDT:USDT")

    def test_infer_symbol_none_df(self):
        """Noneデータフレームでのシンボル推定テスト - エラーが発生するか確認"""
        with self.assertRaises(AttributeError) as context:
            self.orchestrator._infer_symbol_from_data(None)

        self.assertIn('df is None', str(context.exception))

    def test_infer_timeframe_1m(self):
        """1分タイムフレームの推定テスト"""
        df = pd.DataFrame({
            'Close': list(range(100))
        })
        # わずかにずれた1分ごとのタイムスタンプ
        df.index = pd.date_range('2023-01-01', periods=100, freq='1min')

        result = self.orchestrator._infer_timeframe_from_data(df)

        self.assertEqual(result, "1m")

    def test_infer_timeframe_1h(self):
        """1時間タイムフレームの推定テスト"""
        df = pd.DataFrame({
            'Close': list(range(100))
        })
        df.index = pd.date_range('2023-01-01', periods=100, freq='1H')

        result = self.orchestrator._infer_timeframe_from_data(df)

        self.assertEqual(result, "1h")

    def test_expand_predictions_to_data_length(self):
        """予測値拡張テスト"""
        predictions = {'up': 0.6, 'down': 0.3, 'range': 0.1}
        data_length = 5

        result = self.orchestrator._expand_predictions_to_data_length(predictions, data_length)

        self.assertEqual(len(result['ML_UP_PROB']), data_length)
        self.assertTrue(np.all(result['ML_UP_PROB'] == 0.6))
        self.assertTrue(np.all(result['ML_DOWN_PROB'] == 0.3))

    def test_validate_ml_indicators_invalid_values(self):
        """ML指標（無効な値）検証テスト - エラーが発生するか確認"""
        ml_indicators = {
            'ML_UP_PROB': np.array([1.5, 2.0]),  # 範囲外の値
            'ML_DOWN_PROB': np.array([0.3, 0.4]),
            'ML_RANGE_PROB': np.array([0.3, 0.4])
        }

        with self.assertRaises(MLValidationError) as context:
            self.orchestrator._validate_ml_indicators(ml_indicators)

        self.assertIn('値が範囲外です', str(context.exception))

    def test_try_load_latest_model_no_model(self):
        """最新モデルがない場合の自動読み込みテスト"""
        with patch('backend.app.services.auto_strategy.services.ml_orchestrator.model_manager') as mock_manager:
            mock_manager.get_latest_model.return_value = None

            result = self.orchestrator._try_load_latest_model()

            self.assertFalse(result)
            self.assertFalse(self.orchestrator.is_model_loaded)


if __name__ == '__main__':
    unittest.main()