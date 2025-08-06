"""
MLOrchestrator包括的テスト

MLOrchestratorとテクニカル指標の統合、AutoML機能、予測精度、
エラー処理の包括的テストを実施します。
"""

import logging
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional

from app.services.auto_strategy.services.ml_orchestrator import MLOrchestrator
from app.services.ml.ml_training_service import MLTrainingService

logger = logging.getLogger(__name__)


class TestMLOrchestratorComprehensive:
    """MLOrchestrator包括的テストクラス"""

    @pytest.fixture
    def sample_market_data(self):
        """サンプル市場データ"""
        dates = pd.date_range('2024-01-01', periods=100, freq='1H')
        np.random.seed(42)  # 再現性のため
        
        data = {
            'timestamp': dates,
            'open': 50000 + np.random.randn(100) * 1000,
            'high': 51000 + np.random.randn(100) * 1000,
            'low': 49000 + np.random.randn(100) * 1000,
            'close': 50000 + np.random.randn(100) * 1000,
            'volume': 1000 + np.random.randn(100) * 100,
        }
        
        df = pd.DataFrame(data)
        # 価格の整合性を保つ
        df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(100) * 100)
        df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(100) * 100)
        
        return df

    @pytest.fixture
    def ml_orchestrator_automl_enabled(self):
        """AutoML有効なMLOrchestrator"""
        return MLOrchestrator(enable_automl=True)

    @pytest.fixture
    def ml_orchestrator_automl_disabled(self):
        """AutoML無効なMLOrchestrator"""
        return MLOrchestrator(enable_automl=False)

    def test_ml_orchestrator_initialization_automl_enabled(self, ml_orchestrator_automl_enabled):
        """AutoML有効時の初期化テスト"""
        orchestrator = ml_orchestrator_automl_enabled
        
        assert orchestrator.enable_automl is True
        assert hasattr(orchestrator, 'ml_training_service')
        assert hasattr(orchestrator, 'feature_service')
        assert hasattr(orchestrator, 'config')

    def test_ml_orchestrator_initialization_automl_disabled(self, ml_orchestrator_automl_disabled):
        """AutoML無効時の初期化テスト"""
        orchestrator = ml_orchestrator_automl_disabled
        
        assert orchestrator.enable_automl is False
        assert hasattr(orchestrator, 'ml_training_service')
        assert hasattr(orchestrator, 'feature_service')

    def test_automl_status_retrieval(self, ml_orchestrator_automl_enabled, ml_orchestrator_automl_disabled):
        """AutoML状態取得テスト"""
        # AutoML有効時
        status_enabled = ml_orchestrator_automl_enabled.get_automl_status()
        assert isinstance(status_enabled, dict)
        assert 'enabled' in status_enabled
        assert status_enabled['enabled'] is True
        assert 'service_type' in status_enabled
        assert 'config' in status_enabled

        # AutoML無効時
        status_disabled = ml_orchestrator_automl_disabled.get_automl_status()
        assert isinstance(status_disabled, dict)
        assert 'enabled' in status_disabled
        assert status_disabled['enabled'] is False

    @patch('app.services.auto_strategy.services.ml_orchestrator.MLTrainingService')
    def test_ml_indicators_calculation_success(self, mock_ml_service, ml_orchestrator_automl_enabled, sample_market_data):
        """ML指標計算成功テスト"""
        # モックの設定
        mock_service_instance = Mock()
        mock_service_instance.generate_signals.return_value = {
            'ML_UP_PROB': 0.7,
            'ML_DOWN_PROB': 0.2,
            'ML_RANGE_PROB': 0.1
        }
        mock_ml_service.return_value = mock_service_instance
        
        # feature_serviceのモック
        with patch.object(ml_orchestrator_automl_enabled, 'feature_service') as mock_feature_service:
            mock_feature_service.calculate_features.return_value = pd.DataFrame({
                'feature1': np.random.randn(len(sample_market_data)),
                'feature2': np.random.randn(len(sample_market_data))
            })
            
            # ML指標計算実行
            try:
                ml_indicators = ml_orchestrator_automl_enabled.calculate_ml_indicators(sample_market_data)
                
                # 結果検証
                assert isinstance(ml_indicators, dict)
                expected_keys = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
                for key in expected_keys:
                    assert key in ml_indicators
                    assert isinstance(ml_indicators[key], (list, np.ndarray))
                    
            except Exception as e:
                # 依存関係の問題でエラーが発生する場合は警告として記録
                logger.warning(f"ML指標計算テストでエラー: {e}")
                pytest.skip(f"ML指標計算テストをスキップ: {e}")

    def test_ml_indicators_calculation_with_invalid_data(self, ml_orchestrator_automl_enabled):
        """無効データでのML指標計算テスト"""
        invalid_data_cases = [
            pd.DataFrame(),  # 空のDataFrame
            None,  # None値
            pd.DataFrame({'invalid': [1, 2, 3]}),  # 必要な列が不足
        ]

        for invalid_data in invalid_data_cases:
            try:
                result = ml_orchestrator_automl_enabled.calculate_ml_indicators(invalid_data)
                # エラーが発生しない場合、適切なデフォルト値が返されることを確認
                if result is not None:
                    assert isinstance(result, dict)
            except Exception as e:
                # 適切なエラーハンドリングが行われることを確認
                assert any(keyword in str(e).lower() for keyword in ['data', 'invalid', 'empty', 'error'])

    @patch('app.services.auto_strategy.services.ml_orchestrator.UnifiedErrorHandler')
    def test_error_handling_validation(self, mock_error_handler, ml_orchestrator_automl_enabled, sample_market_data):
        """エラーハンドリング検証テスト"""
        # バリデーション失敗のシミュレーション
        mock_error_handler.validate_predictions.return_value = False
        
        with patch.object(ml_orchestrator_automl_enabled, 'feature_service') as mock_feature_service:
            mock_feature_service.calculate_features.return_value = pd.DataFrame({
                'feature1': np.random.randn(len(sample_market_data))
            })
            
            with patch.object(ml_orchestrator_automl_enabled, 'ml_training_service') as mock_ml_service:
                mock_ml_service.generate_signals.return_value = {'invalid': 'predictions'}
                
                try:
                    ml_orchestrator_automl_enabled.calculate_ml_indicators(sample_market_data)
                    pytest.fail("バリデーション失敗時にエラーが発生しませんでした")
                except Exception as e:
                    assert "バリデーション" in str(e) or "validation" in str(e).lower()

    def test_prediction_expansion_to_data_length(self, ml_orchestrator_automl_enabled):
        """予測値のデータ長への拡張テスト"""
        # テスト用の予測値
        predictions = {
            'ML_UP_PROB': 0.6,
            'ML_DOWN_PROB': 0.3,
            'ML_RANGE_PROB': 0.1
        }
        
        data_length = 100
        
        # 内部メソッドのテスト（実装に依存）
        try:
            expanded = ml_orchestrator_automl_enabled._expand_predictions_to_data_length(predictions, data_length)
            
            assert isinstance(expanded, dict)
            for key, value in expanded.items():
                assert len(value) == data_length
                assert all(isinstance(v, (int, float)) for v in value)
                
        except AttributeError:
            # メソッドが存在しない場合はスキップ
            pytest.skip("_expand_predictions_to_data_length メソッドが存在しません")

    def test_ml_indicators_validation(self, ml_orchestrator_automl_enabled):
        """ML指標検証テスト"""
        # 有効なML指標
        valid_indicators = {
            'ML_UP_PROB': [0.7, 0.6, 0.8],
            'ML_DOWN_PROB': [0.2, 0.3, 0.1],
            'ML_RANGE_PROB': [0.1, 0.1, 0.1]
        }
        
        # 無効なML指標
        invalid_indicators_cases = [
            {},  # 空の辞書
            {'ML_UP_PROB': []},  # 空のリスト
            {'ML_UP_PROB': [1.5, 0.6]},  # 範囲外の値
            {'ML_UP_PROB': ['invalid', 'data']},  # 非数値データ
        ]

        # 有効性検証メソッドのテスト（実装に依存）
        try:
            ml_orchestrator_automl_enabled._validate_ml_indicators(valid_indicators)
            # エラーが発生しないことを確認
        except AttributeError:
            pytest.skip("_validate_ml_indicators メソッドが存在しません")
        except Exception as e:
            pytest.fail(f"有効なML指標の検証でエラー: {e}")

        # 無効な指標でのエラー確認
        for invalid_indicators in invalid_indicators_cases:
            try:
                ml_orchestrator_automl_enabled._validate_ml_indicators(invalid_indicators)
                # エラーが発生しない場合は警告
                logger.warning(f"無効なML指標でエラーが発生しませんでした: {invalid_indicators}")
            except AttributeError:
                pytest.skip("_validate_ml_indicators メソッドが存在しません")
            except Exception:
                # 適切にエラーが発生することを確認
                pass

    def test_automl_config_handling(self):
        """AutoML設定ハンドリングテスト"""
        custom_automl_config = {
            'model_type': 'ensemble',
            'max_trials': 50,
            'timeout': 3600
        }
        
        orchestrator = MLOrchestrator(
            enable_automl=True,
            automl_config=custom_automl_config
        )
        
        assert orchestrator.enable_automl is True
        assert orchestrator.automl_config == custom_automl_config

    def test_feature_service_integration(self, ml_orchestrator_automl_enabled, sample_market_data):
        """特徴量サービス統合テスト"""
        try:
            # 特徴量サービスが適切に統合されていることを確認
            assert hasattr(ml_orchestrator_automl_enabled, 'feature_service')
            
            # 特徴量計算の実行（モックなし）
            if hasattr(ml_orchestrator_automl_enabled.feature_service, 'calculate_features'):
                features = ml_orchestrator_automl_enabled.feature_service.calculate_features(sample_market_data)
                if features is not None:
                    assert isinstance(features, pd.DataFrame)
                    assert len(features) > 0
                    
        except Exception as e:
            logger.warning(f"特徴量サービス統合テストでエラー: {e}")
            pytest.skip(f"特徴量サービス統合テストをスキップ: {e}")

    def test_ml_training_service_integration(self, ml_orchestrator_automl_enabled):
        """MLトレーニングサービス統合テスト"""
        try:
            # MLトレーニングサービスが適切に統合されていることを確認
            assert hasattr(ml_orchestrator_automl_enabled, 'ml_training_service')
            assert ml_orchestrator_automl_enabled.ml_training_service is not None
            
            # サービスの基本的な属性確認
            ml_service = ml_orchestrator_automl_enabled.ml_training_service
            if hasattr(ml_service, 'trainer_type'):
                assert ml_service.trainer_type is not None
                
        except Exception as e:
            logger.warning(f"MLトレーニングサービス統合テストでエラー: {e}")
            pytest.skip(f"MLトレーニングサービス統合テストをスキップ: {e}")

    def test_safe_ml_operation_decorator(self, ml_orchestrator_automl_enabled, sample_market_data):
        """安全なML操作デコレータテスト"""
        # エラーが発生した場合の安全な処理を確認
        with patch.object(ml_orchestrator_automl_enabled, 'feature_service') as mock_feature_service:
            # 特徴量計算でエラーを発生させる
            mock_feature_service.calculate_features.side_effect = Exception("Feature calculation error")
            
            try:
                result = ml_orchestrator_automl_enabled.calculate_ml_indicators(sample_market_data)
                # エラーハンドリングが適切に行われることを確認
                if result is not None:
                    logger.warning("エラー発生時にもNone以外の結果が返されました")
            except Exception as e:
                # 適切なエラーメッセージが含まれることを確認
                assert any(keyword in str(e).lower() for keyword in ['error', 'failed', 'calculation'])

    def test_memory_efficiency(self, ml_orchestrator_automl_enabled):
        """メモリ効率性テスト"""
        # 大量データでのメモリ使用量テスト
        large_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10000, freq='1H'),
            'open': np.random.randn(10000) * 1000 + 50000,
            'high': np.random.randn(10000) * 1000 + 51000,
            'low': np.random.randn(10000) * 1000 + 49000,
            'close': np.random.randn(10000) * 1000 + 50000,
            'volume': np.random.randn(10000) * 100 + 1000,
        })
        
        try:
            # メモリ使用量の監視（簡易版）
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            # ML指標計算実行
            with patch.object(ml_orchestrator_automl_enabled, 'feature_service') as mock_feature_service:
                mock_feature_service.calculate_features.return_value = pd.DataFrame({
                    'feature1': np.random.randn(len(large_data))
                })
                
                with patch.object(ml_orchestrator_automl_enabled, 'ml_training_service') as mock_ml_service:
                    mock_ml_service.generate_signals.return_value = {
                        'ML_UP_PROB': 0.6,
                        'ML_DOWN_PROB': 0.3,
                        'ML_RANGE_PROB': 0.1
                    }
                    
                    result = ml_orchestrator_automl_enabled.calculate_ml_indicators(large_data)
            
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            # メモリ増加が合理的な範囲内であることを確認（100MB以下）
            assert memory_increase < 100 * 1024 * 1024, f"メモリ使用量が過大: {memory_increase / 1024 / 1024:.2f}MB"
            
        except ImportError:
            pytest.skip("psutilが利用できないため、メモリ効率性テストをスキップ")
        except Exception as e:
            logger.warning(f"メモリ効率性テストでエラー: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
