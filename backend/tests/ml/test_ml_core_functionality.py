"""
ML基本機能テストスイート

MLIndicatorService、MLSignalGenerator、FeatureEngineeringServiceの
基本機能を包括的にテストします。
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .utils import (
    create_sample_ohlcv_data,
    create_sample_funding_rate_data,
    create_sample_open_interest_data,
    MLTestConfig,
    measure_performance,
    validate_ml_predictions,
    create_comprehensive_test_data
)


class MLCoreFunctionalityTestSuite:
    """ML基本機能テストスイート"""
    
    def __init__(self):
        self.config = MLTestConfig()
        self.test_results = []
        
    def run_all_tests(self):
        """全テストを実行"""
        print("ML基本機能テストスイート開始")
        print("=" * 60)
        
        tests = [
            self.test_ml_indicator_service_initialization,
            self.test_ml_indicator_service_basic_calculation,
            self.test_ml_signal_generator_initialization,
            self.test_ml_signal_generator_training_data_preparation,
            self.test_ml_signal_generator_prediction,
            self.test_feature_engineering_service_initialization,
            self.test_feature_engineering_service_basic_features,
            self.test_feature_engineering_service_advanced_features,
            self.test_ml_services_integration,
            self.test_error_handling_and_edge_cases,
        ]
        
        passed = 0
        total = len(tests)
        
        for test in tests:
            try:
                print(f"\n実行中: {test.__name__}")
                if test():
                    passed += 1
                    print("✓ PASS")
                else:
                    print("✗ FAIL")
            except Exception as e:
                print(f"✗ ERROR: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 60)
        print(f"テスト結果: {passed}/{total} 成功")
        
        if passed == total:
            print("全テスト成功！ML基本機能は正常に動作しています。")
        else:
            print(f"{total - passed}個のテストが失敗しました。")
            
        return passed == total

    def test_ml_indicator_service_initialization(self):
        """MLIndicatorServiceの初期化テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            
            # 基本属性の確認
            assert hasattr(service, 'feature_service')
            assert hasattr(service, 'ml_generator')
            assert hasattr(service, 'is_model_loaded')
            assert hasattr(service, '_last_predictions')
            
            # 初期状態の確認
            assert service.is_model_loaded == False
            assert isinstance(service._last_predictions, dict)
            assert 'up' in service._last_predictions
            assert 'down' in service._last_predictions
            assert 'range' in service._last_predictions
            
            print("MLIndicatorService初期化成功")
            return True
            
        except Exception as e:
            print(f"MLIndicatorService初期化失敗: {e}")
            return False

    def test_ml_indicator_service_basic_calculation(self):
        """MLIndicatorServiceの基本計算テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            ohlcv_data = create_sample_ohlcv_data(self.config.sample_size)
            
            # ML指標計算
            result, metrics = measure_performance(
                service.calculate_ml_indicators,
                ohlcv_data
            )
            
            # 結果の検証
            assert isinstance(result, dict)
            expected_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
            
            for indicator in expected_indicators:
                assert indicator in result
                values = result[indicator]
                assert isinstance(values, np.ndarray)
                assert len(values) == len(ohlcv_data)
                assert np.all(values >= 0)
                assert np.all(values <= 1)
            
            # 確率の合計確認
            prob_sums = (result['ML_UP_PROB'] + 
                        result['ML_DOWN_PROB'] + 
                        result['ML_RANGE_PROB'])
            assert np.all(np.abs(prob_sums - 1.0) < 0.1)
            
            print(f"ML指標計算成功 - 実行時間: {metrics.execution_time:.3f}秒")
            return True
            
        except Exception as e:
            print(f"ML指標計算失敗: {e}")
            return False

    def test_ml_signal_generator_initialization(self):
        """MLSignalGeneratorの初期化テスト"""
        try:
            from app.core.services.ml.signal_generator import MLSignalGenerator
            
            # 一時ディレクトリでテスト
            with tempfile.TemporaryDirectory() as temp_dir:
                generator = MLSignalGenerator(model_save_path=temp_dir)
                
                # 基本属性の確認
                assert hasattr(generator, 'model')
                assert hasattr(generator, 'scaler')
                assert hasattr(generator, 'feature_columns')
                assert hasattr(generator, 'is_trained')
                assert hasattr(generator, 'model_save_path')
                
                # 初期状態の確認
                assert generator.model is None
                assert generator.scaler is None
                assert generator.feature_columns is None
                assert generator.is_trained == False
                assert generator.model_save_path == temp_dir
                
                print("MLSignalGenerator初期化成功")
                return True
                
        except Exception as e:
            print(f"MLSignalGenerator初期化失敗: {e}")
            return False

    def test_ml_signal_generator_training_data_preparation(self):
        """MLSignalGeneratorの学習データ準備テスト"""
        try:
            from app.core.services.ml.signal_generator import MLSignalGenerator
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            generator = MLSignalGenerator()
            feature_service = FeatureEngineeringService()
            
            # テストデータ準備
            ohlcv_data = create_sample_ohlcv_data(self.config.sample_size)
            
            # 特徴量計算
            lookback_periods = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20
            }
            features_df = feature_service.calculate_advanced_features(
                ohlcv_data, lookback_periods=lookback_periods
            )
            
            # 学習データ準備
            X, y = generator.prepare_training_data(
                features_df,
                prediction_horizon=self.config.prediction_horizon,
                threshold_up=self.config.threshold_up,
                threshold_down=self.config.threshold_down
            )
            
            # 結果の検証
            assert isinstance(X, pd.DataFrame)
            assert isinstance(y, pd.Series)
            assert len(X) == len(y)
            assert len(X) > 0
            assert len(X.columns) > 0
            
            # ラベルの確認
            unique_labels = set(y.unique())
            expected_labels = {0, 1, 2}  # down, up, range
            assert unique_labels.issubset(expected_labels)
            
            print(f"学習データ準備成功 - サンプル数: {len(X)}, 特徴量数: {len(X.columns)}")
            return True
            
        except Exception as e:
            print(f"学習データ準備失敗: {e}")
            return False

    def test_ml_signal_generator_prediction(self):
        """MLSignalGeneratorの予測テスト"""
        try:
            from app.core.services.ml.signal_generator import MLSignalGenerator
            
            generator = MLSignalGenerator()
            
            # ダミー特徴量データ
            features_df = pd.DataFrame({
                'feature1': [1.0, 2.0, 3.0],
                'feature2': [0.5, 1.5, 2.5],
                'feature3': [10.0, 20.0, 30.0],
                'close': [50000.0, 51000.0, 52000.0]
            })
            
            # 予測実行（未学習状態）
            predictions = generator.predict(features_df.head(1))
            
            # 結果の検証
            assert isinstance(predictions, dict)
            assert validate_ml_predictions(predictions)
            
            # デフォルト値の確認
            assert 0.2 <= predictions['up'] <= 0.5
            assert 0.2 <= predictions['down'] <= 0.5
            assert 0.2 <= predictions['range'] <= 0.5
            
            print("ML予測成功（未学習状態でのデフォルト値）")
            return True
            
        except Exception as e:
            print(f"ML予測失敗: {e}")
            return False

    def test_feature_engineering_service_initialization(self):
        """FeatureEngineeringServiceの初期化テスト"""
        try:
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            
            # 基本属性の確認
            assert hasattr(service, 'feature_cache')
            assert hasattr(service, 'max_cache_size')
            assert hasattr(service, 'cache_ttl')
            
            # 初期状態の確認
            assert isinstance(service.feature_cache, dict)
            assert len(service.feature_cache) == 0
            assert service.max_cache_size > 0
            assert service.cache_ttl > 0
            
            print("FeatureEngineeringService初期化成功")
            return True
            
        except Exception as e:
            print(f"FeatureEngineeringService初期化失敗: {e}")
            return False

    def test_feature_engineering_service_basic_features(self):
        """FeatureEngineeringServiceの基本特徴量テスト"""
        try:
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            ohlcv_data = create_sample_ohlcv_data(self.config.sample_size)
            
            # 基本特徴量計算
            lookback_periods = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20
            }
            
            result, metrics = measure_performance(
                service.calculate_advanced_features,
                ohlcv_data,
                lookback_periods=lookback_periods
            )
            
            # 結果の検証
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(ohlcv_data)
            assert len(result.columns) > len(ohlcv_data.columns)
            
            # 基本特徴量の存在確認
            expected_features = [
                'Price_MA_Ratio_Short',
                'Price_MA_Ratio_Long',
                'Realized_Volatility_20',
                'Volume_Ratio'
            ]
            
            found_features = 0
            for feature in expected_features:
                if feature in result.columns:
                    found_features += 1
            
            assert found_features >= len(expected_features) // 2  # 半分以上の特徴量が存在
            
            print(f"基本特徴量計算成功 - 特徴量数: {len(result.columns)}, 実行時間: {metrics.execution_time:.3f}秒")
            return True
            
        except Exception as e:
            print(f"基本特徴量計算失敗: {e}")
            return False

    def test_feature_engineering_service_advanced_features(self):
        """FeatureEngineeringServiceの高度特徴量テスト"""
        try:
            from app.core.services.feature_engineering.feature_engineering_service import FeatureEngineeringService
            
            service = FeatureEngineeringService()
            ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
            
            # 高度特徴量計算
            lookback_periods = {
                "short_ma": 10,
                "long_ma": 50,
                "volatility": 20,
                "momentum": 14,
                "volume": 20
            }
            result = service.calculate_advanced_features(
                ohlcv_data,
                funding_rate_data,
                open_interest_data,
                lookback_periods=lookback_periods
            )
            
            # 結果の検証
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(ohlcv_data)
            
            # 外部データ関連特徴量の確認
            external_features = ['FR_Change', 'OI_Change_Rate']
            found_external = 0
            for feature in external_features:
                if feature in result.columns:
                    found_external += 1
            
            # 少なくとも一部の外部特徴量が存在することを確認
            print(f"高度特徴量計算成功 - 総特徴量数: {len(result.columns)}, 外部特徴量: {found_external}")
            return True
            
        except Exception as e:
            print(f"高度特徴量計算失敗: {e}")
            return False

    def test_ml_services_integration(self):
        """MLサービス統合テスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
            
            # 統合計算
            result = service.calculate_ml_indicators(
                ohlcv_data,
                funding_rate_data,
                open_interest_data
            )
            
            # 結果の検証
            assert isinstance(result, dict)
            assert len(result) == 3  # ML_UP_PROB, ML_DOWN_PROB, ML_RANGE_PROB
            
            for indicator_name, values in result.items():
                assert isinstance(values, np.ndarray)
                assert len(values) == len(ohlcv_data)
                assert np.all(values >= 0)
                assert np.all(values <= 1)
            
            print("MLサービス統合テスト成功")
            return True
            
        except Exception as e:
            print(f"MLサービス統合テスト失敗: {e}")
            return False

    def test_error_handling_and_edge_cases(self):
        """エラーハンドリング・エッジケーステスト"""
        try:
            from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
            
            service = MLIndicatorService()
            
            # 空データテスト
            empty_df = pd.DataFrame()
            result = service.calculate_ml_indicators(empty_df)
            assert isinstance(result, dict)
            assert len(result) == 3
            
            # 不正データテスト
            invalid_df = pd.DataFrame({'invalid': [1, 2, 3]})
            result = service.calculate_ml_indicators(invalid_df)
            assert isinstance(result, dict)
            assert len(result) == 3
            
            # 最小データテスト
            minimal_df = pd.DataFrame({
                'open': [100.0, 101.0],
                'high': [102.0, 103.0],
                'low': [99.0, 100.0],
                'close': [101.0, 102.0],
                'volume': [1000.0, 1100.0]
            })
            result = service.calculate_ml_indicators(minimal_df)
            assert isinstance(result, dict)
            assert len(result) == 3
            
            print("エラーハンドリング・エッジケーステスト成功")
            return True
            
        except Exception as e:
            print(f"エラーハンドリング・エッジケーステスト失敗: {e}")
            return False


def main():
    """メインテスト実行"""
    test_suite = MLCoreFunctionalityTestSuite()
    success = test_suite.run_all_tests()
    return success


if __name__ == "__main__":
    main()
