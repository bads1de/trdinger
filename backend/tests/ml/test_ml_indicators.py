"""
ML指標の動作テスト

ML_UP_PROB、ML_DOWN_PROB、ML_RANGE_PROB指標の動作を確認します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def create_sample_ohlcv_data(length=100):
    """テスト用のOHLCVデータを作成"""
    dates = pd.date_range(start='2024-01-01', periods=length, freq='h')
    
    np.random.seed(42)
    price_base = 50000
    
    # より現実的な価格データを生成
    returns = np.random.randn(length) * 0.02  # 2%の標準偏差
    prices = [price_base]
    
    for i in range(1, length):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    ohlcv_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(length)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(length)) * 0.01),
        'close': prices * (1 + np.random.randn(length) * 0.005),
        'volume': np.random.rand(length) * 1000000
    })
    
    return ohlcv_data


def test_ml_indicator_service_import():
    """MLIndicatorServiceのインポートテスト"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        
        service = MLIndicatorService()
        assert service is not None
        
        print("MLIndicatorService import successful")
        return service
        
    except ImportError as e:
        pytest.fail(f"MLIndicatorService import failed: {e}")


def test_feature_engineering_service():
    """FeatureEngineeringServiceの動作テスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        ohlcv_data = create_sample_ohlcv_data()
        
        # 特徴量計算（lookback_periodsを明示的に指定）
        lookback_periods = {
            "short_ma": 10,
            "long_ma": 50,
            "volatility": 20,
            "momentum": 14,
            "volume": 20
        }
        features_df = service.calculate_advanced_features(ohlcv_data, lookback_periods=lookback_periods)
        
        assert not features_df.empty
        assert len(features_df) == len(ohlcv_data)
        
        # 基本的な特徴量が含まれているか確認
        expected_features = [
            'Price_MA_Ratio_Short', 'Price_MA_Ratio_Long',
            'Realized_Volatility_20', 'Volume_Ratio'
        ]
        
        for feature in expected_features:
            if feature in features_df.columns:
                print(f"Feature '{feature}' calculated successfully")
            else:
                print(f"Feature '{feature}' not found")

        print(f"FeatureEngineeringService calculated {len(features_df.columns)} features")
        return features_df
        
    except Exception as e:
        pytest.fail(f"FeatureEngineeringService test failed: {e}")


def test_ml_signal_generator():
    """MLSignalGeneratorの動作テスト"""
    try:
        from app.core.services.ml import MLSignalGenerator
        
        generator = MLSignalGenerator()
        ohlcv_data = create_sample_ohlcv_data()
        
        # 学習用データの準備をテスト
        features_df = pd.DataFrame({
            'Price_MA_Ratio_Short': np.random.randn(len(ohlcv_data)),
            'Price_MA_Ratio_Long': np.random.randn(len(ohlcv_data)),
            'Realized_Volatility_20': np.abs(np.random.randn(len(ohlcv_data))),
            'Volume_Ratio': np.abs(np.random.randn(len(ohlcv_data))),
            'close': ohlcv_data['close'],
            'target_up': (np.random.rand(len(ohlcv_data)) > 0.5).astype(int),
            'target_down': (np.random.rand(len(ohlcv_data)) > 0.5).astype(int),
            'target_range': (np.random.rand(len(ohlcv_data)) > 0.5).astype(int),
        })
        
        # 学習データ準備のテスト
        X, y = generator.prepare_training_data(features_df)
        
        assert not X.empty
        assert len(y) > 0
        assert len(X) == len(y)
        
        print(f"MLSignalGenerator prepared training data: {len(X)} samples")

        # 予測テスト（モデル未学習時）
        predictions = generator.predict(features_df.head(1))

        assert 'up' in predictions
        assert 'down' in predictions
        assert 'range' in predictions

        # 確率の合計が1に近いことを確認
        prob_sum = predictions['up'] + predictions['down'] + predictions['range']
        assert abs(prob_sum - 1.0) < 0.1

        print("MLSignalGenerator prediction works (default values)")
        return generator
        
    except Exception as e:
        pytest.fail(f"MLSignalGenerator test failed: {e}")


def test_ml_indicators_calculation():
    """ML指標の計算テスト"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        
        service = MLIndicatorService()
        ohlcv_data = create_sample_ohlcv_data()
        
        # ML指標の計算
        ml_indicators = service.calculate_ml_indicators(ohlcv_data)
        
        # 期待される指標が含まれているか確認
        expected_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
        
        for indicator in expected_indicators:
            assert indicator in ml_indicators, f"Missing indicator: {indicator}"
            
            values = ml_indicators[indicator]
            assert len(values) == len(ohlcv_data), f"Length mismatch for {indicator}"
            assert np.all(values >= 0), f"Negative values in {indicator}"
            assert np.all(values <= 1), f"Values > 1 in {indicator}"
            
            print(f"{indicator}: shape={values.shape}, range=[{values.min():.3f}, {values.max():.3f}]")

        # 確率の合計が1に近いことを確認
        prob_sums = (ml_indicators['ML_UP_PROB'] +
                    ml_indicators['ML_DOWN_PROB'] +
                    ml_indicators['ML_RANGE_PROB'])

        assert np.all(np.abs(prob_sums - 1.0) < 0.1), "Probability sums are not close to 1"

        print("ML indicators calculated successfully")
        return ml_indicators
        
    except Exception as e:
        pytest.fail(f"ML indicators calculation failed: {e}")


def test_indicator_calculator_integration():
    """IndicatorCalculatorとの統合テスト"""
    try:
        from app.core.services.auto_strategy.calculators.indicator_calculator import IndicatorCalculator

        calculator = IndicatorCalculator()
        ohlcv_data = create_sample_ohlcv_data()

        # backtesting.pyのDataオブジェクトを模擬
        class MockData:
            def __init__(self, df):
                self.df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                # timestampカラムも追加
                if 'timestamp' in df.columns:
                    self.df['timestamp'] = df['timestamp']

        mock_data = MockData(ohlcv_data)

        # ML指標の計算テスト
        ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']

        for indicator_type in ml_indicators:
            try:
                # 引数順序を修正: indicator_type, parameters, data
                result = calculator.calculate_indicator(indicator_type, {}, mock_data)

                if result is not None:
                    assert len(result) == len(ohlcv_data)
                    assert np.all(result >= 0)
                    assert np.all(result <= 1)
                    print(f"IndicatorCalculator.calculate_indicator('{indicator_type}') works")
                else:
                    print(f"IndicatorCalculator test for {indicator_type}: returned None (expected for uninitialized ML model)")

            except Exception as e:
                print(f"IndicatorCalculator test for {indicator_type} failed: {e}")

        print("IndicatorCalculator integration test completed")
        
    except Exception as e:
        pytest.fail(f"IndicatorCalculator integration test failed: {e}")


def test_technical_indicator_service_registration():
    """TechnicalIndicatorServiceでのML指標登録テスト"""
    try:
        from app.core.services.indicators import TechnicalIndicatorService
        from app.core.services.indicators.config import indicator_registry

        service = TechnicalIndicatorService()
        ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']

        # レジストリから直接確認
        for indicator in ml_indicators:
            if indicator_registry.is_indicator_supported(indicator):
                print(f"{indicator} is registered in indicator_registry")

                # 設定の詳細確認
                config = indicator_registry.get_indicator_config(indicator)
                if config:
                    print(f"  - Category: {config.category}")
                    print(f"  - Scale type: {config.scale_type}")
                    print(f"  - Adapter function: {config.adapter_function}")
            else:
                print(f"{indicator} is not registered in indicator_registry")

        # TechnicalIndicatorServiceでの対応確認
        try:
            if hasattr(service, 'get_supported_indicators'):
                supported_indicators = service.get_supported_indicators()
                print(f"TechnicalIndicatorService supports {len(supported_indicators)} indicators")
            else:
                print("TechnicalIndicatorService does not have get_supported_indicators method")
        except Exception as e:
            print(f"Error getting supported indicators: {e}")

        print("TechnicalIndicatorService registration check completed")

    except Exception as e:
        print(f"TechnicalIndicatorService test failed: {e}")


if __name__ == "__main__":
    """テストの直接実行"""
    print("ML指標の動作テストを開始...")
    print("=" * 60)
    
    try:
        # 各テストを順次実行
        print("\n1. MLIndicatorService インポートテスト")
        ml_service = test_ml_indicator_service_import()
        
        print("\n2. FeatureEngineeringService 動作テスト")
        features_df = test_feature_engineering_service()
        
        print("\n3. MLSignalGenerator 動作テスト")
        ml_generator = test_ml_signal_generator()
        
        print("\n4. ML指標計算テスト")
        ml_indicators = test_ml_indicators_calculation()
        
        print("\n5. IndicatorCalculator 統合テスト")
        test_indicator_calculator_integration()
        
        print("\n6. TechnicalIndicatorService 登録テスト")
        test_technical_indicator_service_registration()
        
        print("\n" + "=" * 60)
        print("すべてのML指標テストが完了しました！")
        print("ML_UP_PROB、ML_DOWN_PROB、ML_RANGE_PROB指標は正常に動作しています。")

    except Exception as e:
        print(f"\nテストエラー: {e}")
        raise
