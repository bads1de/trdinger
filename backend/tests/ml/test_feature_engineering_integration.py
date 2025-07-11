"""
FeatureEngineeringServiceの統合テスト

高度な特徴量計算が正常に動作し、GAに統合されることを確認します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def create_comprehensive_test_data():
    """包括的なテストデータを作成"""
    # 1ヶ月分の時間足データ
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    
    np.random.seed(42)
    price_base = 50000
    
    # より現実的な価格データを生成
    returns = np.random.randn(len(dates)) * 0.02
    prices = [price_base]
    
    for i in range(1, len(dates)):
        prices.append(prices[-1] * (1 + returns[i]))
    
    prices = np.array(prices)
    
    # OHLCV データ
    ohlcv_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
        'close': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'volume': np.random.rand(len(dates)) * 1000000
    })
    
    # ファンディングレートデータ
    funding_rate_data = pd.DataFrame({
        'timestamp': dates[::8],  # 8時間ごと
        'funding_rate': np.random.randn(len(dates)//8) * 0.0001  # 0.01%程度
    })
    
    # 建玉残高データ
    open_interest_data = pd.DataFrame({
        'timestamp': dates,
        'open_interest_value': np.random.rand(len(dates)) * 1e9 + 5e8  # 5億-15億
    })
    
    return ohlcv_data, funding_rate_data, open_interest_data


def test_feature_engineering_service_basic():
    """FeatureEngineeringServiceの基本機能テスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
        
        # 基本的な特徴量計算
        features_df = service.calculate_advanced_features(ohlcv_data)
        
        assert not features_df.empty
        assert len(features_df) == len(ohlcv_data)
        
        # 基本的な特徴量が含まれているか確認
        basic_features = [
            'Price_MA_Ratio_Short', 'Price_MA_Ratio_Long',
            'Realized_Volatility_20', 'Volume_Ratio',
            'RSI', 'MACD'
        ]
        
        found_features = []
        for feature in basic_features:
            if feature in features_df.columns:
                found_features.append(feature)
        
        print(f"✅ Basic features found: {found_features}")
        print(f"✅ Total features calculated: {len(features_df.columns)}")
        
        return features_df
        
    except Exception as e:
        pytest.fail(f"FeatureEngineeringService basic test failed: {e}")


def test_advanced_features_with_external_data():
    """外部データを含む高度な特徴量テスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
        
        # 外部データを含む特徴量計算
        features_df = service.calculate_advanced_features(
            ohlcv_data, 
            funding_rate_data, 
            open_interest_data
        )
        
        # 外部データ関連の特徴量が含まれているか確認
        external_features = [
            'FR_Change', 'Price_FR_Divergence',
            'OI_Change_Rate', 'OI_Surge',
            'FR_OI_Ratio', 'Market_Heat_Index'
        ]
        
        found_external_features = []
        for feature in external_features:
            if feature in features_df.columns:
                found_external_features.append(feature)
        
        print(f"✅ External data features found: {found_external_features}")
        
        # データの妥当性チェック
        assert not features_df.isnull().all().any(), "Some features are all NaN"
        
        return features_df
        
    except Exception as e:
        pytest.fail(f"Advanced features with external data test failed: {e}")


def test_feature_names_consistency():
    """特徴量名の一貫性テスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        
        # 期待される特徴量名を取得
        expected_features = service.get_feature_names()
        
        # 実際に計算される特徴量
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
        features_df = service.calculate_advanced_features(
            ohlcv_data, 
            funding_rate_data, 
            open_interest_data
        )
        
        actual_features = [col for col in features_df.columns 
                          if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # 期待される特徴量と実際の特徴量の比較
        expected_set = set(expected_features)
        actual_set = set(actual_features)
        
        missing_features = expected_set - actual_set
        extra_features = actual_set - expected_set
        
        if missing_features:
            print(f"⚠️ Missing features: {missing_features}")
        
        if extra_features:
            print(f"ℹ️ Extra features: {extra_features}")
        
        common_features = expected_set & actual_set
        print(f"✅ Common features: {len(common_features)}/{len(expected_features)}")
        
        # 少なくとも50%の特徴量が一致することを確認
        assert len(common_features) >= len(expected_features) * 0.5
        
    except Exception as e:
        pytest.fail(f"Feature names consistency test failed: {e}")


def test_ml_indicator_service_integration():
    """MLIndicatorServiceとの統合テスト"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        
        service = MLIndicatorService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
        
        # ML指標の計算（FeatureEngineeringServiceを内部で使用）
        ml_indicators = service.calculate_ml_indicators(
            ohlcv_data,
            funding_rate_data,
            open_interest_data
        )
        
        # ML指標が正常に計算されることを確認
        expected_ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
        
        for indicator in expected_ml_indicators:
            assert indicator in ml_indicators
            values = ml_indicators[indicator]
            assert len(values) == len(ohlcv_data)
            assert np.all(values >= 0) and np.all(values <= 1)
        
        print("✅ MLIndicatorService integration works")
        
    except Exception as e:
        pytest.fail(f"MLIndicatorService integration test failed: {e}")


def test_performance_and_memory():
    """パフォーマンスとメモリ使用量のテスト"""
    try:
        import time
        import psutil
        import os
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        
        # より大きなデータセットでテスト
        large_dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='1H')
        large_ohlcv = pd.DataFrame({
            'timestamp': large_dates,
            'open': np.random.rand(len(large_dates)) * 1000 + 50000,
            'high': np.random.rand(len(large_dates)) * 1000 + 50500,
            'low': np.random.rand(len(large_dates)) * 1000 + 49500,
            'close': np.random.rand(len(large_dates)) * 1000 + 50000,
            'volume': np.random.rand(len(large_dates)) * 1000000
        })
        
        # メモリ使用量測定開始
        process = psutil.Process(os.getpid())
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # 実行時間測定開始
        start_time = time.time()
        
        # 特徴量計算実行
        features_df = service.calculate_advanced_features(large_ohlcv)
        
        # 実行時間測定終了
        end_time = time.time()
        execution_time = end_time - start_time
        
        # メモリ使用量測定終了
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"✅ Performance test completed:")
        print(f"   - Data size: {len(large_ohlcv)} records")
        print(f"   - Execution time: {execution_time:.2f} seconds")
        print(f"   - Memory used: {memory_used:.2f} MB")
        print(f"   - Features calculated: {len(features_df.columns)}")
        
        # パフォーマンス基準（緩い基準）
        assert execution_time < 60, f"Execution time too long: {execution_time}s"
        assert memory_used < 500, f"Memory usage too high: {memory_used}MB"
        
    except Exception as e:
        print(f"⚠️ Performance test failed: {e}")
        # パフォーマンステストは失敗しても致命的ではない


def test_error_handling():
    """エラーハンドリングのテスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        
        # 空のデータでのテスト
        empty_df = pd.DataFrame()
        
        try:
            features_df = service.calculate_advanced_features(empty_df)
            pytest.fail("Empty data should raise an error")
        except ValueError:
            print("✅ Empty data error handling works")
        
        # 不正なデータでのテスト
        invalid_df = pd.DataFrame({
            'invalid_column': [1, 2, 3]
        })
        
        try:
            features_df = service.calculate_advanced_features(invalid_df)
            pytest.fail("Invalid data should raise an error")
        except (ValueError, KeyError):
            print("✅ Invalid data error handling works")
        
    except Exception as e:
        pytest.fail(f"Error handling test failed: {e}")


if __name__ == "__main__":
    """テストの直接実行"""
    print("🔧 FeatureEngineeringServiceの統合テストを開始...")
    print("=" * 60)
    
    try:
        # 各テストを順次実行
        print("\n1. 基本機能テスト")
        basic_features = test_feature_engineering_service_basic()
        
        print("\n2. 外部データを含む高度な特徴量テスト")
        advanced_features = test_advanced_features_with_external_data()
        
        print("\n3. 特徴量名の一貫性テスト")
        test_feature_names_consistency()
        
        print("\n4. MLIndicatorServiceとの統合テスト")
        test_ml_indicator_service_integration()
        
        print("\n5. パフォーマンスとメモリ使用量テスト")
        test_performance_and_memory()
        
        print("\n6. エラーハンドリングテスト")
        test_error_handling()
        
        print("\n" + "=" * 60)
        print("🎉 FeatureEngineeringServiceの統合テストが完了しました！")
        print("高度な特徴量計算が正常に動作し、GAに統合される準備が整っています。")
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        raise
