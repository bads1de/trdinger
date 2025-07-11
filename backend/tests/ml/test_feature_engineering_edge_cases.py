"""
FeatureEngineeringServiceのエッジケーステスト

非常に小さいデータセットや欠損値を含むデータに対するFeatureEngineeringServiceの挙動を検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 既存のテストファイルからcreate_comprehensive_test_dataをコピー
def create_comprehensive_test_data(num_records=100, include_nan=False, all_same_value=False):
    """包括的なテストデータを作成

    Args:
        num_records (int): 生成するレコード数。
        include_nan (bool): データにNaNを含めるかどうか。
        all_same_value (bool): 全ての価格データを同じ値にするかどうか。
    """
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)
    price_base = 50000

    if all_same_value:
        prices = np.full(num_records, price_base)
    else:
        returns = np.random.randn(num_records) * 0.02
        prices = [price_base]
        for i in range(1, num_records):
            prices.append(prices[-1] * (1 + returns[i]))
        prices = np.array(prices)

    ohlcv_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(num_records)) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(num_records)) * 0.01),
        'close': prices * (1 + np.random.randn(num_records) * 0.005),
        'volume': np.random.rand(num_records) * 1000000
    })

    if include_nan:
        # ランダムにNaNを挿入
        for col in ['open', 'high', 'low', 'close', 'volume']:
            nan_indices = np.random.choice(ohlcv_data.index, size=int(num_records * 0.1), replace=False)
            ohlcv_data.loc[nan_indices, col] = np.nan

    funding_dates = dates[::8]
    funding_rate_data = pd.DataFrame({
        'timestamp': funding_dates,
        'funding_rate': np.random.randn(len(funding_dates)) * 0.0001
    })
    open_interest_data = pd.DataFrame({
        'timestamp': dates,
        'open_interest_value': np.random.rand(len(dates)) * 1e9 + 5e8
    })
    return ohlcv_data, funding_rate_data, open_interest_data

def test_feature_engineering_with_single_record():
    """単一レコードのOHLCVデータに対する特徴量計算のテスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        ohlcv_data, _, _ = create_comprehensive_test_data(num_records=1)
        
        features_df = service.calculate_advanced_features(ohlcv_data)
        
        assert not features_df.empty, "Features DataFrame should not be empty for single record"
        assert len(features_df) == 1, "Output DataFrame should have 1 row"
        # 単一レコードの場合、多くの指標はNaNになることを許容
        print("✅ Single record data handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"Single record test failed: {e}")

def test_feature_engineering_with_nan_values():
    """欠損値を含むOHLCVデータに対する特徴量計算のテスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data(num_records=100, include_nan=True)
        
        features_df = service.calculate_advanced_features(
            ohlcv_data, 
            funding_rate_data, 
            open_interest_data
        )
        
        assert not features_df.empty, "Features DataFrame should not be empty for NaN data"
        assert len(features_df) == len(ohlcv_data), "Output DataFrame row count mismatch for NaN data"
        # NaNが含まれていてもエラーにならないことを確認
        print("✅ NaN values in input data handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"NaN values test failed: {e}")

def test_feature_engineering_with_all_same_values():
    """全ての値が同じOHLCVデータに対する特徴量計算のテスト"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        ohlcv_data, _, _ = create_comprehensive_test_data(num_records=50, all_same_value=True)
        
        features_df = service.calculate_advanced_features(ohlcv_data)
        
        assert not features_df.empty, "Features DataFrame should not be empty for all same values data"
        assert len(features_df) == len(ohlcv_data), "Output DataFrame row count mismatch for all same values data"
        # 全て同じ値の場合、多くの指標はNaNまたはゼロになることを許容
        print("✅ All same values data handled gracefully.")
        
    except Exception as e:
        pytest.fail(f"All same values test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
