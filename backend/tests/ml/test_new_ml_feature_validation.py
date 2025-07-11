"""
新しいML特徴量検証テスト

FeatureEngineeringServiceによって生成される特徴量の形状とデータ型を検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 既存のテストファイルからcreate_comprehensive_test_dataをインポート
# test_feature_engineering_integration.pyからコピー
def create_comprehensive_test_data():
    """包括的なテストデータを作成"""
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='h')
    np.random.seed(42)
    price_base = 50000
    returns = np.random.randn(len(dates)) * 0.02
    prices = [price_base]
    for i in range(1, len(dates)):
        prices.append(prices[-1] * (1 + returns[i]))
    prices = np.array(prices)
    ohlcv_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(len(dates))) * 0.01),
        'low': prices * (1 - np.abs(np.random.randn(len(dates))) * 0.01),
        'close': prices * (1 + np.random.randn(len(dates)) * 0.005),
        'volume': np.random.rand(len(dates)) * 1000000
    })
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

def test_feature_engineering_output_shape_and_types():
    """FeatureEngineeringServiceの出力DataFrameの形状とデータ型を検証"""
    try:
        from app.core.services.feature_engineering import FeatureEngineeringService
        
        service = FeatureEngineeringService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
        
        # 特徴量計算
        features_df = service.calculate_advanced_features(
            ohlcv_data, 
            funding_rate_data, 
            open_interest_data
        )
        
        # 1. 出力DataFrameが空でないことを確認
        assert not features_df.empty, "Features DataFrame should not be empty"
        
        # 2. 行数が入力OHLCVデータと同じであることを確認
        assert len(features_df) == len(ohlcv_data), "Output DataFrame row count mismatch"
        
        # 3. 期待される特徴量の一部が存在し、数値型であることを確認
        expected_numeric_features = [
            'Price_MA_Ratio_Short', 'RSI', 'MACD', 'FR_Change', 'OI_Change_Rate'
        ]
        
        for feature in expected_numeric_features:
            assert feature in features_df.columns, f"Feature '{feature}' not found in output"
            # NaNが含まれる可能性があるので、数値型であることを確認
            assert pd.api.types.is_numeric_dtype(features_df[feature]), f"Feature '{feature}' is not numeric"
            
        print("✅ FeatureEngineeringService output shape and types are valid.")
        
    except Exception as e:
        pytest.fail(f"Feature engineering output validation test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
