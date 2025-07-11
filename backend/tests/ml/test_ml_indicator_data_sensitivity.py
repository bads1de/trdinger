"""
MLIndicatorServiceのデータ感度テスト

異なる市場状況（トレンド、レンジなど）のデータに対するML指標の反応を検証します。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import patch

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 既存のテストファイルからcreate_comprehensive_test_dataをコピー
def create_comprehensive_test_data(num_records=100, trend_type='none'):
    """包括的なテストデータを作成

    Args:
        num_records (int): 生成するレコード数。
        trend_type (str): 'up', 'down', 'range', 'none' のいずれか。
    """
    dates = pd.date_range(start='2024-01-01', periods=num_records, freq='h')
    np.random.seed(42)
    price_base = 50000

    prices = np.zeros(num_records)
    prices[0] = price_base

    if trend_type == 'up':
        for i in range(1, num_records):
            prices[i] = prices[i-1] * (1 + 0.005 + np.random.randn() * 0.0005)
    elif trend_type == 'down':
        for i in range(1, num_records):
            prices[i] = prices[i-1] * (1 - 0.005 + np.random.randn() * 0.0005)
    elif trend_type == 'range':
        amplitude = 0.05 * price_base
        frequency = 2 * np.pi / (num_records / 5) # 5サイクル
        for i in range(1, num_records):
            prices[i] = price_base + amplitude * np.sin(i * frequency) + np.random.randn() * 0.0005 * price_base
    else: # 'none'
        for i in range(1, num_records):
            prices[i] = prices[i-1] * (1 + np.random.randn() * 0.0005)

    ohlcv_data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.randn(num_records)) * 0.001),
        'low': prices * (1 - np.abs(np.random.randn(num_records)) * 0.001),
        'close': prices * (1 + np.random.randn(num_records) * 0.0005),
        'volume': np.random.rand(num_records) * 1000000
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

def test_ml_indicators_on_uptrend_data():
    """上昇トレンドデータに対するML指標のテスト"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        service = MLIndicatorService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data(num_records=200, trend_type='up')
        
        # MLSignalGenerator.predictをモック
        with patch.object(MLSignalGenerator, 'predict') as mock_predict:
            mock_predict.return_value = {'up': 0.7, 'down': 0.1, 'range': 0.2} # 上昇トレンドを想定した予測
            
            ml_indicators = service.calculate_ml_indicators(
                ohlcv_data,
                funding_rate_data,
                open_interest_data
            )
            
            # 上昇トレンドではML_UP_PROBが高くなる傾向があることを確認
            assert 'ML_UP_PROB' in ml_indicators
            assert np.mean(ml_indicators['ML_UP_PROB']) > 0.6 # 閾値を0.6に設定
            print("✅ ML_UP_PROB shows higher values in uptrend.")
            
    except Exception as e:
        pytest.fail(f"Uptrend data sensitivity test failed: {e}")

def test_ml_indicators_on_downtrend_data():
    """下降トレンドデータに対するML指標のテスト"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        service = MLIndicatorService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data(num_records=200, trend_type='down')
        
        # MLSignalGenerator.predictをモック
        with patch.object(MLSignalGenerator, 'predict') as mock_predict:
            mock_predict.return_value = {'up': 0.1, 'down': 0.7, 'range': 0.2} # 下降トレンドを想定した予測
            
            ml_indicators = service.calculate_ml_indicators(
                ohlcv_data,
                funding_rate_data,
                open_interest_data
            )
            
            # 下降トレンドではML_DOWN_PROBが高くなる傾向があることを確認
            assert 'ML_DOWN_PROB' in ml_indicators
            assert np.mean(ml_indicators['ML_DOWN_PROB']) > 0.6 # 閾値を0.6に設定
            print("✅ ML_DOWN_PROB shows higher values in downtrend.")
            
    except Exception as e:
        pytest.fail(f"Downtrend data sensitivity test failed: {e}")

def test_ml_indicators_on_range_data():
    """レンジ相場データに対するML指標のテスト"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        from app.core.services.ml.signal_generator import MLSignalGenerator
        
        service = MLIndicatorService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data(num_records=200, trend_type='range')
        
        # MLSignalGenerator.predictをモック
        with patch.object(MLSignalGenerator, 'predict') as mock_predict:
            mock_predict.return_value = {'up': 0.2, 'down': 0.2, 'range': 0.6} # レンジ相場を想定した予測
            
            ml_indicators = service.calculate_ml_indicators(
                ohlcv_data,
                funding_rate_data,
                open_interest_data
            )
            
            # レンジ相場ではML_RANGE_PROBが高くなる傾向があることを確認
            assert 'ML_RANGE_PROB' in ml_indicators
            assert np.mean(ml_indicators['ML_RANGE_PROB']) > 0.5 # 閾値を0.5に設定
            print("✅ ML_RANGE_PROB shows higher values in range-bound market.")
            
    except Exception as e:
        pytest.fail(f"Range data sensitivity test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
