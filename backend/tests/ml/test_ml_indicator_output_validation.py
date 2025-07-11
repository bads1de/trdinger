"""
MLIndicatorServiceの出力検証テスト

MLIndicatorServiceによって計算される確率値の妥当性を検証します。
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

def test_ml_indicator_probabilities_sum_to_one():
    """MLIndicatorServiceの確率値が合計で1になり、0-1の範囲内であることを検証"""
    try:
        from app.core.services.auto_strategy.services.ml_indicator_service import MLIndicatorService
        
        service = MLIndicatorService()
        ohlcv_data, funding_rate_data, open_interest_data = create_comprehensive_test_data()
        
        ml_indicators = service.calculate_ml_indicators(
            ohlcv_data,
            funding_rate_data,
            open_interest_data
        )
        
        # 期待される確率指標が存在することを確認
        prob_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
        for indicator in prob_indicators:
            assert indicator in ml_indicators, f"Missing ML indicator: {indicator}"
            assert len(ml_indicators[indicator]) == len(ohlcv_data), f"Length mismatch for {indicator}"
            
            # 各確率値が0から1の範囲内であることを確認
            assert np.all(ml_indicators[indicator] >= 0), f"{indicator} contains values less than 0"
            assert np.all(ml_indicators[indicator] <= 1), f"{indicator} contains values greater than 1"

        # 各時点での確率の合計が約1であることを確認
        total_prob = (
            ml_indicators['ML_UP_PROB'] +
            ml_indicators['ML_DOWN_PROB'] +
            ml_indicators['ML_RANGE_PROB']
        )
        
        # 浮動小数点数の比較のため、allcloseを使用
        assert np.allclose(total_prob, 1.0, atol=1e-6), "Sum of probabilities does not approximate 1.0"
        
        print("✅ MLIndicatorService probabilities are valid (sum to 1 and within 0-1 range).")
        
    except Exception as e:
        pytest.fail(f"ML indicator output validation test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
