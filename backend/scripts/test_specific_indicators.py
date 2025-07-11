#!/usr/bin/env python3
"""
特定の指標の実行テスト

エラーが発生している指標を直接テストします。
"""

import sys
import os
import pandas as pd
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app.core.services.indicators import TechnicalIndicatorService


def create_test_data() -> pd.DataFrame:
    """テスト用のOHLCVデータを作成"""
    np.random.seed(42)
    
    n = 100
    base_price = 100
    
    # ランダムウォークで価格データを生成
    returns = np.random.normal(0, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV データを生成
    data = {
        'Open': prices * (1 + np.random.normal(0, 0.001, n)),
        'High': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        'Close': prices,
        'Volume': np.random.randint(1000, 10000, n),
    }
    
    # 価格の論理的整合性を保証
    for i in range(n):
        data['High'][i] = max(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
        data['Low'][i] = min(data['Open'][i], data['High'][i], data['Low'][i], data['Close'][i])
    
    return pd.DataFrame(data)


def test_indicators():
    """問題のある指標をテスト"""
    service = TechnicalIndicatorService()
    test_data = create_test_data()
    
    # エラーが発生している指標をテスト
    problem_indicators = [
        ("MIDPOINT", {"period": 14}),
        ("HT_TRENDLINE", {}),
        ("AVGPRICE", {}),
        ("HT_DCPERIOD", {}),
        ("ACOS", {}),
        ("CDL_DOJI", {}),
    ]
    
    print("=== 問題指標のテスト ===")
    
    for indicator_name, params in problem_indicators:
        try:
            print(f"\n{indicator_name} をテスト中...")
            result = service.calculate_indicator(test_data, indicator_name, params)
            print(f"✅ {indicator_name}: 成功 - 結果タイプ: {type(result)}, 長さ: {len(result) if hasattr(result, '__len__') else 'N/A'}")
        except Exception as e:
            print(f"❌ {indicator_name}: エラー - {str(e)}")


if __name__ == "__main__":
    test_indicators()
