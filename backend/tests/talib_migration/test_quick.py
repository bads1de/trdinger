#!/usr/bin/env python3
"""
クイックテスト
"""

import sys
import os
import pandas as pd
import numpy as np

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def quick_test():
    """クイックテスト"""
    print("🧪 クイックテスト開始")
    
    try:
        # TALibAdapterのテスト
        from app.core.services.indicators.talib_adapter import TALibAdapter
        print("✅ TALibAdapter インポート成功")
        
        # 更新された指標のテスト
        from app.core.services.indicators.trend_indicators import SMAIndicator
        print("✅ SMAIndicator インポート成功")
        
        # テストデータ作成
        dates = pd.date_range('2024-01-01', periods=50, freq='D')
        np.random.seed(42)
        
        base_price = 50000
        returns = np.random.normal(0, 0.02, 50)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        test_data = pd.DataFrame({
            'open': close_prices,
            'high': close_prices * 1.01,
            'low': close_prices * 0.99,
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 50)
        }, index=dates)
        
        print(f"📊 テストデータ作成: {len(test_data)}件")
        
        # SMAテスト
        sma_indicator = SMAIndicator()
        sma_result = sma_indicator.calculate(test_data, period=20)
        
        print(f"✅ SMA計算成功: {sma_result.iloc[-1]:.2f}")
        print(f"📊 結果の型: {type(sma_result)}")
        print(f"🏷️ 名前: {sma_result.name}")
        
        print("\n🎉 クイックテスト成功！")
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    quick_test()
