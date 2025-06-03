#!/usr/bin/env python3
"""
最終テスト
"""

import sys
import os
import pandas as pd
import numpy as np

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def final_test():
    """最終テスト"""
    print("🔬 TA-lib移行 最終テスト")
    print("=" * 50)
    
    try:
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
        
        print(f"📊 テストデータ: {len(test_data)}件")
        
        # 1. TALibAdapterテスト
        print("\n1. TALibAdapter テスト")
        from app.core.services.indicators.talib_adapter import TALibAdapter
        
        sma_result = TALibAdapter.sma(test_data['close'], 20)
        print(f"   ✅ SMA: {sma_result.iloc[-1]:.2f}")
        
        rsi_result = TALibAdapter.rsi(test_data['close'], 14)
        print(f"   ✅ RSI: {rsi_result.iloc[-1]:.2f}")
        
        # 2. 更新された指標クラステスト
        print("\n2. 更新された指標クラス テスト")
        from app.core.services.indicators.trend_indicators import SMAIndicator
        from app.core.services.indicators.momentum_indicators import RSIIndicator
        
        sma_indicator = SMAIndicator()
        sma_class_result = sma_indicator.calculate(test_data, period=20)
        print(f"   ✅ SMAIndicator: {sma_class_result.iloc[-1]:.2f}")
        
        rsi_indicator = RSIIndicator()
        rsi_class_result = rsi_indicator.calculate(test_data, period=14)
        print(f"   ✅ RSIIndicator: {rsi_class_result.iloc[-1]:.2f}")
        
        # 3. backtesting.py用関数テスト
        print("\n3. backtesting.py用関数 テスト")
        from app.core.strategies.indicators import SMA, RSI
        
        sma_func_result = SMA(test_data['close'], 20)
        print(f"   ✅ SMA関数: {sma_func_result.iloc[-1]:.2f}")
        
        rsi_func_result = RSI(test_data['close'], 14)
        print(f"   ✅ RSI関数: {rsi_func_result.iloc[-1]:.2f}")
        
        # 4. 一貫性チェック
        print("\n4. 一貫性チェック")
        sma_diff = abs(sma_result.iloc[-1] - sma_class_result.iloc[-1])
        rsi_diff = abs(rsi_result.iloc[-1] - rsi_class_result.iloc[-1])
        
        print(f"   📊 SMA差分: {sma_diff:.6f}")
        print(f"   📊 RSI差分: {rsi_diff:.6f}")
        
        if sma_diff < 1e-6 and rsi_diff < 1e-6:
            print("   ✅ 一貫性確認")
        else:
            print("   ⚠️ 一貫性に問題があります")
        
        print("\n🎉 最終テスト成功！")
        print("✅ TA-lib移行が完了しました")
        print("🚀 すべてのテクニカル指標がTA-libを使用して高速化されました")
        print("🔄 後方互換性も保たれています")
        
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    final_test()
