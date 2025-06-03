#!/usr/bin/env python3
"""
更新されたテクニカル指標のテストスクリプト
"""

import sys
import os
import pandas as pd
import numpy as np

# バックエンドのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def test_updated_trend_indicators():
    """更新されたトレンド系指標のテスト"""
    print("🧪 更新されたトレンド系指標テスト開始")
    print("=" * 50)
    
    try:
        from app.core.services.indicators.trend_indicators import (
            SMAIndicator, EMAIndicator, MACDIndicator
        )
        print("✅ 更新された指標クラス インポート成功")
        
        # テストデータ作成
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        close_prices = base_price * np.exp(np.cumsum(returns))
        
        # OHLCV データを生成
        test_data = pd.DataFrame({
            'open': close_prices * (1 + np.random.normal(0, 0.001, 100)),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.01, 100))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.01, 100))),
            'close': close_prices,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        print(f"📊 テストデータ作成: {len(test_data)}件")
        
        # SMAIndicatorテスト
        print("\n1. SMAIndicator テスト")
        try:
            sma_indicator = SMAIndicator()
            sma_result = sma_indicator.calculate(test_data, period=20)
            
            print(f"   ✅ SMA計算成功")
            print(f"   📈 結果の型: {type(sma_result)}")
            print(f"   📊 データ長: {len(sma_result)}")
            print(f"   🏷️ 名前: {sma_result.name}")
            print(f"   📈 最後の値: {sma_result.iloc[-1]:.2f}")
            
            # 基本的な検証
            assert isinstance(sma_result, pd.Series)
            assert len(sma_result) == len(test_data)
            assert sma_result.index.equals(test_data.index)
            print("   ✅ SMA検証完了")
            
        except Exception as e:
            print(f"   ❌ SMAテスト失敗: {e}")
            return False
        
        # EMAIndicatorテスト
        print("\n2. EMAIndicator テスト")
        try:
            ema_indicator = EMAIndicator()
            ema_result = ema_indicator.calculate(test_data, period=20)
            
            print(f"   ✅ EMA計算成功")
            print(f"   📈 結果の型: {type(ema_result)}")
            print(f"   🏷️ 名前: {ema_result.name}")
            print(f"   📈 最後の値: {ema_result.iloc[-1]:.2f}")
            
            assert isinstance(ema_result, pd.Series)
            print("   ✅ EMA検証完了")
            
        except Exception as e:
            print(f"   ❌ EMAテスト失敗: {e}")
            return False
        
        # MACDIndicatorテスト
        print("\n3. MACDIndicator テスト")
        try:
            macd_indicator = MACDIndicator()
            macd_result = macd_indicator.calculate(test_data, period=12)
            
            print(f"   ✅ MACD計算成功")
            print(f"   📈 結果の型: {type(macd_result)}")
            print(f"   🔑 カラム: {list(macd_result.columns)}")
            
            assert isinstance(macd_result, pd.DataFrame)
            assert 'macd_line' in macd_result.columns
            assert 'signal_line' in macd_result.columns
            assert 'histogram' in macd_result.columns
            
            for col in macd_result.columns:
                print(f"   📊 {col}: {macd_result[col].iloc[-1]:.4f}")
            
            print("   ✅ MACD検証完了")
            
        except Exception as e:
            print(f"   ❌ MACDテスト失敗: {e}")
            return False
        
        print("\n🎉 全ての更新された指標テストが成功しました！")
        return True
        
    except ImportError as e:
        print(f"❌ インポートエラー: {e}")
        return False
    except Exception as e:
        print(f"❌ 予期しないエラー: {e}")
        return False

def test_backward_compatibility():
    """後方互換性のテスト"""
    print("\n🔄 後方互換性テスト")
    print("=" * 50)
    
    try:
        from app.core.services.indicators.trend_indicators import (
            SMAIndicator, EMAIndicator, MACDIndicator
        )
        
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
        
        # 既存のAPIが変更されていないことを確認
        print("1. API互換性確認")
        
        # SMA
        sma_indicator = SMAIndicator()
        sma_result = sma_indicator.calculate(test_data, period=20)
        assert isinstance(sma_result, pd.Series)
        print("   ✅ SMA API互換性確認")
        
        # EMA
        ema_indicator = EMAIndicator()
        ema_result = ema_indicator.calculate(test_data, period=20)
        assert isinstance(ema_result, pd.Series)
        print("   ✅ EMA API互換性確認")
        
        # MACD
        macd_indicator = MACDIndicator()
        macd_result = macd_indicator.calculate(test_data, period=12)
        assert isinstance(macd_result, pd.DataFrame)
        assert set(macd_result.columns) == {'macd_line', 'signal_line', 'histogram'}
        print("   ✅ MACD API互換性確認")
        
        print("\n✅ 後方互換性テスト成功")
        return True
        
    except Exception as e:
        print(f"❌ 後方互換性テスト失敗: {e}")
        return False

def main():
    """メインテスト実行"""
    print("🔬 更新されたテクニカル指標 包括的テスト")
    print("=" * 60)
    
    # 基本機能テスト
    basic_success = test_updated_trend_indicators()
    
    # 後方互換性テスト
    if basic_success:
        compat_success = test_backward_compatibility()
    else:
        compat_success = False
    
    # 結果サマリー
    print("\n📋 テスト結果サマリー")
    print("=" * 60)
    print(f"基本機能テスト: {'✅ 成功' if basic_success else '❌ 失敗'}")
    print(f"後方互換性テスト: {'✅ 成功' if compat_success else '❌ 失敗'}")
    
    if basic_success and compat_success:
        print("\n🎉 トレンド系指標のTA-lib移行が成功しました！")
        print("次のステップ: モメンタム系指標の移行")
    else:
        print("\n⚠️ 問題が発見されました。修正が必要です。")

if __name__ == "__main__":
    main()
