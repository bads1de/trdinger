"""
リファクタリング後のテスト

分割されたアダプターとファサードパターンが正常に動作することを確認します。
"""

import pandas as pd
import numpy as np
import sys
import os

# プロジェクトルートをパスに追加
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

def test_adapters():
    """分割されたアダプターのテスト"""
    print("=== アダプターテスト開始 ===")
    
    # テストデータの作成
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.Series(prices, index=dates, name='close')
    
    try:
        # TrendAdapterのテスト
        from backend.app.core.services.indicators.adapters.trend_adapter import TrendAdapter
        
        sma_result = TrendAdapter.sma(data, 20)
        print(f"✓ TrendAdapter.sma: {len(sma_result)} データポイント")
        
        ema_result = TrendAdapter.ema(data, 20)
        print(f"✓ TrendAdapter.ema: {len(ema_result)} データポイント")
        
        # MomentumAdapterのテスト
        from backend.app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
        
        rsi_result = MomentumAdapter.rsi(data, 14)
        print(f"✓ MomentumAdapter.rsi: {len(rsi_result)} データポイント")
        
        # VolatilityAdapterのテスト
        from backend.app.core.services.indicators.adapters.volatility_adapter import VolatilityAdapter
        
        # 高値・安値データを作成
        high = data + np.random.rand(len(data)) * 2
        low = data - np.random.rand(len(data)) * 2
        
        atr_result = VolatilityAdapter.atr(high, low, data, 14)
        print(f"✓ VolatilityAdapter.atr: {len(atr_result)} データポイント")
        
        # VolumeAdapterのテスト
        from backend.app.core.services.indicators.adapters.volume_adapter import VolumeAdapter
        
        volume = pd.Series(np.random.randint(1000, 10000, len(data)), index=dates)
        obv_result = VolumeAdapter.obv(data, volume)
        print(f"✓ VolumeAdapter.obv: {len(obv_result)} データポイント")
        
        print("=== アダプターテスト完了 ===\n")
        return True
        
    except Exception as e:
        print(f"❌ アダプターテストエラー: {e}")
        return False

def test_facade():
    """ファサードパターンのテスト"""
    print("=== ファサードテスト開始 ===")
    
    # テストデータの作成
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    data = pd.Series(prices, index=dates, name='close')
    
    try:
        # ファサードクラスのテスト
        from backend.app.core.services.indicators.talib_adapter import TALibAdapter
        
        # トレンド系指標
        sma_result = TALibAdapter.sma(data, 20)
        print(f"✓ TALibAdapter.sma: {len(sma_result)} データポイント")
        
        ema_result = TALibAdapter.ema(data, 20)
        print(f"✓ TALibAdapter.ema: {len(ema_result)} データポイント")
        
        # モメンタム系指標
        rsi_result = TALibAdapter.rsi(data, 14)
        print(f"✓ TALibAdapter.rsi: {len(rsi_result)} データポイント")
        
        # ボラティリティ系指標
        high = data + np.random.rand(len(data)) * 2
        low = data - np.random.rand(len(data)) * 2
        
        atr_result = TALibAdapter.atr(high, low, data, 14)
        print(f"✓ TALibAdapter.atr: {len(atr_result)} データポイント")
        
        # ボリューム系指標
        volume = pd.Series(np.random.randint(1000, 10000, len(data)), index=dates)
        obv_result = TALibAdapter.obv(data, volume)
        print(f"✓ TALibAdapter.obv: {len(obv_result)} データポイント")
        
        print("=== ファサードテスト完了 ===\n")
        return True
        
    except Exception as e:
        print(f"❌ ファサードテストエラー: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compatibility():
    """後方互換性のテスト"""
    print("=== 互換性テスト開始 ===")
    
    try:
        # 後方互換性関数のテスト
        from backend.app.core.services.indicators.talib_adapter import safe_talib_calculation
        from backend.app.core.services.indicators.adapters.base_adapter import TALibCalculationError
        
        print("✓ safe_talib_calculation関数のインポート成功")
        print("✓ TALibCalculationErrorクラスのインポート成功")
        
        print("=== 互換性テスト完了 ===\n")
        return True
        
    except Exception as e:
        print(f"❌ 互換性テストエラー: {e}")
        return False

if __name__ == "__main__":
    print("テクニカル指標リファクタリングテスト\n")
    
    # 各テストの実行
    adapter_ok = test_adapters()
    facade_ok = test_facade()
    compatibility_ok = test_compatibility()
    
    # 結果の表示
    print("=== テスト結果 ===")
    print(f"アダプターテスト: {'✓ 成功' if adapter_ok else '❌ 失敗'}")
    print(f"ファサードテスト: {'✓ 成功' if facade_ok else '❌ 失敗'}")
    print(f"互換性テスト: {'✓ 成功' if compatibility_ok else '❌ 失敗'}")
    
    if adapter_ok and facade_ok and compatibility_ok:
        print("\n🎉 全てのテストが成功しました！")
        print("リファクタリングは正常に完了しています。")
    else:
        print("\n⚠️ 一部のテストが失敗しました。")
        print("問題を修正してください。")
