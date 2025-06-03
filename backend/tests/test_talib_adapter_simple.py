"""
TALibAdapterの新しい指標の簡単なテスト

データベース依存を避けて、TALibAdapterの基本機能をテストします。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

# TALibAdapterクラスを直接インポート
from backend.app.core.services.indicators.talib_adapter import TALibAdapter, TALibCalculationError


class TestTALibAdapterNewIndicators:
    """TALibAdapterの新しい指標メソッドのテスト"""

    def setup_method(self):
        """テスト用のサンプルデータを準備"""
        np.random.seed(42)
        self.sample_size = 100
        
        # サンプルOHLCVデータ
        dates = pd.date_range('2023-01-01', periods=self.sample_size, freq='D')
        
        # より現実的な価格データを生成
        base_price = 100
        price_changes = np.random.normal(0, 1, self.sample_size).cumsum()
        close_prices = base_price + price_changes
        
        self.high = pd.Series(close_prices + np.random.uniform(0, 2, self.sample_size), index=dates)
        self.low = pd.Series(close_prices - np.random.uniform(0, 2, self.sample_size), index=dates)
        self.close = pd.Series(close_prices, index=dates)
        self.volume = pd.Series(np.random.uniform(1000, 10000, self.sample_size), index=dates)

    def test_adx_calculation(self):
        """ADX計算のテスト"""
        period = 14
        result = TALibAdapter.adx(self.high, self.low, self.close, period)
        
        # 基本的な検証
        assert isinstance(result, pd.Series)
        assert result.name == f"ADX_{period}"
        assert len(result) == len(self.close)
        
        # ADXは0-100の範囲（NaN値を除く）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(0 <= val <= 100 for val in valid_values), f"ADX値が範囲外: {valid_values.min()}-{valid_values.max()}"
        
        print(f"ADX計算成功: {len(valid_values)}個の有効値")

    def test_aroon_calculation(self):
        """Aroon計算のテスト"""
        period = 14
        result = TALibAdapter.aroon(self.high, self.low, period)
        
        # 基本的な検証
        assert isinstance(result, dict)
        assert "aroon_down" in result
        assert "aroon_up" in result
        
        aroon_down = result["aroon_down"]
        aroon_up = result["aroon_up"]
        
        assert isinstance(aroon_down, pd.Series)
        assert isinstance(aroon_up, pd.Series)
        assert len(aroon_down) == len(self.high)
        assert len(aroon_up) == len(self.high)
        
        # Aroonは0-100の範囲（NaN値を除く）
        valid_down = aroon_down.dropna()
        valid_up = aroon_up.dropna()
        
        if len(valid_down) > 0:
            assert all(0 <= val <= 100 for val in valid_down), f"Aroon Down値が範囲外: {valid_down.min()}-{valid_down.max()}"
        if len(valid_up) > 0:
            assert all(0 <= val <= 100 for val in valid_up), f"Aroon Up値が範囲外: {valid_up.min()}-{valid_up.max()}"
        
        print(f"Aroon計算成功: Down={len(valid_down)}個, Up={len(valid_up)}個の有効値")

    def test_mfi_calculation(self):
        """MFI計算のテスト"""
        period = 14
        result = TALibAdapter.mfi(self.high, self.low, self.close, self.volume, period)
        
        # 基本的な検証
        assert isinstance(result, pd.Series)
        assert result.name == f"MFI_{period}"
        assert len(result) == len(self.close)
        
        # MFIは0-100の範囲（NaN値を除く）
        valid_values = result.dropna()
        if len(valid_values) > 0:
            assert all(0 <= val <= 100 for val in valid_values), f"MFI値が範囲外: {valid_values.min()}-{valid_values.max()}"
        
        print(f"MFI計算成功: {len(valid_values)}個の有効値")

    def test_adx_input_validation(self):
        """ADX入力検証のテスト"""
        # データ長不一致
        short_high = self.high[:50]
        with pytest.raises(TALibCalculationError, match="データ長が一致しません"):
            TALibAdapter.adx(short_high, self.low, self.close, 14)
        
        # 期間が長すぎる
        with pytest.raises(TALibCalculationError, match="データ長.*が期間.*より短いです"):
            TALibAdapter.adx(self.high[:10], self.low[:10], self.close[:10], 14)
        
        print("ADX入力検証テスト成功")

    def test_aroon_input_validation(self):
        """Aroon入力検証のテスト"""
        # データ長不一致
        short_low = self.low[:50]
        with pytest.raises(TALibCalculationError, match="データ長が一致しません"):
            TALibAdapter.aroon(self.high, short_low, 14)
        
        # 期間が長すぎる
        with pytest.raises(TALibCalculationError, match="データ長.*が期間.*より短いです"):
            TALibAdapter.aroon(self.high[:10], self.low[:10], 14)
        
        print("Aroon入力検証テスト成功")

    def test_mfi_input_validation(self):
        """MFI入力検証のテスト"""
        # データ長不一致
        short_volume = self.volume[:50]
        with pytest.raises(TALibCalculationError, match="データ長が一致しません"):
            TALibAdapter.mfi(self.high, self.low, self.close, short_volume, 14)
        
        # 期間が長すぎる
        with pytest.raises(TALibCalculationError, match="データ長.*が期間.*より短いです"):
            TALibAdapter.mfi(self.high[:10], self.low[:10], self.close[:10], self.volume[:10], 14)
        
        print("MFI入力検証テスト成功")

    def test_existing_indicators_still_work(self):
        """既存の指標が正常に動作することを確認"""
        # RSI
        rsi_result = TALibAdapter.rsi(self.close, 14)
        assert isinstance(rsi_result, pd.Series)
        assert len(rsi_result) == len(self.close)
        
        # SMA
        sma_result = TALibAdapter.sma(self.close, 20)
        assert isinstance(sma_result, pd.Series)
        assert len(sma_result) == len(self.close)
        
        # MACD
        macd_result = TALibAdapter.macd(self.close)
        assert isinstance(macd_result, dict)
        assert "macd_line" in macd_result
        assert "signal_line" in macd_result
        assert "histogram" in macd_result
        
        print("既存指標の動作確認成功")

    def test_comprehensive_calculation(self):
        """包括的な計算テスト"""
        print("\n=== 包括的な計算テスト ===")
        
        # 各指標を計算して結果を表示
        adx = TALibAdapter.adx(self.high, self.low, self.close, 14)
        aroon = TALibAdapter.aroon(self.high, self.low, 14)
        mfi = TALibAdapter.mfi(self.high, self.low, self.close, self.volume, 14)
        
        print(f"ADX: 有効値数={len(adx.dropna())}, 最新値={adx.iloc[-1]:.2f if not pd.isna(adx.iloc[-1]) else 'NaN'}")
        print(f"Aroon Up: 有効値数={len(aroon['aroon_up'].dropna())}, 最新値={aroon['aroon_up'].iloc[-1]:.2f if not pd.isna(aroon['aroon_up'].iloc[-1]) else 'NaN'}")
        print(f"Aroon Down: 有効値数={len(aroon['aroon_down'].dropna())}, 最新値={aroon['aroon_down'].iloc[-1]:.2f if not pd.isna(aroon['aroon_down'].iloc[-1]) else 'NaN'}")
        print(f"MFI: 有効値数={len(mfi.dropna())}, 最新値={mfi.iloc[-1]:.2f if not pd.isna(mfi.iloc[-1]) else 'NaN'}")
        
        # 全て正常に計算されていることを確認
        assert len(adx) == self.sample_size
        assert len(aroon['aroon_up']) == self.sample_size
        assert len(aroon['aroon_down']) == self.sample_size
        assert len(mfi) == self.sample_size


if __name__ == "__main__":
    # 個別にテストを実行
    test_instance = TestTALibAdapterNewIndicators()
    test_instance.setup_method()
    
    try:
        test_instance.test_adx_calculation()
        test_instance.test_aroon_calculation()
        test_instance.test_mfi_calculation()
        test_instance.test_adx_input_validation()
        test_instance.test_aroon_input_validation()
        test_instance.test_mfi_input_validation()
        test_instance.test_existing_indicators_still_work()
        test_instance.test_comprehensive_calculation()
        
        print("\n✅ 全てのテストが成功しました！")
        
    except Exception as e:
        print(f"\n❌ テストエラー: {e}")
        import traceback
        traceback.print_exc()
