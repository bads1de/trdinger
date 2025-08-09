"""
pandas-taライブラリの基本使用方法確認とTA-lib移行テスト

このファイルでは以下をテストします：
1. pandas-taの基本的な使用方法
2. DataFrameとnumpy配列の相互変換
3. TA-libとpandas-taの結果比較
4. backtesting.pyとの互換性確認
"""

import numpy as np
import pandas as pd
import pytest
import talib
import pandas_ta as ta
from typing import Tuple, Optional


class TestPandasTAMigration:
    """pandas-ta移行テストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self) -> pd.DataFrame:
        """テスト用のOHLCVデータを生成"""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # 現実的な価格データを生成
        base_price = 100.0
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # OHLC生成
        close = np.array(prices)
        high = close * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        low = close * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        open_price = np.roll(close, 1)
        open_price[0] = close[0]
        volume = np.random.randint(1000, 10000, 100)
        
        return pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)

    def test_pandas_ta_basic_usage(self, sample_ohlcv_data: pd.DataFrame):
        """pandas-taの基本的な使用方法をテスト"""
        df = sample_ohlcv_data.copy()
        
        # 基本的な指標の計算
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=20, append=True)
        df.ta.rsi(length=14, append=True)
        
        # 結果の確認
        assert 'SMA_20' in df.columns
        assert 'EMA_20' in df.columns
        assert 'RSI_14' in df.columns
        
        # NaN値の確認（初期値は期待される）
        assert not df['SMA_20'].iloc[-10:].isna().any()
        assert not df['EMA_20'].iloc[-10:].isna().any()
        assert not df['RSI_14'].iloc[-10:].isna().any()

    def test_talib_vs_pandas_ta_comparison(self, sample_ohlcv_data: pd.DataFrame):
        """TA-libとpandas-taの結果比較"""
        df = sample_ohlcv_data.copy()
        close = df['Close'].values.astype(np.float64)
        
        # TA-libでSMA計算
        talib_sma = talib.SMA(close, timeperiod=20)
        
        # pandas-taでSMA計算
        pandas_ta_sma = ta.sma(df['Close'], length=20)
        
        # 結果比較（NaN以外の部分）
        valid_mask = ~(np.isnan(talib_sma) | np.isnan(pandas_ta_sma))
        np.testing.assert_allclose(
            talib_sma[valid_mask], 
            pandas_ta_sma[valid_mask], 
            rtol=1e-10,
            err_msg="TA-libとpandas-taのSMA結果が一致しません"
        )

    def test_dataframe_numpy_conversion(self, sample_ohlcv_data: pd.DataFrame):
        """DataFrameとnumpy配列の相互変換テスト"""
        df = sample_ohlcv_data.copy()
        
        # pandas-taで計算
        sma_series = ta.sma(df['Close'], length=20)
        
        # numpy配列に変換
        sma_array = sma_series.values
        assert isinstance(sma_array, np.ndarray)
        
        # pandas Seriesに戻す
        sma_series_restored = pd.Series(sma_array, index=df.index)
        
        # 元のSeriesと比較（NaN処理を考慮）
        pd.testing.assert_series_equal(
            sma_series.fillna(0), 
            sma_series_restored.fillna(0),
            check_names=False
        )

    def test_multiple_indicators_calculation(self, sample_ohlcv_data: pd.DataFrame):
        """複数指標の同時計算テスト"""
        df = sample_ohlcv_data.copy()
        
        # 複数指標を一度に計算
        df.ta.sma(length=10, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.ema(length=12, append=True)
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        
        # 期待される列が存在することを確認
        expected_columns = ['SMA_10', 'SMA_20', 'EMA_12', 'RSI_14', 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9']
        for col in expected_columns:
            assert col in df.columns, f"列 {col} が見つかりません"

    def test_bollinger_bands_comparison(self, sample_ohlcv_data: pd.DataFrame):
        """Bollinger Bandsの比較テスト"""
        df = sample_ohlcv_data.copy()
        close = df['Close'].values.astype(np.float64)
        
        # TA-libでBollinger Bands計算
        talib_upper, talib_middle, talib_lower = talib.BBANDS(
            close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
        )
        
        # pandas-taでBollinger Bands計算
        bb_result = ta.bbands(df['Close'], length=20, std=2)
        
        # 結果比較（NaN以外の部分）
        valid_mask = ~np.isnan(talib_upper)
        
        np.testing.assert_allclose(
            talib_upper[valid_mask], 
            bb_result['BBU_20_2.0'][valid_mask], 
            rtol=1e-10,
            err_msg="Bollinger Bands上限が一致しません"
        )
        
        np.testing.assert_allclose(
            talib_middle[valid_mask], 
            bb_result['BBM_20_2.0'][valid_mask], 
            rtol=1e-10,
            err_msg="Bollinger Bands中央線が一致しません"
        )
        
        np.testing.assert_allclose(
            talib_lower[valid_mask], 
            bb_result['BBL_20_2.0'][valid_mask], 
            rtol=1e-10,
            err_msg="Bollinger Bands下限が一致しません"
        )

    def test_rsi_comparison(self, sample_ohlcv_data: pd.DataFrame):
        """RSIの比較テスト"""
        df = sample_ohlcv_data.copy()
        close = df['Close'].values.astype(np.float64)
        
        # TA-libでRSI計算
        talib_rsi = talib.RSI(close, timeperiod=14)
        
        # pandas-taでRSI計算
        pandas_ta_rsi = ta.rsi(df['Close'], length=14)
        
        # 結果比較（NaN以外の部分）
        valid_mask = ~(np.isnan(talib_rsi) | np.isnan(pandas_ta_rsi))
        
        np.testing.assert_allclose(
            talib_rsi[valid_mask], 
            pandas_ta_rsi[valid_mask], 
            rtol=1e-10,
            err_msg="RSI結果が一致しません"
        )

    def test_macd_comparison(self, sample_ohlcv_data: pd.DataFrame):
        """MACDの比較テスト"""
        df = sample_ohlcv_data.copy()
        close = df['Close'].values.astype(np.float64)
        
        # TA-libでMACD計算
        talib_macd, talib_signal, talib_hist = talib.MACD(
            close, fastperiod=12, slowperiod=26, signalperiod=9
        )
        
        # pandas-taでMACD計算
        macd_result = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        
        # 結果比較（NaN以外の部分）
        valid_mask = ~np.isnan(talib_macd)
        
        np.testing.assert_allclose(
            talib_macd[valid_mask], 
            macd_result['MACD_12_26_9'][valid_mask], 
            rtol=1e-10,
            err_msg="MACD線が一致しません"
        )
        
        np.testing.assert_allclose(
            talib_signal[valid_mask], 
            macd_result['MACDs_12_26_9'][valid_mask], 
            rtol=1e-10,
            err_msg="MACDシグナル線が一致しません"
        )
        
        np.testing.assert_allclose(
            talib_hist[valid_mask], 
            macd_result['MACDh_12_26_9'][valid_mask], 
            rtol=1e-10,
            err_msg="MACDヒストグラムが一致しません"
        )

    def test_backtesting_compatibility(self, sample_ohlcv_data: pd.DataFrame):
        """backtesting.py互換性テスト"""
        df = sample_ohlcv_data.copy()
        
        # pandas-taで計算した結果をnumpy配列として取得
        sma_20 = ta.sma(df['Close'], length=20).values
        ema_12 = ta.ema(df['Close'], length=12).values
        rsi_14 = ta.rsi(df['Close'], length=14).values
        
        # numpy配列として正しく取得できることを確認
        assert isinstance(sma_20, np.ndarray)
        assert isinstance(ema_12, np.ndarray)
        assert isinstance(rsi_14, np.ndarray)
        
        # 長さが元データと一致することを確認
        assert len(sma_20) == len(df)
        assert len(ema_12) == len(df)
        assert len(rsi_14) == len(df)
        
        # 有効な値が存在することを確認
        assert not np.isnan(sma_20[-10:]).all()
        assert not np.isnan(ema_12[-10:]).all()
        assert not np.isnan(rsi_14[-10:]).all()


if __name__ == "__main__":
    # 基本的なテストを実行
    pytest.main([__file__, "-v"])
