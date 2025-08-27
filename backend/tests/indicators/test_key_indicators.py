"""
主要インジケータの動作テスト
"""
import pytest
import pandas as pd
import numpy as np


class TestKeyIndicators:
    """主要インジケータの動作テスト"""

    def test_rsi_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """RSIインジケータの計算テスト"""
        df = sample_ohlcv_data.copy()

        # RSIの計算
        result = technical_indicator_service.calculate_indicator(
            df, "RSI", {"length": 14}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)
        assert 0 <= result[-1] <= 100  # RSIは0-100の範囲

    def test_sma_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """SMAインジケータの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "SMA", {"length": 20}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)
        # 最初の19個はNaNになるはず
        assert np.isnan(result[:19]).all()
        assert not np.isnan(result[19:]).all()

    def test_ema_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """EMAインジケータの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "EMA", {"length": 20}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)

    def test_macd_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """MACDインジケータの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "MACD", {"fast": 12, "slow": 26, "signal": 9}
        )

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3  # MACD, Signal, Histogram

    def test_bb_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """ボリンジャーバンドの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "BB", {"period": 20, "std": 2.0}
        )

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 3  # Upper, Middle, Lower

        upper, middle, lower = result
        assert len(upper) == len(df)
        assert len(middle) == len(df)
        assert len(lower) == len(df)

    def test_stoch_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """ストキャスティクスの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "STOCH", {"fastk_period": 5, "slowk_period": 3, "slowd_period": 3}
        )

        assert result is not None
        assert isinstance(result, tuple)
        assert len(result) == 2  # %K, %D

    def test_adx_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """ADXの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "ADX", {"length": 14}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)

    def test_cci_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """CCIの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "CCI", {"period": 14}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)

    def test_stc_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """STCの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "STC", {"length": 10}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)

    def test_atr_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """ATRの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "ATR", {"period": 14}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)

    def test_mfi_calculation(self, sample_ohlcv_data, technical_indicator_service):
        """MFIの計算テスト"""
        df = sample_ohlcv_data.copy()

        result = technical_indicator_service.calculate_indicator(
            df, "MFI", {"length": 14}
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert len(result) == len(df)
        assert 0 <= result[-1] <= 100  # MFIは0-100の範囲