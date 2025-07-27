"""
新しいnumpy配列ベースTa-lib指標クラスのテスト

オートストラテジー最適化版の指標クラスの動作確認と
既存実装との結果比較を行います。
"""

import pytest
import numpy as np
import pandas as pd
import talib
from unittest.mock import patch

# 新しい指標クラスをインポート
from backend.app.services.indicators.technical_indicators.trend import TrendIndicators
from backend.app.services.indicators.technical_indicators.momentum import MomentumIndicators
from backend.app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from backend.app.services.indicators.utils import (
    TALibError,
    validate_input,
    ensure_numpy_array,
)


class TestTrendIndicators:
    """トレンド系指標のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        return np.random.random(100) * 100 + 50

    @pytest.fixture
    def sample_ohlc_data(self):
        """OHLC形式のテスト用データ"""
        np.random.seed(42)
        close = np.random.random(100) * 100 + 50
        high = close + np.random.random(100) * 5
        low = close - np.random.random(100) * 5
        return high, low, close

    def test_sma_basic(self, sample_data):
        """SMAの基本動作テスト"""
        period = 20
        result = TrendIndicators.sma(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data)
        assert not np.all(np.isnan(result))

        # Ta-lib直接呼び出しとの比較
        expected = talib.SMA(sample_data, timeperiod=period)
        np.testing.assert_array_almost_equal(result, expected)

    def test_ema_basic(self, sample_data):
        """EMAの基本動作テスト"""
        period = 20
        result = TrendIndicators.ema(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data)

        # Ta-lib直接呼び出しとの比較
        expected = talib.EMA(sample_data, timeperiod=period)
        np.testing.assert_array_almost_equal(result, expected)

    def test_tema_basic(self, sample_data):
        """TEMAの基本動作テスト"""
        period = 20
        result = TrendIndicators.tema(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data)

        # Ta-lib直接呼び出しとの比較
        expected = talib.TEMA(sample_data, timeperiod=period)
        np.testing.assert_array_almost_equal(result, expected)

    def test_sar_basic(self, sample_ohlc_data):
        """SARの基本動作テスト"""
        high, low, close = sample_ohlc_data
        result = TrendIndicators.sar(high, low)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)

        # Ta-lib直接呼び出しとの比較
        expected = talib.SAR(high, low)
        np.testing.assert_array_almost_equal(result, expected)

    def test_invalid_period(self, sample_data):
        """無効な期間でのエラーテスト"""
        with pytest.raises(TALibError):
            TrendIndicators.sma(sample_data, 0)

        with pytest.raises(TALibError):
            TrendIndicators.sma(sample_data, -1)

    def test_insufficient_data(self):
        """データ不足でのエラーテスト"""
        short_data = np.array([1, 2, 3])
        with pytest.raises(TALibError):
            TrendIndicators.sma(short_data, 20)


class TestMomentumIndicators:
    """モメンタム系指標のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        return np.random.random(100) * 100 + 50

    @pytest.fixture
    def sample_ohlc_data(self):
        """OHLC形式のテスト用データ"""
        np.random.seed(42)
        close = np.random.random(100) * 100 + 50
        high = close + np.random.random(100) * 5
        low = close - np.random.random(100) * 5
        return high, low, close

    def test_rsi_basic(self, sample_data):
        """RSIの基本動作テスト"""
        period = 14
        result = MomentumIndicators.rsi(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_data)

        # RSIの値域チェック（0-100）
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

        # Ta-lib直接呼び出しとの比較
        expected = talib.RSI(sample_data, timeperiod=period)
        np.testing.assert_array_almost_equal(result, expected)

    def test_macd_basic(self, sample_data):
        """MACDの基本動作テスト"""
        macd, signal, histogram = MomentumIndicators.macd(sample_data)

        # 結果の基本検証
        assert isinstance(macd, np.ndarray)
        assert isinstance(signal, np.ndarray)
        assert isinstance(histogram, np.ndarray)
        assert len(macd) == len(sample_data)
        assert len(signal) == len(sample_data)
        assert len(histogram) == len(sample_data)

        # Ta-lib直接呼び出しとの比較
        expected_macd, expected_signal, expected_hist = talib.MACD(sample_data)
        np.testing.assert_array_almost_equal(macd, expected_macd)
        np.testing.assert_array_almost_equal(signal, expected_signal)
        np.testing.assert_array_almost_equal(histogram, expected_hist)

    def test_stoch_basic(self, sample_ohlc_data):
        """ストキャスティクスの基本動作テスト"""
        high, low, close = sample_ohlc_data
        slowk, slowd = MomentumIndicators.stoch(high, low, close)

        # 結果の基本検証
        assert isinstance(slowk, np.ndarray)
        assert isinstance(slowd, np.ndarray)
        assert len(slowk) == len(high)
        assert len(slowd) == len(high)

        # Ta-lib直接呼び出しとの比較
        expected_k, expected_d = talib.STOCH(high, low, close)
        np.testing.assert_array_almost_equal(slowk, expected_k)
        np.testing.assert_array_almost_equal(slowd, expected_d)

    def test_williams_r_basic(self, sample_ohlc_data):
        """Williams %Rの基本動作テスト"""
        high, low, close = sample_ohlc_data
        result = MomentumIndicators.williams_r(high, low, close)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)

        # Williams %Rの値域チェック（-100 to 0）
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -100)
        assert np.all(valid_values <= 0)

        # Ta-lib直接呼び出しとの比較
        expected = talib.WILLR(high, low, close)
        np.testing.assert_array_almost_equal(result, expected)


class TestVolatilityIndicators:
    """ボラティリティ系指標のテスト"""

    @pytest.fixture
    def sample_data(self):
        """テスト用サンプルデータ"""
        np.random.seed(42)
        return np.random.random(100) * 100 + 50

    @pytest.fixture
    def sample_ohlc_data(self):
        """OHLC形式のテスト用データ"""
        np.random.seed(42)
        close = np.random.random(100) * 100 + 50
        high = close + np.random.random(100) * 5
        low = close - np.random.random(100) * 5
        return high, low, close

    def test_atr_basic(self, sample_ohlc_data):
        """ATRの基本動作テスト"""
        high, low, close = sample_ohlc_data
        result = VolatilityIndicators.atr(high, low, close)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)

        # ATRは正の値のみ
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)

        # Ta-lib直接呼び出しとの比較
        expected = talib.ATR(high, low, close)
        np.testing.assert_array_almost_equal(result, expected)

    def test_bollinger_bands_basic(self, sample_data):
        """ボリンジャーバンドの基本動作テスト"""
        period = 20
        std_dev = 2.0
        upper, middle, lower = VolatilityIndicators.bollinger_bands(
            sample_data, period=period, std_dev=std_dev
        )

        # 結果の基本検証
        assert isinstance(upper, np.ndarray)
        assert isinstance(middle, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert len(upper) == len(sample_data)
        assert len(middle) == len(sample_data)
        assert len(lower) == len(sample_data)

        # バンドの順序チェック（upper >= middle >= lower）
        valid_indices = ~(np.isnan(upper) | np.isnan(middle) | np.isnan(lower))
        assert np.all(upper[valid_indices] >= middle[valid_indices])
        assert np.all(middle[valid_indices] >= lower[valid_indices])

        # Ta-lib直接呼び出しとの比較（同じパラメータで）
        expected_upper, expected_middle, expected_lower = talib.BBANDS(
            sample_data, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev, matype=0
        )
        np.testing.assert_array_almost_equal(upper, expected_upper)
        np.testing.assert_array_almost_equal(middle, expected_middle)
        np.testing.assert_array_almost_equal(lower, expected_lower)

    def test_adx_basic(self, sample_ohlc_data):
        """ADXの基本動作テスト"""
        high, low, close = sample_ohlc_data
        result = VolatilityIndicators.adx(high, low, close)

        # 結果の基本検証
        assert isinstance(result, np.ndarray)
        assert len(result) == len(high)

        # ADXの値域チェック（0-100）
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= 0)
        assert np.all(valid_values <= 100)

        # Ta-lib直接呼び出しとの比較
        expected = talib.ADX(high, low, close)
        np.testing.assert_array_almost_equal(result, expected)


class TestUtilityFunctions:
    """ユーティリティ関数のテスト"""

    def test_ensure_numpy_array_with_numpy(self):
        """numpy配列の場合のテスト"""
        data = np.array([1, 2, 3, 4, 5])
        result = ensure_numpy_array(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data)

    def test_ensure_numpy_array_with_list(self):
        """リストの場合のテスト"""
        data = [1, 2, 3, 4, 5]
        result = ensure_numpy_array(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.array(data))

    def test_ensure_numpy_array_with_pandas_series(self):
        """pandas Seriesの場合のテスト"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = ensure_numpy_array(data)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, data.values)

    def test_validate_input_valid(self):
        """有効な入力の検証テスト"""
        data = np.array([1, 2, 3, 4, 5])
        period = 3
        # エラーが発生しないことを確認
        validate_input(data, period)

    def test_validate_input_invalid_period(self):
        """無効な期間の検証テスト"""
        data = np.array([1, 2, 3, 4, 5])

        with pytest.raises(TALibError):
            validate_input(data, 0)

        with pytest.raises(TALibError):
            validate_input(data, -1)

    def test_validate_input_insufficient_data(self):
        """データ不足の検証テスト"""
        data = np.array([1, 2])
        period = 5

        with pytest.raises(TALibError):
            validate_input(data, period)

    def test_validate_input_empty_data(self):
        """空データの検証テスト"""
        data = np.array([])
        period = 5

        with pytest.raises(TALibError):
            validate_input(data, period)

    def test_validate_input_none_data(self):
        """Noneデータの検証テスト"""
        data = None
        period = 5

        with pytest.raises(TALibError):
            validate_input(data, period)
