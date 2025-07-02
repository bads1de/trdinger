"""
インジケーター計算ロジック統一のテスト

TDDアプローチで、indicators/ディレクトリ内の各インジケータークラスが
TALibAdapterを経由してTA-Lib関数を呼び出すことを検証するテスト
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from app.core.services.indicators.talib_adapter import TALibAdapter
from app.core.services.indicators.trend import TrendIndicators
from app.core.services.indicators.momentum import MomentumIndicators
from app.core.services.indicators.volatility import VolatilityIndicators


class TestIndicatorCalculationRefactoring:
    """インジケーター計算ロジック統一のテストクラス"""

    @pytest.fixture
    def sample_ohlcv_data(self):
        """テスト用のOHLCVデータを生成"""
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        np.random.seed(42)

        # 価格データを生成（現実的な値）
        close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
        high_prices = close_prices + np.random.uniform(0, 2, 100)
        low_prices = close_prices - np.random.uniform(0, 2, 100)
        open_prices = close_prices + np.random.uniform(-1, 1, 100)
        volume = np.random.randint(1000, 10000, 100)

        return pd.DataFrame(
            {
                "open": open_prices,
                "high": high_prices,
                "low": low_prices,
                "close": close_prices,
                "volume": volume,
            },
            index=dates,
        )

    def test_sma_indicator_uses_talib_adapter(self, sample_ohlcv_data):
        """SMAIndicatorがTALibAdapterを使用することを検証"""
        period = 20

        with patch.object(TALibAdapter, "sma") as mock_sma:
            expected_result = pd.Series(
                [100.0] * len(sample_ohlcv_data),
                index=sample_ohlcv_data.index,
                name="SMA_20",
            )
            mock_sma.return_value = expected_result

            result = TrendIndicators.sma(
                sample_ohlcv_data["close"].values, period=period
            )

            mock_sma.assert_called_once()
            call_args = mock_sma.call_args
            pd.testing.assert_series_equal(
                pd.Series(call_args[0][0]), sample_ohlcv_data["close"]
            )
            assert call_args[0][1] == period
            pd.testing.assert_series_equal(pd.Series(result), expected_result)

    def test_ema_indicator_uses_talib_adapter(self, sample_ohlcv_data):
        """EMAIndicatorがTALibAdapterを使用することを検証"""
        period = 20

        with patch.object(TALibAdapter, "ema") as mock_ema:
            expected_result = pd.Series(
                [100.0] * len(sample_ohlcv_data),
                index=sample_ohlcv_data.index,
                name="EMA_20",
            )
            mock_ema.return_value = expected_result

            result = TrendIndicators.ema(
                sample_ohlcv_data["close"].values, period=period
            )

            mock_ema.assert_called_once()
            call_args = mock_ema.call_args
            pd.testing.assert_series_equal(
                pd.Series(call_args[0][0]), sample_ohlcv_data["close"]
            )
            assert call_args[0][1] == period
            pd.testing.assert_series_equal(pd.Series(result), expected_result)

    def test_rsi_indicator_uses_talib_adapter(self, sample_ohlcv_data):
        """RSIIndicatorがTALibAdapterを使用することを検証"""
        period = 14

        with patch.object(TALibAdapter, "rsi") as mock_rsi:
            expected_result = pd.Series(
                [50.0] * len(sample_ohlcv_data),
                index=sample_ohlcv_data.index,
                name="RSI_14",
            )
            mock_rsi.return_value = expected_result

            result = MomentumIndicators.rsi(
                sample_ohlcv_data["close"].values, period=period
            )

            mock_rsi.assert_called_once()
            call_args = mock_rsi.call_args
            pd.testing.assert_series_equal(
                pd.Series(call_args[0][0]), sample_ohlcv_data["close"]
            )
            assert call_args[0][1] == period
            pd.testing.assert_series_equal(pd.Series(result), expected_result)

    def test_atr_indicator_uses_talib_adapter(self, sample_ohlcv_data):
        """ATRIndicatorがTALibAdapterを使用することを検証"""
        period = 14

        with patch.object(TALibAdapter, "atr") as mock_atr:
            expected_result = pd.Series(
                [1.0] * len(sample_ohlcv_data),
                index=sample_ohlcv_data.index,
                name="ATR_14",
            )
            mock_atr.return_value = expected_result

            result = VolatilityIndicators.atr(
                sample_ohlcv_data["high"].values,
                sample_ohlcv_data["low"].values,
                sample_ohlcv_data["close"].values,
                period=period,
            )

            mock_atr.assert_called_once()
            call_args = mock_atr.call_args
            pd.testing.assert_series_equal(
                pd.Series(call_args[0][0]), sample_ohlcv_data["high"]
            )
            pd.testing.assert_series_equal(
                pd.Series(call_args[0][1]), sample_ohlcv_data["low"]
            )
            pd.testing.assert_series_equal(
                pd.Series(call_args[0][2]), sample_ohlcv_data["close"]
            )
            assert call_args[0][3] == period
            pd.testing.assert_series_equal(pd.Series(result), expected_result)

    def test_bollinger_bands_indicator_uses_talib_adapter(self, sample_ohlcv_data):
        """BollingerBandsIndicatorがTALibAdapterを使用することを検証"""
        period = 20

        with patch.object(TALibAdapter, "bollinger_bands") as mock_bb:
            expected_result = {
                "upper": pd.Series(
                    [105.0] * len(sample_ohlcv_data), index=sample_ohlcv_data.index
                ),
                "middle": pd.Series(
                    [100.0] * len(sample_ohlcv_data), index=sample_ohlcv_data.index
                ),
                "lower": pd.Series(
                    [95.0] * len(sample_ohlcv_data), index=sample_ohlcv_data.index
                ),
            }
            mock_bb.return_value = expected_result

            result = VolatilityIndicators.bollinger_bands(
                sample_ohlcv_data["close"].values, period=period
            )

            mock_bb.assert_called_once()
            call_args = mock_bb.call_args
            pd.testing.assert_series_equal(
                pd.Series(call_args[0][0]), sample_ohlcv_data["close"]
            )
            assert call_args[1]["period"] == period
            assert call_args[1]["std_dev"] == 2.0

            # 結果がDataFrameで正しい列を持つことを確認
            assert isinstance(result, pd.DataFrame)
            assert "upper" in result.columns
            assert "middle" in result.columns
            assert "lower" in result.columns

    def test_no_direct_talib_imports_in_indicators(self):
        """インジケータークラスが直接TA-Libをインポートしていないことを検証"""
        import inspect
        from app.core.services.indicators.trend import TrendIndicators
        from app.core.services.indicators.momentum import MomentumIndicators
        from app.core.services.indicators.volatility import VolatilityIndicators

        # 各モジュールのソースコードを取得
        modules = [TrendIndicators, MomentumIndicators, VolatilityIndicators]

        for module in modules:
            source = inspect.getsource(module)
            # 直接的なTA-Libインポートがないことを確認
            assert (
                "import talib" not in source
            ), f"{module.__name__} contains direct talib import"
            assert (
                "from talib" not in source
            ), f"{module.__name__} contains direct talib import"

    def test_adapters_not_used_in_indicators(self):
        """インジケータークラスが個別アダプター（TrendAdapter等）を使用していないことを検証"""
        import inspect
        from app.core.services.indicators.trend import TrendIndicators
        from app.core.services.indicators.momentum import MomentumIndicators
        from app.core.services.indicators.volatility import VolatilityIndicators

        modules = [TrendIndicators, MomentumIndicators, VolatilityIndicators]

        for module in modules:
            source = inspect.getsource(module)
            # 個別アダプターの使用がないことを確認
            assert (
                "TrendAdapter" not in source
            ), f"{module.__name__} still uses TrendAdapter"
            assert (
                "MomentumAdapter" not in source
            ), f"{module.__name__} still uses MomentumAdapter"
            assert (
                "VolatilityAdapter" not in source
            ), f"{module.__name__} still uses VolatilityAdapter"

    def test_talib_adapter_usage_in_indicators(self):
        """インジケータークラスがTALibAdapterを使用していることを検証"""
        import inspect
        from app.core.services.indicators.trend import TrendIndicators
        from app.core.services.indicators.momentum import MomentumIndicators
        from app.core.services.indicators.volatility import VolatilityIndicators

        modules = [TrendIndicators, MomentumIndicators, VolatilityIndicators]

        for module in modules:
            source = inspect.getsource(module)
            # TALibAdapterの使用があることを確認
            assert (
                "TALibAdapter" in source
            ), f"{module.__name__} does not use TALibAdapter"
