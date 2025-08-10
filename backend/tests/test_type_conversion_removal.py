"""
型変換なし実装の包括的テスト

型変換を削除した実装が正常に動作することを確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.technical_indicators.trend import TrendIndicators
from app.services.indicators.technical_indicators.momentum import MomentumIndicators
from app.services.indicators.technical_indicators.volatility import VolatilityIndicators
from app.services.indicators.technical_indicators.math_operators import (
    MathOperatorsIndicators,
)
from app.services.indicators.technical_indicators.statistics import StatisticsIndicators
from app.services.auto_strategy.calculators.indicator_calculator import (
    IndicatorCalculator,
)


class TestTypeConversionRemoval:
    """型変換なし実装のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ（50個のデータポイント）"""
        np.random.seed(42)
        size = 50
        base_price = 100

        # より現実的な価格データを生成
        returns = np.random.normal(0, 0.01, size)
        prices = [base_price]
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 10))

        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.005, size)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.005, size)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(1000, 2000, size)

        return pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            }
        )

    @pytest.fixture
    def mock_backtest_data(self, sample_data):
        """backtesting.py用のモックデータ"""

        class MockData:
            def __init__(self, df):
                self.df = df

        return MockData(sample_data)

    def test_pandas_series_direct_usage(self, sample_data):
        """pandas.Seriesを直接使用するテスト"""
        close_series = sample_data["Close"]

        # SMAテスト
        sma_result = TrendIndicators.sma(close_series, 5)
        assert isinstance(sma_result, np.ndarray)
        assert len(sma_result) == len(close_series)

        # RSIテスト
        rsi_result = MomentumIndicators.rsi(close_series, 5)
        assert isinstance(rsi_result, np.ndarray)
        assert len(rsi_result) == len(close_series)

        # ATRテスト（複数データ）
        atr_result = VolatilityIndicators.atr(
            sample_data["High"], sample_data["Low"], sample_data["Close"], 5
        )
        assert isinstance(atr_result, np.ndarray)
        assert len(atr_result) == len(close_series)

    def test_numpy_array_compatibility(self, sample_data):
        """numpy配列との互換性テスト"""
        close_array = sample_data["Close"].to_numpy()

        # SMAテスト
        sma_result = TrendIndicators.sma(close_array, 5)
        assert isinstance(sma_result, np.ndarray)
        assert len(sma_result) == len(close_array)

        # RSIテスト
        rsi_result = MomentumIndicators.rsi(close_array, 5)
        assert isinstance(rsi_result, np.ndarray)
        assert len(rsi_result) == len(close_array)

    def test_math_operators_with_series(self, sample_data):
        """数学演算子系インジケーターでのSeries対応テスト"""
        data1 = sample_data["Close"]
        data2 = sample_data["Volume"] / 1000

        # ADDテスト
        add_result = MathOperatorsIndicators.add(data1, data2)
        assert isinstance(add_result, np.ndarray)
        assert len(add_result) == len(data1)

        # SUBテスト
        sub_result = MathOperatorsIndicators.sub(data1, data2)
        assert isinstance(sub_result, np.ndarray)
        assert len(sub_result) == len(data1)

        # MULTテスト
        mult_result = MathOperatorsIndicators.mult(data1, data2)
        assert isinstance(mult_result, np.ndarray)
        assert len(mult_result) == len(data1)

    def test_statistics_indicators_with_series(self, sample_data):
        """統計系インジケーターでのSeries対応テスト"""
        high_series = sample_data["High"]
        low_series = sample_data["Low"]

        # BETAテスト
        beta_result = StatisticsIndicators.beta(high_series, low_series, 5)
        assert isinstance(beta_result, np.ndarray)
        assert len(beta_result) == len(high_series)

        # CORRELテスト
        correl_result = StatisticsIndicators.correl(high_series, low_series, 5)
        assert isinstance(correl_result, np.ndarray)
        assert len(correl_result) == len(high_series)

    def test_technical_indicator_service(self, sample_data):
        """TechnicalIndicatorServiceのテスト"""
        service = TechnicalIndicatorService()

        # SMAテスト
        sma_result = service.calculate_indicator(sample_data, "SMA", {"period": 5})
        assert isinstance(sma_result, np.ndarray)
        assert len(sma_result) == len(sample_data)

        # RSIテスト
        rsi_result = service.calculate_indicator(sample_data, "RSI", {"period": 5})
        assert isinstance(rsi_result, np.ndarray)
        assert len(rsi_result) == len(sample_data)

        # MACDテスト（複数出力）
        macd_result = service.calculate_indicator(
            sample_data, "MACD", {"fast": 12, "slow": 26, "signal": 9}
        )
        assert isinstance(macd_result, tuple)
        assert len(macd_result) == 3  # MACD, Signal, Histogram

    def test_auto_strategy_indicator_calculator(self, mock_backtest_data):
        """オートストラテジーのIndicatorCalculatorテスト"""
        calculator = IndicatorCalculator()

        # SMAテスト
        sma_result = calculator.calculate_indicator(
            mock_backtest_data, "SMA", {"period": 5}
        )
        assert isinstance(sma_result, np.ndarray)
        assert len(sma_result) == len(mock_backtest_data.df)

        # RSIテスト
        rsi_result = calculator.calculate_indicator(
            mock_backtest_data, "RSI", {"period": 5}
        )
        assert isinstance(rsi_result, np.ndarray)
        assert len(rsi_result) == len(mock_backtest_data.df)

    def test_parameter_mapping(self, sample_data):
        """パラメータマッピング（period → length）のテスト"""
        service = TechnicalIndicatorService()

        # periodパラメータがlengthに正しくマッピングされることを確認
        sma_result = service.calculate_indicator(sample_data, "SMA", {"period": 10})
        assert isinstance(sma_result, np.ndarray)
        assert len(sma_result) == len(sample_data)

        # 直接lengthパラメータでも動作することを確認
        sma_direct = TrendIndicators.sma(sample_data["Close"], 10)
        assert isinstance(sma_direct, np.ndarray)
        assert len(sma_direct) == len(sample_data)

    def test_mixed_data_types(self, sample_data):
        """混合データ型でのテスト"""
        close_series = sample_data["Close"]
        close_array = close_series.to_numpy()

        # Series + Array
        add_result = MathOperatorsIndicators.add(close_series, close_array)
        assert isinstance(add_result, np.ndarray)
        assert len(add_result) == len(close_series)

        # Array + Series
        sub_result = MathOperatorsIndicators.sub(close_array, close_series)
        assert isinstance(sub_result, np.ndarray)
        assert len(sub_result) == len(close_series)

    def test_error_handling(self, sample_data):
        """エラーハンドリングのテスト"""
        close_series = sample_data["Close"]

        # 期間が長すぎる場合
        with pytest.raises(Exception):
            TrendIndicators.sma(close_series, 20)  # データ長10に対して期間20

        # 無効なパラメータ
        with pytest.raises(Exception):
            TrendIndicators.sma(close_series, 0)  # 期間0

        # 空のデータ
        empty_series = pd.Series([])
        with pytest.raises(Exception):
            TrendIndicators.sma(empty_series, 5)

    def test_performance_comparison(self, sample_data):
        """パフォーマンス比較テスト"""
        import time

        close_series = sample_data["Close"]
        close_array = close_series.to_numpy()

        # pandas.Seriesでの処理時間
        start_time = time.time()
        for _ in range(100):
            TrendIndicators.sma(close_series, 5)
        series_time = time.time() - start_time

        # numpy配列での処理時間
        start_time = time.time()
        for _ in range(100):
            TrendIndicators.sma(close_array, 5)
        array_time = time.time() - start_time

        # 性能差が大きすぎないことを確認（10倍以内）
        assert abs(series_time - array_time) / min(series_time, array_time) < 10
