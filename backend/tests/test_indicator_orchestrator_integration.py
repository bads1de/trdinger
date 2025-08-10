"""
IndicatorOrchestratorの統合テスト

型変換なし実装でのIndicatorOrchestratorの動作を確認するテストです。
"""

import pytest
import pandas as pd
import numpy as np

from app.services.indicators.indicator_orchestrator import TechnicalIndicatorService
from app.services.indicators.config.indicator_definitions import (
    initialize_all_indicators,
)


class TestIndicatorOrchestratorIntegration:
    """IndicatorOrchestratorの統合テストクラス"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """テスト前の初期化"""
        initialize_all_indicators()

    @pytest.fixture
    def comprehensive_data(self):
        """包括的なテストデータ"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="1H")

        # より現実的な価格データを生成
        base_price = 100
        price_changes = np.random.normal(0, 0.02, 100)
        prices = [base_price]

        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 1))  # 価格が負にならないように

        # OHLCV データを生成
        closes = np.array(prices)
        highs = closes * (1 + np.abs(np.random.normal(0, 0.01, 100)))
        lows = closes * (1 - np.abs(np.random.normal(0, 0.01, 100)))
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        volumes = np.random.randint(1000, 10000, 100)

        return pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volumes,
            },
            index=dates,
        )

    def test_all_trend_indicators(self, comprehensive_data):
        """全てのトレンド系指標のテスト"""
        service = TechnicalIndicatorService()

        trend_indicators = [
            ("SMA", {"period": 10}),
            ("EMA", {"period": 10}),
            ("WMA", {"period": 10}),
            ("TRIMA", {"period": 10}),
            ("KAMA", {"period": 10}),
            ("MA", {"period": 10}),
            ("MIDPOINT", {"period": 10}),
            ("MIDPRICE", {"period": 10}),
        ]

        for indicator_name, params in trend_indicators:
            try:
                result = service.calculate_indicator(
                    comprehensive_data, indicator_name, params
                )
                assert isinstance(
                    result, np.ndarray
                ), f"{indicator_name} should return numpy array"
                assert len(result) == len(
                    comprehensive_data
                ), f"{indicator_name} length mismatch"
                assert not np.all(
                    np.isnan(result)
                ), f"{indicator_name} returned all NaN"
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {e}")

    def test_all_momentum_indicators(self, comprehensive_data):
        """全てのモメンタム系指標のテスト"""
        service = TechnicalIndicatorService()

        momentum_indicators = [
            ("RSI", {"period": 14}),
            ("CCI", {"period": 14}),
            ("ADX", {"period": 14}),
            ("WILLR", {"period": 14}),
            ("MFI", {"period": 14}),
            ("BOP", {}),
            ("ROC", {"period": 10}),
            ("TRIX", {"period": 14}),
            ("ULTOSC", {"period1": 7, "period2": 14, "period3": 28}),
        ]

        for indicator_name, params in momentum_indicators:
            try:
                result = service.calculate_indicator(
                    comprehensive_data, indicator_name, params
                )
                assert isinstance(
                    result, np.ndarray
                ), f"{indicator_name} should return numpy array"
                assert len(result) == len(
                    comprehensive_data
                ), f"{indicator_name} length mismatch"
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {e}")

    def test_all_volatility_indicators(self, comprehensive_data):
        """全てのボラティリティ系指標のテスト"""
        service = TechnicalIndicatorService()

        volatility_indicators = [
            ("ATR", {"period": 14}),
            ("NATR", {"period": 14}),
            ("TRANGE", {}),
        ]

        for indicator_name, params in volatility_indicators:
            try:
                result = service.calculate_indicator(
                    comprehensive_data, indicator_name, params
                )
                assert isinstance(
                    result, np.ndarray
                ), f"{indicator_name} should return numpy array"
                assert len(result) == len(
                    comprehensive_data
                ), f"{indicator_name} length mismatch"
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {e}")

    def test_complex_output_indicators(self, comprehensive_data):
        """複数出力を持つ指標のテスト"""
        service = TechnicalIndicatorService()

        complex_indicators = [
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}),
            ("STOCH", {"k_period": 14, "d_period": 3}),
            ("BB", {"period": 20, "std": 2}),
            ("AROON", {"period": 14}),
        ]

        for indicator_name, params in complex_indicators:
            try:
                result = service.calculate_indicator(
                    comprehensive_data, indicator_name, params
                )
                assert isinstance(
                    result, tuple
                ), f"{indicator_name} should return tuple"
                assert (
                    len(result) >= 2
                ), f"{indicator_name} should have multiple outputs"

                for i, output in enumerate(result):
                    assert isinstance(
                        output, np.ndarray
                    ), f"{indicator_name} output {i} should be numpy array"
                    assert len(output) == len(
                        comprehensive_data
                    ), f"{indicator_name} output {i} length mismatch"
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {e}")

    def test_math_operators_indicators(self, comprehensive_data):
        """数学演算子系指標のテスト"""
        service = TechnicalIndicatorService()

        math_indicators = [
            (
                "ADD",
                {
                    "data0": comprehensive_data["Close"],
                    "data1": comprehensive_data["Volume"] / 1000,
                },
            ),
            (
                "SUB",
                {
                    "data0": comprehensive_data["High"],
                    "data1": comprehensive_data["Low"],
                },
            ),
            ("MULT", {"data0": comprehensive_data["Close"], "data1": 1.1}),
            (
                "DIV",
                {
                    "data0": comprehensive_data["Close"],
                    "data1": comprehensive_data["Open"],
                },
            ),
            ("MAX", {"period": 10}),
            ("MIN", {"period": 10}),
            ("SUM", {"period": 10}),
        ]

        for indicator_name, params in math_indicators:
            try:
                result = service.calculate_indicator(
                    comprehensive_data, indicator_name, params
                )
                assert isinstance(
                    result, np.ndarray
                ), f"{indicator_name} should return numpy array"
                assert len(result) == len(
                    comprehensive_data
                ), f"{indicator_name} length mismatch"
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {e}")

    def test_statistics_indicators(self, comprehensive_data):
        """統計系指標のテスト"""
        service = TechnicalIndicatorService()

        stats_indicators = [
            ("BETA", {"period": 10}),
            ("CORREL", {"period": 10}),
            ("LINEARREG", {"period": 10}),
            ("LINEARREG_SLOPE", {"period": 10}),
            ("STDDEV", {"period": 10}),
            ("VAR", {"period": 10}),
        ]

        for indicator_name, params in stats_indicators:
            try:
                result = service.calculate_indicator(
                    comprehensive_data, indicator_name, params
                )
                assert isinstance(
                    result, np.ndarray
                ), f"{indicator_name} should return numpy array"
                assert len(result) == len(
                    comprehensive_data
                ), f"{indicator_name} length mismatch"
            except Exception as e:
                pytest.fail(f"{indicator_name} failed: {e}")

    def test_parameter_validation(self, comprehensive_data):
        """パラメータ検証のテスト"""
        service = TechnicalIndicatorService()

        # 正常なパラメータ
        result = service.calculate_indicator(comprehensive_data, "SMA", {"period": 10})
        assert isinstance(result, np.ndarray)

        # 無効なパラメータ（期間が0以下）
        with pytest.raises(Exception):
            service.calculate_indicator(comprehensive_data, "SMA", {"period": 0})

        # 無効なパラメータ（期間がデータ長より長い）
        with pytest.raises(Exception):
            service.calculate_indicator(comprehensive_data, "SMA", {"period": 200})

        # 不足パラメータ
        with pytest.raises(Exception):
            service.calculate_indicator(comprehensive_data, "SMA", {})

    def test_data_type_consistency(self, comprehensive_data):
        """データ型の一貫性テスト"""
        service = TechnicalIndicatorService()

        # pandas.DataFrameでの計算
        df_result = service.calculate_indicator(
            comprehensive_data, "SMA", {"period": 10}
        )

        # numpy配列での計算（直接指標クラスを使用）
        from app.services.indicators.technical_indicators.trend import TrendIndicators

        array_result = TrendIndicators.sma(comprehensive_data["Close"].to_numpy(), 10)

        # 結果が同じであることを確認（NaN部分を除く）
        valid_mask = ~(np.isnan(df_result) | np.isnan(array_result))
        np.testing.assert_array_almost_equal(
            df_result[valid_mask], array_result[valid_mask], decimal=10
        )

    def test_memory_efficiency(self, comprehensive_data):
        """メモリ効率性のテスト"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        service = TechnicalIndicatorService()

        # 大量の計算を実行
        for _ in range(100):
            service.calculate_indicator(comprehensive_data, "SMA", {"period": 10})
            service.calculate_indicator(comprehensive_data, "RSI", {"period": 14})
            service.calculate_indicator(
                comprehensive_data, "MACD", {"fast": 12, "slow": 26, "signal": 9}
            )

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # メモリ増加が100MB以下であることを確認
        assert (
            memory_increase < 100 * 1024 * 1024
        ), f"Memory increase too large: {memory_increase / 1024 / 1024:.2f} MB"

    def test_concurrent_access(self, comprehensive_data):
        """並行アクセスのテスト"""
        import threading
        import time

        service = TechnicalIndicatorService()
        results = []
        errors = []

        def calculate_indicator():
            try:
                result = service.calculate_indicator(
                    comprehensive_data, "SMA", {"period": 10}
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # 複数スレッドで同時実行
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=calculate_indicator)
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # エラーが発生していないことを確認
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 10, "Not all threads completed successfully"

        # 全ての結果が同じであることを確認
        for result in results[1:]:
            np.testing.assert_array_equal(results[0], result)
