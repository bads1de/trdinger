"""
Ichimoku Cloudインジケーターのテスト
"""

import numpy as np
import pandas as pd
import pytest

from app.services.indicators.technical_indicators.overlap import OverlapIndicators


class TestIchimokuIndicators:
    """Ichimoku Cloudインジケーターのテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータを生成"""
        # 簡単なテストデータを生成
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # トレンドのあるデータを生成
        base_price = 100
        trend = np.linspace(0, 10, 100)
        noise = np.random.normal(0, 2, 100)
        close_prices = base_price + trend + noise

        # HighとLowはCloseから適当にずらす
        high_prices = close_prices + np.random.normal(1, 0.5, 100)
        low_prices = close_prices - np.random.normal(1, 0.5, 100)

        df = pd.DataFrame(
            {"high": high_prices, "low": low_prices, "close": close_prices}, index=dates
        )

        return df

    def test_ichimoku_basic_calculation(self, sample_data):
        """Ichimoku Cloudの基本計算をテスト"""
        result = OverlapIndicators.ichimoku(
            high=sample_data["high"], low=sample_data["low"], close=sample_data["close"]
        )

        # 結果が辞書形式であることを確認
        assert isinstance(result, dict)

        # 必要なコンポーネントが含まれていることを確認
        expected_components = [
            "tenkan_sen",  # 転換線
            "kijun_sen",  # 基準線
            "senkou_span_a",  # 先行スパンA
            "senkou_span_b",  # 先行スパンB
            "chikou_span",  # 遅行スパン
        ]

        for component in expected_components:
            assert component in result
            assert isinstance(result[component], pd.Series)
            assert len(result[component]) == len(sample_data)

    def test_ichimoku_with_custom_parameters(self, sample_data):
        """カスタムパラメータでのIchimoku Cloud計算をテスト"""
        result = OverlapIndicators.ichimoku(
            high=sample_data["high"],
            low=sample_data["low"],
            close=sample_data["close"],
            tenkan_period=10,
            kijun_period=30,
            senkou_span_b_period=60,
        )

        assert isinstance(result, dict)
        expected_components = [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ]

        for component in expected_components:
            assert component in result
            assert isinstance(result[component], pd.Series)

    def test_ichimoku_empty_data(self):
        """空のデータでのテスト"""
        empty_series = pd.Series([], dtype=float)

        result = OverlapIndicators.ichimoku(
            high=empty_series, low=empty_series, close=empty_series
        )

        assert isinstance(result, dict)
        expected_components = [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ]

        for component in expected_components:
            assert component in result
            assert isinstance(result[component], pd.Series)
            assert len(result[component]) == 0

    def test_ichimoku_insufficient_data(self):
        """データが不足している場合のテスト"""
        # 最小データ長より少ないデータ
        short_data = pd.Series([100, 101, 102, 103, 104])

        result = OverlapIndicators.ichimoku(
            high=short_data, low=short_data - 1, close=short_data - 0.5
        )

        assert isinstance(result, dict)

        # 各コンポーネントがSeriesであることを確認
        for component in [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ]:
            assert component in result
            assert isinstance(result[component], pd.Series)

    def test_ichimoku_data_type_validation(self):
        """データ型の検証をテスト"""
        # 適切なデータ
        high = pd.Series([100, 101, 102])
        low = pd.Series([98, 99, 100])
        close = pd.Series([99, 100, 101])

        # 正常な計算
        result = OverlapIndicators.ichimoku(high=high, low=low, close=close)
        assert isinstance(result, dict)

        # pandas Series以外のデータ型でテスト（エラーになることを確認）
        with pytest.raises(TypeError):
            OverlapIndicators.ichimoku(
                high=[100, 101, 102], low=[98, 99, 100], close=[99, 100, 101]
            )

    def test_ichimoku_series_length_validation(self):
        """系列長の検証をテスト"""
        high = pd.Series([100, 101, 102])
        low = pd.Series([98, 99])  # 長さが異なる
        close = pd.Series([99, 100, 101])

        # 長さが異なる場合はエラーになることを確認
        with pytest.raises(ValueError):
            OverlapIndicators.ichimoku(high=high, low=low, close=close)

    def test_ichimoku_component_properties(self, sample_data):
        """各コンポーネントの特性をテスト"""
        result = OverlapIndicators.ichimoku(
            high=sample_data["high"], low=sample_data["low"], close=sample_data["close"]
        )

        # tenkan_sen (転換線) は短期の平均
        tenkan = result["tenkan_sen"]
        assert isinstance(tenkan, pd.Series)
        assert tenkan.notna().sum() > 0  # 一部はNaNだが、有効な値もある

        # kijun_sen (基準線) は長期の平均
        kijun = result["kijun_sen"]
        assert isinstance(kijun, pd.Series)
        assert kijun.notna().sum() > 0

        # senkou_span_a はtenkanとkijunの平均を前方にずらしたもの
        senkou_a = result["senkou_span_a"]
        assert isinstance(senkou_a, pd.Series)

        # senkou_span_b は長期の平均を前方にずらしたもの
        senkou_b = result["senkou_span_b"]
        assert isinstance(senkou_b, pd.Series)

        # chikou_span はcloseを後方にずらしたもの
        chikou = result["chikou_span"]
        assert isinstance(chikou, pd.Series)

    def test_ichimoku_parameter_ranges(self, sample_data):
        """パラメータ範囲のテスト"""
        high = sample_data["high"]
        low = sample_data["low"]
        close = sample_data["close"]

        # 正常な範囲のパラメータ
        result1 = OverlapIndicators.ichimoku(
            high=high,
            low=low,
            close=close,
            tenkan_period=9,
            kijun_period=26,
            senkou_span_b_period=52,
        )
        assert isinstance(result1, dict)

        # 小さなパラメータ
        result2 = OverlapIndicators.ichimoku(
            high=high,
            low=low,
            close=close,
            tenkan_period=3,
            kijun_period=5,
            senkou_span_b_period=10,
        )
        assert isinstance(result2, dict)

        # 大きなパラメータ
        result3 = OverlapIndicators.ichimoku(
            high=high,
            low=low,
            close=close,
            tenkan_period=50,
            kijun_period=100,
            senkou_span_b_period=200,
        )
        assert isinstance(result3, dict)

    def test_ichimoku_nan_handling(self, sample_data):
        """NaN値の処理をテスト"""
        # NaNを含むデータ
        high_with_nan = sample_data["high"].copy()
        low_with_nan = sample_data["low"].copy()
        close_with_nan = sample_data["close"].copy()

        high_with_nan.iloc[10:15] = np.nan
        low_with_nan.iloc[20:25] = np.nan
        close_with_nan.iloc[30:35] = np.nan

        result = OverlapIndicators.ichimoku(
            high=high_with_nan, low=low_with_nan, close=close_with_nan
        )

        assert isinstance(result, dict)
        for component in [
            "tenkan_sen",
            "kijun_sen",
            "senkou_span_a",
            "senkou_span_b",
            "chikou_span",
        ]:
            assert component in result
            assert isinstance(result[component], pd.Series)
            # NaNが適切に処理されていることを確認
            assert len(result[component]) == len(sample_data)
