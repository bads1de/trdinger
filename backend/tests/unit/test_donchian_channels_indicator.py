#!/usr/bin/env python3
"""
Donchian Channels 指標のテスト

TDD方式でDonchian Channelsの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# テスト対象のインポート（実装前なので失敗する予定）
try:
    from app.core.services.indicators.volatility_indicators import (
        DonchianChannelsIndicator,
    )
    from app.core.services.indicators.adapters import VolatilityAdapter
except ImportError:
    # 実装前なので期待されるエラー
    DonchianChannelsIndicator = None
    VolatilityAdapter = None


class TestDonchianChannelsIndicator:
    """Donchian Channels指標のテストクラス"""

    @pytest.fixture
    def sample_data(self):
        """テスト用のサンプルデータ"""
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # より現実的な価格データを生成
        base_price = 50000
        returns = np.random.normal(0, 0.02, 100)
        prices = [base_price]

        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))

        # OHLC データを生成
        highs = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
        lows = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]

        return pd.DataFrame(
            {
                "open": prices,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_donchian_channels_indicator_initialization(self):
        """DonchianChannelsIndicatorの初期化テスト"""
        if DonchianChannelsIndicator is None:
            pytest.skip("DonchianChannelsIndicator not implemented yet")

        indicator = DonchianChannelsIndicator()

        # 基本属性の確認
        assert indicator.indicator_type == "DONCHIAN"
        assert hasattr(indicator, "supported_periods")
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0

        # 期待される期間が含まれているか
        expected_periods = [10, 20, 50]
        for period in expected_periods:
            assert period in indicator.supported_periods

    def test_donchian_channels_calculation_basic(self, sample_data):
        """Donchian Channels基本計算のテスト"""
        if DonchianChannelsIndicator is None:
            pytest.skip("DonchianChannelsIndicator not implemented yet")

        indicator = DonchianChannelsIndicator()
        period = 20

        # Donchian Channels計算実行
        result = indicator.calculate(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, dict)
        assert "upper" in result
        assert "lower" in result
        assert "middle" in result

        # 各チャネルがpandas Seriesであることを確認
        for key in ["upper", "lower", "middle"]:
            assert isinstance(result[key], pd.Series)
            assert len(result[key]) == len(sample_data)

    def test_donchian_channels_calculation_different_periods(self, sample_data):
        """異なる期間でのDonchian Channels計算テスト"""
        if DonchianChannelsIndicator is None:
            pytest.skip("DonchianChannelsIndicator not implemented yet")

        indicator = DonchianChannelsIndicator()
        periods = [10, 20, 50]

        results = {}
        for period in periods:
            results[period] = indicator.calculate(sample_data, period)

        # 各期間の結果が計算されていることを確認
        for period in periods:
            assert "upper" in results[period]
            assert "lower" in results[period]
            assert "middle" in results[period]

            # 有効な値が存在することを確認
            assert len(results[period]["upper"].dropna()) > 0
            assert len(results[period]["lower"].dropna()) > 0
            assert len(results[period]["middle"].dropna()) > 0

        # 期間が長いほど、より多くのNaN値があることを確認（期間が長いほど計算開始が遅い）
        valid_counts = {}
        for period in periods:
            valid_counts[period] = len(results[period]["upper"].dropna())

        # 期間が短いほど有効な値が多いことを確認
        assert valid_counts[10] >= valid_counts[20] >= valid_counts[50]

    def test_donchian_channels_mathematical_properties(self, sample_data):
        """Donchian Channelsの数学的特性のテスト"""
        if DonchianChannelsIndicator is None:
            pytest.skip("DonchianChannelsIndicator not implemented yet")

        indicator = DonchianChannelsIndicator()
        period = 20

        result = indicator.calculate(sample_data, period)

        # 有効なデータのみで検証
        valid_indices = ~(
            pd.isna(result["upper"])
            | pd.isna(result["lower"])
            | pd.isna(result["middle"])
        )

        if valid_indices.sum() > 0:
            upper_values = result["upper"][valid_indices]
            lower_values = result["lower"][valid_indices]
            middle_values = result["middle"][valid_indices]

            # 上限 >= 中央 >= 下限の関係を確認
            assert all(upper_values >= middle_values), "上限が中央より低い値があります"
            assert all(middle_values >= lower_values), "中央が下限より低い値があります"

            # 中央値が上限と下限の平均であることを確認
            expected_middle = (upper_values + lower_values) / 2
            pd.testing.assert_series_equal(
                middle_values, expected_middle, check_names=False
            )

    def test_donchian_channels_adapter_calculation(self, sample_data):
        """VolatilityAdapterのDonchian Channels計算テスト"""
        if VolatilityAdapter is None:
            pytest.skip(
                "VolatilityAdapter donchian_channels method not implemented yet"
            )

        period = 20

        # VolatilityAdapter経由でのDonchian Channels計算
        result = VolatilityAdapter.donchian_channels(
            sample_data["high"], sample_data["low"], period
        )

        # 結果の基本検証
        assert isinstance(result, dict)
        assert "upper" in result
        assert "lower" in result
        assert "middle" in result

    def test_donchian_channels_error_handling(self, sample_data):
        """Donchian Channelsのエラーハンドリングテスト"""
        if DonchianChannelsIndicator is None:
            pytest.skip("DonchianChannelsIndicator not implemented yet")

        from app.core.services.indicators.adapters.base_adapter import (
            TALibCalculationError,
        )

        indicator = DonchianChannelsIndicator()

        # 無効な期間でのテスト
        with pytest.raises((ValueError, TypeError, TALibCalculationError)):
            indicator.calculate(sample_data, -1)

        with pytest.raises((ValueError, TypeError, TALibCalculationError)):
            indicator.calculate(sample_data, 0)

        # データ不足のテスト
        small_data = sample_data.head(5)
        with pytest.raises((ValueError, IndexError, TALibCalculationError)):
            indicator.calculate(small_data, 20)

    def test_donchian_channels_description(self):
        """Donchian Channels説明文のテスト"""
        if DonchianChannelsIndicator is None:
            pytest.skip("DonchianChannelsIndicator not implemented yet")

        indicator = DonchianChannelsIndicator()
        description = indicator.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert (
            "Donchian" in description
            or "ドンチャン" in description
            or "チャネル" in description
        )


class TestDonchianChannelsIntegration:
    """Donchian Channels統合テストクラス"""

    def test_donchian_channels_in_volatility_indicators_info(self):
        """VOLATILITY_INDICATORS_INFOにDonchian Channelsが含まれているかテスト"""
        try:
            from app.core.services.indicators.volatility_indicators import (
                VOLATILITY_INDICATORS_INFO,
            )

            assert "DONCHIAN" in VOLATILITY_INDICATORS_INFO
            donchian_info = VOLATILITY_INDICATORS_INFO["DONCHIAN"]

            assert "periods" in donchian_info
            assert "description" in donchian_info
            assert "category" in donchian_info
            assert donchian_info["category"] == "volatility"

        except ImportError:
            pytest.skip("VOLATILITY_INDICATORS_INFO not available yet")

    def test_donchian_channels_factory_function(self):
        """get_volatility_indicator関数でDonchian Channelsが取得できるかテスト"""
        try:
            from app.core.services.indicators.volatility_indicators import (
                get_volatility_indicator,
            )

            donchian_indicator = get_volatility_indicator("DONCHIAN")
            assert donchian_indicator is not None
            assert donchian_indicator.indicator_type == "DONCHIAN"

        except ImportError:
            pytest.skip("get_volatility_indicator not available yet")

    def test_donchian_channels_in_main_module(self):
        """メインモジュールからDonchian Channelsがインポートできるかテスト"""
        try:
            from app.core.services.indicators import DonchianChannelsIndicator

            indicator = DonchianChannelsIndicator()
            assert indicator.indicator_type == "DONCHIAN"

        except ImportError:
            pytest.skip("DonchianChannelsIndicator not exported from main module yet")


class TestDonchianChannelsAlgorithm:
    """Donchian Channelsアルゴリズムの詳細テスト"""

    @pytest.fixture
    def range_data(self):
        """レンジ相場のテストデータ"""
        dates = pd.date_range("2023-01-01", periods=50, freq="D")

        # レンジ相場のデータ（100-120の範囲で推移）
        base_price = 110
        range_amplitude = 10
        noise = np.random.normal(0, 2, 50)
        prices = (
            base_price + range_amplitude * np.sin(np.linspace(0, 4 * np.pi, 50)) + noise
        )

        # OHLC データを生成
        highs = [max(p * 1.01, p + 1) for p in prices]
        lows = [min(p * 0.99, p - 1) for p in prices]

        return pd.DataFrame({"high": highs, "low": lows, "close": prices}, index=dates)

    def test_donchian_channels_range_detection(self, range_data):
        """Donchian Channelsのレンジ検出能力のテスト"""
        if DonchianChannelsIndicator is None:
            pytest.skip("DonchianChannelsIndicator not implemented yet")

        indicator = DonchianChannelsIndicator()
        period = 20

        result = indicator.calculate(range_data, period)

        # レンジ相場では上限と下限の差が比較的安定しているはず
        valid_indices = ~(pd.isna(result["upper"]) | pd.isna(result["lower"]))

        if valid_indices.sum() > 10:
            upper_values = result["upper"][valid_indices]
            lower_values = result["lower"][valid_indices]
            channel_width = upper_values - lower_values

            # チャネル幅の変動係数が小さいことを確認（レンジ相場の特徴）
            cv = channel_width.std() / channel_width.mean()
            assert cv < 1.0, f"チャネル幅の変動が大きすぎます: CV={cv:.4f}"


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
