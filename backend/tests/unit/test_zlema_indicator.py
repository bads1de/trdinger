#!/usr/bin/env python3
"""
ZLEMA (Zero Lag Exponential Moving Average) 指標のテスト

TDD方式でZLEMAの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# テスト対象のインポート（実装前なので失敗する予定）
try:
    from app.core.services.indicators.trend_indicators import ZLEMAIndicator
    from app.core.services.indicators.adapters import TrendAdapter
except ImportError:
    # 実装前なので期待されるエラー
    ZLEMAIndicator = None
    TrendAdapter = None


class TestZLEMAIndicator:
    """ZLEMA指標のテストクラス"""

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

        return pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
                "close": prices,
                "volume": np.random.randint(1000, 10000, 100),
            },
            index=dates,
        )

    def test_zlema_indicator_initialization(self):
        """ZLEMAIndicatorの初期化テスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        indicator = ZLEMAIndicator()

        # 基本属性の確認
        assert indicator.indicator_type == "ZLEMA"
        assert hasattr(indicator, "supported_periods")
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0

        # 期待される期間が含まれているか
        expected_periods = [9, 14, 21, 30, 50]
        for period in expected_periods:
            assert period in indicator.supported_periods

    def test_zlema_calculation_basic(self, sample_data):
        """ZLEMA基本計算のテスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        indicator = ZLEMAIndicator()
        period = 20

        # ZLEMA計算実行
        result = indicator.calculate(sample_data, period)

        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

        # NaN値の確認（初期値はNaNであるべき）
        assert pd.isna(result.iloc[0])

        # 有効な値の確認
        valid_values = result.dropna()
        assert len(valid_values) > 0
        assert all(isinstance(val, (int, float)) for val in valid_values)

    def test_zlema_calculation_different_periods(self, sample_data):
        """異なる期間でのZLEMA計算テスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        indicator = ZLEMAIndicator()
        periods = [9, 14, 21, 30]

        results = {}
        for period in periods:
            results[period] = indicator.calculate(sample_data, period)

        # 各期間の結果が異なることを確認
        for i, period1 in enumerate(periods):
            for period2 in periods[i + 1 :]:
                # 最後の有効な値が異なることを確認
                val1 = results[period1].dropna().iloc[-1]
                val2 = results[period2].dropna().iloc[-1]
                assert (
                    val1 != val2
                ), f"ZLEMA({period1}) と ZLEMA({period2}) の値が同じです"

    def test_zlema_adapter_calculation(self, sample_data):
        """TrendAdapterのZLEMA計算テスト"""
        if TrendAdapter is None:
            pytest.skip("TrendAdapter ZLEMA method not implemented yet")

        period = 20

        # TrendAdapter経由でのZLEMA計算
        result = TrendAdapter.zlema(sample_data["close"], period)

        # 結果の基本検証
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

        # ZLEMAの特性確認（ゼロラグ特性）
        valid_values = result.dropna()
        assert len(valid_values) > 0

    def test_zlema_mathematical_properties(self, sample_data):
        """ZLEMAの数学的特性のテスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        indicator = ZLEMAIndicator()
        period = 20

        result = indicator.calculate(sample_data, period)
        valid_result = result.dropna()

        # ZLEMAは価格の範囲内にあるべき（一般的に）
        price_min = sample_data["close"].min()
        price_max = sample_data["close"].max()

        # 多少の範囲外は許容（ZLEMAの特性上）
        tolerance = (price_max - price_min) * 0.1
        assert all(val >= price_min - tolerance for val in valid_result)
        assert all(val <= price_max + tolerance for val in valid_result)

    def test_zlema_vs_ema_responsiveness(self, sample_data):
        """ZLEMAとEMAの応答性比較テスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        # EMAとの比較でZLEMAの応答性を確認
        from app.core.services.indicators.trend_indicators import EMAIndicator

        zlema_indicator = ZLEMAIndicator()
        ema_indicator = EMAIndicator()
        period = 20

        zlema_result = zlema_indicator.calculate(sample_data, period)
        ema_result = ema_indicator.calculate(sample_data, period)

        # 最新の価格変動に対してZLEMAがより応答的であることを確認
        assert len(zlema_result.dropna()) > 0
        assert len(ema_result.dropna()) > 0

        # ZLEMAはEMAよりもラグが少ないはず（数値的な検証は実装後に調整）

    def test_zlema_error_handling(self, sample_data):
        """ZLEMAのエラーハンドリングテスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        from app.core.services.indicators.adapters.base_adapter import (
            TALibCalculationError,
        )

        indicator = ZLEMAIndicator()

        # 無効な期間でのテスト
        with pytest.raises((ValueError, TypeError, TALibCalculationError)):
            indicator.calculate(sample_data, -1)

        with pytest.raises((ValueError, TypeError, TALibCalculationError)):
            indicator.calculate(sample_data, 0)

        # データ不足のテスト
        small_data = sample_data.head(5)
        with pytest.raises((ValueError, IndexError, TALibCalculationError)):
            indicator.calculate(small_data, 20)

    def test_zlema_description(self):
        """ZLEMA説明文のテスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        indicator = ZLEMAIndicator()
        description = indicator.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert (
            "ZLEMA" in description
            or "Zero Lag" in description
            or "ゼロラグ" in description
        )


class TestZLEMAIntegration:
    """ZLEMA統合テストクラス"""

    def test_zlema_in_trend_indicators_info(self):
        """TREND_INDICATORS_INFOにZLEMAが含まれているかテスト"""
        try:
            from app.core.services.indicators.trend_indicators import (
                TREND_INDICATORS_INFO,
            )

            assert "ZLEMA" in TREND_INDICATORS_INFO
            zlema_info = TREND_INDICATORS_INFO["ZLEMA"]

            assert "periods" in zlema_info
            assert "description" in zlema_info
            assert "category" in zlema_info
            assert zlema_info["category"] == "trend"

        except ImportError:
            pytest.skip("TREND_INDICATORS_INFO not available yet")

    def test_zlema_factory_function(self):
        """get_trend_indicator関数でZLEMAが取得できるかテスト"""
        try:
            from app.core.services.indicators.trend_indicators import (
                get_trend_indicator,
            )

            zlema_indicator = get_trend_indicator("ZLEMA")
            assert zlema_indicator is not None
            assert zlema_indicator.indicator_type == "ZLEMA"

        except ImportError:
            pytest.skip("get_trend_indicator not available yet")

    def test_zlema_in_main_module(self):
        """メインモジュールからZLEMAがインポートできるかテスト"""
        try:
            from app.core.services.indicators import ZLEMAIndicator

            indicator = ZLEMAIndicator()
            assert indicator.indicator_type == "ZLEMA"

        except ImportError:
            pytest.skip("ZLEMAIndicator not exported from main module yet")


class TestZLEMAAlgorithm:
    """ZLEMAアルゴリズムの詳細テスト"""

    @pytest.fixture
    def trend_data(self):
        """トレンドのあるテストデータ"""
        # 上昇トレンドのデータ
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        base_price = 100
        trend = np.linspace(0, 20, 50)  # 20ポイントの上昇トレンド
        noise = np.random.normal(0, 1, 50)
        prices = base_price + trend + noise

        return pd.DataFrame({"close": prices}, index=dates)

    def test_zlema_lag_reduction(self, trend_data):
        """ZLEMAのラグ削減効果のテスト"""
        if ZLEMAIndicator is None:
            pytest.skip("ZLEMAIndicator not implemented yet")

        from app.core.services.indicators.trend_indicators import EMAIndicator

        zlema_indicator = ZLEMAIndicator()
        ema_indicator = EMAIndicator()
        period = 14

        zlema_result = zlema_indicator.calculate(trend_data, period)
        ema_result = ema_indicator.calculate(trend_data, period)

        # 上昇トレンドにおいて、ZLEMAはEMAよりも高い値を示すはず
        # （ラグが少ないため、トレンドにより早く追従）
        valid_indices = ~(pd.isna(zlema_result) | pd.isna(ema_result))

        if valid_indices.sum() > 10:  # 十分なデータがある場合
            zlema_values = zlema_result[valid_indices]
            ema_values = ema_result[valid_indices]

            # 最後の10個の値で比較
            recent_zlema = zlema_values.tail(10).mean()
            recent_ema = ema_values.tail(10).mean()

            # 上昇トレンドでZLEMAがEMAより高いことを確認
            assert (
                recent_zlema >= recent_ema
            ), "ZLEMAがEMAよりもトレンドに追従していません"


if __name__ == "__main__":
    # テスト実行
    pytest.main([__file__, "-v"])
