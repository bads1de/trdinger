"""
Stochastic RSI 指標のテスト

TDD方式でStochasticRSIIndicatorクラスの実装をテストします。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

# テスト対象のインポート（まだ実装されていないのでImportErrorが発生する予定）
try:
    from app.core.services.indicators.momentum_indicators import StochasticRSIIndicator
    from app.core.services.indicators.adapters.momentum_adapter import MomentumAdapter
except ImportError:
    # まだ実装されていない場合はNoneを設定
    StochasticRSIIndicator = None
    MomentumAdapter = None


class TestStochasticRSIIndicator:
    """StochasticRSIIndicatorクラスのテスト"""

    def setup_method(self):
        """各テストメソッドの前に実行される初期化"""
        # テストデータの作成
        self.dates = pd.date_range("2023-01-01", periods=100, freq="D")

        # より現実的な価格データを生成
        base_price = 100
        price_trend = np.linspace(0, 10, 100)
        price_noise = np.random.normal(0, 1, 100)
        close_prices = base_price + price_trend + price_noise

        self.test_data = pd.DataFrame(
            {
                "open": close_prices + np.random.uniform(-0.5, 0.5, 100),
                "high": close_prices + np.random.uniform(0.5, 1.5, 100),
                "low": close_prices - np.random.uniform(0.5, 1.5, 100),
                "close": close_prices,
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=self.dates,
        )

    def test_stochastic_rsi_indicator_import(self):
        """StochasticRSIIndicatorクラスがインポートできることをテスト"""
        # Red: まだ実装されていないのでNoneになっているはず
        assert (
            StochasticRSIIndicator is not None
        ), "StochasticRSIIndicatorクラスが実装されていません"

    def test_stochastic_rsi_indicator_initialization(self):
        """StochasticRSIIndicatorの初期化テスト"""
        if StochasticRSIIndicator is None:
            pytest.skip("StochasticRSIIndicatorが実装されていません")

        indicator = StochasticRSIIndicator()

        # 基本属性の確認
        assert indicator.indicator_type == "STOCHRSI"
        assert isinstance(indicator.supported_periods, list)
        assert len(indicator.supported_periods) > 0

        # 期待される期間が含まれているか
        expected_periods = [14, 21]
        for period in expected_periods:
            assert period in indicator.supported_periods

    def test_stochastic_rsi_calculation_basic(self):
        """Stochastic RSI計算の基本テスト"""
        if StochasticRSIIndicator is None:
            pytest.skip("StochasticRSIIndicatorが実装されていません")

        indicator = StochasticRSIIndicator()
        period = 14

        # モックを使用してMomentumAdapter.stochastic_rsiをテスト
        with patch.object(MomentumAdapter, "stochastic_rsi") as mock_stoch_rsi:
            # モックの戻り値を設定
            expected_result = pd.DataFrame(
                {
                    "fastk": np.random.uniform(0, 100, 100),
                    "fastd": np.random.uniform(0, 100, 100),
                },
                index=self.test_data.index,
            )
            mock_stoch_rsi.return_value = expected_result

            # Stochastic RSI計算を実行
            result = indicator.calculate(self.test_data, period)

            # 結果の検証
            assert isinstance(result, pd.DataFrame)
            assert len(result) == len(self.test_data)
            assert "fastk" in result.columns
            assert "fastd" in result.columns

            # MomentumAdapter.stochastic_rsiが正しい引数で呼ばれたか確認
            mock_stoch_rsi.assert_called_once_with(
                self.test_data["close"], period, 3, 3
            )

    def test_stochastic_rsi_calculation_different_periods(self):
        """異なる期間でのStochastic RSI計算テスト"""
        if StochasticRSIIndicator is None:
            pytest.skip("StochasticRSIIndicatorが実装されていません")

        indicator = StochasticRSIIndicator()

        for period in [14, 21]:
            with patch.object(MomentumAdapter, "stochastic_rsi") as mock_stoch_rsi:
                expected_result = pd.DataFrame(
                    {
                        "fastk": np.random.uniform(0, 100, 100),
                        "fastd": np.random.uniform(0, 100, 100),
                    },
                    index=self.test_data.index,
                )
                mock_stoch_rsi.return_value = expected_result

                result = indicator.calculate(self.test_data, period)

                assert isinstance(result, pd.DataFrame)
                assert "fastk" in result.columns
                assert "fastd" in result.columns
                mock_stoch_rsi.assert_called_once_with(
                    self.test_data["close"], period, 3, 3
                )

    def test_stochastic_rsi_description(self):
        """Stochastic RSI説明文のテスト"""
        if StochasticRSIIndicator is None:
            pytest.skip("StochasticRSIIndicatorが実装されていません")

        indicator = StochasticRSIIndicator()
        description = indicator.get_description()

        assert isinstance(description, str)
        assert len(description) > 0
        assert "Stochastic RSI" in description or "ストキャスティクスRSI" in description

    def test_stochastic_rsi_parameter_validation(self):
        """Stochastic RSIパラメータ検証のテスト"""
        if StochasticRSIIndicator is None:
            pytest.skip("StochasticRSIIndicatorが実装されていません")

        indicator = StochasticRSIIndicator()

        # 無効な期間でのテスト
        with pytest.raises(Exception):
            indicator.calculate(self.test_data, 0)

        with pytest.raises(Exception):
            indicator.calculate(self.test_data, -1)

    def test_stochastic_rsi_empty_data(self):
        """空データでのStochastic RSIテスト"""
        if StochasticRSIIndicator is None:
            pytest.skip("StochasticRSIIndicatorが実装されていません")

        indicator = StochasticRSIIndicator()
        empty_data = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        with pytest.raises(Exception):
            indicator.calculate(empty_data, 14)


class TestMomentumAdapterStochasticRSI:
    """MomentumAdapterのStochastic RSIメソッドのテスト"""

    def setup_method(self):
        """テスト初期化"""
        self.test_close = pd.Series(
            np.random.uniform(100, 110, 100),
            index=pd.date_range("2023-01-01", periods=100, freq="D"),
            name="close",
        )

    def test_momentum_adapter_stochastic_rsi_method_exists(self):
        """MomentumAdapter.stochastic_rsiメソッドが存在することをテスト"""
        # Red: まだ実装されていないのでAttributeErrorが発生する予定
        assert hasattr(
            MomentumAdapter, "stochastic_rsi"
        ), "MomentumAdapter.stochastic_rsiメソッドが実装されていません"

    def test_momentum_adapter_stochastic_rsi_calculation(self):
        """MomentumAdapter.stochastic_rsiの計算テスト"""
        if not hasattr(MomentumAdapter, "stochastic_rsi"):
            pytest.skip("MomentumAdapter.stochastic_rsiが実装されていません")

        period = 14
        result = MomentumAdapter.stochastic_rsi(self.test_close, period)

        # 結果の検証
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(self.test_close)
        assert "fastk" in result.columns
        assert "fastd" in result.columns

    def test_momentum_adapter_stochastic_rsi_different_periods(self):
        """MomentumAdapter.stochastic_rsiの異なる期間でのテスト"""
        if not hasattr(MomentumAdapter, "stochastic_rsi"):
            pytest.skip("MomentumAdapter.stochastic_rsiが実装されていません")

        for period in [14, 21]:
            result = MomentumAdapter.stochastic_rsi(self.test_close, period)

            assert isinstance(result, pd.DataFrame)
            assert "fastk" in result.columns
            assert "fastd" in result.columns

    def test_momentum_adapter_stochastic_rsi_parameter_validation(self):
        """MomentumAdapter.stochastic_rsiのパラメータ検証テスト"""
        if not hasattr(MomentumAdapter, "stochastic_rsi"):
            pytest.skip("MomentumAdapter.stochastic_rsiが実装されていません")

        # 無効なパラメータでのテスト
        with pytest.raises(Exception):
            MomentumAdapter.stochastic_rsi(self.test_close, 0)

        with pytest.raises(Exception):
            MomentumAdapter.stochastic_rsi(self.test_close, -1)


class TestStochasticRSIIntegration:
    """Stochastic RSIの統合テスト"""

    def test_stochastic_rsi_in_momentum_indicators_factory(self):
        """get_momentum_indicator関数でStochastic RSIが取得できることをテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import (
                get_momentum_indicator,
            )

            # Red: まだStochastic RSIが追加されていないのでValueErrorが発生する予定
            indicator = get_momentum_indicator("STOCHRSI")
            assert indicator.indicator_type == "STOCHRSI"

        except (ImportError, ValueError):
            pytest.fail(
                "Stochastic RSIがget_momentum_indicator関数に追加されていません"
            )

    def test_stochastic_rsi_in_indicators_info(self):
        """MOMENTUM_INDICATORS_INFOにStochastic RSIが含まれることをテスト"""
        try:
            from app.core.services.indicators.momentum_indicators import (
                MOMENTUM_INDICATORS_INFO,
            )

            # Red: まだStochastic RSIが追加されていないのでKeyErrorが発生する予定
            assert "STOCHRSI" in MOMENTUM_INDICATORS_INFO

            stoch_rsi_info = MOMENTUM_INDICATORS_INFO["STOCHRSI"]
            assert "periods" in stoch_rsi_info
            assert "description" in stoch_rsi_info
            assert "category" in stoch_rsi_info
            assert stoch_rsi_info["category"] == "momentum"

        except (ImportError, KeyError):
            pytest.fail("Stochastic RSIがMOMENTUM_INDICATORS_INFOに追加されていません")

    def test_stochastic_rsi_in_main_indicators_module(self):
        """メインのindicatorsモジュールでStochastic RSIが利用できることをテスト"""
        try:
            from app.core.services.indicators import (
                StochasticRSIIndicator,
                get_indicator_by_type,
            )

            # StochasticRSIIndicatorの直接インポート
            assert StochasticRSIIndicator is not None

            # ファクトリー関数経由での取得
            indicator = get_indicator_by_type("STOCHRSI")
            assert indicator.indicator_type == "STOCHRSI"

        except (ImportError, ValueError):
            pytest.fail(
                "Stochastic RSIがメインのindicatorsモジュールに統合されていません"
            )


if __name__ == "__main__":
    # テストの実行
    pytest.main([__file__, "-v"])
