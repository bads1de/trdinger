#!/usr/bin/env python3
"""
不足しているテクニカル指標のテスト（TDD Red Phase）

設計ドキュメントで要求されている58種類の指標が実装されているかをテストする
"""

import pytest
from database.connection import get_db
from app.core.services.strategy_builder_service import StrategyBuilderService


class TestMissingIndicators:
    """不足指標のテストクラス"""

    def setup_method(self):
        """テスト前の初期化"""
        self.db = next(get_db())
        self.service = StrategyBuilderService(self.db)

    def teardown_method(self):
        """テスト後のクリーンアップ"""
        self.db.close()

    def test_total_indicator_count_should_be_at_least_58(self):
        """総指標数が58個以上であることをテスト"""
        indicators = self.service.get_available_indicators()

        total_count = sum(len(indicator_list) for indicator_list in indicators.values())

        assert (
            total_count >= 58
        ), f"期待される指標数は最低58個ですが、実際は{total_count}個です"

    def test_trend_indicators_should_be_14(self):
        """トレンド系指標が14個であることをテスト"""
        indicators = self.service.get_available_indicators()
        trend_indicators = indicators.get("trend", [])

        assert (
            len(trend_indicators) == 14
        ), f"トレンド系指標は14個必要ですが、実際は{len(trend_indicators)}個です"

    def test_momentum_indicators_should_be_at_least_19(self):
        """モメンタム系指標が19個以上であることをテスト"""
        indicators = self.service.get_available_indicators()
        momentum_indicators = indicators.get("momentum", [])

        assert (
            len(momentum_indicators) >= 19
        ), f"モメンタム系指標は最低19個必要ですが、実際は{len(momentum_indicators)}個です"

    def test_volatility_indicators_should_be_8(self):
        """ボラティリティ系指標が8個であることをテスト"""
        indicators = self.service.get_available_indicators()
        volatility_indicators = indicators.get("volatility", [])

        assert (
            len(volatility_indicators) == 8
        ), f"ボラティリティ系指標は8個必要ですが、実際は{len(volatility_indicators)}個です"

    def test_volume_indicators_should_be_3(self):
        """ボリューム系指標が3個であることをテスト"""
        indicators = self.service.get_available_indicators()
        volume_indicators = indicators.get("volume", [])

        assert (
            len(volume_indicators) == 3
        ), f"ボリューム系指標は3個必要ですが、実際は{len(volume_indicators)}個です"

    def test_price_transform_indicators_should_be_at_least_13(self):
        """価格変換系指標が13個以上であることをテスト"""
        indicators = self.service.get_available_indicators()
        price_transform_indicators = indicators.get("price_transform", [])

        assert (
            len(price_transform_indicators) >= 13
        ), f"価格変換系指標は最低13個必要ですが、実際は{len(price_transform_indicators)}個です"

    def test_specific_volume_indicators_exist(self):
        """特定のボリューム系指標が存在することをテスト"""
        indicators = self.service.get_available_indicators()
        volume_indicators = indicators.get("volume", [])
        volume_types = [indicator["type"] for indicator in volume_indicators]

        required_volume_indicators = ["OBV", "AD", "ADOSC"]

        for indicator_type in required_volume_indicators:
            assert (
                indicator_type in volume_types
            ), f"ボリューム系指標 '{indicator_type}' が見つかりません"

    def test_specific_price_transform_indicators_exist(self):
        """特定の価格変換系指標が存在することをテスト"""
        indicators = self.service.get_available_indicators()
        price_transform_indicators = indicators.get("price_transform", [])
        price_transform_types = [
            indicator["type"] for indicator in price_transform_indicators
        ]

        required_price_transform_indicators = [
            "AVGPRICE",
            "MEDPRICE",
            "TYPPRICE",
            "WCLPRICE",
            "HT_DCPERIOD",
            "HT_DCPHASE",
            "HT_PHASOR",
            "HT_SINE",
            "HT_TRENDMODE",
            "FAMA",
            "SAREXT",
            "SAR",
            "MAMA",
        ]

        for indicator_type in required_price_transform_indicators:
            assert (
                indicator_type in price_transform_types
            ), f"価格変換系指標 '{indicator_type}' が見つかりません"

    def test_specific_volatility_indicators_exist(self):
        """特定のボラティリティ系指標が存在することをテスト"""
        indicators = self.service.get_available_indicators()
        volatility_indicators = indicators.get("volatility", [])
        volatility_types = [indicator["type"] for indicator in volatility_indicators]

        # 設計ドキュメントから不足している指標を特定
        required_volatility_indicators = [
            "ATR",
            "NATR",
            "TRANGE",
            "BB",
            "STDDEV",
            "VAR",
            "BETA",
            "CORREL",
        ]

        for indicator_type in required_volatility_indicators:
            assert (
                indicator_type in volatility_types
            ), f"ボラティリティ系指標 '{indicator_type}' が見つかりません"

    def test_specific_momentum_indicators_exist(self):
        """特定のモメンタム系指標が存在することをテスト"""
        indicators = self.service.get_available_indicators()
        momentum_indicators = indicators.get("momentum", [])
        momentum_types = [indicator["type"] for indicator in momentum_indicators]

        # 設計ドキュメントから不足している指標を特定
        required_momentum_indicators = [
            "RSI",
            "STOCH",
            "CCI",
            "WILLR",
            "MOM",
            "ROC",
            "ADX",
            "AROON",
            "MFI",
            "STOCHRSI",
            "ULTOSC",
            "CMO",
            "BOP",
            "PPO",
            "ROCP",
            "ROCR",
            "STOCHF",
            "PLUS_DI",
            "MINUS_DI",
        ]

        for indicator_type in required_momentum_indicators:
            assert (
                indicator_type in momentum_types
            ), f"モメンタム系指標 '{indicator_type}' が見つかりません"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
