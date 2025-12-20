"""
DynamicIndicatorDiscovery のテスト

動的インジケーター検出と特別オーバーライドのテストを行います。
"""

import pytest
from app.services.indicators.config.discovery import DynamicIndicatorDiscovery
from app.services.indicators.config.indicator_config import (
    IndicatorConfig,
    IndicatorResultType,
    IndicatorScaleType,
)


class TestApplySpecialOverrides:
    """_apply_special_overrides のテスト"""

    @pytest.fixture
    def base_config(self):
        """テスト用基本設定ファクトリ"""

        def _create(indicator_name: str) -> IndicatorConfig:
            return IndicatorConfig(
                indicator_name=indicator_name,
                category="test",
                required_data=["close"],
                param_map={},
                parameters={},
                default_values={},
                result_type=IndicatorResultType.SINGLE,
                returns="single",
                scale_type=IndicatorScaleType.PRICE_ABSOLUTE,
            )

        return _create

    @pytest.mark.parametrize(
        "indicator_name,expected_col_count,pattern",
        [
            # 動的取得のため、カラム数とパターンで検証
            ("STOCH", 2, "STOCH"),  # STOCHk, STOCHd
            ("MACD", 3, "MACD"),  # MACD, MACDh, MACDs
            ("BBANDS", 5, "BB"),  # BBL, BBM, BBU, BBB, BBP
            ("VORTEX", 2, "VTX"),  # VTXP, VTXM
            ("FISHER", 2, "FISHER"),  # FISHERT, FISHERTs
            ("TSI", 2, "TSI"),  # TSI, TSIs
        ],
    )
    def test_return_cols_dynamic(
        self, base_config, indicator_name: str, expected_col_count: int, pattern: str
    ):
        """return_cols がイントロスペクションで動的に取得されることを確認"""
        config = base_config(indicator_name)
        DynamicIndicatorDiscovery._apply_special_overrides(config)

        # return_cols が設定されている
        assert config.return_cols is not None
        # カラム数が期待通り
        assert len(config.return_cols) == expected_col_count
        # パターンがカラム名に含まれる
        assert all(pattern in col for col in config.return_cols)

    def test_bbands_aliases(self, base_config):
        """BBANDS の aliases が正しく設定されることを確認"""
        config = base_config("BBANDS")
        DynamicIndicatorDiscovery._apply_special_overrides(config)
        assert set(config.aliases) == {"BB", "BOLLINGER"}

    @pytest.mark.parametrize(
        "indicator_name,params,expected_min_length",
        [
            # STOCH: max(k, d, smooth_k)
            ("STOCH", {"k": 14, "d": 3, "smooth_k": 3}, 14),  # max(14, 3, 3)
            ("STOCH", {"k": 10, "d": 5, "smooth_k": 2}, 10),  # max(10, 5, 2)
            ("STOCH", {}, 14),  # デフォルト: max(14, 3, 3)
            # MACD: max(fast, slow, signal)
            ("MACD", {"fast": 12, "slow": 26, "signal": 9}, 26),  # max(12, 26, 9)
            ("MACD", {"fast": 10, "slow": 50, "signal": 20}, 50),  # max(10, 50, 20)
            ("MACD", {}, 26),  # デフォルト: max(12, 26, 9)
            # RSI: length
            ("RSI", {"length": 14}, 14),
            ("RSI", {"length": 20}, 20),
            ("RSI", {}, 14),  # デフォルト: 14
            # ATR: length
            ("ATR", {"length": 14}, 14),
            ("ATR", {"length": 20}, 20),
            ("ATR", {}, 14),  # デフォルト: 14
        ],
    )
    def test_min_length_func(
        self, base_config, indicator_name: str, params: dict, expected_min_length: int
    ):
        """min_length_func が正しく動作することを確認（イントロスペクションベース）"""
        config = base_config(indicator_name)
        DynamicIndicatorDiscovery._apply_special_overrides(config)

        if config.min_length_func is not None:
            result = config.min_length_func(params)
            assert result == expected_min_length

    def test_unknown_indicator_no_override(self, base_config):
        """未登録のインジケーターにはオーバーライドが適用されないこと"""
        config = base_config("UNKNOWN_INDICATOR")
        original_return_cols = config.return_cols

        DynamicIndicatorDiscovery._apply_special_overrides(config)

        # 変更されていないこと
        assert config.return_cols == original_return_cols
        assert config.min_length_func is None
        assert config.aliases is None


class TestDiscoverAll:
    """discover_all のテスト"""

    def test_discover_all_returns_list(self):
        """discover_all がリストを返すこと"""
        result = DynamicIndicatorDiscovery.discover_all()
        assert isinstance(result, list)

    def test_discover_all_contains_common_indicators(self):
        """一般的なインジケーターが検出されること"""
        configs = DynamicIndicatorDiscovery.discover_all()
        names = {c.indicator_name for c in configs}

        # 一般的なインジケーターが含まれていることを確認
        common_indicators = {"RSI", "SMA", "EMA", "MACD", "BBANDS", "ATR"}
        for ind in common_indicators:
            assert ind in names, f"{ind} が検出されていません"

    def test_discover_all_no_duplicates(self):
        """重複する設定がないこと"""
        configs = DynamicIndicatorDiscovery.discover_all()
        names = [c.indicator_name for c in configs]

        assert len(names) == len(set(names)), "重複するインジケーター設定があります"


class TestAnalyzeFunction:
    """_analyze_function のテスト"""

    def test_analyze_simple_function(self):
        """単純な関数の解析"""

        def simple_indicator(close, length=14):
            pass

        config = DynamicIndicatorDiscovery._analyze_function(
            "simple", simple_indicator, "test"
        )

        assert config is not None
        assert config.indicator_name == "SIMPLE"
        assert "close" in config.required_data
        assert "length" in config.parameters

    def test_analyze_multi_data_function(self):
        """複数データ引数を持つ関数の解析"""

        def multi_data_indicator(high, low, close, length=14):
            pass

        config = DynamicIndicatorDiscovery._analyze_function(
            "multi", multi_data_indicator, "test"
        )

        assert config is not None
        assert set(config.required_data) == {"high", "low", "close"}


class TestCreateParameterConfig:
    """_create_parameter_config のテスト"""

    def test_parameter_range_calculated_from_default(self):
        """パラメータ範囲がデフォルト値から動的に計算されること"""
        # 期間パラメータ
        config = DynamicIndicatorDiscovery._create_parameter_config("length", 14)
        assert config is not None
        assert config.default_value == 14
        # min = max(2, 14*0.2) = 2, max = max(14*5, 50) = 70
        assert config.min_value == 2
        assert config.max_value == 70

    def test_std_parameter_special_range(self):
        """std パラメータに特殊な範囲が適用されること"""
        config = DynamicIndicatorDiscovery._create_parameter_config("std", 2.0)
        assert config is not None
        assert config.default_value == 2.0
        # std: min=max(0.1, 2.0*0.1)=0.2, max=2.0*3.0=6.0
        assert config.min_value == 0.2
        assert config.max_value == 6.0

    def test_numeric_default_creates_config(self):
        """数値デフォルト値がある場合にパラメータ設定が生成されること"""
        config = DynamicIndicatorDiscovery._create_parameter_config(
            "custom_param", 10.0
        )

        assert config is not None
        assert config.default_value == 10.0
        # 一般パラメータ: min=max(2, 10*0.2)=2, max=max(10*5, 50)=50
        assert config.min_value == 2
        assert config.max_value == 50

    def test_non_numeric_returns_none(self):
        """非数値パラメータでは None が返ること"""
        config = DynamicIndicatorDiscovery._create_parameter_config(
            "string_param", "some_string"
        )
        assert config is None
