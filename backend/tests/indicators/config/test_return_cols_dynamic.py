"""
return_cols 動的取得のテスト

pandas-ta から戻り値カラム名を動的に取得する機能のテスト
"""


class TestDynamicReturnCols:
    """return_cols 動的取得のテスト"""

    def test_get_macd_return_cols(self):
        """MACD の戻り値カラム名を動的に取得"""
        from app.services.indicators.config.pandas_ta_introspection import (
            get_return_column_names,
        )

        cols = get_return_column_names("macd")
        assert cols is not None
        assert len(cols) == 3
        # MACD, Signal, Histogram の順
        assert any("MACD" in c.upper() for c in cols)

    def test_get_stoch_return_cols(self):
        """STOCH の戻り値カラム名を動的に取得"""
        from app.services.indicators.config.pandas_ta_introspection import (
            get_return_column_names,
        )

        cols = get_return_column_names("stoch")
        assert cols is not None
        assert len(cols) == 2
        # STOCHk, STOCHd
        assert any("k" in c.lower() for c in cols)
        assert any("d" in c.lower() for c in cols)

    def test_get_bbands_return_cols(self):
        """BBANDS の戻り値カラム名を動的に取得"""
        from app.services.indicators.config.pandas_ta_introspection import (
            get_return_column_names,
        )

        cols = get_return_column_names("bbands")
        assert cols is not None
        # BBL, BBM, BBU, BBB, BBP の5つ
        assert len(cols) >= 3

    def test_single_return_indicator_returns_none(self):
        """単一値を返す指標はNoneを返す"""
        from app.services.indicators.config.pandas_ta_introspection import (
            get_return_column_names,
        )

        # RSI は単一値を返す
        cols = get_return_column_names("rsi")
        # 単一値指標は None または 長さ1のリスト
        assert cols is None or len(cols) == 1


class TestReducedIndicatorOverrides:
    """INDICATOR_OVERRIDES が完全に削除されていることのテスト"""

    def test_indicator_overrides_is_removed(self):
        """INDICATOR_OVERRIDES が完全に削除されていること"""
        from app.services.indicators.config.discovery import DynamicIndicatorDiscovery

        # 定数は削除され、ロジックに移行
        assert not hasattr(
            DynamicIndicatorDiscovery, "INDICATOR_OVERRIDES"
        ), "INDICATOR_OVERRIDES should be removed"
