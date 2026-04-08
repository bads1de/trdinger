"""
pandas_taパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.indicators.technical_indicators.pandas_ta as pandas_ta_package


class TestPandasTAInitExports:
    """pandas_ta/__init__.pyのエクスポートテスト"""

    def test_momentum_indicators_exported(self):
        """MomentumIndicatorsがエクスポートされている"""
        assert hasattr(pandas_ta_package, "MomentumIndicators")

    def test_overlap_indicators_exported(self):
        """OverlapIndicatorsがエクスポートされている"""
        assert hasattr(pandas_ta_package, "OverlapIndicators")

    def test_trend_indicators_exported(self):
        """TrendIndicatorsがエクスポートされている"""
        assert hasattr(pandas_ta_package, "TrendIndicators")

    def test_volatility_indicators_exported(self):
        """VolatilityIndicatorsがエクスポートされている"""
        assert hasattr(pandas_ta_package, "VolatilityIndicators")

    def test_volume_indicators_exported(self):
        """VolumeIndicatorsがエクスポートされている"""
        assert hasattr(pandas_ta_package, "VolumeIndicators")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "MomentumIndicators",
            "OverlapIndicators",
            "TrendIndicators",
            "VolatilityIndicators",
            "VolumeIndicators",
        ]

        for item in expected_items:
            assert item in pandas_ta_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(pandas_ta_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert pandas_ta_package.__doc__ is not None
        assert len(pandas_ta_package.__doc__) > 0
