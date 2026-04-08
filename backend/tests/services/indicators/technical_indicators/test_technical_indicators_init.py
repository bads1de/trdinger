"""
technical_indicatorsパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.indicators.technical_indicators as ti_package


class TestTechnicalIndicatorsInitExports:
    """technical_indicators/__init__.pyのエクスポートテスト"""

    def test_trend_indicators_exported(self):
        """TrendIndicatorsがエクスポートされている"""
        assert hasattr(ti_package, "TrendIndicators")

    def test_momentum_indicators_exported(self):
        """MomentumIndicatorsがエクスポートされている"""
        assert hasattr(ti_package, "MomentumIndicators")

    def test_volatility_indicators_exported(self):
        """VolatilityIndicatorsがエクスポートされている"""
        assert hasattr(ti_package, "VolatilityIndicators")

    def test_volume_indicators_exported(self):
        """VolumeIndicatorsがエクスポートされている"""
        assert hasattr(ti_package, "VolumeIndicators")

    def test_overlap_indicators_exported(self):
        """OverlapIndicatorsがエクスポートされている"""
        assert hasattr(ti_package, "OverlapIndicators")

    def test_advanced_features_exported(self):
        """AdvancedFeaturesがエクスポートされている"""
        assert hasattr(ti_package, "AdvancedFeatures")

    def test_original_indicators_exported(self):
        """OriginalIndicatorsがエクスポートされている"""
        assert hasattr(ti_package, "OriginalIndicators")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = [
            "TrendIndicators",
            "MomentumIndicators",
            "VolatilityIndicators",
            "VolumeIndicators",
            "OverlapIndicators",
            "AdvancedFeatures",
            "OriginalIndicators",
        ]

        for item in expected_items:
            assert item in ti_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(ti_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert ti_package.__doc__ is not None
        assert len(ti_package.__doc__) > 0
