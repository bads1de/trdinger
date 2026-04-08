"""
originalパッケージの__init__.pyのテスト

エクスポート定義を確認します。
"""

import pytest

import app.services.indicators.technical_indicators.original as original_package


class TestOriginalInitExports:
    """original/__init__.pyのエクスポートテスト"""

    def test_original_indicators_exported(self):
        """OriginalIndicatorsがエクスポートされている"""
        assert hasattr(original_package, "OriginalIndicators")

    def test_original_indicators_class(self):
        """OriginalIndicatorsがクラスである"""
        assert isinstance(original_package.OriginalIndicators, type)

    def test_original_indicators_has_methods(self):
        """OriginalIndicatorsにメソッドがある"""
        indicators = original_package.OriginalIndicators

        assert hasattr(indicators, "frama")
        assert hasattr(indicators, "prime_oscillator")
        assert hasattr(indicators, "fibonacci_cycle")
        assert hasattr(indicators, "adaptive_entropy")
        assert hasattr(indicators, "quantum_flow")
        assert hasattr(indicators, "harmonic_resonance")
        assert hasattr(indicators, "chaos_fractal_dimension")
        assert hasattr(indicators, "trend_intensity_index")
        assert hasattr(indicators, "connors_rsi")
        assert hasattr(indicators, "gri")
        assert hasattr(indicators, "damiani_volatmeter")
        assert hasattr(indicators, "demarker")
        assert hasattr(indicators, "kairi_relative_index")
        assert hasattr(indicators, "entropy_volatility_index")
        assert hasattr(indicators, "rmi")
        assert hasattr(indicators, "pfe")
        assert hasattr(indicators, "mmi")
        assert hasattr(indicators, "ehlers_cyber_cycle")
        assert hasattr(indicators, "ehlers_instantaneous_trendline")
        assert hasattr(indicators, "smoothed_adaptive_momentum")
        assert hasattr(indicators, "vortex_rsi")
        assert hasattr(indicators, "ttf")
        assert hasattr(indicators, "rwi")
        assert hasattr(indicators, "hurst_exponent")

    def test_all_contains_expected_items(self):
        """__all__に期待されるアイテムが含まれる"""
        expected_items = ["OriginalIndicators"]

        for item in expected_items:
            assert item in original_package.__all__, f"{item} not in __all__"

    def test_all_is_list(self):
        """__all__がリストである"""
        assert isinstance(original_package.__all__, list)

    def test_module_has_docstring(self):
        """モジュールにドキュメント文字列がある"""
        assert original_package.__doc__ is not None
        assert len(original_package.__doc__) > 0
