from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.config.indicator_universe import (
    IndicatorUniverseMode,
    get_indicator_universe_names,
    get_non_price_indicator_names,
)
from app.services.auto_strategy.genes import generate_random_indicators


class TestIndicatorUniverse:
    def test_curated_universe_returns_fixed_subset(self):
        indicators = set(get_indicator_universe_names(IndicatorUniverseMode.CURATED))

        assert "SMA" in indicators
        assert "EMA" in indicators
        assert "BBANDS" in indicators
        assert "LIQUIDATION_CASCADE_SCORE" in indicators
        assert "OI_PRICE_CONFIRMATION" in indicators

    def test_curated_universe_excludes_market_cap_indicator(self):
        indicators = set(get_indicator_universe_names(IndicatorUniverseMode.CURATED))

        assert "LEVERAGE_RATIO" not in indicators

    def test_curated_universe_filters_non_standard_or_non_condition_indicators(self):
        indicators = set(get_indicator_universe_names(IndicatorUniverseMode.CURATED))

        assert "CRYPTO_LEVERAGE_INDEX" not in indicators
        assert "WHALE_DIVERGENCE" not in indicators
        assert "REGIME_QUADRANT" not in indicators

    def test_experimental_all_keeps_broader_supported_set(self):
        indicators = set(
            get_indicator_universe_names(IndicatorUniverseMode.EXPERIMENTAL_ALL)
        )

        assert "LEVERAGE_RATIO" in indicators
        assert "CRYPTO_LEVERAGE_INDEX" in indicators

    def test_random_indicator_generation_uses_curated_universe_by_default(self):
        config = GAConfig(min_indicators=1, max_indicators=3, random_state=7)
        curated = set(get_indicator_universe_names(IndicatorUniverseMode.CURATED))

        indicators = generate_random_indicators(config)

        assert indicators
        assert all(indicator.type in curated for indicator in indicators)

    def test_non_price_indicator_names_returns_oifrlsr_derived(self):
        non_price = get_non_price_indicator_names()

        assert "OI_WEIGHTED_FUNDING_RATE" in non_price
        assert "LONG_SHORT_RATIO_ZSCORE" in non_price
        assert "OI_PRICE_CONFIRMATION" in non_price
        # 価格ベースの指標は含まれない
        assert "SMA" not in non_price
        assert "RSI" not in non_price

    def test_min_non_price_indicators_config_field(self):
        config = GAConfig(min_non_price_indicators=2)

        assert config.min_non_price_indicators == 2
        assert "min_non_price_indicators" in config.to_dict()

    def test_min_non_price_guaranteed_in_generation(self):
        config = GAConfig(
            min_indicators=1,
            max_indicators=5,
            min_non_price_indicators=1,
        )
        non_price = set(get_non_price_indicator_names())

        for _ in range(30):
            indicators = generate_random_indicators(config)
            assert any(indicator.type in non_price for indicator in indicators)


class TestCuratedVolumeAndRegimeIndicators:
    """curatedカタログへの出来高系・レンジ/統計系追加のテスト"""

    def test_curated_universe_contains_volume_indicators(self):
        indicators = set(get_indicator_universe_names(IndicatorUniverseMode.CURATED))

        assert "VWAP" in indicators
        assert "MFI" in indicators
        assert "CMF" in indicators
        assert "PVO" in indicators
        assert "VFI" in indicators

    def test_curated_universe_contains_regime_stat_indicators(self):
        indicators = set(get_indicator_universe_names(IndicatorUniverseMode.CURATED))

        assert "SQUEEZE" in indicators
        assert "RWI" in indicators
        assert "ZSCORE" in indicators
