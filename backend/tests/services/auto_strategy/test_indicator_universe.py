from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.indicator_universe import (
    IndicatorUniverseMode,
    get_indicator_universe_names,
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
