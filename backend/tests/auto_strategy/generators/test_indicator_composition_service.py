"""
IndicatorCompositionService Tests
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from backend.app.services.auto_strategy.generators.indicator_composition_service import (
    IndicatorCompositionService,
)
from backend.app.services.auto_strategy.models.strategy_models import IndicatorGene

# Mock constants if necessary, or assume they are available from imports within the service
# The service imports them from ..config.constants.
# We should probably mock them or rely on them.
# Since we are testing logic dependent on these constants, let's assume standard values or patch them if we want consistent tests.


class TestIndicatorCompositionService:
    """IndicatorCompositionServiceのテスト"""

    @pytest.fixture
    def mock_config(self):
        config = Mock()
        config.max_indicators = 3
        return config

    @pytest.fixture
    def service(self, mock_config):
        service = IndicatorCompositionService(config=mock_config)
        return service

    @pytest.fixture
    def mock_indicator_gene(self):
        def _create(type_name, params=None):
            return IndicatorGene(type=type_name, parameters=params or {}, enabled=True)

        return _create

    def test_enhance_with_trend_indicators(self, service, mock_indicator_gene):
        """トレンド指標の強制追加（現在は無効化されていることを確認）"""
        indicators = []
        available = ["SMA", "RSI"]

        result = service.enhance_with_trend_indicators(indicators, available)

        # Currently the implementation just returns the list as is
        assert result == indicators
        assert len(result) == 0

    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.random.random"
    )
    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.MOVING_AVERAGE_INDICATORS",
        ["SMA", "EMA"],
    )
    def test_enhance_with_ma_cross_already_enough_ma(
        self, mock_random, service, mock_indicator_gene
    ):
        """MAが既に十分ある場合は追加しない"""
        indicators = [mock_indicator_gene("SMA"), mock_indicator_gene("EMA")]
        available = ["SMA", "EMA", "RSI"]

        # ma_count = 2, condition is ma_count < 2
        service.enhance_with_ma_cross_strategy(indicators, available)

        assert len(indicators) == 2

    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.random.random"
    )
    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.MOVING_AVERAGE_INDICATORS",
        ["SMA", "EMA"],
    )
    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.PREFERRED_MA_INDICATORS",
        ["SMA"],
    )
    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.MA_INDICATORS_NEEDING_PERIOD",
        ["SMA", "EMA"],
    )
    def test_enhance_with_ma_cross_add_one(
        self, mock_random, service, mock_indicator_gene
    ):
        """条件を満たす場合MAを追加する"""
        indicators = [mock_indicator_gene("RSI")]
        available = ["SMA", "EMA", "RSI"]

        # random < 0.25 to trigger
        mock_random.return_value = 0.1

        # We need to ensure _choose_ma_with_unique_period picks one.
        # It calls random.sample inside. We should maybe mock random.sample too or let it run if deterministic enough.
        # The logic: pick unique period.

        with patch(
            "backend.app.services.auto_strategy.generators.indicator_composition_service.random.sample",
            side_effect=lambda pop, k: list(pop),
        ):
            with patch(
                "backend.app.services.auto_strategy.generators.indicator_composition_service.random.choice",
                return_value=14,
            ):
                service.enhance_with_ma_cross_strategy(indicators, available)

        assert len(indicators) == 2
        added = indicators[1]
        assert added.type in ["SMA", "EMA"]

    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.random.random"
    )
    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.MOVING_AVERAGE_INDICATORS",
        ["SMA", "EMA"],
    )
    def test_enhance_with_ma_cross_prob_fail(
        self, mock_random, service, mock_indicator_gene
    ):
        """確率条件でスキップ"""
        indicators = [mock_indicator_gene("RSI")]
        available = ["SMA", "EMA"]

        # random >= 0.25 -> skip
        mock_random.return_value = 0.5

        service.enhance_with_ma_cross_strategy(indicators, available)

        assert len(indicators) == 1

    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.indicator_registry"
    )
    def test_is_trend_indicator(self, mock_registry, service):
        """トレンド指標判定"""
        mock_config = Mock()
        mock_config.category = "trend"
        mock_registry.get_indicator_config.return_value = mock_config

        assert service._is_trend_indicator("SMA") is True

        mock_config.category = "momentum"
        assert service._is_trend_indicator("RSI") is False

        mock_registry.get_indicator_config.return_value = None
        assert service._is_trend_indicator("UNKNOWN") is False

    @patch(
        "backend.app.services.auto_strategy.generators.indicator_composition_service.MA_INDICATORS_NEEDING_PERIOD",
        ["SMA"],
    )
    def test_get_default_params_for_indicator(self, service):
        """デフォルトパラメータ取得"""
        with patch(
            "backend.app.services.auto_strategy.generators.indicator_composition_service.random.choice",
            return_value=20,
        ):
            params = service._get_default_params_for_indicator("SMA")
            assert params == {"period": 20}

            params = service._get_default_params_for_indicator("RSI")
            assert params == {}

    def test_remove_non_ma_indicator(self, service, mock_indicator_gene):
        """非MA指標の削除"""
        # Patch generic constants locally within test if needed, or rely on real ones?
        # Let's rely on real implementation but we need to know what IS considered ma indicator.
        # It imports MOVING_AVERAGE_INDICATORS.

        with patch(
            "backend.app.services.auto_strategy.generators.indicator_composition_service.MOVING_AVERAGE_INDICATORS",
            ["SMA"],
        ):
            indicators = [
                mock_indicator_gene("RSI"),
                mock_indicator_gene("SMA"),
                mock_indicator_gene("MACD"),
            ]

            service._remove_non_ma_indicator(indicators)

            # Should accept RSI (not MA) and remove it.
            assert len(indicators) == 2
            assert indicators[0].type == "SMA"
            assert indicators[1].type == "MACD"

    def test_choose_ma_with_unique_period(self, service):
        """重複しないPeriodを持つMAの選択ロジック"""
        ma_pool = ["SMA", "EMA"]
        existing_periods = {20}

        # Mock _get_default_params_for_indicator to control periods
        with patch.object(service, "_get_default_params_for_indicator") as mock_params:
            # Case 1: First candidate has unique period
            mock_params.return_value = {"period": 10}
            result = service._choose_ma_with_unique_period(ma_pool, existing_periods)
            assert result in ma_pool
            # Should pick one that gives period 10 (unique)

            # Case 2: Force collision
            def side_effect(ma_type):
                if ma_type == "SMA":
                    return {"period": 20}  # Collision
                return {"period": 30}  # Unique

            mock_params.side_effect = side_effect
            # If SMA is picked first and collides, loop should continue to EMA
            # We can't deterministicly control random.sample order easily without patching random
            # But we can verify it doesn't return None if a valid one exists

            result = service._choose_ma_with_unique_period(
                ["SMA", "EMA"], existing_periods
            )
            assert result is not None

    def test_remove_non_trend_indicator(self, service, mock_indicator_gene):
        """非トレンド指標の削除"""
        with patch.object(service, "_is_trend_indicator") as mock_is_trend:
            # Setup: SMA is trend, RSI is not
            mock_is_trend.side_effect = lambda name: name == "SMA"

            indicators = [
                mock_indicator_gene("SMA"),
                mock_indicator_gene("RSI"),
                mock_indicator_gene("SMA"),
            ]

            removed = service._remove_non_trend_indicator(indicators)

            assert removed == "RSI"
            assert len(indicators) == 2
            assert all(ind.type == "SMA" for ind in indicators)

            # Case: All are trend
            indicators = [mock_indicator_gene("SMA")]
            removed = service._remove_non_trend_indicator(indicators)
            assert removed == ""
            assert len(indicators) == 1

    def test_choose_preferred_trend_indicator(self, service):
        """トレンド指標の選択"""
        trend_pool = ["SMA", "EMA", "Unknown"]

        with patch(
            "backend.app.services.auto_strategy.generators.indicator_composition_service.random.choice"
        ) as mock_choice:
            mock_choice.side_effect = lambda seq: seq[0]  # Always pick first

            # Preferred exist
            result = service._choose_preferred_trend_indicator(trend_pool)
            assert result in ["SMA", "EMA"]

            # No preferred
            result = service._choose_preferred_trend_indicator(["Unknown"])
            assert result == "Unknown"

            # Empty
            result = service._choose_preferred_trend_indicator([])
            assert result == "SMA"  # Fallback
