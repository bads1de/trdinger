"""
Indicator Utils Tests
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.auto_strategy.utils.indicator_utils import (
    indicators_by_category,
    get_all_indicators,
    get_all_indicator_ids,
    get_valid_indicator_types,
)


class TestIndicatorUtils:
    """Indicator Utilsのテスト"""

    @pytest.fixture
    def mock_registry(self):
        with patch(
            "app.services.auto_strategy.utils.indicator_utils._load_indicator_registry"
        ) as mock_load:
            registry = MagicMock()
            registry._configs = {
                "RSI": MagicMock(category="momentum", indicator_name="RSI"),
                "SMA": MagicMock(category="trend", indicator_name="SMA"),
                "EMA": MagicMock(category="trend", indicator_name="EMA"),
                "ATR": MagicMock(category="volatility", indicator_name="ATR"),
                "OBV": MagicMock(category="volume", indicator_name="OBV"),
                "Unknown": MagicMock(category="other", indicator_name="Unknown"),
                # Config with no category should be skipped or handled gracefully
                "Broken": None,
            }
            mock_load.return_value = registry
            yield registry

    def test_indicators_by_category(self, mock_registry):
        """カテゴリ別の指標取得"""
        trend = indicators_by_category("trend")
        assert "SMA" in trend
        assert "EMA" in trend
        assert "RSI" not in trend

        momentum = indicators_by_category("momentum")
        assert "RSI" in momentum
        assert "SMA" not in momentum

    @patch("app.services.auto_strategy.utils.indicator_utils.get_volume_indicators")
    @patch("app.services.auto_strategy.utils.indicator_utils.get_momentum_indicators")
    @patch("app.services.auto_strategy.utils.indicator_utils.get_trend_indicators")
    @patch("app.services.auto_strategy.utils.indicator_utils.get_volatility_indicators")
    def test_get_all_indicators(
        self, mock_volatility, mock_trend, mock_momentum, mock_volume
    ):
        """全指標取得"""
        mock_volume.return_value = ["OBV"]
        mock_momentum.return_value = ["RSI"]
        mock_trend.return_value = ["SMA"]
        mock_volatility.return_value = ["ATR"]

        # Test needs to handle import of COMPOSITE_INDICATORS inside the function
        # We can patch the module import if needed, but since it imports from ..config.constants,
        # it might be easier to check if results contain what we expect + composites if environment has them.

        all_inds = get_all_indicators()

        assert "OBV" in all_inds
        assert "RSI" in all_inds
        assert "SMA" in all_inds
        assert "ATR" in all_inds

    @patch("app.services.auto_strategy.utils.indicator_utils.TechnicalIndicatorService")
    def test_get_all_indicator_ids(self, MockService):
        """指標IDマッピング取得"""
        service = MockService.return_value
        service.get_supported_indicators.return_value = {"RSI": {}, "SMA": {}}

        ids = get_all_indicator_ids()

        assert ids[""] == 0
        assert "RSI" in ids
        assert "SMA" in ids
        assert ids["RSI"] > 0

    def test_get_valid_indicator_types(self, mock_registry):
        """有効な指標タイプ一覧"""
        # mock_registry patch is applied via fixture, which affects indicators_by_category used inside
        # get_valid_indicator_types -> get_trend_indicators etc -> indicators_by_category

        valid = get_valid_indicator_types()

        assert "SMA" in valid
        assert "RSI" in valid
        assert "OBV" in valid
