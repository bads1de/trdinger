"""
Indicator Utils Tests

指標関連ユーティリティとYAML設定ユーティリティの統合テスト
"""

import pytest
from unittest.mock import patch

from app.services.auto_strategy.utils.indicator_utils import (
    indicators_by_category,
    get_all_indicators,
    get_all_indicator_ids,
    get_valid_indicator_types,
)


# =============================================================================
# 指標リスト取得関連テスト
# =============================================================================


class TestIndicatorUtils:
    """Indicator Utilsのテスト"""

    def test_indicators_by_category(self):
        """カテゴリ別の指標取得"""
        # 実データに基づいたテスト（多くの指標がcustomに分類されている現状に合わせる）
        custom = indicators_by_category("custom")
        assert len(custom) > 0
        assert any(x in custom for x in ["SMA", "RSI"])

    def test_get_all_indicators(self):
        """全指標取得"""
        all_inds = get_all_indicators()
        assert "RSI" in all_inds
        assert "SMA" in all_inds

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

    def test_get_valid_indicator_types(self):
        """有効な指標タイプ一覧"""
        valid = get_valid_indicator_types()
        assert "SMA" in valid
        assert "RSI" in valid
