"""
Indicator Utils Tests

指標関連ユーティリティとYAML設定ユーティリティの統合テスト
"""

from app.services.auto_strategy.utils.indicators import (
    get_all_indicators,
    get_valid_indicator_types,
)

# =============================================================================
# 指標リスト取得関連テスト
# =============================================================================


class TestIndicatorUtils:
    """Indicator Utilsのテスト"""

    def test_get_all_indicators(self):
        """全指標取得"""
        all_inds = get_all_indicators()
        assert "RSI" in all_inds
        assert "SMA" in all_inds
        assert "EXP_MA" not in all_inds
        assert "SIMPLE_MA" not in all_inds

    def test_get_valid_indicator_types(self):
        """有効な指標タイプ一覧"""
        valid = get_valid_indicator_types()
        assert "SMA" in valid
        assert "RSI" in valid
