"""
オペランドグループ化機能のテストモジュール

OperandGroupingSystemの機能をテストする。
"""

import pytest
from unittest.mock import patch, Mock

from backend.app.services.auto_strategy.core.operand_grouping import (
    OperandGroupingSystem,
    OperandGroup,
    operand_grouping_system,
    get_operand_group,
    get_compatibility_score,
    validate_condition,
)


class TestOperandGroupingSystem:
    """OperandGroupingSystemのテスト"""

    @pytest.fixture
    def grouping_system(self):
        """OperandGroupingSystemインスタンス"""
        return OperandGroupingSystem()

    def test_get_operand_group_price_based(self, grouping_system):
        """PRICE_BASEDグループのオペランド分類"""
        # OHLCVデータ
        assert grouping_system.get_operand_group("close") == OperandGroup.PRICE_BASED
        assert grouping_system.get_operand_group("open") == OperandGroup.PRICE_BASED
        assert grouping_system.get_operand_group("high") == OperandGroup.PRICE_BASED
        assert grouping_system.get_operand_group("low") == OperandGroup.PRICE_BASED

        # 移動平均系
        assert grouping_system.get_operand_group("SMA") == OperandGroup.PRICE_BASED
        assert grouping_system.get_operand_group("EMA") == OperandGroup.PRICE_BASED

    def test_get_operand_group_percentage_0_100(self, grouping_system):
        """PERCENTAGE_0_100グループのオペランド分類"""
        # RSI系
        assert grouping_system.get_operand_group("RSI") == OperandGroup.PERCENTAGE_0_100
        assert grouping_system.get_operand_group("STOCH") == OperandGroup.PERCENTAGE_0_100

        # 他のパーセント型
        assert grouping_system.get_operand_group("ACCBANDS") == OperandGroup.PERCENTAGE_0_100

    def test_get_operand_group_zero_centered(self, grouping_system):
        """ZERO_CENTEREDグループのオペランド分類"""
        # MACD系
        assert grouping_system.get_operand_group("MACD") == OperandGroup.ZERO_CENTERED
        assert grouping_system.get_operand_group("EFI") == OperandGroup.ZERO_CENTERED

    def test_get_operand_group_volume_based(self, grouping_system):
        """VOLUME_BASEDグループのオペランド分類"""
        assert grouping_system.get_operand_group("VOLUME") == OperandGroup.VOLUME_BASED
        assert grouping_system.get_operand_group("OPENINTEREST") == OperandGroup.VOLUME_BASED
        assert grouping_system.get_operand_group("FUNDING_RATE") == OperandGroup.VOLUME_BASED

    def test_get_compatibility_score_perfect_match(self, grouping_system):
        """完全一致するオペランド間の互換性スコア"""
        score = grouping_system.get_compatibility_score("close", "open")
        assert score == 1.0

        score = grouping_system.get_compatibility_score("RSI", "STOCH")
        assert score == 1.0

    def test_get_compatibility_score_moderate_match(self, grouping_system):
        """中程度の互換性を持つオペランド間のスコア"""
        score = grouping_system.get_compatibility_score("close", "RSI")
        assert 0.5 <= score < 1.0

    def test_get_compatibility_score_low_match(self, grouping_system):
        """低い互換性を持つオペランド間のスコア"""
        score = grouping_system.get_compatibility_score("VOLUME", "RSI")
        assert score < 0.5

    def test_get_compatible_operands_with_min_compatibility(self, grouping_system):
        """最小互換性スコアに基づく互換オペランドの取得"""
        available_operands = ["close", "open", "RSI", "VOLUME", "MACD"]

        # 高互換性のみ
        compatible = grouping_system.get_compatible_operands(
            "close", available_operands, min_compatibility=0.8
        )
        assert "open" in compatible
        assert "RSI" not in compatible  # 互換性が低い

        # 中程度互換性も含む
        compatible = grouping_system.get_compatible_operands(
            "close", available_operands, min_compatibility=0.5
        )
        assert "open" in compatible
        assert "RSI" in compatible

    def test_validate_condition_numeric_right_operand(self, grouping_system):
        """数値右オペランドとの条件検証"""
        is_valid, message = grouping_system.validate_condition("close", 100.0)
        assert is_valid is True
        assert message == ""

        is_valid, message = grouping_system.validate_condition("RSI", 70)
        assert is_valid is True

    def test_validate_condition_string_right_operand(self, grouping_system):
        """文字列右オペランドとの条件検証"""
        # 互換性が高い
        is_valid, message = grouping_system.validate_condition("close", "open")
        assert is_valid is True

        # 互換性が低い
        is_valid, message = grouping_system.validate_condition("VOLUME", "RSI")
        assert is_valid is False

    def test_error_handling_invalid_operand(self, grouping_system):
        """無効なオペランドでのエラー処理"""
        # 空文字列
        with patch.object(grouping_system, '_classify_by_pattern', side_effect=Exception("Pattern error")):
            group = grouping_system.get_operand_group("")
            assert group == OperandGroup.PRICE_BASED  # デフォルト

    def test_get_compatibility_score_error_handling(self, grouping_system):
        """互換性スコア計算時のエラー処理"""
        # 両方が無効な場合
        score = grouping_system.get_compatibility_score("", "")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestGlobalInterface:
    """グローバルインターフェース関数のテスト"""

    def test_get_operand_group_global_function(self):
        """グローバルget_operand_group関数のテスト"""
        group = get_operand_group("close")
        assert isinstance(group, OperandGroup)

    def test_get_compatibility_score_global_function(self):
        """グローバルget_compatibility_score関数のテスト"""
        score = get_compatibility_score("close", "open")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_validate_condition_global_function(self):
        """グローバルvalidate_condition関数のテスト"""
        is_valid, message = validate_condition("close", "open")
        assert isinstance(is_valid, bool)
        assert isinstance(message, str)

    def test_operand_grouping_system_singleton(self):
        """operand_grouping_systemがシングルトンであることをテスト"""
        from backend.app.services.auto_strategy.core.operand_grouping import operand_grouping_system
        instance1 = operand_grouping_system
        instance2 = operand_grouping_system
        assert instance1 is instance2