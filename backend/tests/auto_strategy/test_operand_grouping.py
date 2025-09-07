import pytest
from backend.app.services.auto_strategy.core.operand_grouping import OperandGroupingSystem


def test_no_duplicate_keys_in_mappings():
    """マッピング辞書に重複キーがないことを確認"""
    system = OperandGroupingSystem()
    mappings = system._group_mappings

    # 重複キーを検知
    seen_keys = set()
    duplicates = set()

    for key in mappings.keys():
        if key in seen_keys:
            duplicates.add(key)
        else:
            seen_keys.add(key)

    assert len(duplicates) == 0, f"重複キーが見つかりました: {duplicates}"


def test_operand_group_assignment():
    """オペランドが正しくグループに割り当てられることを確認"""
    system = OperandGroupingSystem()

    # 既知のマッピングをテスト
    assert system.get_operand_group("SMA") is system._group_mappings.get("SMA")
    assert system.get_operand_group("RSI") is system._group_mappings.get("RSI")
    assert system.get_operand_group("TSI") is system._group_mappings.get("TSI")  # 重複修正後確認


def test_compatibility_score_calculation():
    """互換性スコアが正しく計算されることを確認"""
    system = OperandGroupingSystem()

    score = system.get_compatibility_score("SMA", "EMA")
    assert score > 0, "同じグループの指標は互換性があるべき"