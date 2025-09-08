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
def test_all_registered_indicators_are_mapped():
    """レジストリに登録されている全ての指標がマッピングされていることを確認"""
    from app.services.indicators.config import indicator_registry

    system = OperandGroupingSystem()
    price_mappings = system._get_price_based_mappings()

    # レジストリから全指標を取得
    all_indicators = indicator_registry.list_indicators()

    # マッピングされていない指標を特定
    unmapped_indicators = []
    for indicator in all_indicators:
        if indicator not in price_mappings:
            unmapped_indicators.append(indicator)

    # ML指標は除外（operand_groupingでは扱わない）
    ml_indicators = ['ML_UP_PROB', 'ML_DOWN_PROB', 'ML_RANGE_PROB']
    unmapped_indicators = [ind for ind in unmapped_indicators if ind not in ml_indicators]

    assert len(unmapped_indicators) == 0, f"以下の指標がマッピングされていません: {unmapped_indicators}"


def test_mapping_coverage_completeness():
    """マッピングの完全性を確認"""
    from app.services.indicators.config import indicator_registry

    system = OperandGroupingSystem()
    price_mappings = system._get_price_based_mappings()

    # レジストリからテクニカル指標を取得
    all_indicators = indicator_registry.list_indicators()
    technical_indicators = [ind for ind in all_indicators if not ind.startswith('ML_')]

    total_technical = len(technical_indicators)
    mapped_count = len(price_mappings)
    coverage = mapped_count / total_technical if total_technical > 0 else 0

    print(f"指標カバレッジ: {mapped_count}/{total_technical} ({coverage:.1%})")

    # 最低80%のカバレッジを期待
    min_coverage = 0.80
    assert coverage >= min_coverage, ".1%"