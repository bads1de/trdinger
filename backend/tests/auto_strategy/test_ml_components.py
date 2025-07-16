import pytest
import sys
import os
import logging

# プロジェクトのルートディレクトリをsys.pathに追加
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, project_root)

from backend.app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from backend.app.core.services.auto_strategy.models.gene_strategy import IndicatorGene, Condition

@pytest.fixture
def smart_condition_generator():
    """SmartConditionGeneratorのフィクスチャ"""
    return SmartConditionGenerator(enable_smart_generation=True)

@pytest.fixture
def sample_indicators():
    """テスト用のIndicatorGeneのリスト"""
    return [
        IndicatorGene(type='RSI', parameters={'period': 14}, enabled=True),
        IndicatorGene(type='SMA', parameters={'period': 50}, enabled=True),
        IndicatorGene(type='BB', parameters={'period': 20}, enabled=True),
    ]

@pytest.fixture
def sample_indicators_with_ml():
    """ML指標を含むテスト用のIndicatorGeneのリスト"""
    return [
        IndicatorGene(type='RSI', parameters={'period': 14}, enabled=True),
        IndicatorGene(type='SMA', parameters={'period': 50}, enabled=True),
        IndicatorGene(type='ML_DOWN_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_UP_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_RANGE_PROB', parameters={},
                    enabled=True),
    ]

@pytest.fixture
def sample_ml_only_indicators():
    """ML指標のみのテスト用のIndicatorGeneのリスト"""
    return [
        IndicatorGene(type='ML_DOWN_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_UP_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_RANGE_PROB', parameters={},
                    enabled=True),
    ]

def test_generate_balanced_conditions(smart_condition_generator, sample_indicators):
    """バランスの取れた条件生成をテストする"""
    long_conds, short_conds, exit_conds = smart_condition_generator.generate_balanced_conditions(sample_indicators)

    assert isinstance(long_conds, list)
    assert isinstance(short_conds, list)
    assert isinstance(exit_conds, list)

    # 条件が少なくとも1つは生成されることを確認
    assert len(long_conds) > 0
    assert len(short_conds) > 0

    print(f"Generated long conditions: {len(long_conds)}")
    print(f"Generated short conditions: {len(short_conds)}")

def test_ml_indicator_integration(smart_condition_generator, sample_indicators_with_ml):
    """ML指標が条件生成に統合されているかをテストする"""
    _, short_conds, _ = smart_condition_generator.generate_balanced_conditions(sample_indicators_with_ml)

    # ML指標に基づくショート条件が生成されているかを確認
    ml_short_condition_found = False
    for cond in short_conds:
        if 'ML_DOWN_PROB' in cond.left_operand and cond.operator == '>' and cond.right_operand > 0.5:
            ml_short_condition_found = True
            break
    
    assert ml_short_condition_found, "ML指標(ML_DOWN_PROB)を考慮したショート条件が見つかりません"
    print(f"ML integration test: Found ML-based short condition in {len(short_conds)} short conditions.")

def test_ml_up_prob_integration(smart_condition_generator, sample_indicators_with_ml):
    """ML指標(ML_UP_PROB)が条件生成に統合されているかをテストする"""
    long_conds, _, _ = smart_condition_generator.generate_balanced_conditions(sample_indicators_with_ml)

    ml_long_condition_found = False
    for cond in long_conds:
        if 'ML_UP_PROB' in cond.left_operand and cond.operator == '>' and cond.right_operand > 0.5:
            ml_long_condition_found = True
            break
    
    assert ml_long_condition_found, "ML指標(ML_UP_PROB)を考慮したロング条件が見つかりません"
    print(f"ML_UP_PROB integration test: Found ML-based long condition in {len(long_conds)} long conditions.")

def test_ml_range_prob_integration(smart_condition_generator, sample_indicators_with_ml, caplog):
    """ML指標(ML_RANGE_PROB)が条件生成に統合されているかをテストする"""
    with caplog.at_level(logging.DEBUG):
        long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(sample_indicators_with_ml)

        ml_range_long_condition_found = False
        for cond in long_conds:
            if 'ML_RANGE_PROB' in cond.left_operand and cond.operator == '<' and cond.right_operand <= 0.3:
                ml_range_long_condition_found = True
                break
        
        ml_range_short_condition_found = False
        for cond in short_conds:
            if 'ML_RANGE_PROB' in cond.left_operand and cond.operator == '<' and cond.right_operand <= 0.3:
                ml_range_short_condition_found = True
                break
        
        assert ml_range_long_condition_found or ml_range_short_condition_found, "ML指標(ML_RANGE_PROB)を考慮した条件が見つかりません"
        print(f"ML_RANGE_PROB integration test: Found ML-based range condition in {len(long_conds)} long conditions and {len(short_conds)} short conditions.")

        # ログの内容を確認
        assert "Added ML_RANGE_PROB < 0.3 to long conditions" in caplog.text or \
               "Added ML_RANGE_PROB < 0.3 to short conditions" in caplog.text

def test_ml_and_technical_indicator_combination(smart_condition_generator):
    """ML指標とテクニカル指標が混在する場合の条件生成をテストする"""
    indicators = [
        IndicatorGene(type='RSI', parameters={'period': 14}, enabled=True),
        IndicatorGene(type='ML_UP_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='SMA', parameters={'period': 50}, enabled=True),
        IndicatorGene(type='ML_DOWN_PROB', parameters={},
                    enabled=True),
    ]
    long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(indicators)

    # ML指標とテクニカル指標の両方が含まれていることを確認
    ml_long_found = any('ML_UP_PROB' in cond.left_operand for cond in long_conds)
    tech_long_found = any('RSI' in cond.left_operand or 'SMA' in cond.left_operand for cond in long_conds)
    ml_short_found = any('ML_DOWN_PROB' in cond.left_operand for cond in short_conds)
    tech_short_found = any('RSI' in cond.left_operand or 'SMA' in cond.left_operand for cond in short_conds)

    assert ml_long_found, "MLロング条件が見つかりません"
    assert tech_long_found, "テクニカルロング条件が見つかりません"
    assert ml_short_found, "MLショート条件が見つかりません"
    assert tech_short_found, "テクニカルショート条件が見つかりません"

    print(f"ML and Technical Combination Test: Long conditions: {len(long_conds)}, Short conditions: {len(short_conds)}")

def test_ml_only_strategy(smart_condition_generator, sample_ml_only_indicators):
    """ML指標のみが与えられた場合の条件生成をテストする"""
    long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(sample_ml_only_indicators)

    # ML指標のみの条件が生成されていることを確認
    ml_long_found = any(cond.left_operand.startswith('ML_') for cond in long_conds)
    ml_short_found = any(cond.left_operand.startswith('ML_') for cond in short_conds)

    assert ml_long_found, "MLのみのロング条件が見つかりません"
    assert ml_short_found, "MLのみのショート条件が見つかりません"

    # テクニカル指標が含まれていないことを確認
    tech_long_not_found = not any(not cond.left_operand.startswith('ML_') for cond in long_conds)
    tech_short_not_found = not any(not cond.left_operand.startswith('ML_') for cond in short_conds)

    assert tech_long_not_found, "MLのみのシナリオでテクニカルロング条件が見つかりました"
    assert tech_short_not_found, "MLのみのシナリオでテクニカルショート条件が見つかりました"

    print(f"ML Only Strategy Test: Long conditions: {len(long_conds)}, Short conditions: {len(short_conds)}")

def test_ml_threshold_conditions(smart_condition_generator, sample_ml_only_indicators):
    """ML指標が閾値に基づいて条件を生成することをテストする"""
    long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(sample_ml_only_indicators)

    # ML_UP_PROB > 0.6 の条件が見つかることを確認
    up_prob_long_found = any(cond.left_operand == 'ML_UP_PROB' and cond.operator == '>' and cond.right_operand == 0.6 for cond in long_conds)
    assert up_prob_long_found, "ML_UP_PROB > 0.6 のロング条件が見つかりません"

    # ML_DOWN_PROB < 0.4 の条件が見つかることを確認
    down_prob_long_found = any(cond.left_operand == 'ML_DOWN_PROB' and cond.operator == '<' and cond.right_operand == 0.4 for cond in long_conds)
    assert down_prob_long_found, "ML_DOWN_PROB < 0.4 のロング条件が見つかりません"

    # ML_DOWN_PROB > 0.6 の条件が見つかることを確認
    down_prob_short_found = any(cond.left_operand == 'ML_DOWN_PROB' and cond.operator == '>' and cond.right_operand == 0.6 for cond in short_conds)
    assert down_prob_short_found, "ML_DOWN_PROB > 0.6 のショート条件が見つかりません"

    # ML_UP_PROB < 0.4 の条件が見つかることを確認
    up_prob_short_found = any(cond.left_operand == 'ML_UP_PROB' and cond.operator == '<' and cond.right_operand == 0.4 for cond in short_conds)
    assert up_prob_short_found, "ML_UP_PROB < 0.4 のショート条件が見つかりません"

    print(f"ML Threshold Conditions Test: Long conditions: {len(long_conds)}, Short conditions: {len(short_conds)}")

def test_ml_combined_conditions(smart_condition_generator, sample_ml_only_indicators):
    """複数のML指標を組み合わせた条件生成をテストする"""
    long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(sample_ml_only_indicators)

    # ML_UP_PROB > ML_DOWN_PROB の条件が見つかることを確認
    combined_long_found = any(cond.left_operand == 'ML_UP_PROB' and cond.operator == '>' and cond.right_operand == 'ML_DOWN_PROB' for cond in long_conds)
    assert combined_long_found, "ML_UP_PROB > ML_DOWN_PROB の組み合わせロング条件が見つかりません"

    # ML_DOWN_PROB > ML_UP_PROB の条件が見つかることを確認 (ショート条件として)
    combined_short_found = any(cond.left_operand == 'ML_DOWN_PROB' and cond.operator == '>' and cond.right_operand == 'ML_UP_PROB' for cond in short_conds)
    assert combined_short_found, "ML_DOWN_PROB > ML_UP_PROB の組み合わせショート条件が見つかりません"

    print(f"ML Combined Conditions Test: Long conditions: {len(long_conds)}, Short conditions: {len(short_conds)}")

def test_ml_indicator_edge_cases(smart_condition_generator):
    """ML指標の極端な値が与えられた場合の条件生成をテストする"""
    # ML_UP_PROB が非常に高い場合
    high_up_prob_indicators = [
        IndicatorGene(type='ML_UP_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_DOWN_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_RANGE_PROB', parameters={},
                    enabled=True),
    ]
    # SmartConditionGeneratorは実際の値ではなく、IndicatorGeneの存在とタイプに基づいて条件を生成するため、
    # ここではテストフィクスチャのML指標の定義を変更するのではなく、
    # 生成される条件が期待される閾値に基づいているかを確認する。
    # 実際のML予測値はバックテスト時に適用されるため、ここでは条件生成ロジックのみをテストする。

    long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(high_up_prob_indicators)

    # ML_UP_PROB > 0.8 の高信頼度条件が生成されることを確認
    high_confidence_long_found = any(cond.left_operand == 'ML_UP_PROB' and cond.operator == '>' and cond.right_operand == 0.8 for cond in long_conds)
    assert high_confidence_long_found, "高信頼度ML_UP_PROBロング条件が見つかりません"

    # ML_DOWN_PROB が非常に低い場合
    low_down_prob_indicators = [
        IndicatorGene(type='ML_UP_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_DOWN_PROB', parameters={},
                    enabled=True),
        IndicatorGene(type='ML_RANGE_PROB', parameters={},
                    enabled=True),
    ]
    long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(low_down_prob_indicators)

    # ML_DOWN_PROB < 0.3 の条件が生成されることを確認
    low_down_prob_long_found = any(cond.left_operand == 'ML_DOWN_PROB' and cond.operator == '<' and cond.right_operand == 0.3 for cond in long_conds)
    assert low_down_prob_long_found, "低ML_DOWN_PROBロング条件が見つかりません"

    print(f"ML Edge Cases Test: Long conditions: {len(long_conds)}, Short conditions: {len(short_conds)}")

def test_ml_with_time_separation_strategy(smart_condition_generator):
    """ML指標が時間軸分離戦略と組み合わされた場合の条件生成をテストする"""
    # 同じML指標を異なる期間として表現（SmartConditionGeneratorは期間を考慮しないが、テストの意図を示す）
    indicators = [
        IndicatorGene(type='ML_UP_PROB', parameters={'period': 10}, enabled=True),
        IndicatorGene(type='ML_UP_PROB', parameters={'period': 50}, enabled=True),
        IndicatorGene(type='ML_DOWN_PROB', parameters={'period': 10}, enabled=True),
        IndicatorGene(type='ML_DOWN_PROB', parameters={'period': 50}, enabled=True),
    ]

    long_conds, short_conds, _ = smart_condition_generator.generate_balanced_conditions(indicators)

    # 時間軸分離戦略が選択された場合、ML指標の条件が生成されることを確認
    # SmartConditionGeneratorの_select_strategy_typeメソッドのロジックにより、
    # 同じ指標が複数ある場合はStrategyType.TIME_SEPARATIONが選択される可能性がある。
    # しかし、ML指標はperiodパラメータを持たないため、実際には異なる指標の組み合わせ戦略が選択される可能性が高い。
    # ここでは、ML指標の条件が生成されることを確認する。

    ml_long_found = any(cond.left_operand.startswith('ML_') for cond in long_conds)
    ml_short_found = any(cond.left_operand.startswith('ML_') for cond in short_conds)

    assert ml_long_found, "時間軸分離戦略でMLロング条件が見つかりません"
    assert ml_short_found, "時間軸分離戦略でMLショート条件が見つかりません"

    print(f"ML with Time Separation Strategy Test: Long conditions: {len(long_conds)}, Short conditions: {len(short_conds)}")