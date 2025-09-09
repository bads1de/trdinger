"""
定数importテスト
"""
import pytest

# 定数のimportテスト
def test_constants_import():
    """定数が正しくimportできることをテスト"""
    from app.services.auto_strategy.constants import (
        DEFAULT_SYMBOL,
        IndicatorType,
        StrategyType,
        OPERATORS,
        DATA_SOURCES,
    )

    assert DEFAULT_SYMBOL == "BTC/USDT:USDT"
    assert isinstance(OPERATORS, list)
    assert isinstance(DATA_SOURCES, list)
    assert IndicatorType.MOMENTUM.value == "momentum"

def test_generators_import():
    """ジェネレーターモジュールのimportテスト"""
    from app.services.auto_strategy.generators.random_gene_generator import RandomGeneGenerator
    from app.services.auto_strategy.generators.condition_generator import ConditionGenerator

    assert RandomGeneGenerator is not None
    assert ConditionGenerator is not None
    assert hasattr(RandomGeneGenerator, 'generate_random_gene')
    assert hasattr(ConditionGenerator, 'generate_balanced_conditions')

def test_services_import():
    """サービスモジュールのimportテスト"""
    from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

    assert AutoStrategyService is not None
    assert hasattr(AutoStrategyService, 'start_strategy_generation')
    assert hasattr(AutoStrategyService, 'list_experiments')

def test_models_import():
    """モデルモジュールのimportテスト"""
    from app.services.auto_strategy.models.strategy_models import (
        StrategyGene,
        IndicatorGene,
        Condition,
        PositionSizingGene,
        TPSLGene
    )

    assert StrategyGene is not None
    assert IndicatorGene is not None

def test_core_import():
    """コアモジュールのimportテスト"""
    from app.services.auto_strategy.core.operand_grouping import operand_grouping_system

    assert operand_grouping_system is not None
    assert hasattr(operand_grouping_system, 'get_compatibility_score')