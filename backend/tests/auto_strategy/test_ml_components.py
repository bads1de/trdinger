import pytest
import sys
import os

# backendディレクトリをsys.pathに追加
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, backend_dir)

from app.core.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator
from app.core.services.auto_strategy.models.gene_strategy import IndicatorGene, Condition

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