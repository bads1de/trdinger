
import pytest
import random
from unittest.mock import patch

from app.services.auto_strategy.core.genetic_operators import crossover_strategy_genes_pure, mutate_strategy_gene_pure
from app.services.auto_strategy.models.strategy_models import StrategyGene, IndicatorGene, Condition, TPSLGene, PositionSizingGene

# --- ヘルパー関数 --- 
def create_test_strategy_gene(id_suffix, indicator_count=2, period_base=10) -> StrategyGene:
    """テスト用のStrategyGeneオブジェクトを生成する"""
    indicators = [
        IndicatorGene(type='SMA', parameters={'period': period_base + i})
        for i in range(indicator_count)
    ]
    entry_conditions = [Condition(left_operand=f'SMA_{i}', operator='>', right_operand=f'SMA_{i+1}') for i in range(indicator_count-1)]
    exit_conditions = [Condition(left_operand=f'SMA_{i}', operator='<', right_operand=f'SMA_{i+1}') for i in range(indicator_count-1)]
    risk_management = {'position_size': 0.1 * id_suffix}
    tpsl_gene = TPSLGene(enabled=True, stop_loss_pct=0.05 * id_suffix)
    position_sizing_gene = PositionSizingGene(enabled=True, risk_per_trade=0.01 * id_suffix)
    
    return StrategyGene(
        id=f'gene_{id_suffix}',
        indicators=indicators,
        entry_conditions=entry_conditions,
        long_entry_conditions=entry_conditions, # テスト簡略化のため同じものを設定
        short_entry_conditions=exit_conditions, # テスト簡略化のため
        exit_conditions=exit_conditions,
        risk_management=risk_management,
        tpsl_gene=tpsl_gene,
        position_sizing_gene=position_sizing_gene,
        metadata={'source': f'test_{id_suffix}'}
    )

# --- テストクラス --- 
class TestGeneticOperators:

    def test_crossover_strategy_genes_pure(self):
        """交叉ロジックが期待通りに動作することを確認するテスト"""
        # 準備
        parent1 = create_test_strategy_gene(1, indicator_count=3, period_base=10)
        parent2 = create_test_strategy_gene(2, indicator_count=4, period_base=50)

        # 実行
        with patch('random.randint', return_value=2), patch('random.random', return_value=0.4):
             child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        # 検証
        # IDが新しくなっているか
        assert child1.id != parent1.id and child1.id != parent2.id
        assert child2.id != parent1.id and child2.id != parent2.id

        # 指標の交叉 (一点交叉、crossover_point=2)
        assert len(child1.indicators) == 4 # min(3,4) -> 2 + (4-2) = 4
        assert child1.indicators[0].parameters['period'] == 10 # from parent1
        assert child1.indicators[1].parameters['period'] == 11 # from parent1
        assert child1.indicators[2].parameters['period'] == 52 # from parent2
        assert child1.indicators[3].parameters['period'] == 53 # from parent2

        assert len(child2.indicators) == 3 # min(3,4) -> 2 + (3-2) = 3
        assert child2.indicators[0].parameters['period'] == 50 # from parent2
        assert child2.indicators[1].parameters['period'] == 51 # from parent2
        assert child2.indicators[2].parameters['period'] == 12 # from parent1

        # 条件の交叉 (random.random < 0.5 なので parent1 -> child1)
        assert child1.entry_conditions[0].left_operand == 'SMA_0'
        assert child2.entry_conditions[0].left_operand == 'SMA_0' # parent2も同じ構造

        # リスク管理の交叉 (平均値)
        assert child1.risk_management['position_size'] == pytest.approx((0.1 + 0.2) / 2)
        assert child2.risk_management['position_size'] == pytest.approx((0.1 + 0.2) / 2)

        # TP/SL遺伝子の交叉 (専用関数が呼ばれる想定)
        assert child1.tpsl_gene is not None
        assert child2.tpsl_gene is not None
        assert child1.tpsl_gene.stop_loss_pct != parent1.tpsl_gene.stop_loss_pct


    def test_mutate_strategy_gene_pure(self):
        """突然変異ロジックが期待通りに動作することを確認するテスト"""
        # 準備
        original_gene = create_test_strategy_gene(1, indicator_count=3, period_base=20)
        
        # 実行 (mutation_rate=1.0 で全ての要素が変更されるように仕向ける)
        with patch('random.random', return_value=0.05), patch('random.uniform', return_value=1.1):
            mutated_gene = mutate_strategy_gene_pure(original_gene, mutation_rate=0.1)

        # 検証
        # IDが新しくなっているか
        assert mutated_gene.id != original_gene.id
        assert mutated_gene.metadata.get('mutated') is True

        # 指標パラメータの突然変異
        assert mutated_gene.indicators[0].parameters['period'] != original_gene.indicators[0].parameters['period']
        assert mutated_gene.indicators[0].parameters['period'] == int(20 * 1.1)

        # 指標の追加 (低確率、random.random < 0.1 * 0.3)
        # このテストでは発生しないようにrandom.randomの戻り値を設定

        # 条件の突然変異 (低確率、random.random < 0.1 * 0.5)
        # このテストでは発生しないようにrandom.randomの戻り値を設定

        # リスク管理の突然変異
        assert mutated_gene.risk_management['position_size'] != original_gene.risk_management['position_size']
        assert mutated_gene.risk_management['position_size'] == pytest.approx(0.1 * 1.1)

        # TP/SL遺伝子の突然変異
        assert mutated_gene.tpsl_gene.stop_loss_pct != original_gene.tpsl_gene.stop_loss_pct

    def test_crossover_does_not_exceed_max_indicators(self):
        """交叉によって指標が最大数を超えないことを確認するテスト"""
        parent1 = create_test_strategy_gene(1, indicator_count=5)
        parent2 = create_test_strategy_gene(2, indicator_count=5)
        # StrategyGene.MAX_INDICATORS はデフォルトで5

        with patch('random.randint', return_value=3):
            child1, child2 = crossover_strategy_genes_pure(parent1, parent2)

        assert len(child1.indicators) <= 5
        assert len(child2.indicators) <= 5

    def test_mutation_does_not_break_gene_structure(self):
        """突然変異が遺伝子の基本構造を破壊しないことを確認するテスト"""
        # indicator_count=2 にして、entry_conditions が必ず1つ存在するようにする
        original_gene = create_test_strategy_gene(1, indicator_count=2)
        
        # 突然変異を100回実行
        for _ in range(100):
            mutated_gene = mutate_strategy_gene_pure(original_gene, mutation_rate=0.5)
            assert isinstance(mutated_gene, StrategyGene)
            assert len(mutated_gene.indicators) > 0 # 指標が0になることはないはず
            assert len(mutated_gene.indicators) <= 5 # MAX_INDICATORS
            assert isinstance(mutated_gene.indicators[0], IndicatorGene)
            # entry_conditionsが空でないことを確認
            assert len(mutated_gene.entry_conditions) > 0
            assert isinstance(mutated_gene.entry_conditions[0], Condition)
            assert isinstance(mutated_gene.risk_management, dict)
