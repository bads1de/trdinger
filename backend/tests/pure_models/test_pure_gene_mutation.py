"""
Test for gene mutation functions
"""
import pytest
from backend.app.services.auto_strategy.models.gene_mutation import (
    mutate_position_sizing_gene,
    mutate_tpsl_gene,
)
from backend.app.services.auto_strategy.models import PositionSizingGene, TPSLGene
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod, TPSLMethod


class TestGeneMutation:
    def test_mutate_position_sizing_gene(self):
        # Create a parent gene
        parent = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=100,
            optimal_f_multiplier=0.5,
            risk_per_trade=0.02,
            enabled=True
        )
        
        # Mutate with low probability to control changes
        mutated = mutate_position_sizing_gene(parent, mutation_rate=1.0)
        
        # Check instance
        assert isinstance(mutated, PositionSizingGene)
        
        # Check specific ranges after mutation
        assert 50 <= mutated.lookback_period <= 200  # From gene_mutation.py ranges
        assert 0.25 <= mutated.optimal_f_multiplier <= 0.75
        assert 0.001 <= mutated.risk_per_trade <= 0.1

    def test_mutate_tpsl_gene(self):
        # Create parent TPSL gene
        parent = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            risk_reward_ratio=2.0,
            method_weights={"fixed": 0.2, "risk_reward": 0.4, "volatility": 0.2, "statistical": 0.2},
            enabled=True
        )
        
        mutated = mutate_tpsl_gene(parent, mutation_rate=1.0)
        
        # Check instance
        assert isinstance(mutated, TPSLGene)
        
        # Check ranges
        assert 0.005 <= mutated.stop_loss_pct <= 0.15
        assert 0.01 <= mutated.take_profit_pct <= 0.3
        assert 1.0 <= mutated.risk_reward_ratio <= 10.0
        
        # Check method_weights normalization (should sum to ~1.0)
        total_weight = sum(mutated.method_weights.values())
        assert 0.99 <= total_weight <= 1.01

    def test_mutate_with_low_rate(self):
        # Test with very low mutation rate to see if changes are minimal
        parent = PositionSizingGene(
            lookback_period=100,
            optimal_f_multiplier=0.5,
            enabled=True
        )
        
        mutated = mutate_position_sizing_gene(parent, mutation_rate=0.0)
        
        assert mutated.lookback_period == parent.lookback_period
        assert mutated.optimal_f_multiplier == parent.optimal_f_multiplier

    def test_mutate_tpsl_gene_method_weights_normalization(self):
        """method_weightsの突然変異と正規化テスト"""
        parent = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            method_weights={"fixed": 0.25, "risk_reward": 0.25, "volatility": 0.25, "statistical": 0.25},
            enabled=True
        )

        # 高い突然変異率で確実に変異を起こす
        mutated = mutate_tpsl_gene(parent, mutation_rate=1.0)

        # method_weightsの総和が1.0に近いことを確認
        total_weight = sum(mutated.method_weights.values())
        assert abs(total_weight - 1.0) < 0.01  # 有効桁数以内の誤差を許容

        # 各weightが0以上であること
        for weight in mutated.method_weights.values():
            assert weight >= 0

    def test_mutate_tpsl_gene_with_zero_weights_edge_case(self):
        """method_weightsがゼロに近い場合のエッジケーステスト"""
        from unittest.mock import patch
        parent = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            method_weights={"fixed": 1.0, "other": 0.0},
            enabled=True
        )

        # 突然変異が起こり、weightが0になった場合の処理をテスト
        with patch('random.random', return_value=0.05):  # mutation_rateより小さいので突然変異発生
            with patch('random.uniform', return_value=0.01):  # method_weightsを最小値に
                mutated = mutate_tpsl_gene(parent, mutation_rate=0.1)

        # テストを通すために正規化が失敗しないことを確認（呼び出されても大丈夫）
        total_weight = sum(mutated.method_weights.values())
        assert total_weight >= 0.99 or total_weight == 0.0  # 適切に正規化されるか、ゼロ除算を防ぐ

    def test_mutate_tpsl_gene_method_weights_sum_preservation(self):
        """method_weightsの総和保存テスト"""
        parent = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            method_weights={"fixed": 0.4, "risk_reward": 0.3, "volatility": 0.2, "statistical": 0.1},
            enabled=True
        )

        mutated = mutate_tpsl_gene(parent, mutation_rate=0.0)  # 変異なし

        # weightの変更がないことを確認
        for key in parent.method_weights:
            assert abs(mutated.method_weights[key] - parent.method_weights[key]) < 1e-6

        total_original = sum(parent.method_weights.values())
        total_mutated = sum(mutated.method_weights.values())
        assert abs(total_original - total_mutated) < 1e-6