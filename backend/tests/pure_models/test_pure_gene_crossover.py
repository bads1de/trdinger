"""
Test for gene crossover functions
"""
import pytest
from backend.app.services.auto_strategy.models.gene_crossover import (
    crossover_position_sizing_genes,
    crossover_tpsl_genes,
)
from backend.app.services.auto_strategy.models import PositionSizingGene, TPSLGene
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod, TPSLMethod


class TestGeneCrossover:
    def test_crossover_position_sizing_genes(self):
        parent1 = PositionSizingGene(
            method=PositionSizingMethod.VOLATILITY_BASED,
            lookback_period=100,
            optimal_f_multiplier=0.5,
            enabled=True
        )
        parent2 = PositionSizingGene(
            method=PositionSizingMethod.FIXED_RATIO,
            lookback_period=150,
            optimal_f_multiplier=0.7,
            enabled=True
        )
        
        child1, child2 = crossover_position_sizing_genes(parent1, parent2)
        
        # Check that returned genes are valid PositionSizingGene instances
        assert isinstance(child1, PositionSizingGene)
        assert isinstance(child2, PositionSizingGene)
        
        # Check that crossovers have mixed properties
        # Assuming crossover mixes properties, but since we don't know the exact implementation,
        # just check they are different from parents or valid
        assert child1.method in [PositionSizingMethod.VOLATILITY_BASED, PositionSizingMethod.FIXED_RATIO]
        assert child2.method in [PositionSizingMethod.VOLATILITY_BASED, PositionSizingMethod.FIXED_RATIO]

    def test_crossover_tpsl_genes(self):
        parent1 = TPSLGene(
            method=TPSLMethod.RISK_REWARD_RATIO,
            stop_loss_pct=0.03,
            method_weights={"fixed": 0.2, "risk_reward": 0.4, "volatility": 0.2, "statistical": 0.2},
            enabled=True
        )
        parent2 = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            stop_loss_pct=0.05,
            method_weights={"fixed": 0.3, "risk_reward": 0.3, "volatility": 0.3, "statistical": 0.1},
            enabled=True
        )
        
        child1, child2 = crossover_tpsl_genes(parent1, parent2)
        
        # Check instances
        assert isinstance(child1, TPSLGene)
        assert isinstance(child2, TPSLGene)
        
        # Check method is valid enum
        assert isinstance(child1.method, TPSLMethod)
        assert isinstance(child2.method, TPSLMethod)

        # Check method_weights crossover
        # Both parents should have all keys
        expected_keys = set(parent1.method_weights.keys()) | set(parent2.method_weights.keys())
        assert set(child1.method_weights.keys()) == expected_keys
        assert set(child2.method_weights.keys()) == expected_keys

        # For keys in both parents, values should be averages (allow tolerance for potential shared reference bug)
        for key in expected_keys:
            if key in parent1.method_weights and key in parent2.method_weights:
                actual_avg = (parent1.method_weights[key] + parent2.method_weights[key]) / 2
                assert abs(child1.method_weights[key] - actual_avg) < 1e-6  # Exact equality after bug fix
                assert abs(child2.method_weights[key] - actual_avg) < 1e-6  # Exact equality after bug fix
            # For keys in only one parent, both children should inherit that value
            elif key in parent1.method_weights:
                assert child1.method_weights[key] == parent1.method_weights[key]
                assert child2.method_weights[key] == parent1.method_weights[key]
            else:  # key only in parent2
                assert child1.method_weights[key] == parent2.method_weights[key]
                assert child2.method_weights[key] == parent2.method_weights[key]

    def test_crossover_tpsl_genes_with_partial_method_weights(self):
        """部分的なmethod_weightsでの交叉テスト"""
        parent1 = TPSLGene(
            method=TPSLMethod.FIXED_PERCENTAGE,
            method_weights={"fixed": 0.5, "risk_reward": 0.5},  # partial keys
            enabled=True
        )
        parent2 = TPSLGene(
            method=TPSLMethod.STATISTICAL,
            method_weights={"statistical": 0.3, "volatility": 0.7},  # different keys
            enabled=True
        )

        child1, child2 = crossover_tpsl_genes(parent1, parent2)

        # Check that children have all keys from both parents
        all_keys = set(parent1.method_weights.keys()) | set(parent2.method_weights.keys())
        assert set(child1.method_weights.keys()) == all_keys
        assert set(child2.method_weights.keys()) == all_keys
