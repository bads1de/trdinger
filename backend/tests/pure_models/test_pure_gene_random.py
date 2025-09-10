"""
Test for random gene creation functions
"""
import pytest
from backend.app.services.auto_strategy.models.gene_random import (
    create_random_position_sizing_gene,
    create_random_tpsl_gene,
)
from backend.app.services.auto_strategy.models import PositionSizingGene, TPSLGene
from backend.app.services.auto_strategy.models.enums import PositionSizingMethod, TPSLMethod


class TestGeneRandom:
    def test_create_random_position_sizing_gene(self):
        gene = create_random_position_sizing_gene()
        
        assert isinstance(gene, PositionSizingGene)
        assert isinstance(gene.method, PositionSizingMethod)
        
        # Check numeric ranges from gene_random.py
        assert 50 <= gene.lookback_period <= 200
        assert 0.25 <= gene.optimal_f_multiplier <= 0.75
        assert 0.01 <= gene.risk_per_trade <= 0.05  # Adjusted based on code
        assert 0.001 <= gene.atr_multiplier <= 4.0
        assert 0.05 <= gene.fixed_ratio <= 0.3
        assert 0.1 <= gene.fixed_quantity <= 10.0
        assert 0.01 <= gene.min_position_size <= 0.05
        assert 5.0 <= gene.max_position_size <= 50.0
        
        # Check enabled and priority
        assert isinstance(gene.enabled, bool)
        assert 0.5 <= gene.priority <= 1.5

    def test_create_random_tpsl_gene(self):
        gene = create_random_tpsl_gene()
        
        assert isinstance(gene, TPSLGene)
        assert isinstance(gene.method, TPSLMethod)
        
        # Check ranges
        assert 0.01 <= gene.stop_loss_pct <= 0.08
        assert 0.02 <= gene.take_profit_pct <= 0.15
        assert 1.2 <= gene.risk_reward_ratio <= 4.0
        assert 50 <= gene.lookback_period <= 200
        assert 10 <= gene.atr_period <= 30
        assert 0.5 <= gene.confidence_threshold <= 0.9
        
        # Check method_weights is dict with sum ~1.0
        assert isinstance(gene.method_weights, dict)
        total_weight = sum(gene.method_weights.values())
        assert 0.98 <= total_weight <= 1.02  # Allow small float error
        
        assert isinstance(gene.enabled, bool)
        assert 0.5 <= gene.priority <= 1.5

    def test_random_generation_produces_different_results(self):
        # Test that multiple calls produce different results
        genes = [create_random_position_sizing_gene() for _ in range(5)]
        
        # All should be different in some property
        lookback_values = [g.lookback_period for g in genes]
        assert len(set(lookback_values)) > 1  # At least one different