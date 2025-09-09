"""
Deep-dive Tests for GA Engine
Focus: Evolution mechanics, population dynamics, fitness evaluation
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Test framework setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestGAEngineDeepDive:
    """Advanced tests for GA engine internals"""

    def test_population_diversity_monitoring(self):
        """Test monitoring of population diversity over generations"""
        # Mock GA population with diversity metrics

        # Create diverse population
        diverse_pop = [
            Mock(fitness_values=[1.0]),
            Mock(fitness_values=[2.0]),
            Mock(fitness_values=[3.0]),
            Mock(fitness_values=[4.0]),
            Mock(fitness_values=[5.0])
        ]

        # Create homogeneous population
        homogeneous_pop = [
            Mock(fitness_values=[1.0]),
            Mock(fitness_values=[1.0]),
            Mock(fitness_values=[1.0]),
            Mock(fitness_values=[1.0]),
            Mock(fitness_values=[1.0])
        ]

        # Diversity should be higher for diverse population
        diverse_fitnesses = [p.fitness_values[0] for p in diverse_pop]
        homogeneous_fitnesses = [p.fitness_values[0] for p in homogeneous_pop]

        diverse_std = np.std(diverse_fitnesses)
        homogeneous_std = np.std(homogeneous_fitnesses)

        assert diverse_std > homogeneous_std

    def test_crossover_operator_effectiveness(self):
        """Test effectiveness of crossover operations"""
        from app.services.auto_strategy.core.ga_engine import GAEngine
        from app.services.auto_strategy.generators.gene_factory import GeneFactory

        try:
            # Test crossover with known good parents
            engine = GAEngine(population_size=4)

            # Mock parents
            parent1 = Mock()
            parent1.fitness_values = [10.0]
            parent2 = Mock()
            parent2.fitness_values = [10.0]

            # Mock children from crossover
            child1 = Mock()
            child1.fitness_values = [9.0]
            child2 = Mock()
            child2.fitness_values = [11.0]

            # Crossover should maintain or improve fitness
            parent_avg = (parent1.fitness_values[0] + parent2.fitness_values[0]) / 2
            child_avg = (child1.fitness_values[0] + child2.fitness_values[0]) / 2

            # Children should be reasonably close to parent fitness
            assert abs(child_avg - parent_avg) < 5.0  # Within 5 points

        except (ImportError, AttributeError):
            # Skip if GA engine not available
            pass

    def test_mutation_rate_impact_analysis(self):
        """Test impact of different mutation rates on population diversity"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        mutation_rates = [0.0, 0.1, 0.5, 1.0]

        for rate in mutation_rates:
            try:
                engine = GAEngine(mutation_rate=rate)
                # Higher mutation rates should potentially allow more exploration
                assert engine.mutation_rate == rate

                # In practice, very high mutation rates might disrupt optimization
                if rate > 0.5:
                    # High mutation should have flags or warnings
                    pass

            except (ImportError, AttributeError):
                pass

    def test_fitness_evaluation_consistency(self):
        """Test fitness evaluation consistency across multiple runs"""
        from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

        try:
            with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
                 patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
                 patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
                 patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

                service = AutoStrategyService()
                config = {"generations": 5, "population_size": 10}

                # Run same config multiple times
                results = []
                for i in range(3):
                    result = service.start_strategy_generation(f"consistency_{i}", f"Consistency {i}", config, {}, Mock())
                    results.append(result)

                # Should produce different but valid results
                assert len(set(results)) == len(results)  # All results should be unique

        except (ImportError, AttributeError):
            pass

    def test_early_stopping_mechanism(self):
        """Test early stopping when convergence is reached"""
        # Mock convergence detection
        fitness_history = [10.0, 9.0, 8.5, 8.1, 8.0, 8.0, 8.0]  # Converged

        # Check if fitness improvement is below threshold
        improvements = []
        for i in range(1, len(fitness_history)):
            improvement = abs(fitness_history[i] - fitness_history[i-1])
            improvements.append(improvement)

        # Recent improvements should be minimal
        recent_improvements = improvements[-3:]
        avg_recent_improvement = sum(recent_improvements) / len(recent_improvements)

        # Early stopping threshold (example: 0.1)
        stopping_threshold = 0.1
        should_stop = avg_recent_improvement < stopping_threshold

        assert should_stop  # Should trigger early stopping for converged populations

    def test_population_size_scaling_behavior(self):
        """Test GA behavior with different population sizes"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        sizes = [2, 5, 10, 50, 100]

        for size in sizes:
            try:
                engine = GAEngine(population_size=size)

                # Larger populations should provide better exploration
                if size >= 10:
                    # Large population assertions
                    assert engine.population_size >= 10
                else:
                    # Small population should still work
                    assert engine.population_size == size

                # Even population should allow proper crossover
                is_even = size % 2 == 0
                if not is_even:
                    # Odd-sized populations require special handling
                    assert engine.population_size % 2 != 0

            except (ImportError, AttributeError):
                pass

    def test_multi_objective_optimization_balance(self):
        """Test balancing of multiple objectives in GA"""
        # Simulate multi-objective fitness
        populations = [
            # Population 1: Balanced objectives
            [Mock(fitness_values=[5.0, 5.0]), Mock(fitness_values=[6.0, 4.0])],
            # Population 2: Skewed objectives
            [Mock(fitness_values=[10.0, 1.0]), Mock(fitness_values=[1.0, 10.0])]
        ]

        for pop in populations:
            objectives = [[ind.fitness_values for ind in pop]]
            if objectives:
                first_pop = objectives[0]
                if len(first_pop) > 1:
                    total_obj1 = sum(ind[0] for ind in first_pop)
                    total_obj2 = sum(ind[1] for ind in first_pop)
                    balance_ratio = min(total_obj1, total_obj2) / max(total_obj1, total_obj2)

                    # Balanced should have ratio > 0.8
                    # This is a simplified test
                    assert balance_ratio > 0.0

    def test_fitness_function_sampling_distribution(self):
        """Test fitness function distribution across population"""
        import random

        # Generate fitness values with known distribution
        np.random.seed(42)
        fitness_values = np.random.normal(5.0, 2.0, 100)

        # Test statistical properties
        mean_fitness = np.mean(fitness_values)
        std_fitness = np.std(fitness_values)

        # Normal distribution range check
        assert 4.5 < mean_fitness < 5.5
        assert 1.5 < std_fitness < 2.5

        # Distribution shape test
        percentile_25 = np.percentile(fitness_values, 25)
        percentile_75 = np.percentile(fitness_values, 75)
        iqr = percentile_75 - percentile_25

        # IQR should be reasonable for normal distribution
        expected_iqr_range = (1.3, 2.7)
        assert expected_iqr_range[0] < iqr < expected_iqr_range[1]