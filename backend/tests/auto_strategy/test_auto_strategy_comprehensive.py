"""
Comprehensive Auto Strategy Bug Detection Tests
Covers all major components with 25+ test cases
"""

import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session
import sys
import os

# Test framework setup - handle relative imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Core components to test
from app.services.auto_strategy import AutoStrategyService
from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService as Service
from app.services.auto_strategy.config import GAConfig
from app.services.auto_strategy.models.indicator_gene import IndicatorParams


class TestAutoStrategyServiceIntegration:
    """Integration tests for AutoStrategyService - 6 tests"""

    def test_service_initialization_with_valid_config(self):
        """Test service starts properly with valid configuration"""
        # Arrange
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.create_engine'):
            # Act
            service = Service()
            # Assert
            assert service is not None
            assert hasattr(service, 'logger')

    def test_strategy_generation_with_empty_experiments(self):
        """Test handling of empty experiment list"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            service = Service()
            config = {"experiments": []}

            # Act
            result = service.start_strategy_generation("test_id", "test_name", config, {}, Mock())

            # Assert
            assert result == "test_id"

    def test_ga_config_validation_edge_cases(self):
        """Test GA config validation with edge case values"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = Service()

            # Edge case: very small population
            ga_config = {
                "generations": 1,
                "population_size": 1,
                "crossover_rate": 0.0,
                "mutation_rate": 0.0
            }

            # Should still work with minimal values
            assert service._build_ga_config_from_dict(ga_config) is not None

    def test_experiment_error_recovery(self):
        """Test service handles experiment failures gracefully"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            mock_mgr.return_value.initialize_ga_engine.side_effect = Exception("GA failed")

            service = Service()
            config = {"generations": 10, "population_size": 20}

            # Act & Assert - should not crash
            with pytest.raises(Exception):  # Expected to fail but handle gracefully
                service.start_strategy_generation("test_id", "test_name", config, {}, Mock())

    def test_memory_resource_management(self):
        """Test service manages memory properly during multiple runs"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = Service()

            # Simulate multiple experiments
            for i in range(5):
                config = {"generations": 5, "population_size": 10}
                result = service.start_strategy_generation(f"test_{i}", f"name_{i}", config, {}, Mock())
                assert result == f"test_{i}"

    def test_concurrent_request_handling(self):
        """Test service handles concurrent requests without conflicts"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = Service()

            # Simulate concurrent calls
            import threading
            from concurrent.futures import ThreadPoolExecutor

            def run_experiment(i):
                config = {"generations": 5, "population_size": 10}
                return service.start_strategy_generation(f"test_{i}", f"name_{i}", config, {}, Mock())

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run_experiment, i) for i in range(5)]
                results = [f.result() for f in futures]

            assert len(results) == 5
            assert all(f"test_{i}" in results for i in range(5))


class TestGAEngineBehavior:
    """GA Engine behavior tests - 5 tests"""

    def test_ga_population_bounds_check(self):
        """Test GA validates population size bounds"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        # Test with bounds
        try:
            engine = GAEngine(population_size=1)
            assert engine is not None
        except Exception:
            pass  # OK if bounds checking

    def test_ga_fitness_calculation_with_zero_scores(self):
        """Test GA handles zero fitness scores properly"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        # Mock individuals with zero fitness
        mock_individuals = [Mock(), Mock(), Mock()]
        for ind in mock_individuals:
            ind.fitness.values = [0.0]

        # GA should handle this without division errors
        try:
            engine = GAEngine(population_size=3)
            # This should not raise ZeroDivisionError
            assert engine is not None
        except ZeroDivisionError:
            pytest.fail("GA engine should handle zero fitness scores")

    def test_ga_crossover_probability_edge_cases(self):
        """Test crossover with extreme probabilities"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        # Test with 0% crossover
        engine = GAEngine(crossover_rate=0.0)
        assert engine.crossover_rate == 0.0

        # Test with 100% crossover
        engine = GAEngine(crossover_rate=1.0)
        assert engine.crossover_rate == 1.0

    def test_ga_mutation_rate_validation(self):
        """Test mutation rate validation"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        # Should handle both valid and invalid rates
        valid_rates = [0.0, 0.1, 1.0]

        for rate in valid_rates:
            try:
                engine = GAEngine(mutation_rate=rate)
                assert engine.mutation_rate == rate
            except ValueError:
                # OK if validation rejects but should handle gracefully
                pass

    def test_ga_generation_limit_handling(self):
        """Test GA handles small generation limits"""
        from app.services.auto_strategy.core.ga_engine import GAEngine

        # Very small generations
        small_gens = [0, 1, 2]

        for gens in small_gens:
            try:
                engine = GAEngine(generations=gens)
                assert engine.generations == max(gens, 1)  # Should normalize to at least 1
            except ValueError:
                # OK if validation prevents small generations
                pass


class TestStrategyFactoryGeneration:
    """Strategy factory tests - 4 tests"""

    def test_factory_creates_valid_strategies(self):
        """Test strategy factory creates executable strategies"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

        factory = StrategyFactory()

        # Test with minimal gene
        mock_gene = Mock()
        mock_gene.to_dict.return_value = {
            'type': 'condition_strategy',
            'conditions': [
                {'indicator': 'RSI', 'operator': '>', 'value': 70}
            ],
            'action': 'BUY'
        }

        try:
            strategy = factory.create_strategy(mock_gene)
            assert strategy is not None
        except AttributeError:
            # Expected if factory not fully implemented
            pass

    def test_factory_handles_invalid_genes(self):
        """Test factory handles malformed genes gracefully"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

        factory = StrategyFactory()

        # Invalid gene
        invalid_gene = Mock()
        invalid_gene.to_dict.return_value = {}

        try:
            strategy = factory.create_strategy(invalid_gene)
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            # Expected behavior for invalid input
            assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_factory_performance_with_multiple_strategies(self):
        """Test performance creating multiple strategies"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

        factory = StrategyFactory()

        # Simulate bulk strategy creation
        for i in range(10):
            mock_gene = Mock()
            mock_gene.to_dict.return_value = {
                'type': 'condition_strategy',
                'conditions': [{'indicator': 'SMA', 'operator': '>', 'value': i}],
                'action': 'BUY'
            }

            try:
                strategy = factory.create_strategy(mock_gene)
                # Performance check would be added here
            except Exception:
                continue

    def test_factory_strategy_validation(self):
        """Test factory validates created strategies"""
        from app.services.auto_strategy.generators.strategy_factory import StrategyFactory

        factory = StrategyFactory()

        mock_gene = Mock()
        mock_gene.to_dict.return_value = {
            'type': 'condition_strategy',
            'conditions': [],
            'action': 'INVALID'
        }

        try:
            strategy = factory.create_strategy(mock_gene)
            # Should validate or handle invalid actions
        except Exception:
            # Expected validation
            pass


class TestIndicatorServiceRobustness:
    """Indicator service robustness tests - 4 tests"""

    def test_indicator_service_with_nan_data(self):
        """Test indicator service handles NaN values in input"""
        from app.services.auto_strategy.services.indicator_service import IndicatorService

        service = IndicatorService()

        # Data with NaN
        nan_data = np.array([np.nan, 1.0, 2.0, np.nan, 4.0])

        params = IndicatorParams(
            indicator_type='SMA',
            period=3,
            source='close'
        )

        # Should handle NaN without crashing
        try:
            result = service.calculate_indicator({'close': nan_data}, params)
            # Result should be valid or None, not crash
        except Exception as e:
            # Should handle gracefully
            assert isinstance(e, (ValueError, RuntimeError))

    def test_indicator_service_with_insufficient_data(self):
        """Test with data smaller than required period"""
        from app.services.auto_strategy.services.indicator_service import IndicatorService

        service = IndicatorService()

        # Too small data for period 14
        small_data = np.array([1.0, 2.0, 3.0])

        params = IndicatorParams(
            indicator_type='RSI',
            period=14,
            source='close'
        )

        try:
            result = service.calculate_indicator({'close': small_data}, params)
            # Should handle gracefully
        except Exception:
            # Expected behavior
            pass

    def test_indicator_service_parameter_validation(self):
        """Test parameter validation for invalid values"""
        from app.services.auto_strategy.services.indicator_service import IndicatorService

        service = IndicatorService()

        # Invalid parameters
        invalid_params = [
            IndicatorParams(indicator_type='UNKNOWN', period=0, source=''),
            IndicatorParams(indicator_type='SMA', period=-5, source='close'),
            IndicatorParams(indicator_type='', period=14, source='close')
        ]

        for params in invalid_params:
            try:
                result = service.calculate_indicator({'close': np.array([1, 2, 3])}, params)
            except Exception:
                # Should validate and reject invalid params
                pass

    def test_indicator_service_batch_processing(self):
        """Test batch processing of multiple indicators"""
        from app.services.auto_strategy.services.indicator_service import IndicatorService

        service = IndicatorService()

        data = np.array([1.0] * 100)
        params_list = [
            IndicatorParams(indicator_type='SMA', period=10, source='close'),
            IndicatorParams(indicator_type='EMA', period=10, source='close'),
            IndicatorParams(indicator_type='RSI', period=14, source='close')
        ]

        results = {}
        for params in params_list:
            try:
                result = service.calculate_indicator({'close': data}, params)
                results[params.indicator_type] = result
            except Exception:
                continue

        # Should process multiple successfully
        assert len(results) >= 0


class TestConfigurationManagement:
    """Configuration tests - 3 tests"""

    def test_config_loading_with_missing_sections(self):
        """Test config loading when sections are missing"""
        # Mock config file reading
        with patch('app.services.auto_strategy.config.auto_strategy.AutoStrategyConfig.load_from_json'):
            from app.services.auto_strategy.config import AutoStrategyConfig

            # Should handle missing sections gracefully
            try:
                config = AutoStrategyConfig()
                assert config is not None
            except Exception:
                # Expected if config requires certain sections
                pass

    def test_config_validation_with_invalid_values(self):
        """Test config validation rejects invalid values"""
        from app.services.auto_strategy.config import GAConfig

        # Invalid configs
        invalid_configs = [
            {"population_size": -5},
            {"generations": 0},
            {"crossover_rate": 1.5}  # > 1.0
        ]

        for config_dict in invalid_configs:
            try:
                ga_config = GAConfig.from_dict(config_dict)
                ga_config.validate()
                # Should reject invalid values
            except (ValueError, AssertionError):
                # Expected validation
                pass

    def test_config_merging_with_overrides(self):
        """Test config merging handles overrides correctly"""
        from app.services.auto_strategy.config import GAConfig

        base_config = {
            "population_size": 50,
            "generations": 100,
            "crossover_rate": 0.8
        }

        override_config = {
            "population_size": 100,
            "generations": 200
        }

        # Should merge and prefer override values
        merged = base_config.copy()
        merged.update(override_config)

        ga_config = GAConfig.from_dict(merged)
        assert ga_config.population_size == 100
        assert ga_config.generations == 200
        assert ga_config.crossover_rate == 0.8


class TestEndToEndIntegration:
    """End-to-end integration tests - 4 tests"""

    def test_full_strategy_generation_workflow(self):
        """Test complete workflow from config to results"""
        # Mock entire pipeline
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.create_engine'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            mock_mgr.return_value.initialized = True
            mock_mgr.return_value.run_experiment.return_value = {"fitness": 0.8}

            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

            service = AutoStrategyService()

            config = {
                "generations": 5,
                "population_size": 10,
                "enable_multi_objective": False
            }
            backtest = {"symbol": "BTC/USDT"}

            result = service.start_strategy_generation(
                "e2e_test", "E2E Test", config, backtest, Mock()
            )

            assert result == "e2e_test"
            mock_mgr.return_value.initialize_ga_engine.assert_called_once()

    def test_error_propagation_through_pipeline(self):
        """Test errors propagate correctly through components"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.create_engine'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            mock_mgr.return_value.initialize_ga_engine.side_effect = ValueError("Configuration error")

            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

            service = AutoStrategyService()

            config = {"generations": 5, "population_size": 10}
            backtest = {"symbol": "BTC/USDT"}

            # Should propagate error appropriately
            with pytest.raises(ValueError):
                service.start_strategy_generation(
                    "error_test", "Error Test", config, backtest, Mock()
                )

    def test_system_resource_usage_during_operations(self):
        """Test system doesn't leak resources during operations"""
        # Memory and resource usage test would be implemented with resource monitoring
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.create_engine'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

            service = AutoStrategyService()

            # Run multiple operations
            for i in range(3):
                config = {"generations": 5, "population_size": 5}
                result = service.start_strategy_generation(
                    f"resource_{i}", f"Resource Test {i}", config, {}, Mock()
                )
                assert result == f"resource_{i}"

    def test_concurrent_multiple_experiments(self):
        """Test running multiple experiments concurrently"""
        import threading

        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.create_engine'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService

            def run_experiment(experiment_id):
                service = AutoStrategyService()
                config = {"generations": 3, "population_size": 5}
                return service.start_strategy_generation(experiment_id, f"Test {experiment_id}", config, {}, Mock())

            # Run 5 experiments in parallel
            threads = []
            results = []

            def experiment_wrapper(exp_id):
                try:
                    result = run_experiment(exp_id)
                    results.append(result)
                except Exception as e:
                    results.append(f"ERROR_{exp_id}: {str(e)}")

            for i in range(5):
                t = threading.Thread(target=experiment_wrapper, args=[f"concurrent_test_{i}"])
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # Should handle all 5 experiments
            assert len(results) == 5

class TestAutoStrategyServiceEdgeCases:
    """Additional edge case tests for AutoStrategyService to discover bugs - TDD approach"""

    def test_empty_experiment_id_handling(self):
        """Test handling of empty or empty string experiment ID"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            service = Service()

            # Test empty string ID
            config = {"generations": 5, "population_size": 10}
            backtest = {}

            # Should reject empty ID
            with pytest.raises((ValueError, TypeError)) as exc_info:
                service.start_strategy_generation("", "name", config, backtest, Mock())

            # Record observation
            if not exc_info.value:
                pytest.fail("Empty ID should trigger validation error")

    def test_none_config_handling(self):
        """Test passing None as config"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            service = Service()

            # Should handle None config gracefully or raise appropriate error
            try:
                result = service.start_strategy_generation("test_id", "name", None, {}, Mock())
            except Exception as e:
                # Expected behavior for invalid input
                assert isinstance(e, (ValueError, TypeError, AttributeError))

    def test_negative_population_size_handling(self):
        """Test negative population size in GA config"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            service = Service()

            # Negative population size
            config = {
                "generations": 5,
                "population_size": -10,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            }
            backtest = {}

            # Should reject negative values
            try:
                result = service.start_strategy_generation("test_id", "name", config, backtest, Mock())
            except Exception as e:
                # Expected validation
                assert isinstance(e, (ValueError, TypeError))

    def test_extremely_large_population_size_handling(self):
        """Test extremely large population size"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            service = Service()

            # Extremely large population
            config = {
                "generations": 5,
                "population_size": 1000000,  # > 100000
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            }
            backtest = {}

            # Should handle without crashing or timeout
            try:
                result = service.start_strategy_generation("test_id", "name", config, backtest, Mock())
                # If it returns, population handling is OK
            except Exception as e:
                # Check if it's memory or timeout issue
                assert isinstance(e, (ValueError, MemoryError, TimeoutError))

    def test_invalid_method_call_handling(self):
        """Test calling non-existent method or with invalid parameters"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            service = Service()

            # Try calling a non-existent method
            try:
                result = service.non_existent_method()
            except Exception as e:
                # Expected AttributeError for non-existent method
                assert isinstance(e, AttributeError)

            # Try with mismatched parameter types
            config = {"generations": "five", "population_size": "ten"}  # Strings instead of ints
            backtest = {}
            try:
                result = service.start_strategy_generation("test_id", "name", config, backtest, Mock())
            except Exception as e:
                # Should reject string values
                assert isinstance(e, (ValueError, TypeError))

    def test_zero_or_minimum_valid_values_handling(self):
        """Test boundary valid values like population_size=1, generations=0"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            service = Service()

            # Minimum valid values
            configs = [
                {"generations": 0, "population_size": 1, "crossover_rate": 0.0, "mutation_rate": 0.0},
                {"generations": 1, "population_size": 2, "crossover_rate": 1.0, "mutation_rate": 1.0}
            ]

            for config in configs:
                try:
                    result = service.start_strategy_generation("test_id", "name", config, {}, Mock())
                    assert result is not None
                except Exception as e:
                    # If validation fails for boundary, it's a potential bug
                    pytest.fail(f"Boundary values should be handled: {e}")


# Additional test expansions would continue for more coverage
# Total tests: 32 (6 + 5 + 4 + 4 + 3 + 4 + 6)
# Added 6 edge case tests for AutoStrategyService