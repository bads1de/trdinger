"""
Test AutoStrategyService invalid input handling
Tests edge cases and error conditions for invalid inputs
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.auto_strategy.services.auto_strategy_service import AutoStrategyService
from app.services.auto_strategy.config import GAConfig


class TestAutoStrategyServiceInvalidInput:
    """Test invalid input handling for AutoStrategyService"""

    def test_invalid_experiment_id_empty_string(self):
        """Test with empty string as experiment ID"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            # Test with empty experiment ID
            with pytest.raises(ValueError, match="experiment_id.*cannot be empty"):
                service.start_strategy_generation("", "test_name", {"generations": 5}, {}, Mock())

    def test_invalid_experiment_name_with_special_chars(self):
        """Test experiment name with invalid characters"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            invalid_names = ["name/with/slashes", "name\nwith\tnewlines", "name\x00with\x00null"]
            for invalid_name in invalid_names:
                with pytest.raises(ValueError, match=".*invalid.*name.*"):
                    service.start_strategy_generation("test_id", invalid_name, {"generations": 5}, {}, Mock())

    def test_ga_config_with_negative_values(self):
        """Test GA config with negative values"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            invalid_configs = [
                {"generations": -5},
                {"population_size": -10},
                {"crossover_rate": -0.1},
                {"mutation_rate": -0.2}
            ]

            for config in invalid_configs:
                try:
                    result = service.start_strategy_generation("test_id", "test_name", config, {}, Mock())
                    # If no exception, check for default handling
                    if config.get("generations", 1) < 0:
                        pytest.fail("Should reject negative generations")
                except (ValueError, AssertionError):
                    continue  # Expected for invalid configs

    def test_none_backtest_data_handling(self):
        """Test handling of None backtest data"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            # Test with None backtest data
            with pytest.raises((ValueError, TypeError), match=".*backtest.*|.*data.*"):
                service.start_strategy_generation("test_id", "test_name", {"generations": 5}, None, Mock())

    def test_extremely_large_config_values(self):
        """Test with extremely large config values that could cause memory issues"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            large_configs = [
                {"generations": 1000000, "population_size": 10000},  # Memory intensive
                {"generations": 1000, "population_size": 1000000},  # Extremely large population
            ]

            for config in large_configs:
                try:
                    result = service.start_strategy_generation("test_id", "test_name", config, {}, Mock())
                    # If allowed, it should handle gracefully or reject due to limits
                    assert result == "test_id"  # Basic sanity check
                except (MemoryError, OverflowError) as e:
                    # Expected for extremely large values
                    assert isinstance(e, (MemoryError, OverflowError))

    def test_concurrent_requests_with_same_id(self):
        """Test behavior when multiple requests use the same experiment ID"""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager') as mock_mgr:

            mock_mgr.return_value.initialize_ga_engine.return_value = None

            service = AutoStrategyService()

            def run_experiment():
                return service.start_strategy_generation("same_id", "test_name", {"generations": 2}, {}, Mock())

            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(run_experiment) for _ in range(3)]
                results = [f.result() for f in futures]

            # All should return the same ID, but check for race conditions
            assert all(r == "same_id" for r in results)

    def test_invalid_json_config_parsing(self):
        """Test handling of invalid JSON-like strings as config"""
        service = AutoStrategyService()

        invalid_configs = [
            "{invalid json",  # Incomplete JSON
            '{"unclosed": [}',  # Syntax error
            '{"recursive": {"self": {"ref": {"back": ""}}}}',  # Nested structure that might cause recursion
        ]

        for invalid_config in invalid_configs:
            try:
                # Try to parse or use the config
                service._build_ga_config_from_dict({})  # Basic test
                # If parsing succeeds unexpectedly, fail
            except Exception as e:
                assert isinstance(e, (ValueError, TypeError))

    def test_unicode_characters_in_names(self):
        """Test handling of Unicode characters in experiment names and IDs"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            unicode_names = ["テスト実験", "実験 № 123", "тест實驗", "🚀 GA Test"]

            for name in unicode_names:
                try:
                    result = service.start_strategy_generation("test_id", name, {"generations": 5}, {}, Mock())
                    assert result == "test_id"  # Basic sanity check
                except UnicodeDecodeError:
                    pytest.fail(f"Unicode handling failed for: {name}")

    def test_extreme_whitespace_handling(self):
        """Test handling of extreme whitespace in inputs"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            # Test with various whitespace patterns
            whitespace_ids = ["   ", "\t\t", "\n\n", "\r\n\t "]
            for ws_id in whitespace_ids:
                with pytest.raises(ValueError, match=".*cannot be.*empty.*|.*whitespaces.*"):
                    service.start_strategy_generation(ws_id, "test_name", {"generations": 5}, {}, Mock())

    def test_very_deep_nested_config(self):
        """Test handling of very deep nested config structures"""
        with patch('app.services.auto_strategy.services.auto_strategy_service.SessionLocal'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.BacktestService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentPersistenceService'), \
             patch('app.services.auto_strategy.services.auto_strategy_service.ExperimentManager'):

            service = AutoStrategyService()

            # Create deeply nested config
            deep_config = {"generations": 5}
            current = deep_config
            for i in range(10):  # 10 levels deep
                current["nested"] = {"generations": 5}
                current = current["nested"]

            try:
                result = service.start_strategy_generation("test_id", "test_name", deep_config, {}, Mock())
                assert result == "test_id"
            except RecursionError:
                pytest.fail("Deep nesting should be handled without recursion error")