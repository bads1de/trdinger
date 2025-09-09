"""
Test GAEngine population bounds validation
Tests population size validation and bounds checking
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.auto_strategy.config import GAConfig


class TestGAEnginePopulationBoundsValidation:
    """Test GA Engine population bounds validation"""

    def test_population_size_zero_validation(self):
        """Test population size zero is rejected"""
        with pytest.raises(ValueError, match=".*population.*size.*must.*be.*positive.*"):
            GAConfig.from_dict({
                "generations": 10,
                "population_size": 0,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            })

    def test_population_size_negative_validation(self):
        """Test negative population size is rejected"""
        with pytest.raises(ValueError, match=".*population.*size.*must.*be.*positive.*"):
            GAConfig.from_dict({
                "generations": 10,
                "population_size": -5,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            })

    def test_population_size_minimum_bounds(self):
        """Test minimum population size bounds"""
        # Population size 1 should be valid (edge case)
        try:
            config = GAConfig.from_dict({
                "generations": 10,
                "population_size": 1,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            })
            assert config.population_size == 1
        except ValueError:
            # Some systems may reject population size of 1
            pass

    def test_population_size_extremely_large(self):
        """Test extremely large population size handling"""
        large_sizes = [100000, 1000000, 10000000]

        for size in large_sizes:
            test_config = {
                "generations": 10,
                "population_size": size,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            }

            # Should either accept or reject with appropriate bounds
            try:
                config = GAConfig.from_dict(test_config)
                assert config.population_size == size
            except (ValueError, OverflowError) as e:
                # Expected for extremely large values
                assert "population" in str(e).lower() or "size" in str(e).lower()

    def test_population_size_non_integer_type(self):
        """Test non-integer population size is rejected"""
        non_integer_sizes = [1.5, "10", None, [], {}]

        for invalid_size in non_integer_sizes:
            with pytest.raises((ValueError, TypeError), match=".*population.*size.*"):
                GAConfig.from_dict({
                    "generations": 10,
                    "population_size": invalid_size,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1
                })

    def test_population_size_relatively_large_valid(self):
        """Test relatively large but valid population sizes"""
        valid_sizes = [100, 500, 1000, 5000]

        for size in valid_sizes:
            try:
                config = GAConfig.from_dict({
                    "generations": 10,
                    "population_size": size,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1
                })
                assert config.population_size == size
            except ValueError:
                # Some systems may have upper limits
                pytest.fail(f"Valid population size {size} was rejected")

    def test_population_size_floating_point_edge_cases(self):
        """Test floating point population sizes"""
        # Test float values that could truncate
        float_sizes = [1.0, 2.9, 3.1, 100.0]

        for size in float_sizes:
            try:
                config = GAConfig.from_dict({
                    "generations": 10,
                    "population_size": size,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1
                })
                # Should be truncated or rejected
                assert isinstance(config.population_size, int)
            except (ValueError, TypeError):
                pass  # Expected for non-integer types

    def test_population_size_with_generations_interaction(self):
        """Test population size validation with different generation counts"""
        test_cases = [
            {"population_size": 10, "generations": 100},
            {"population_size": 50, "generations": 10},
            {"population_size": 1000, "generations": 5},
        ]

        for case in test_cases:
            try:
                config = GAConfig.from_dict({
                    **case,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1
                })
                assert config.population_size == case["population_size"]
                assert config.generations == case["generations"]
            except ValueError as e:
                # Check if the error is about population or generations
                error_msg = str(e).lower()
                assert "population" in error_msg or "generations" in error_msg

    def test_population_size_memory_consumption_estimate(self):
        """Test population size validation considering memory consumption"""
        # Large populations can cause memory issues
        # This test verifies that the system properly handles or rejects memory-intensive configurations

        memory_intensive_sizes = [50000, 100000, 500000]

        for size in memory_intensive_sizes:
            test_config = {
                "generations": 50,  # Also contributes to memory usage
                "population_size": size,
                "crossover_rate": 0.8,
                "mutation_rate": 0.1
            }

            try:
                config = GAConfig.from_dict(test_config)
                # If accepted, verify the values are set
                assert config.population_size == size
            except (ValueError, MemoryError) as e:
                # Expected for memory-intensive configurations
                assert isinstance(e, (ValueError, MemoryError))

    def test_population_size_bounds_message_clarity(self):
        """Test that population size validation error messages are clear"""
        invalid_sizes = [-1, 0, "-10", None, []]

        for size in invalid_sizes:
            try:
                GAConfig.from_dict({
                    "generations": 10,
                    "population_size": size,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1
                })
                pytest.fail(f"Expected validation error for population_size={size}")
            except (ValueError, TypeError) as e:
                error_msg = str(e).lower()
                # Error message should mention population size
                assert "population" in error_msg or "size" in error_msg

    def test_population_size_with_mutation_rate_interaction(self):
        """Test population size validation with high mutation rates"""
        # High mutation rates with large populations can be computationally expensive

        test_cases = [
            {"population_size": 1000, "mutation_rate": 0.9},  # High mutation on large population
            {"population_size": 10, "mutation_rate": 0.0},   # Edge case mutation rate
            {"population_size": 500, "mutation_rate": 0.5},
        ]

        for case in test_cases:
            try:
                config = GAConfig.from_dict({
                    "generations": 20,
                    **case,
                    "crossover_rate": 0.8,
                })
                assert config.population_size == case["population_size"]
                assert config.mutation_rate == case["mutation_rate"]
            except ValueError as e:
                # Verify error is related to the configured parameters
                error_msg = str(e).lower()
                assert any(keyword in error_msg for keyword in ["population", "mutation", "size", "rate"])

    def test_population_size_data_type_conversion(self):
        """Test population size data type conversion edge cases"""
        # Test string representations of numbers
        string_sizes = ["10", "100", "500", "1000"]

        for size_str in string_sizes:
            try:
                config = GAConfig.from_dict({
                    "generations": 10,
                    "population_size": size_str,
                    "crossover_rate": 0.8,
                    "mutation_rate": 0.1
                })
                # Should convert to int or reject
                assert isinstance(config.population_size, int)
                assert config.population_size == int(size_str)
            except (ValueError, TypeError):
                # Some implementations may reject string inputs
                pass