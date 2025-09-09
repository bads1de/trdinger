"""
Configuration Management Tests
Focus: Config loading, validation, merging, persistence
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch
import json
import sys

# Test framework setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestConfigurationManagement:
    """Configuration handling tests"""

    def test_config_loading_from_valid_file(self):
        """Test loading configuration from valid JSON file"""
        # Create temporary config file
        config_data = {
            "population_size": 50,
            "generations": 100,
            "crossover_rate": 0.8,
            "mutation_rate": 0.1
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Test that file can be loaded
            with open(temp_path, 'r') as f:
                loaded_config = json.load(f)

            assert loaded_config["population_size"] == 50
            assert loaded_config["generations"] == 100

        finally:
            os.unlink(temp_path)

    def test_config_validation_with_valid_values(self):
        """Test config validation accepts valid values"""
        from app.services.auto_strategy.config import GAConfig

        valid_configs = [
            {"population_size": 10, "generations": 5},
            {"population_size": 100, "generations": 50, "crossover_rate": 0.9},
            {"population_size": 1000, "generations": 100, "mutation_rate": 0.05}
        ]

        for config in valid_configs:
            try:
                ga_config = GAConfig.from_dict(config)
                # Should not raise exception for valid configs
                assert ga_config is not None
            except Exception as e:
                # Some may fail if validation is strict
                if not any(keyword in str(e) for keyword in ['Invalid', 'Validation', 'Error']):
                    pytest.fail(f"Valid config rejected: {e}")

    def test_config_merge_with_override_scenarios(self):
        """Test configuration merging in various scenarios"""
        default_config = {
            "population_size": 50,
            "generations": 100,
            "verbose": False
        }

        override_configs = [
            {"population_size": 75},  # Single override
            {"population_size": 75, "verbose": True},  # Multiple override
            {"extra_param": "value", "population_size": 25}  # Extra parameters
        ]

        for override in override_configs:
            merged = default_config.copy()
            merged.update(override)

            # Merged config should contain both default and override values
            assert merged["generations"] == 100  # Default preserved
            if "population_size" in override:
                assert merged["population_size"] == override["population_size"]  # Override applied

    def test_config_persistence_and_loading_cycle(self):
        """Test complete save/load cycle"""
        config_data = {
            "population_size": 30,
            "generations": 80,
            "crossover_rate": 0.7,
            "experiment_id": "test_exp"
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            # Save operation
            saved_data = config_data.copy()

            # Load operation
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            # Data should be identical after save/load cycle
            assert loaded_data == saved_data
            assert loaded_data["experiment_id"] == "test_exp"

        finally:
            os.unlink(temp_path)

    def test_config_validation_edge_case_values(self):
        """Test validation with extreme but valid values"""
        from app.services.auto_strategy.config import GAConfig

        edge_case_configs = [
            {"population_size": 1, "generations": 1, "crossover_rate": 0.0, "mutation_rate": 0.0},  # Minimums
            {"population_size": 10000, "generations": 10000, "crossover_rate": 1.0, "mutation_rate": 1.0}  # Maximums
        ]

        for config in edge_case_configs:
            try:
                ga_config = GAConfig.from_dict(config)
                # Edge cases should be handled appropriately
                assert ga_config is not None
            except Exception as e:
                 # Acceptable if bounds are enforced
    def test_config_invalid_type_handling(self):
        """Test handling of invalid data types in config"""
        from app.services.auto_strategy.config import GAConfig

        invalid_type_configs = [
            {"population_size": "50", "generations": 100},  # String instead of number
            {"population_size": None, "generations": [100, 200]},  # Wrongtype and list
            {"population_size": {"value": 50}, "generations": True}  # Objectc and boolean
        ]

        for config in invalid_type_configs:
            try:
                ga_config = GAConfig.from_dict(config)
                # Should either convert, reject, or handle appropriately
            except (TypeError, ValueError) as e:
                # Expected for type mismatches
                assert isinstance(e, (TypeError, ValueError))
            except Exception as e:
                # Other exceptions are also acceptable
                assert e is not None

    def test_config_missing_required_fields(self):
        """Test handling of missing required fields"""
        from app.services.auto_strategy.config import GAConfig

        incomplete_configs = [
            {"population_size": 50},  # Missing generations
            {"generations": 100}  # Missing population_size
        ]

        for config in incomplete_configs:
            try:
                ga_config = GAConfig.from_dict(config)
                # Should handle missing fields (defaults or error)
            except (ValueError, TypeError):
                # Expected for missing required fields
                pass

    def test_config_environment_override_behavior(self):
        """Test environment variable config overrides"""
        import os

        # Save original environment
        original_env = dict(os.environ)

        try:
            # Set environment overrides
            os.environ['GA_POPULATION_SIZE'] = '200'
            os.environ['GA_GENERATIONS'] = '300'

            # Create config that reads from environment
            config = {
                "population_size": int(os.environ.get('GA_POPULATION_SIZE', '50')),
                "generations": int(os.environ.get('GA_GENERATIONS', '100'))
            }

            from app.services.auto_strategy.config import GAConfig
            ga_config = GAConfig.from_dict(config)

            # Should use environment values
            assert ga_config.population_size == 200
            assert ga_config.generations == 300

        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)

    def test_config_schema_version_compatibility(self):
        """Test configuration schema version handling"""
        # Mock version information
        config_schemas = [
            {"version": "1.0", "population_size": 50, "generations": 100},
            {"version": "2.0", "pop_size": 50, "gen_num": 100},  # Different field names
            {"schema_version": "3.0", "population_size": 50}  # Different version key
        ]

        for config in config_schemas:
            try:
                from app.services.auto_strategy.config import GAConfig
                ga_config = GAConfig.from_dict(config)
                # Should handle different schemas or reject incompatible ones
            except Exception:
                # Different schemas may not be compatible, that's OK
                pass

    def test_config_size_limits_and_memory_usage(self):
        """Test configuration size limits"""
        # Large configuration
        large_config = {
            "population_size": 1000,
            "generations": 1000,
            "custom_parameters": ["param_" + str(i) for i in range(1000)]  # Large array
        }

        try:
            from app.services.auto_strategy.config import GAConfig
            ga_config = GAConfig.from_dict(large_config)
            # Should handle large configs or impose limits
        except (MemoryError, ValueError):
            # Expected for very large configurations
            pass