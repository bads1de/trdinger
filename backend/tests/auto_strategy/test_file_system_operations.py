"""
File System Operations Tests
Focus: Configuration file management, temporary files, persistence
"""

import pytest
import os
import tempfile
import shutil
from unittest.mock import Mock, patch
import json
import time
import sys

# Test framework setup
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class TestFileSystemOperations:
    """File system operation tests"""

    def test_config_file_creation_and_writing(self):
        """Test creating and writing configuration files"""
        config_data = {
            "app_name": "auto_strategy",
            "version": "1.0.0",
            "database": {"connection_string": "sqlite:///test.db"},
            "features": ["ga_strategy", "backtest", "indicator_analysis"]
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write config data
            with open(temp_path, 'w') as f:
                json.dump(config_data, f, indent=2)

            # Verify file was created and writable
            assert os.path.exists(temp_path)
            assert os.access(temp_path, os.W_OK)

            # Read back and verify
            with open(temp_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data["app_name"] == "auto_strategy"
            assert len(loaded_data["features"]) == 3

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_config_file_reading_validation(self):
        """Test reading and validating configuration files"""
        # Create valid config
        valid_config = {
            "generations": 100,
            "population_size": 50,
            "symbol": "BTC/USDT",
            "timeframe": "1h"
        }

        # Create invalid config
        invalid_config = {
            "generations": -5,  # Invalid
            "population_size": "invalid",  # Wrong type
        }

        configs = [
            ("valid", valid_config),
            ("invalid", invalid_config)
        ]

        for config_type, config_data in configs:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Write config
                with open(temp_path, 'w') as f:
                    json.dump(config_data, f)

                # Test reading
                with open(temp_path, 'r') as f:
                    loaded_config = json.load(f)

                # Basic validation
                if config_type == "valid":
                    assert loaded_config["generations"] > 0
                    assert isinstance(loaded_config["population_size"], int)
                elif config_type == "invalid":
                    # Should detect invalid values
                    assert loaded_config["generations"] < 0 or not isinstance(loaded_config["population_size"], int)

            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

    def test_temporary_directory_cleanup(self):
        """Test cleanup of temporary directories and files"""
        temp_dir = tempfile.mkdtemp(prefix='auto_strategy_test_')

        try:
            # Create files in temp directory
            for i in range(5):
                temp_file = os.path.join(temp_dir, f'test_file_{i}.txt')
                with open(temp_file, 'w') as f:
                    f.write(f'Content {i}' * 100)  # Larger content

                assert os.path.exists(temp_file)

            # Verify temp directory exists and has files
            assert os.path.exists(temp_dir)
            assert len(os.listdir(temp_dir)) == 5

            # Simulate cleanup operation
            files_removed = 0
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                os.unlink(file_path)
                files_removed += 1

            assert files_removed == 5
            assert len(os.listdir(temp_dir)) == 0

        finally:
            # Final cleanup
            shutil.rmtree(temp_dir)
            assert not os.path.exists(temp_dir)

    def test_config_file_backup_and_restore(self):
        """Test backup creation and restoration of configuration files"""
        original_config = {
            "setting_1": 42,
            "setting_2": "test_value",
            "setting_3": [1, 2, 3]
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            config_path = config_file.name

        try:
            # Write original config
            with open(config_path, 'w') as f:
                json.dump(original_config, f)

            # Create backup
            backup_path = config_path + '.backup'
            shutil.copy2(config_path, backup_path)

            assert os.path.exists(backup_path)

            # Modify original
            modified_config = original_config.copy()
            modified_config["setting_4"] = "new_value"

            with open(config_path, 'w') as f:
                json.dump(modified_config, f)

            # Restore from backup
            shutil.copy2(backup_path, config_path)

            # Verify restoration
            with open(config_path, 'r') as f:
                restored_config = json.load(f)

            assert restored_config == original_config
            assert "setting_4" not in restored_config

        finally:
            for path in [config_path, backup_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def test_large_config_files_handling(self):
        """Test handling of large configuration files"""
        # Create large config with many entries
        large_config = {}

        # Add many strategy variations
        for i in range(100):
            large_config[f'strategy_{i}'] = {
                'name': f'Strategy_{i}',
                'indicator': 'RSI',
                'period': i % 14 + 1,
                'threshold': (i % 50) / 50.0,
                'description': f'Description text for strategy {i} ' * 10
            }

        # Add large data arrays
        large_config['historical_data'] = list(range(10000))
        large_config['results_matrix'] = [
            [(i + j) % 100 for j in range(100)] for i in range(100)
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write large config
            start_time = time.time()
            with open(temp_path, 'w') as f:
                json.dump(large_config, f, separators=(',', ':'))
            write_time = time.time() - start_time

            # Check file size
            file_size = os.path.getsize(temp_path)
            assert file_size > 500000  # > 500KB

            # Read back
            read_start_time = time.time()
            with open(temp_path, 'r') as f:
                loaded_config = json.load(f)
            read_time = time.time() - read_start_time

            # Verify data integrity
            assert len(loaded_config) > 20
            assert len(loaded_config['historical_data']) == 10000
            assert len(loaded_config['results_matrix']) == 100

            # Performance check: should complete within reasonable time
            assert write_time < 5.0  # Less than 5 seconds to write
            assert read_time < 10.0   # Less than 10 seconds to read

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_config_file_permissions_handling(self):
        """Test proper handling of file permissions"""
        config_data = {"security_level": "high"}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Write config
            with open(temp_path, 'w') as f:
                f.write(json.dumps(config_data))

            # Check permissions
            stat_info = os.stat(temp_path)

            # Verify file is readable and writable by owner
            assert bool(stat_info.st_mode & 0o600)

            # Test chmod (if running on Unix-like system)
            try:
                os.chmod(temp_path, 0o600)  # rw-------
                new_stat = os.stat(temp_path)
                assert bool(new_stat.st_mode & 0o600)
            except OSError:
                # Skip if chmod not supported (Windows)
                pass

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_directory_structure_creation(self):
        """Test creation of directory structures for config organization"""
        base_dir = tempfile.mkdtemp(prefix='config_test_')

        try:
            # Create hierarchical structure
            sub_dirs = [
                'config',
                'config/models',
                'config/databases',
                'config/indicators',
                'logs',
                'logs/experiments',
                'data',
                'data/cache'
            ]

            for sub_dir in sub_dirs:
                dir_path = os.path.join(base_dir, sub_dir)
                os.makedirs(dir_path, exist_ok=True)

                # Create sample file in directory
                sample_file = os.path.join(dir_path, 'sample.txt')
                with open(sample_file, 'w') as f:
                    f.write(f'Sample content for {sub_dir}')

            # Verify structure
            created_dirs = []
            for root, dirs, files in os.walk(base_dir):
                rel_root = os.path.relpath(root, base_dir)
                if rel_root != '.':
                    created_dirs.append(rel_root)

            assert len(created_dirs) == len(sub_dirs)
            assert 'config/models' in created_dirs
            assert 'logs/experiments' in created_dirs

            # Verify file creation
            total_samples = sum(1 for _, _, files in os.walk(base_dir) for f in files if f == 'sample.txt')
            assert total_samples == len(sub_dirs)

        finally:
            shutil.rmtree(base_dir)
            assert not os.path.exists(base_dir)

    def test_file_locking_mechanism(self):
        """Test file locking to prevent concurrent access issues"""
        config_data = {"current_experiment": "test_exp"}
        lockfile_path = None

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file:
            config_path = config_file.name
            lockfile_path = config_path + '.lock'

        try:
            # Write initial config
            with open(config_path, 'w') as f:
                json.dump(config_data, f)

            # Create lock file
            with open(lockfile_path, 'w') as lock_file:
                lock_file.write(str(os.getpid()))

            assert os.path.exists(lockfile_path)

            # Attempt to read config while locked
            if os.path.exists(lockfile_path):
                # Simulate lock detection
                lock_detected = True
            else:
                lock_detected = False

            assert lock_detected

            # Remove lock
            os.unlink(lockfile_path)
            assert not os.path.exists(lockfile_path)

            # Now config should be accessible
            with open(config_path, 'r') as f:
                loaded_data = json.load(f)

            assert loaded_data["current_experiment"] == "test_exp"

        finally:
            for path in [config_path, lockfile_path]:
                if path and os.path.exists(path):
                    os.unlink(path)

    def test_config_migration_between_formats(self):
        """Test config migration between different format versions"""
        # Version 1 format
        v1_config = {
            "version": 1,
            "strategy": "RSI_and_MACD",
            "params": {
                "rsi_period": 14,
                "macd_fast": 12,
                "macd_slow": 26
            }
        }

        # Version 2 format
        v2_config = {
            "version": 2,
            "strategies": [{
                "id": "RSI_and_MACD",
                "indicators": [
                    {"name": "RSI", "period": 14},
                    {"name": "MACD", "fast": 12, "slow": 26}
                ]
            }]
        }

        # Test migration from V1 to V2
        migrated_config = v2_config.copy()

        # Simulate migration logic
        if v1_config["strategy"] == "RSI_and_MACD":
            migrated_config["strategies"] = [{
                "id": "RSI_and_MACD",
                "indicators": [
                    {"name": "RSI", "period": v1_config["params"]["rsi_period"]},
                    {"name": "MACD",
                     "fast": v1_config["params"]["macd_fast"],
                     "slow": v1_config["params"]["macd_slow"]}
                ]
            }]
            migrated_config["migrated_from"] = "v1"

        # Verify migration
        assert migrated_config["version"] == 2
        assert "migrated_from" in migrated_config
        assert len(migrated_config["strategies"][0]["indicators"]) == 2
        assert migrated_config["strategies"][0]["indicators"][0]["period"] == 14

    def test_recursive_file_operations(self):
        """Test recursive file operations in directory trees"""
        base_dir = tempfile.mkdtemp(prefix='recursive_test_')

        try:
            # Create nested directory structure
            structure = {
                'level1': {
                    'level2': {
                        'level3': {
                            'config.json': {'depth': 3, 'path': '/level1/level2/level3'},
                            'data.txt': 'Deep nested data'
                        }
                    },
                    'sibling': {
                        'settings.json': {'theme': 'dark', 'auto_save': True}
                    }
                },
                'root_config.json': {'root': True, 'timestamp': 123456789}
            }

            # Create files recursively
            def create_from_dict(path, data):
                if isinstance(data, dict):
                    for key, value in data.items():
                        sub_path = os.path.join(path, key)
                        if isinstance(value, (dict, str)):
                            if isinstance(value, dict) and not any(k.endswith('.json') for k in value.keys()):
                                # Directory
                                os.makedirs(sub_path, exist_ok=True)
                                create_from_dict(sub_path, value)
                            else:
                                # File
                                with open(sub_path, 'w') as f:
                                    json.dump(value if isinstance(value, dict) else value, f)
                else:
                    with open(path, 'w') as f:
                        json.dump(data, f)

            create_from_dict(base_dir, structure)

            # Verify structure creation
            total_files = sum(len([f for f in files if f.endswith('.json') or f.endswith('.txt')])
                            for _, _, files in os.walk(base_dir))

            # Count directories
            total_dirs = sum(1 for _, dirs, _ in os.walk(base_dir) if dirs) + 1

            assert total_files >= 3  # Should create at least 3 files
            assert total_dirs >= 4   # Base + level1 + level2 + level3 + sibling

        finally:
            shutil.rmtree(base_dir)
            assert not os.path.exists(base_dir)