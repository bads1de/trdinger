"""
yaml_utils.pyのユニットテスト
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../'))

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import tempfile
from typing import Dict, Any

try:
    from backend.app.services.auto_strategy.utils.yaml_utils import (
        YamlIndicatorUtils,
        YamlLoadUtils,
        YamlTestUtils,
        MockIndicatorGene
    )
except ImportError as e:
    # パス問題の場合の代替インポート
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "yaml_utils",
        os.path.join(os.path.dirname(__file__), '../../../app/services/auto_strategy/utils/yaml_utils.py')
    )
    yaml_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(yaml_utils)
    YamlIndicatorUtils = yaml_utils.YamlIndicatorUtils
    YamlLoadUtils = yaml_utils.YamlLoadUtils
    YamlTestUtils = yaml_utils.YamlTestUtils
    MockIndicatorGene = yaml_utils.MockIndicatorGene


class TestYamlIndicatorUtils:
    """YamlIndicatorUtilsクラスのテスト"""

    def test_generate_characteristics_from_yaml_valid_file(self):
        """有効なYAMLファイルからの特性生成テスト"""
        # Arrange
        yaml_content = """
indicators:
  sma:
    type: moving_average
    scale_type: price_absolute
    thresholds:
      normal:
        long_gt: 0.01
        short_lt: -0.01
"""
        with patch('builtins.open', mock_open(read_data=yaml_content)):
            # Act
            result = YamlIndicatorUtils.generate_characteristics_from_yaml("test.yml")

        # Assert
        assert "sma" in result
        assert result["sma"]["type"] == "moving_average"
        assert result["sma"]["scale_type"] == "price_absolute"

    def test_generate_characteristics_from_yaml_file_not_found(self):
        """ファイルが見つからない場合のテスト"""
        # Arrange
        with patch('builtins.open', side_effect=FileNotFoundError):
            # Act
            result = YamlIndicatorUtils.generate_characteristics_from_yaml("nonexistent.yml")

        # Assert
        assert result == {}

    def test_get_indicator_config_from_yaml(self):
        """YAMLから指標設定を取得するテスト"""
        # Arrange
        yaml_config = {
            "indicators": {
                "rsi": {"type": "oscillator", "scale_type": "oscillator_0_100"}
            }
        }

        # Act
        result = YamlIndicatorUtils.get_indicator_config_from_yaml(yaml_config, "rsi")

        # Assert
        assert result == {"type": "oscillator", "scale_type": "oscillator_0_100"}

    def test_get_indicator_config_from_yaml_not_found(self):
        """存在しない指標設定を取得する場合のテスト"""
        # Arrange
        yaml_config = {"indicators": {}}

        # Act
        result = YamlIndicatorUtils.get_indicator_config_from_yaml(yaml_config, "nonexistent")

        # Assert
        assert result is None


class TestYamlLoadUtils:
    """YamlLoadUtilsクラスのテスト"""

    def test_load_yaml_config_success(self):
        """正常なYAML設定読み込みテスト"""
        # Arrange
        yaml_content = {"indicators": {"test": {"type": "test"}}}
        config_path = "test.yml"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(yaml_content, f)
            temp_path = f.name

        try:
            # Act
            result = YamlLoadUtils.load_yaml_config(temp_path)

            # Assert
            assert result == yaml_content
        finally:
            os.unlink(temp_path)

    def test_load_yaml_config_file_not_exists(self):
        """ファイルが存在しない場合のテスト"""
        # Arrange
        config_path = "nonexistent.yml"

        with patch("pathlib.Path") as mock_path:
            mock_path.return_value.exists.return_value = False

            # Act
            result = YamlLoadUtils.load_yaml_config(config_path)

        # Assert
        expected_fallback = {"indicators": {}}
        assert result == expected_fallback

    def test_load_yaml_config_empty_file(self):
        """空のYAMLファイルの場合のテスト"""
        # Arrange
        config_path = "empty.yml"

        with patch("pathlib.Path") as mock_path, \
             patch("builtins.open", mock_open(read_data="")), \
             patch("yaml.safe_load", return_value=None):

            mock_path.return_value.exists.return_value = True

            # Act
            result = YamlLoadUtils.load_yaml_config(config_path)

        # Assert
        expected_fallback = {"indicators": {}}
        assert result == expected_fallback

    def test_validate_yaml_config_valid(self):
        """有効なYAML設定検証テスト"""
        # Arrange
        config = {
            "indicators": {
                "rsi": {
                    "type": "oscillator",
                    "scale_type": "oscillator_0_100",
                    "thresholds": {"normal": {}},
                    "conditions": {"long": "value > 70", "short": "value < 30"}
                }
            }
        }

        # Act
        is_valid, errors = YamlLoadUtils.validate_yaml_config(config)

        # Assert
        assert is_valid is True
        assert errors == []

    def test_validate_yaml_config_missing_indicators(self):
        """indicatorsセクションがない場合のテスト"""
        # Arrange
        config = {"other_section": {}}

        # Act
        is_valid, errors = YamlLoadUtils.validate_yaml_config(config)

        # Assert
        assert is_valid is False
        assert "indicatorsセクションが必須です" in errors[0]

    def test_get_indicator_config(self):
        """特定のindicator設定取得テスト"""
        # Arrange
        yaml_config = {
            "indicators": {
                "sma": {"type": "moving_average"},
                "rsi": {"type": "oscillator"}
            }
        }

        # Act
        result = YamlLoadUtils.get_indicator_config(yaml_config, "sma")

        # Assert
        assert result == {"type": "moving_average"}

    def test_get_all_indicator_names(self):
        """全indicator名取得テスト"""
        # Arrange
        yaml_config = {
            "indicators": {
                "sma": {"type": "moving_average"},
                "rsi": {"type": "oscillator"},
                "macd": {"type": "momentum"}
            }
        }

        # Act
        result = YamlLoadUtils.get_all_indicator_names(yaml_config)

        # Assert
        expected = ["sma", "rsi", "macd"]
        assert set(result) == set(expected)


class TestYamlTestUtils:
    """YamlTestUtilsクラスのテスト"""

    def test_test_yaml_based_conditions_success(self):
        """YAMLベース条件テスト成功ケース"""
        # Arrange
        yaml_config = {
            "indicators": {
                "sma": {"type": "moving_average", "scale_type": "price_absolute"}
            }
        }

        mock_generator = MagicMock()
        mock_generator.yaml_config = yaml_config
        mock_generator._generate_yaml_based_conditions.return_value = ["test_condition"]

        mock_condition_generator_class = MagicMock(return_value=mock_generator)

        with patch('backend.tests.auto_strategy.utils.test_yaml_utils.YamlLoadUtils.get_all_indicator_names') as mock_get_names, \
             patch('backend.tests.auto_strategy.utils.test_yaml_utils.YamlLoadUtils.get_indicator_config') as mock_get_config:

            mock_get_names.return_value = ["sma"]
            mock_get_config.return_value = yaml_config["indicators"]["sma"]

            # Act
            result = YamlTestUtils.test_yaml_based_conditions(
                yaml_config, mock_condition_generator_class
            )

            # Assert
            assert result["success"] is True
            assert result["summary"]["total_tested"] == 1
            assert result["summary"]["successful"] == 1
            assert len(result["tested_indicators"]) == 1

    def test_test_yaml_based_conditions_no_indicators(self):
        """indicatorがない場合のテスト"""
        # Arrange
        yaml_config = {"indicators": {}}
        mock_generator = MagicMock()
        mock_condition_generator_class = MagicMock(return_value=mock_generator)

        # Act
        result = YamlTestUtils.test_yaml_based_conditions(
            yaml_config, mock_condition_generator_class
        )

        # Assert
        assert result["success"] is False
        assert result["summary"]["total_tested"] == 0


class TestMockIndicatorGene:
    """MockIndicatorGeneクラスのテスト"""

    def test_mock_indicator_gene_initialization(self):
        """初期化テスト"""
        # Arrange & Act
        gene = MockIndicatorGene("sma", enabled=True, parameters={"period": 20})

        # Assert
        assert gene.type == "sma"
        assert gene.enabled is True
        assert gene.parameters == {"period": 20}

    def test_mock_indicator_gene_default_values(self):
        """デフォルト値での初期化テスト"""
        # Arrange & Act
        gene = MockIndicatorGene("rsi")

        # Assert
        assert gene.type == "rsi"
        assert gene.enabled is True
        assert gene.parameters == {}