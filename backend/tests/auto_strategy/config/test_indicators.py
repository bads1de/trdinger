"""
テスト: IndicatorSettingsクラス

IndicatorSettingsクラスの機能をテストします。
TDD準拠で、基本機能からバグ検出のためのエッジケースまでテストします。
"""

import pytest
from unittest.mock import patch, Mock
from typing import Dict, Any, List, Optional

# テスト対象のクラス
from backend.app.services.auto_strategy.config.indicators import IndicatorSettings
from backend.app.services.auto_strategy.constants import (
    ML_INDICATOR_TYPES,
    OPERATORS,
    DATA_SOURCES,
)


class TestIndicatorSettings:
    """IndicatorSettingsクラスのテスト"""

    def test_initialize_default(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """デフォルト初期化テスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()

            assert isinstance(config.valid_indicator_types, list)
            assert isinstance(config.ml_indicator_types, list)
            assert isinstance(config.operators, list)
            assert isinstance(config.data_sources, list)
            assert isinstance(config.multi_output_mappings, dict)

            # 定数がコピーされていることを確認
            assert len(config.ml_indicator_types) > 0
            assert len(config.operators) > 0
            assert len(config.data_sources) > 0

    def test_get_default_values(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """get_default_valuesテスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()
            defaults = config.get_default_values()

            assert isinstance(defaults, dict)
            assert "valid_indicator_types" in defaults
            assert "ml_indicator_types" in defaults
            assert "indicator_characteristics" in defaults
            assert "operators" in defaults

            # フィールド自動生成されたデフォルト値を確認
            assert isinstance(defaults["valid_indicator_types"], list)
            assert isinstance(defaults["operators"], list)

    def test_get_all_indicators(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """全指標取得テスト"""
        expected_technical = ["RSI", "MACD", "SMA"]
        expected_ml = ["MACHINE_LEARNING_1", "MACHINE_LEARNING_2"]

        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=expected_technical), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings(  # ml_indicator_typesをモックできなくなったので直接設定
                valid_indicator_types=expected_technical,
                ml_indicator_types=expected_ml
            )

            all_indicators = config.get_all_indicators()

            assert isinstance(all_indicators, list)
            assert len(all_indicators) == len(expected_technical) + len(expected_ml)
            assert all(indicator in all_indicators for indicator in expected_technical)
            assert all(indicator in all_indicators for indicator in expected_ml)

    def test_get_indicator_characteristics_existing(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """既存指標特性取得テスト"""
        test_indicator = "RSI"
        expected_chars = {"periods": [14], "outputs": ["rsi"], "type": "oscillator"}

        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()

            # テストケースがmock_indicator_characteristicsにアクセスを追加
            config.indicator_characteristics[test_indicator] = expected_chars

            result = config.get_indicator_characteristics(test_indicator)

            assert result is not None
            assert result == expected_chars
            assert result["type"] == "oscillator"

    def test_get_indicator_characteristics_nonexistent(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """存在しない指標特性取得テスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()

            result = config.get_indicator_characteristics("NONEXISTENT_INDICATOR")

            assert result is None

    def test_validate_success(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """正常検証テスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()
            is_valid, errors = config.validate()

            assert is_valid is True
            assert errors == []

    def test_to_dict_success(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """正常な辞書変換テスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()
            result = config.to_dict()

            assert isinstance(result, dict)
            assert "valid_indicator_types" in result
            assert "ml_indicator_types" in result
            assert "operators" in result

            # 複雑なオブジェクトは適切に処理されることを確認
            assert isinstance(result["ml_indicator_types"], list if result["ml_indicator_types"] is not None else str)

    def test_from_dict_success(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """正常な辞書からの変換テスト"""
        data = {
            "valid_indicator_types": ["RSI", "MACD"],
            "ml_indicator_types": ["ML1", "ML2"],
            "operators": ["==", ">="],
            "data_sources": ["close", "volume"],
        }

        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings.from_dict(data)

            assert isinstance(config, IndicatorSettings)
            assert config.valid_indicator_types == ["RSI", "MACD"]
            assert config.ml_indicator_types == ["ML1", "ML2"]
            assert config.operators == ["==", ">="]
            assert config.data_sources == ["close", "volume"]

    def test_multi_output_mappings(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """マルチ出力マッピングテスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()

            assert isinstance(config.multi_output_mappings, dict)
            assert "AROON" in config.multi_output_mappings
            assert "MACD" in config.multi_output_mappings
            assert "STOCH" in config.multi_output_mappings

            # マップされた値を確認
            assert config.multi_output_mappings["AROON"] == "AROON_0"
            assert config.multi_output_mappings["MACD"] == "MACD_0"

    def test_indicator_characteristics_initialization(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """指標特性初期化テスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()

            assert isinstance(config.indicator_characteristics, dict)
            assert len(config.indicator_characteristics) > 0

            # 特定の指標の特性が含まれていることを確認可能にする
            # (実際の特性データはmockで制御される)

    def test_copy_behavior(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """定数コピーの動作テスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()

            # 定数がコピーされていることを確認（共有されていないことを確認）
            assert config.ml_indicator_types is not ML_INDICATOR_TYPES  # 別インスタンス
            assert config.operators is not OPERATORS  # 別インスタンス
            assert config.data_sources is not DATA_SOURCES  # 別インスタンス

    def test_empty_initializations_handling(self):
        """空初期化処理テスト"""
        with patch('backend.app.services.auto_strategy.utils.indicator_utils.get_valid_indicator_types') as mock_get_valid, \
             patch('backend.app.services.auto_strategy.utils.indicator_characteristics.INDICATOR_CHARACTERISTICS', {}):

            mock_get_valid.return_value = []

            # このテストではエラーハンドリングを確認
            # エラーが発生せずに空の設定が作成されることを確認
            config = IndicatorSettings()

            assert isinstance(config.valid_indicator_types, list)
            assert isinstance(config.indicator_characteristics, dict)
            # 具体的な値はユーティリティ関数の動作による

    def test_to_json_from_json(self, mock_indicator_characteristics, mock_valid_indicator_types):
        """JSON変換テスト"""
        with patch('backend.app.services.auto_strategy.config.indicators.get_valid_indicator_types', return_value=mock_valid_indicator_types), \
             patch('backend.app.services.auto_strategy.config.indicators.INDICATOR_CHARACTERISTICS', mock_indicator_characteristics):

            config = IndicatorSettings()

            json_str = config.to_json()
            loaded_config = IndicatorSettings.from_json(json_str)

            assert isinstance(loaded_config, IndicatorSettings)
            # JSON経由で復元できることを確認


# Fixtures for mocking external dependencies
@pytest.fixture
def mock_valid_indicator_types():
    """get_valid_indicator_types関数のモック"""
    return ["RSI", "MACD", "SMA", "EMA", "STOCH", "CCI", "AROON", "WILLR"]


@pytest.fixture
def mock_indicator_characteristics():
    """INDICATOR_CHARACTERISTICSのモック"""
    return {
        "RSI": {
            "periods": [14],
            "outputs": ["rsi"],
            "type": "oscillator",
            "description": "Relative Strength Index"
        },
        "MACD": {
            "periods": [12, 26, 9],
            "outputs": ["macd", "macdsignal", "macdhist"],
            "type": "oscillator",
            "description": "Moving Average Convergence Divergence"
        },
        "SMA": {
            "periods": [20],
            "outputs": ["sma"],
            "type": "trend",
            "description": "Simple Moving Average"
        }
    }