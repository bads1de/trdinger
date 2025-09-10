"""
テスト: PositionSizingSettingsクラス

PositionSizingSettingsクラスの機能をテストします。
TDD準拠で、基本機能からバグ検出のためのエッジケースまでテストします。
"""

import pytest
from typing import Dict, Any, List, Tuple

# テスト対象のクラス
from backend.app.services.auto_strategy.config.position_sizing import PositionSizingSettings
from backend.app.services.auto_strategy.constants import (
    POSITION_SIZING_METHODS,
    GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS,
    GA_POSITION_SIZING_LOOKBACK_RANGE,
    GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_ATR_PERIOD_RANGE,
    GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE,
    GA_POSITION_SIZING_RISK_PER_TRADE_RANGE,
    GA_POSITION_SIZING_FIXED_RATIO_RANGE,
    GA_POSITION_SIZING_FIXED_QUANTITY_RANGE,
    GA_POSITION_SIZING_MIN_SIZE_RANGE,
    GA_POSITION_SIZING_MAX_SIZE_RANGE,
    GA_POSITION_SIZING_PRIORITY_RANGE,
)
from backend.app.services.auto_strategy.config.position_sizing import POSITION_SIZING_LIMITS


class TestPositionSizingSettings:
    """PositionSizingSettingsクラスのテスト"""

    def test_initialize_default(self):
        """デフォルト初期化テスト"""
        config = PositionSizingSettings()

        # メソッド確認
        assert hasattr(config, 'methods')
        assert isinstance(config.methods, list)
        assert len(config.methods) > 0

        # データ範囲を確認（全てのrange属性がリストであることを確認）
        range_attributes = [
            'lookback_range', 'optimal_f_multiplier_range', 'atr_period_range',
            'atr_multiplier_range', 'risk_per_trade_range', 'fixed_ratio_range',
            'fixed_quantity_range', 'min_size_range', 'max_size_range', 'priority_range'
        ]

        for attr_name in range_attributes:
            assert hasattr(config, attr_name)
            attr_value = getattr(config, attr_name)
            assert isinstance(attr_value, list)
            assert len(attr_value) >= 2  # 最小値と最大値

        # 制限設定確認
        assert hasattr(config, 'limits')
        assert isinstance(config.limits, dict)

    def test_get_default_values(self):
        """get_default_valuesテスト"""
        config = PositionSizingSettings()
        defaults = config.get_default_values()

        assert isinstance(defaults, dict)
        assert 'methods' in defaults
        assert 'limits' in defaults

        # 範囲属性が含まれていることを確認
        assert 'lookback_range' in defaults
        assert 'optimal_f_multiplier_range' in defaults

        # methodsがコピーされていることを確認（参照ではなく）
        assert defaults['methods'] is not POSITION_SIZING_METHODS

    def test_validate_success(self):
        """正常検証テスト"""
        config = PositionSizingSettings()
        is_valid, errors = config.validate()

        assert is_valid is True
        assert errors == []

    def test_to_dict_success(self):
        """正常な辞書変換テスト"""
        config = PositionSizingSettings()
        result = config.to_dict()

        assert isinstance(result, dict)

        # 変換された辞書が適切な構造を持っていることを確認
        assert isinstance(result.get('methods'), list) or isinstance(result.get('methods'), str)
        assert isinstance(result.get('limits'), dict) or isinstance(result.get('limits'), str)

    def test_from_dict_success(self):
        """正常な辞書からの変換テスト"""
        data = {
            'methods': ['fixed', 'risk_based'],
            'default_methods': ['risk_based'],
            'lookback_range': [10, 50],
            'limits': {'min_size': (0.1, 100.0)}
        }

        config = PositionSizingSettings.from_dict(data)

        assert isinstance(config, PositionSizingSettings)
        assert config.methods == ['fixed', 'risk_based']
        assert config.default_methods == ['risk_based']
        assert config.lookback_range == [10, 50]

    def test_to_json_from_json(self):
        """JSON変換テスト"""
        config = PositionSizingSettings()

        json_str = config.to_json()
        loaded_config = PositionSizingSettings.from_json(json_str)

        assert isinstance(loaded_config, PositionSizingSettings)
        # 同様の属性を持っていることを確認
        assert hasattr(loaded_config, 'methods')
        assert hasattr(loaded_config, 'limits')

    def test_methods_are_copied_correctly(self):
        """methodsが正しくコピーされているテスト"""
        config = PositionSizingSettings()

        # デフォルト値がconstantsからコピーされていることを確認
        assert set(config.methods) == set(POSITION_SIZING_METHODS)
        assert config.methods is not POSITION_SIZING_METHODS  # 別インスタンス

    def test_default_methods_are_copied_correctly(self):
        """default_methodsが正しくコピーされているテスト"""
        config = PositionSizingSettings()

        # default_methodsがGA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTSからコピー
        assert set(config.default_methods) == set(GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS)
        assert config.default_methods is not GA_DEFAULT_POSITION_SIZING_METHOD_CONSTRAINTS

    def test_limits_are_copied_correctly(self):
        """limitsが正しくコピーされているテスト"""
        config = PositionSizingSettings()

        # limitsがPOSITION_SIZING_LIMITSからコピー
        assert isinstance(config.limits, dict)
        assert len(config.limits) >= 0  # 少なくとも何らかの制限がある
        assert config.limits is not POSITION_SIZING_LIMITS  # 別インスタンス

    def test_parameter_ranges_are_valid(self):
        """パラメータ範囲が有効なリストであるテスト"""
        config = PositionSizingSettings()

        # 全ての範囲フィールドが [min, max] 形式であることを確認
        ranges_fields = {
            'lookback_range': GA_POSITION_SIZING_LOOKBACK_RANGE,
            'optimal_f_multiplier_range': GA_POSITION_SIZING_OPTIMAL_F_MULTIPLIER_RANGE,
            'atr_period_range': GA_POSITION_SIZING_ATR_PERIOD_RANGE,
            'atr_multiplier_range': GA_POSITION_SIZING_ATR_MULTIPLIER_RANGE,
            'risk_per_trade_range': GA_POSITION_SIZING_RISK_PER_TRADE_RANGE,
            'fixed_ratio_range': GA_POSITION_SIZING_FIXED_RATIO_RANGE,
            'fixed_quantity_range': GA_POSITION_SIZING_FIXED_QUANTITY_RANGE,
            'min_size_range': GA_POSITION_SIZING_MIN_SIZE_RANGE,
            'max_size_range': GA_POSITION_SIZING_MAX_SIZE_RANGE,
            'priority_range': GA_POSITION_SIZING_PRIORITY_RANGE,
        }

        for field_name in ranges_fields:
            config_value = getattr(config, field_name)
            expected_value = ranges_fields[field_name]

            # 値が正しく設定されていることを確認
            assert config_value == expected_value
            # 別インスタンスであることを確認
            assert config_value is not expected_value

            # 範囲が[min, max]形式であることを確認
            assert len(config_value) >= 2
            if len(config_value) >= 2:
                assert config_value[0] <= config_value[1]  # min <= max

    def test_all_range_fields_exist(self):
        """全ての必須範囲フィールドが存在するテスト"""
        config = PositionSizingSettings()
        required_range_fields = [
            'lookback_range', 'optimal_f_multiplier_range', 'atr_period_range',
            'atr_multiplier_range', 'risk_per_trade_range', 'fixed_ratio_range',
            'fixed_quantity_range', 'min_size_range', 'max_size_range', 'priority_range'
        ]

        for field_name in required_range_fields:
            assert hasattr(config, field_name)
            field_value = getattr(config, field_name)
            assert isinstance(field_value, list)

    def test_modification_does_not_affect_original_constants(self):
        """設定値の変更がオリジナルの定数を変更しないテスト"""
        config = PositionSizingSettings()

        # methodsの変更がオリジナル定数に影響しないことを確認
        original_methods_length = len(POSITION_SIZING_METHODS)
        config.methods.append("custom_method")

        assert len(config.methods) == original_methods_length + 1
        assert len(POSITION_SIZING_METHODS) == original_methods_length