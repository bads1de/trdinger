"""
テスト: TPSLSettingsクラス

TPSLSettingsクラスの機能をテストします。
TDD準拠で、基本機能からバグ検出のためのエッジケースまでテストします。
"""

import pytest
from typing import Dict, Any, List, Tuple

# テスト対象のクラス
from backend.app.services.auto_strategy.config.tpsl import TPSLSettings
from backend.app.services.auto_strategy.constants import (
    TPSL_METHODS,
    GA_DEFAULT_TPSL_METHOD_CONSTRAINTS,
    GA_TPSL_SL_RANGE,
    GA_TPSL_TP_RANGE,
    GA_TPSL_RR_RANGE,
    GA_TPSL_ATR_MULTIPLIER_RANGE,
)
from backend.app.services.auto_strategy.config.tpsl import TPSL_LIMITS


class TestTPSLSettings:
    """TPSLSettingsクラスのテスト"""

    def test_initialize_default(self):
        """デフォルト初期化テスト"""
        config = TPSLSettings()

        # メソッド確認
        assert hasattr(config, 'methods')
        assert isinstance(config.methods, list)
        assert len(config.methods) > 0

        assert hasattr(config, 'default_tpsl_methods')
        assert isinstance(config.default_tpsl_methods, list)

        # パラメータ範囲確認
        range_attributes = ['sl_range', 'tp_range', 'rr_range', 'atr_multiplier_range']
        for attr_name in range_attributes:
            assert hasattr(config, attr_name)
            attr_value = getattr(config, attr_name)
            assert isinstance(attr_value, list)
            assert len(attr_value) >= 2

        # 制限設定確認
        assert hasattr(config, 'limits')
        assert isinstance(config.limits, dict)

    def test_get_default_values(self):
        """get_default_valuesテスト"""
        config = TPSLSettings()
        defaults = config.get_default_values()

        assert isinstance(defaults, dict)
        assert 'methods' in defaults
        assert 'limits' in defaults
        assert 'sl_range' in defaults

        # 各フィールドが自動生成されたデフォルト値を含んでいることを確認
        assert isinstance(defaults['methods'], list) or defaults['methods'] is None
        assert isinstance(defaults['limits'], dict) or defaults['limits'] is None

    def test_validate_success(self):
        """正常検証テスト"""
        config = TPSLSettings()
        is_valid, errors = config.validate()

        assert is_valid is True
        assert errors == []

    def test_get_limits_for_param_success(self):
        """パラメータ制限取得成功テスト"""
        config = TPSLSettings()

        # limits辞書に含まれているパラメータでテスト
        for param_name in config.limits.keys():
            limits = config.get_limits_for_param(param_name)
            assert isinstance(limits, tuple)
            assert len(limits) == 2
            assert limits[0] <= limits[1]  # min <= max

    def test_get_limits_for_param_invalid_param(self):
        """無効なパラメータ制限取得テスト"""
        config = TPSLSettings()

        invalid_param = "nonexistent_parameter"
        with pytest.raises(ValueError, match=f"不明なパラメータ: {invalid_param}"):
            config.get_limits_for_param(invalid_param)

    def test_get_limits_for_param_edge_cases(self):
        """パラメータ制限取得エッジケーストスト"""
        config = TPSLSettings()

        # 有効なパラメータでのテスト
        if len(config.limits) > 0:
            param_name = list(config.limits.keys())[0]
            limits = config.get_limits_for_param(param_name)

            # 制限が数値であることを確認
            assert isinstance(limits[0], (int, float))
            assert isinstance(limits[1], (int, float))

    def test_to_dict_success(self):
        """正常な辞書変換テスト"""
        config = TPSLSettings()
        result = config.to_dict()

        assert isinstance(result, dict)

        # TPSL特有のフィールドが含まれていることを確認
        expected_fields = ['methods', 'default_tpsl_methods', 'sl_range', 'tp_range', 'limits']
        for field in expected_fields:
            assert field in result

    def test_from_dict_success(self):
        """正常な辞書からの変換テスト"""
        data = {
            'methods': ['atr_based', 'percentage'],
            'default_tpsl_methods': ['percentage'],
            'sl_range': [0.01, 0.05],
            'tp_range': [0.02, 0.08],
            'rr_range': [1.5, 3.0],
            'limits': {'sl_min': (0.005, 0.1)}
        }

        config = TPSLSettings.from_dict(data)

        assert isinstance(config, TPSLSettings)
        assert config.methods == ['atr_based', 'percentage']
        assert config.sl_range == [0.01, 0.05]
        assert config.tp_range == [0.02, 0.08]

    def test_to_json_from_json(self):
        """JSON変換テスト"""
        config = TPSLSettings()

        json_str = config.to_json()
        loaded_config = TPSLSettings.from_json(json_str)

        assert isinstance(loaded_config, TPSLSettings)
        # 必須属性を持っていることを確認
        assert hasattr(loaded_config, 'methods')
        assert hasattr(loaded_config, 'limits')
        assert hasattr(loaded_config, 'sl_range')

    def test_methods_are_copied_correctly(self):
        """methodsが正しくコピーされているテスト"""
        config = TPSLSettings()

        # TPSL_METHODSからコピーされていることを確認
        assert set(config.methods) == set(TPSL_METHODS)
        assert config.methods is not TPSL_METHODS

    def test_default_tpsl_methods_are_copied_correctly(self):
        """default_tpsl_methodsが正しくコピーされているテスト"""
        config = TPSLSettings()

        # GA_DEFAULT_TPSL_METHOD_CONSTRAINTSからコピー
        assert set(config.default_tpsl_methods) == set(GA_DEFAULT_TPSL_METHOD_CONSTRAINTS)
        assert config.default_tpsl_methods is not GA_DEFAULT_TPSL_METHOD_CONSTRAINTS

    def test_limits_are_copied_correctly(self):
        """limitsが正しくコピーされているテスト"""
        config = TPSLSettings()

        # TPSL_LIMITSからコピー
        assert isinstance(config.limits, dict)
        assert len(config.limits) >= 0
        assert config.limits is not TPSL_LIMITS

    def test_parameter_ranges_are_valid(self):
        """パラメータ範囲が有効なリストであるテスト"""
        config = TPSLSettings()

        # 全ての範囲フィールドでテスト
        ranges_fields = {
            'sl_range': GA_TPSL_SL_RANGE,
            'tp_range': GA_TPSL_TP_RANGE,
            'rr_range': GA_TPSL_RR_RANGE,
            'atr_multiplier_range': GA_TPSL_ATR_MULTIPLIER_RANGE,
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
        config = TPSLSettings()
        required_range_fields = ['sl_range', 'tp_range', 'rr_range', 'atr_multiplier_range']

        for field_name in required_range_fields:
            assert hasattr(config, field_name)
            field_value = getattr(config, field_name)
            assert isinstance(field_value, list)

    def test_get_limits_for_param_with_empty_limits(self):
        """空のlimitsでのget_limits_for_paramテスト"""
        config = TPSLSettings()
        # 空のlimitsの場合はgetItemメソッドがスムーズに動作する
        config.limits = {}

        with pytest.raises(ValueError, match="不明なパラメータ: test_param"):
            config.get_limits_for_param("test_param")

    def test_modification_does_not_affect_original_constants(self):
        """設定値の変更がオリジナルの定数を変更しないテスト"""
        config = TPSLSettings()

        original_methods_length = len(TPSL_METHODS)

        # 設定の変更がオリジナル定数に影響しないことを確認
        if isinstance(config.methods, list):
            config.methods.append("custom_tpsl_method")
            assert len(config.methods) == original_methods_length + 1
            assert len(TPSL_METHODS) == original_methods_length

    @pytest.mark.parametrize("param_name", [
        "sl_limit", "tp_limit", "rr_limit"  # config.limits辞書のキーを想定
    ])
    def test_get_limits_for_param_multiple_params(self, param_name):
        """複数パラメータでのget_limits_for_paramテスト"""
        config = TPSLSettings()

        # limits辞書に存在するパラメータのみテスト
        if param_name in config.limits:
            limits = config.get_limits_for_param(param_name)

            # 戻り値が適切なタプルであることを確認
            assert isinstance(limits, tuple)
            assert len(limits) == 2
            assert isinstance(limits[0], (int, float))
            assert isinstance(limits[1], (int, float))

        else:
            # 存在しないパラメータの場合はエラー
            with pytest.raises(ValueError):
                config.get_limits_for_param(param_name)