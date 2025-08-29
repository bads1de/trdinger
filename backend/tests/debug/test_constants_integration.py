"""
定数統合のテスト
Phase 1-1: 定数の統合に関するテスト
"""

import pytest
import sys
import os

# PYTHONPATHを追加してimportを可能にする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.services.auto_strategy.config.constants import (
    INDICATOR_CHARACTERISTICS,
    VALID_INDICATOR_TYPES,
    GA_DEFAULT_CONFIG,
    TPSL_LIMITS,
    POSITION_SIZING_LIMITS
)
from app.services.auto_strategy.generators.smart_condition_generator import SmartConditionGenerator


class TestConstantsIntegration:
    """定数統合テスト"""

    def test_indicator_characteristics_exists_in_constants(self):
        """INDICATOR_CHARACTERISTICSがconstants.pyで定義されていることを確認"""
        # 少なくとも主要な指標（RSI, MACD, SMAなど）が含まれていることを確認
        major_indicators = ["RSI", "MACD", "SMA", "EMA", "BBANDS", "STOCH"]

        for indicator in major_indicators:
            assert indicator in INDICATOR_CHARACTERISTICS, f"指標 {indicator} がINDICATOR_CHARACTERISTICSに含まれていません"

        # RSIの特性が正しく定義されていることを確認
        rsi_char = INDICATOR_CHARACTERISTICS["RSI"]
        assert "type" in rsi_char
        assert "range" in rsi_char
        assert "long_zones" in rsi_char
        assert "short_zones" in rsi_char

    def test_ga_default_config_integration(self):
        """GA_DEFAULT_CONFIGが正しく統合されていることを確認"""
        # 必須パラメータが存在することを確認
        assert "population_size" in GA_DEFAULT_CONFIG
        assert "generations" in GA_DEFAULT_CONFIG
        assert "crossover_rate" in GA_DEFAULT_CONFIG
        assert "mutation_rate" in GA_DEFAULT_CONFIG

        # 値が妥当な範囲であることを確認
        assert GA_DEFAULT_CONFIG["population_size"] > 0
        assert 0 <= GA_DEFAULT_CONFIG["crossover_rate"] <= 1
        assert 0 <= GA_DEFAULT_CONFIG["mutation_rate"] <= 1

    def test_tpsl_limits_integration(self):
        """TPSL_LIMITSが正しく統合されていることを確認"""
        assert "stop_loss_pct" in TPSL_LIMITS
        assert "take_profit_pct" in TPSL_LIMITS

        # 範囲が正しく設定されていることを確認
        sl_range = TPSL_LIMITS["stop_loss_pct"]
        tp_range = TPSL_LIMITS["take_profit_pct"]
        assert len(sl_range) == 2
        assert len(tp_range) == 2
        assert sl_range[0] < sl_range[1]
        assert tp_range[0] < tp_range[1]

    def test_smart_condition_generator_uses_constants(self):
        """SmartConditionGeneratorがconstants経由で定数を使用していることを確認"""
        generator = SmartConditionGenerator()

        # generatorがINDICATOR_CHARACTERISTICSにアクセスできることを確認
        # 実際の使用は内部メソッドで使用されていることをテスト
        test_indicators = [
            type('IndicatorGene', (), {
                'type': 'RSI',
                'enabled': True,
                'parameters': {'period': 14}
            })()
        ]

        # 正常に動作することを確認（定数統合後もエラーが発生しない）
        try:
            longs, shorts, exits = generator.generate_balanced_conditions(test_indicators)
            assert isinstance(longs, list)
            assert isinstance(shorts, list)
            assert isinstance(exits, list)
        except Exception as e:
            pytest.fail(f"定数統合後にSmartConditionGeneratorがエラーを起こしました: {e}")

    def test_no_duplicate_indicator_characteristics(self):
        """INDICATOR_CHARACTERISTICSが重複定義されていないことを確認"""
        # constants.pyとsmart_condition_generator.pyの両方で定義されていないことを確認
        try:
            from backend.app.services.auto_strategy.generators.smart_condition_generator import INDICATOR_CHARACTERISTICS as SC_INDICATOR_CHARACTERISTICS
            # 両方が存在する場合、エラーを発生させる
            pytest.fail("INDICATOR_CHARACTERISTICSがsmart_condition_generator.pyでも定義されています。constants.pyへの統合が必要です。")
        except ImportError:
            # smart_condition_generator.pyから直接importできない場合、定数統合が完了していると判断
            pass

    def test_valid_indicator_types_consistent(self):
        """VALID_INDICATOR_TYPESとINDICATOR_CHARACTERISTICSが整合していることを確認"""
        # INDICATOR_CHARACTERISTICSに含まれる指標がVALID_INDICATOR_TYPESにも含まれていることを確認
        for indicator in INDICATOR_CHARACTERISTICS.keys():
            if "_" not in indicator:  # ML指標以外
                assert indicator in VALID_INDICATOR_TYPES, f"指標 {indicator} がVALID_INDICATOR_TYPESに含まれていません"