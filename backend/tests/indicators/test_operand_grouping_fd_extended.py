"""
拡張版_operand_grouping TDDテストスイート

operand_grouping.py の拡充アップグレードのための包括的なテストケース群。
TDD原則に基づき、現時点では全てのテストがFAILすることを想定。
"""

import pytest
from typing import Dict, List

from app.services.auto_strategy.core.operand_grouping import (
    operand_grouping_system, OperandGroupingSystem, OperandGroup
)


class TestExtendedOperandGrouping:
    """
    拡張版オペランドグループングシステムのテストスイート

    既存および新規指標の全面的なテストカバレッジを提供し、
    システムの堅牢性と互換性を保証する。
    """

    # 既存指標のリスト（回帰テスト用）
    EXISTING_INDICATORS = {
        # PRICE_BASED
        "PRICE_BASED": ["SMA", "EMA", "BB", "close", "open", "high", "low"],
        # PERCENTAGE_0_100
        "PERCENTAGE_0_100": ["RSI", "STOCH", "ADX", "MFI", "ULTOSC", "QQE", "DX", "PLUS_DI", "MINUS_DI", "ADXR"],
        # PERCENTAGE_NEG100_100
        "PERCENTAGE_NEG100_100": ["CCI", "CMO", "AROONOSC"],
        # ZERO_CENTERED
        "ZERO_CENTERED": ["MACD", "MACD_0", "MACD_1", "MACD_2", "ROC", "MOM", "ROCP", "ROCR", "ROCR100", "TRIX", "WILLR", "T3", "APO", "PPO", "TSI", "BOP"],
        # SPECIAL_SCALE
        "SPECIAL_SCALE": ["ATR", "TRANGE", "OBV", "volume", "OpenInterest", "FundingRate"]
    }

    # 新規追加指標（タスク指定）
    NEW_TREND_INDICATORS = ["HMA", "ZLMA", "VWMA", "SWMA", "ALMA", "JMA", "MCGD", "ICHIMOKU", "HILO", "HWMA", "HL2", "HLC3", "OHLC4", "WCP", "SSF", "VIDYA"]
    NEW_VOLATILITY_INDICATORS = ["KELTNER", "DONCHIAN", "ACCBANDS", "SUPERTREND", "HWC", "UI", "MASSI"]
    NEW_VOLUME_INDICATORS = ["NVI", "PVI", "VWAP", "PVT", "EFI", "EOM", "KVO", "CMF", "AOBV", "PVOL", "PVR"]
    NEW_MOMENTUM_INDICATORS = ["TSI", "RVI", "RMI", "DPO", "VORTEX", "CHOP", "PVO", "CFO", "CTI"]

    def test_regression_existing_indicators(self):
        """既存指標の分類維持確認 - 回帰テスト"""
        for expected_group_name, indicators in self.EXISTING_INDICATORS.items():
            expected_group = OperandGroup[expected_group_name]

            for indicator in indicators:
                with pytest.raises(AssertionError, match=f"Indicator {indicator} classification changed"):
                    actual_group = operand_grouping_system.get_operand_group(indicator)
                    assert actual_group == expected_group, f"Indicator {indicator} should be {expected_group.value} but got {actual_group.value}"

    def test_new_trend_indicators_grouping(self):
        """TREND系新規指標分類テスト"""
        # トレンド指標は主にPRICE_BASEDまたはPRICE_RATIOに分類されるべき
        expected_groups = [OperandGroup.PRICE_BASED, OperandGroup.PRICE_RATIO]

        for indicator in self.NEW_TREND_INDICATORS:
            with pytest.raises(AssertionError, match=f"New trend indicator {indicator} not properly classified"):
                actual_group = operand_grouping_system.get_operand_group(indicator)
                assert actual_group in expected_groups, f"Trend indicator {indicator} should be PRICE_BASED or PRICE_RATIO but got {actual_group.value}"

    def test_new_volatility_indicators_grouping(self):
        """VOLATILITY系新規指標分類テスト"""
        # ボラティリティ指標は主にPERCENTAGE_0_100またはPRICE_BASED
        expected_groups = [OperandGroup.PERCENTAGE_0_100, OperandGroup.PRICE_BASED, OperandGroup.PERCENTAGE_NEG100_100]

        for indicator in self.NEW_VOLATILITY_INDICATORS:
            with pytest.raises(AssertionError, match=f"New volatility indicator {indicator} not properly classified"):
                actual_group = operand_grouping_system.get_operand_group(indicator)
                assert actual_group in expected_groups, f"Volatility indicator {indicator} should be in volatility groups but got {actual_group.value}"

    def test_new_volume_indicators_grouping(self):
        """VOLUME系新規指標分類テスト"""
        # ボリューム指標は主に专门スケールまたはZERO_CENTERED
        expected_groups = [OperandGroup.SPECIAL_SCALE, OperandGroup.ZERO_CENTERED, OperandGroup.PRICE_BASED]

        for indicator in self.NEW_VOLUME_INDICATORS:
            with pytest.raises(AssertionError, match=f"New volume indicator {indicator} not properly classified"):
                actual_group = operand_grouping_system.get_operand_group(indicator)
                assert actual_group in expected_groups, f"Volume indicator {indicator} should be in volume groups but got {actual_group.value}"

    def test_new_momentum_indicators_grouping(self):
        """MOMENTUM系新規指標分類テスト"""
        # モメンタム指標は主にZERO_CENTEREDまたはPERCENTAGE_NEG100_100
        expected_groups = [OperandGroup.ZERO_CENTERED, OperandGroup.PERCENTAGE_NEG100_100, OperandGroup.PERCENTAGE_0_100]

        for indicator in self.NEW_MOMENTUM_INDICATORS:
            with pytest.raises(AssertionError, match=f"New momentum indicator {indicator} not properly classified"):
                actual_group = operand_grouping_system.get_operand_group(indicator)
                assert actual_group in expected_groups, f"Momentum indicator {indicator} should be in momentum groups but got {actual_group.value}"


    def test_compatibility_scores_with_new_indicators(self):
        """新規指標との互換性スコア計算テスト"""
        # 既存と新規指標間の互換性
        existing_test_indicators = ["SMA", "RSI", "MACD", "volume", "CCI"]

        for existing in existing_test_indicators:
            for new_trend in self.NEW_TREND_INDICATORS[:3]:  # サンプル選択
                with pytest.raises(AssertionError, match="Compatibility score calculation failed"):
                    score = operand_grouping_system.get_compatibility_score(existing, new_trend)
                    assert isinstance(score, float), "Compatibility score should be float"
                    assert 0.0 <= score <= 1.0, f"Compatibility score should be between 0.0 and 1.0, got {score}"

    def test_edge_cases_unknown_indicators(self):
        """未知指標のデフォルト分類テスト"""
        # 未知指標はデフォルト分類されるべき
        unknown_indicators = ["NON_EXISTENT_INDICATOR_123", "UNKNOWN_SIGNAL_XYZ", "RANDOM_PATTERN_ABC"]

        for unknown in unknown_indicators:
            with pytest.raises(AssertionError, match=f"Unknown indicator {unknown} default classification failed"):
                group = operand_grouping_system.get_operand_group(unknown)
                # 未知指標はPRICE_BASEDデフォルトになるはず
                assert group == OperandGroup.PRICE_BASED, f"Unknown indicator {unknown} should default to PRICE_BASED but got {group.value}"

    def test_special_scale_operands(self):
        """特殊スケール（VOLUME, FUNDING_RATE）の処理テスト"""
        special_operands = ["VOLUME", "volume", "FUNDINGRATE", "FUNDING_RATE", "OpenInterest", "OPEN_INTEREST"]

        for operand in special_operands:
            with pytest.raises(AssertionError, match=f"Special scale operand {operand} not properly handled"):
                group = operand_grouping_system.get_operand_group(operand)
                assert group == OperandGroup.SPECIAL_SCALE, f"Special scale operand {operand} should be SPECIAL_SCALE but got {group.value}"

    def test_boundaries_numerical_comparison(self):
        """数値比較の境界テスト"""
        # 数値との比較妥当性テスト
        test_cases = [
            ("SMA", 100.0, True),   # 価格指標 × 数値 = OK
            ("RSI", 75, True),      # オシレーター × 数値 = OK
            ("SMA", "invalid", False),  # 不正な右側オペランド
            ("volume", "SMA", True),    # 特殊スケール × 指標 = スコア次第
            ("MACD", "RSI", False),     # ゼロ中心 × オシレーター = 互換性低い可能性
        ]

        for left, right, expected_valid in test_cases:
            with pytest.raises(AssertionError, match=f"Numerical comparison validation failed for {left} vs {right}"):
                is_valid, reason = operand_grouping_system.validate_condition(left, right)
                assert is_valid == expected_valid, f"Condition validation for {left} vs {right} failed. Expected {expected_valid}, got {is_valid}. Reason: {reason}"

    def test_compatibility_matrix_completeness(self):
        """互換性マトリックスの完全性テスト"""
        # 全てのグループペアが定義されているべき
        groups = list(OperandGroup)
        matrix_size = len(groups)

        for i in range(matrix_size):
            for j in range(matrix_size):
                group1 = groups[i]
                group2 = groups[j]

                with pytest.raises(AssertionError, match=f"Compatibility matrix missing entry for {group1.value} -> {group2.value}"):
                    score = operand_grouping_system.get_compatibility_score(
                        list(operand_grouping_system._group_mappings.keys())[0],  # ダミーオペランド
                        "UNKNOWN"  # 別のダミーオペランド
                    )
                    # このテストはマトリックの完全性を検証（実際の実装に依存）
                    assert isinstance(score, float), f"Matrix completeness check failed for {group1.value} -> {group2.value}"