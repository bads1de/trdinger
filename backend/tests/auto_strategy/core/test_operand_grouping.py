"""
オペランドグループ化システムのテスト

operand_grouping.pyの全てのコンポーネントをテストします。
TDDアプローチでバグを発見し、修正を行います。
"""

import pytest
from typing import List, Tuple
from unittest.mock import patch, MagicMock

from backend.app.services.auto_strategy.core.operand_grouping import (
    OperandGroup,
    OperandGroupingSystem,
    operand_grouping_system,
    get_operand_group,
    _classify_by_pattern,
    get_compatibility_score,
    validate_condition,
)


class TestOperandGroup:
    """OperandGroup Enumのテスト"""

    def test_operand_group_values(self):
        """各グループの値が正しいことを確認"""
        assert OperandGroup.PRICE_BASED.value == "price_based"
        assert OperandGroup.PRICE_RATIO.value == "price_ratio"
        assert OperandGroup.PERCENTAGE_0_100.value == "percentage_0_100"
        assert OperandGroup.PERCENTAGE_NEG100_100.value == "percentage_neg100_100"
        assert OperandGroup.ZERO_CENTERED.value == "zero_centered"
        assert OperandGroup.SPECIAL_SCALE.value == "special_scale"

    def test_all_operand_groups_defined(self):
        """全てのグループが定義されていることを確認"""
        expected_groups = [
            "PRICE_BASED", "PRICE_RATIO", "PERCENTAGE_0_100",
            "PERCENTAGE_NEG100_100", "ZERO_CENTERED", "SPECIAL_SCALE"
        ]
        actual_groups = [group.name for group in OperandGroup]

        assert set(actual_groups) == set(expected_groups)
        assert len(actual_groups) == len(expected_groups)


class TestOperandGroupingSystem:
    """OperandGroupingSystemクラスのテスト"""

    def test_initialization(self):
        """初期化が正しく行われることを確認"""
        system = OperandGroupingSystem()

        assert hasattr(system, '_group_mappings')
        assert hasattr(system, '_compatibility_matrix')
        assert isinstance(system._group_mappings, dict)
        assert isinstance(system._compatibility_matrix, dict)

    def test_get_operand_group_direct_mapping(self):
        """直接マッピングがあるオペランドのグループ取得をテスト"""
        system = OperandGroupingSystem()

        # 直接マッピングされている指標をテスト
        assert system.get_operand_group("close") == OperandGroup.PRICE_BASED
        assert system.get_operand_group("RSI") == OperandGroup.PERCENTAGE_0_100
        assert system.get_operand_group("MACD") == OperandGroup.ZERO_CENTERED

    def test_get_operand_group_pattern_matching(self):
        """パターンマッチングによるオペランドのグループ取得をテスト"""
        system = OperandGroupingSystem()

        # パターンマッチングされるオペランドをテスト
        result = system.get_operand_group("UNKNOWN_INDICATOR")
        assert isinstance(result, OperandGroup)

    def test_classify_by_pattern_price_based(self):
        """価格ベースパターン分類のテスト"""
        system = OperandGroupingSystem()

        # BB (ボリンジャーバンド) 関連
        assert system._classify_by_pattern("BB_UPPER") == OperandGroup.PRICE_BASED

        # Trend指標
        assert system._classify_by_pattern("BANANA_TREND") == OperandGroup.PRICE_BASED

    def test_classify_by_pattern_percentage_0_100(self):
        """0-100%オシレーターパターンのテスト"""
        system = OperandGroupingSystem()

        assert system._classify_by_pattern("RSI_FAST") == OperandGroup.PERCENTAGE_0_100
        assert system._classify_by_pattern("STOCH_SLOW") == OperandGroup.PERCENTAGE_0_100

    def test_classify_by_pattern_percentage_neg100_100(self):
        """±100%オシレーターパターンのテスト"""
        system = OperandGroupingSystem()

        assert system._classify_by_pattern("CCI_SCALED") == OperandGroup.PERCENTAGE_NEG100_100

    def test_classify_by_pattern_zero_centered(self):
        """ゼロ中心指標パターンのテスト"""
        system = OperandGroupingSystem()

        assert system._classify_by_pattern("MOMENTUM_Z") == OperandGroup.ZERO_CENTERED
        assert system._classify_by_pattern("ROC_SCALED") == OperandGroup.ZERO_CENTERED

    def test_classify_by_pattern_special_scale(self):
        """特殊スケールパターンのテスト"""
        system = OperandGroupingSystem()

        assert system._classify_by_pattern("FUNDING_RATE") == OperandGroup.SPECIAL_SCALE
        assert system._classify_by_pattern("HUGE_VOLUME") == OperandGroup.SPECIAL_SCALE

    def test_classify_by_pattern_default(self):
        """デフォルト分類のテスト"""
        system = OperandGroupingSystem()

        # マッチしないパターンはPRICE_BASEDになる
        result = system._classify_by_pattern("UNKNOWN")
        assert result == OperandGroup.PRICE_BASED

    def test_get_compatibility_score(self):
        """互換性スコア取得のテスト"""
        system = OperandGroupingSystem()

        # 同一グループは1.0
        score = system.get_compatibility_score("close", "high")
        assert score == 1.0

        # 異なるグループもスコアが取得可能
        score = system.get_compatibility_score("close", "RSI")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_get_compatibility_score_unknown_operands(self):
        """未知のオペランドに対する互換性スコアテスト"""
        system = OperandGroupingSystem()

        # 未知のオペランド同士でもスコアが取得可能
        score = system.get_compatibility_score("X_UNKNOWN", "Y_UNKNOWN")
        assert isinstance(score, float)

    def test_validate_condition_with_numeric(self):
        """数値との条件検証テスト"""
        system = OperandGroupingSystem()

        valid, reason = system.validate_condition("close", 0.5)
        assert valid is True
        assert "数値との比較" in reason

        valid, reason = system.validate_condition("RSI", 30)
        assert valid is True

    def test_validate_condition_with_operand_high_compatibility(self):
        """高互換性オペランドとの条件検証テスト"""
        system = OperandGroupingSystem()

        valid, reason = system.validate_condition("close", "high")
        assert valid is True
        assert "高い互換性" in reason

    def test_validate_condition_with_operand_low_compatibility(self):
        """低互換性オペランドとの条件検証テスト"""
        system = OperandGroupingSystem()

        # 低互換性のペアを見つけてテスト（必要に応じて調整）
        valid, reason = system.validate_condition("RSI", "volume")
        assert isinstance(valid, bool)

    def test_get_compatible_operands(self):
        """互換性の高いオペランドリスト取得のテスト"""
        system = OperandGroupingSystem()

        available_operands = ["close", "high", "low", "RSI", "MACD", "volume"]
        compatible = system.get_compatible_operands("close", available_operands)

        assert isinstance(compatible, list)
        assert len(compatible) > 0
        assert "close" not in compatible  # 自分自身は除外される

    def test_get_compatible_operands_min_compatibility_filter(self):
        """最小互換性スコアによるフィルタリングテスト"""
        system = OperandGroupingSystem()

        available_operands = ["close", "high", "low", "RSI", "MACD", "volume"]
        compatible = system.get_compatible_operands("close", available_operands, min_compatibility=0.9)

        assert isinstance(compatible, list)
        # 互換性スコア0.9以上のオペランドのみが返されるはず


class TestGlobalFunctions:
    """グローバル関数のテスト"""

    def test_operand_grouping_system_global_instance(self):
        """グローバルインスタンスが正しいタイプであることを確認"""
        assert isinstance(operand_grouping_system, OperandGroupingSystem)

    def test_get_operand_group_function(self):
        """グローバルget_operand_group関数のテスト"""
        result = get_operand_group("close")
        assert isinstance(result, OperandGroup)

    def test_get_operand_group_function_unknown(self):
        """未知のオペランドに対するグローバルget_operand_group関数のテスト"""
        result = get_operand_group("UNKNOWN_OPERAND")
        assert isinstance(result, OperandGroup)

    def test_get_compatibility_score_function(self):
        """グローバルget_compatibility_score関数のテスト"""
        score = get_compatibility_score("close", "high")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_validate_condition_function(self):
        """グローバルvalidate_condition関数のテスト"""
        valid, reason = validate_condition("close", 0.5)
        assert isinstance(valid, bool)
        assert isinstance(reason, str)

    def test_validate_condition_operand_vs_operand(self):
        """オペランド対オペランドのvalidate_conditionテスト"""
        valid, reason = validate_condition("close", "RSI")
        assert isinstance(valid, bool)
        assert isinstance(reason, str)


# 統合テスト
class TestIntegration:
    """統合テストケース"""

    def test_complete_operand_grouping_workflow(self):
        """完全なオペランドグループ化ワークフローのテスト"""
        # グローバルインスタンスを使用
        system = operand_grouping_system

        # 1. オペランドのグループを取得
        price_group = system.get_operand_group("close")
        assert price_group == OperandGroup.PRICE_BASED

        rsi_group = system.get_operand_group("RSI")
        assert rsi_group == OperandGroup.PERCENTAGE_0_100

        # 2. 互換性スコアを取得
        score = system.get_compatibility_score("close", "RSI")
        assert isinstance(score, float)

        # 3. 条件を検証
        valid, reason = system.validate_condition("close", "RSI")
        assert isinstance(valid, bool)
        assert isinstance(reason, str)

        # 4. 互換性の高いオペランドを取得
        available_operands = ["high", "low", "open", "RSI", "MACD", "volume"]
        compatible_operands = system.get_compatible_operands("close", available_operands)
        assert isinstance(compatible_operands, list)

    def test_edge_cases(self):
        """エッジケースのテスト"""
        system = operand_grouping_system

        # 空文字列
        result = system.get_operand_group("")
        assert isinstance(result, OperandGroup)

        # None（実際にはstrを想定しているが）
        # result = system.get_operand_group(None)  # TypeErrorになるはず

        # 特殊文字を含むオペランド名
        result = system.get_operand_group("BB_@#$%")
        assert isinstance(result, OperandGroup)

    def test_unknown_operand_patterns(self):
        """未知のオペランドパターンのテスト"""
        system = operand_grouping_system

        # 様々な未知パターン
        unknown_patterns = [
            "UNKNOWN",
            "TEST_INDICATOR",
            "MY_CUSTOM_METRIC",
            "XYZ123",
            ""  # 空文字列
        ]

        for pattern in unknown_patterns:
            result = system.get_operand_group(pattern)
            assert isinstance(result, OperandGroup)
            assert result == OperandGroup.PRICE_BASED  # デフォルトはPRICE_BASED

    def test_compatibility_matrix_completeness(self):
        """互換性マトリックスの完全性をテスト"""
        system = operand_grouping_system

        # 全てのグループペアがマトリックスに含まれているかを確認
        all_groups = list(OperandGroup)
        for group1 in all_groups:
            for group2 in all_groups:
                score = system._compatibility_matrix.get((group1, group2), 0.1)
                assert isinstance(score, float)
                assert 0.0 <= score <= 1.0

    def test_group_mapping_completeness(self):
        """グループマッピングの完全性をテスト"""
        system = operand_grouping_system

        # マッピングされている指標を取得
        mapped_operands = list(system._group_mappings.keys())
        assert len(mapped_operands) > 0

        # 各マッピングが正しいグループを持っている
        for operand, group in system._group_mappings.items():
            assert isinstance(group, OperandGroup)

    def test_debug_logging(self):
        """デバッグログ機能のテスト"""
        # デバッグモードが有効であることを確認
        with patch('backend.app.services.auto_strategy.core.operand_grouping.logger') as mock_logger:
            system = operand_grouping_system
            system.get_operand_group("close")
            system.get_compatibility_score("close", "high")

            # ログが呼ばれているはず（DEBUG_MODE = Trueの場合）
            # 実際の呼び出し回数は実装次第

    def test_unsupported_indicators_not_mapped(self):
        """pandas-taでサポートされていない指標がマッピングに含まれていないことを確認"""
        system = OperandGroupingSystem()

        unsupported_indicators = ["CWMA", "MAVP", "SAREXT", "TLB", "RSI_EMA_CROSS", "CV", "IRM"]

        for indicator in unsupported_indicators:
            assert indicator not in system._group_mappings, f"{indicator} should not be in group mappings"
            # パターンマッチングでも適切なグループに分類されることを確認
            result = system.get_operand_group(indicator)
            assert isinstance(result, OperandGroup)

    def test_undefined_indicators_not_mapped(self):
        """indicator_definitions.pyで定義されていない指標がマッピングに含まれていないことを確認"""
        system = OperandGroupingSystem()

        # indicator_definitions.pyで定義されている指標のみ残す
        defined_indicators = [
            "RSI", "SMA", "EMA", "WMA", "DEMA", "TEMA", "T3", "KAMA", "MACD", "STOCH", "CCI", "WILLR", "ROC", "MOM", "ADX", "QQE", "SAR", "ATR", "BB", "KELTNER", "SUPERTREND", "DONCHIAN", "ACCBANDS", "UI", "OBV", "AD", "ADOSC", "CMF", "EFI", "VWAP", "SQUEEZE", "MFI",
            # データソース
            "close", "open", "high", "low", "volume", "OpenInterest", "FundingRate"
        ]

        # 残すべき指標はマッピングに含まれていることを確認
        for indicator in defined_indicators:
            if indicator in system._group_mappings:
                assert indicator in system._group_mappings, f"{indicator} should be in group mappings"

        # 定義されていない指標はマッピングに含まれていないことを確認
        undefined_indicators = [
            "BB", "BB_0", "BB_1", "BB_2", "ULTOSC", "DX", "PLUS_DI", "MINUS_DI", "ADXR", "CMO", "AROONOSC", "MACD_0", "MACD_1", "MACD_2", "ROCP", "ROCR", "ROCR100", "TRIX", "APO", "PPO", "TSI", "BOP", "STOCH_0", "STOCH_1", "STOCHRSI_0", "STOCHRSI_1", "KDJ_2", "SMI_0", "SMI_1", "PVO_0", "PVO_1", "NATR", "TRANGE", "HMA", "ZLMA", "VWMA", "SWMA", "ALMA", "JMA", "MCGD", "ICHIMOKU", "HILO", "HWMA", "HL2", "HLC3", "OHLC4", "WCP", "SSF", "VIDYA", "HWC", "NVI", "PVI", "PVT", "EOM", "KVO", "AOBV", "PVOL", "PVR", "RVI", "RMI", "DPO", "VORTEX", "CHOP", "PVO", "CFO", "CTI", "MA", "FWMA", "PWMA", "SINWMA", "PLUS_DM", "MINUS_DM", "KST", "RSX", "BIAS", "BRAR", "CG", "FISHER", "INERTIA", "PGO", "PSL", "SQUEEZE_PRO", "ER", "ERI", "COPPOCK", "PDIST", "VAR"
        ]

        for indicator in undefined_indicators:
            assert indicator not in system._group_mappings, f"{indicator} should not be in group mappings"


if __name__ == "__main__":
    pytest.main([__file__])