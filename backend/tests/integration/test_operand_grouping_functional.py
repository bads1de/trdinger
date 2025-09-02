"""
オペランドグループ化機能統合テスト

test_operand_check.py と test_operand_e2e.py の機能を統合し、
包括的なオペランドグループ化機能のテストスイートを提供。

機能テスト範囲:
- 全指標の包括的分類テスト（新規・既存・各グループ別）
- 互換性計算・オペランド探索・条件妥当性の完全検証
- 必須グループ（PRICE_BASED, PERCENTAGE_0_100, etc.）のカバー率確認
- エラーハンドリングと堅牢性テスト
"""

import pytest
import sys
import os
from typing import List, Dict, Set

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.services.auto_strategy.core.operand_grouping import (
    operand_grouping_system,
    OperandGroupingSystem,
    OperandGroup
)


class TestOperandGroupingFunctional:
    """
    オペランドグループ化システムの包括的機能テストクラス

    既存・新規指標の分類機能、エラーハンドリング、
    オペランド探索・条件妥当性検証等の全機能をカバー
    """

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """各テストメソッド前のセットアップ"""
        # システムはグローバルインスタンスを使用
        pass

    def test_comprehensive_operand_classification(self):
        """全指標の包括的分類テスト（各グループ別）"""
        # 全指標リストを取得
        all_indicators = list(operand_grouping_system._group_mappings.keys())

        # 各グループの分類を検証
        groups_classified = {
            OperandGroup.PRICE_BASED: [],
            OperandGroup.PERCENTAGE_0_100: [],
            OperandGroup.PERCENTAGE_NEG100_100: [],
            OperandGroup.ZERO_CENTERED: [],
            OperandGroup.SPECIAL_SCALE: [],
        }

        for indicator in all_indicators:
            group = operand_grouping_system.get_operand_group(indicator)
            if group in groups_classified:
                groups_classified[group].append(indicator)

        # 各グループに指標が存在することを確認
        for group, indicators in groups_classified.items():
            assert len(indicators) > 0, f"グループ {group.value} に指標が存在しません"
            print(f"グループ {group.value}: {len(indicators)} 指標")

        # 総指標数を表示
        print(f"全指標数: {len(all_indicators)}")

    def test_new_indicator_classification(self):
        """新規指標分類テスト"""
        new_indicators = [
            # Trend系新規指標
            "HMA", "ZLMA", "VWMA", "SWMA", "ALMA", "JMA",
            # Volatility系新規指標
            "KELTNER", "DONCHIAN", "SUPERTREND",
            # Volume系新規指標
            "NVI", "PVI", "PVT",
            # Momentum系新規指標
            "RVI", "RMI", "DPO",
        ]

        for indicator in new_indicators:
            group = operand_grouping_system.get_operand_group(indicator)
            assert group is not None, f"新指標 {indicator} のグループ分類が失敗"
            assert isinstance(group, OperandGroup), f"無効なグループタイプ: {type(group)}"

    def test_existing_indicator_validation(self):
        """既存指標検証テスト"""
        existing_indicators = [
            "RSI", "MACD", "SMA", "VOLUME", "close", "CCI"
        ]

        # 期待されるグループマッピング（既知のもののみ）
        expected_mapping = {
            "RSI": OperandGroup.PERCENTAGE_0_100,
            "MACD": OperandGroup.ZERO_CENTERED,
            "SMA": OperandGroup.PRICE_BASED,
            "VOLUME": OperandGroup.SPECIAL_SCALE,
            "close": OperandGroup.PRICE_BASED,
            "CCI": OperandGroup.PERCENTAGE_NEG100_100
        }

        for indicator in existing_indicators:
            group = operand_grouping_system.get_operand_group(indicator)
            if indicator in expected_mapping:
                assert group == expected_mapping[indicator], \
                    f"指標 {indicator} のグループが期待値と一致しません: 期待={expected_mapping[indicator].value}, 実際={group.value}"
            else:
                assert isinstance(group, OperandGroup), f"指標 {indicator} のグループ分類が無効"

    def test_compatibility_matrix_validation(self):
        """互換性計算検証テスト"""
        test_pairs = [
            ("RSI", "STOCH", 1.0),  # 同一グループ
            ("RSI", "SMA", 0.1),    # 異なるグループ
            ("close", "EMA", 1.0),  # 同一グループ
            ("RSI", "CCI", 0.3),    # 中程度互換
            ("RSI", "VOLUME", 0.1), # 低い互換
        ]

        for left, right, expected_score in test_pairs:
            actual_score = operand_grouping_system.get_compatibility_score(left, right)
            assert actual_score == expected_score, \
                f"互換性スコア不一致: {left} vs {right}, 期待={expected_score}, 実際={actual_score}"

    def test_operand_discovery_functionality(self):
        """オペランド探索機能テスト"""
        target_operand = "RSI"
        available_operands = ["STOCH", "MACD", "SMA", "VOLUME", "close", "CCI"]

        # 高互換性オペランド探索
        compatible_operands = operand_grouping_system.get_compatible_operands(
            target_operand, available_operands, min_compatibility=0.8
        )

        # RSIと同じPERCENTAGE_0_100グループのSTOCHのみが選択されるはず
        assert "STOCH" in compatible_operands, "高互換オペランド探索が失敗"
        assert "MACD" not in compatible_operands, "低互換オペランドが誤って選択"
        assert len(compatible_operands) == 1, f"互換オペランド数が期待値と異なる: {compatible_operands}"

    def test_condition_validation_comprehensive(self):
        """条件妥当性検証の完全テスト"""
        test_conditions = [
            ("RSI", 75, True, "数値との比較"),  # 数値比較は常に有効
            ("RSI", "STOCH", True, "高い互換性"),  # 高互換ソース比較
            ("RSI", "SMA", False, "低い互換性"),   # 低互換ソース比較
            ("close", "SMA", True, "高い互換性"),  # 価格ベース同士
        ]

        for left, right, expected_valid, expected_reason in test_conditions:
            is_valid, reason = operand_grouping_system.validate_condition(left, right)

            # 検証結果のチェック
            valid_match = is_valid == expected_valid
            reason_match = expected_reason in reason if isinstance(reason, str) else False

            assert valid_match, f"条件妥当性検証不一致: {left} {right}, 期待={expected_valid}, 実際={is_valid}"
            assert reason_match, f"理由不一致: 期待='{expected_reason}', 実際='{reason}'"

    def test_mandatory_group_coverage(self):
        """必須グループカバー率確認テスト"""
        mandatory_groups = {
            OperandGroup.PRICE_BASED,
            OperandGroup.PERCENTAGE_0_100,
            OperandGroup.PERCENTAGE_NEG100_100,
            OperandGroup.ZERO_CENTERED,
            OperandGroup.SPECIAL_SCALE,
        }

        # 全指標グループを取得
        all_indicators = list(operand_grouping_system._group_mappings.keys())
        covered_groups = set()

        for indicator in all_indicators:
            group = operand_grouping_system.get_operand_group(indicator)
            covered_groups.add(group)

        # 全必須グループがカバーされていることを確認
        missing_groups = mandatory_groups - covered_groups
        assert len(missing_groups) == 0, f"必須グループがカバーされていません: {missing_groups}"

        # 各必須グループの指標数を確認
        print("必須グループカバー率:")
        for group in mandatory_groups:
            indicators_in_group = [i for i in all_indicators
                                 if operand_grouping_system.get_operand_group(i) == group]
            assert len(indicators_in_group) > 0, f"グループ {group.value} に指標がありません"
            print(f"  {group.value}: {len(indicators_in_group)} 指標")

    def test_error_handling_robustness(self):
        """エラーハンドリングと堅牢性テスト"""

        # 存在しない指標のテスト
        try:
            nonexistent_group = operand_grouping_system.get_operand_group("NONEXISTENT_INDICATOR")
            assert isinstance(nonexistent_group, OperandGroup), "存在しない指標がデフォルトグループ分類されるはず"
            print(f"存在しない指標のデフォルトグループ: {nonexistent_group.value}")
        except Exception as e:
            pytest.fail(f"存在しない指標で例外が発生: {e}")

        # Noneを渡すテスト
        with pytest.raises((TypeError, AttributeError)):
            operand_grouping_system.get_operand_group(None)

        # 空文字を渡すテスト
        try:
            empty_group = operand_grouping_system.get_operand_group("")
            assert isinstance(empty_group, OperandGroup), "空文字がデフォルトグループ分類されるはず"
        except Exception as e:
            pytest.fail(f"空文字で例外が発生: {e}")

        # 互換性スコアのエラーケース
        try:
            score = operand_grouping_system.get_compatibility_score("RSI", "NONEXISTENT")
            assert isinstance(score, float), "互換性スコアが浮動小数点数として返されるはず"
            assert 0.0 <= score <= 1.0, f"互換性スコア範囲外: {score}"
        except Exception as e:
            pytest.fail(f"互換性スコア計算で例外が発生: {e}")

        # 条件検証のエラーケース
        try:
            is_valid, reason = operand_grouping_system.validate_condition("NONEXISTENT", 100)
            assert isinstance(is_valid, bool), "条件検証結果がブール値として返されるはず"
            assert isinstance(reason, str), "理由が文字列として返されるはず"
        except Exception as e:
            pytest.fail(f"条件検証で例外が発生: {e}")

    def test_e2e_workflow_verification(self):
        """エンドツーエンドワークフロー検証"""

        # 1. Trend指標グループ分類
        trend_indicators = ["SMA", "EMA", "HMA", "close"]
        trend_groups = [operand_grouping_system.get_operand_group(ind) for ind in trend_indicators]

        # 全トレンド指標がPRICE_BASEDに分類されることを確認
        for indicator, group in zip(trend_indicators, trend_groups):
            assert group == OperandGroup.PRICE_BASED, f"トレンド指標 {indicator} がPRICE_BASEDグループではない: {group.value}"

        # 2. Volume指標グループ分類
        volume_indicators = ["volume", "NVI", "OBV"]
        volume_groups = [operand_grouping_system.get_operand_group(ind) for ind in volume_indicators]

        expected_volume_groups = [OperandGroup.SPECIAL_SCALE, OperandGroup.ZERO_CENTERED, OperandGroup.ZERO_CENTERED]
        for indicator, actual, expected in zip(volume_indicators, volume_groups, expected_volume_groups):
            assert actual == expected, f"Volume指標 {indicator} のグループ分類不一致: 期待={expected.value}, 実際={actual.value}"


        # 4. 互換性スコア検証
        test_pairs = [
            ("SMA", "EMA", 1.0),     # 同一グループ
            ("RSI", "STOCH", 1.0),   # 同一グループ
            ("RSI", "MACD", 0.3),    # 異なるグループ
            ("volume", "NVI", 0.3),  # 異なるグループ
        ]

        for left, right, expected in test_pairs:
            actual = operand_grouping_system.get_compatibility_score(left, right)
            assert actual == expected, f"互換性スコア不一致: {left} vs {right}, 期待={expected}, 実際={actual}"

        print("エンドツーエンドワークフロー検証完了")


if __name__ == "__main__":
    # テスト実行用のメインブロック
    pytest.main([__file__, "-v"])