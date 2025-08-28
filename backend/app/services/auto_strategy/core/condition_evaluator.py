"""
条件評価器

戦略の条件評価ロジックを担当します。
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np

from ..models.strategy_models import Condition, ConditionGroup

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """
    条件評価器

    戦略の条件評価ロジックを担当します。
    """

    def evaluate_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]], strategy_instance
    ) -> bool:
        """
        条件評価（AND）

        - 通常のConditionはそのまま評価（AND）
        - ConditionGroupは内部のORで評価し、グループ全体を1つの条件として扱う
        """
        try:
            if not conditions:
                return True

            for cond in conditions:
                if isinstance(cond, ConditionGroup):
                    if not self._evaluate_condition_group(cond, strategy_instance):
                        return False
                else:
                    if not self.evaluate_single_condition(cond, strategy_instance):
                        return False
            return True

        except Exception as e:
            logger.error(f"条件評価エラー: {e}")
            return False

    def _evaluate_condition_group(
        self, group: ConditionGroup, strategy_instance
    ) -> bool:
        """ORグループ: 内部のいずれかがTrueならTrue"""
        try:
            if not group or group.is_empty():
                return False
            for c in group.conditions:
                if self.evaluate_single_condition(c, strategy_instance):
                    return True
            return False
        except Exception as e:
            logger.error(f"条件グループ評価エラー: {e}")
            return False

    def evaluate_single_condition(
        self, condition: Condition, strategy_instance
    ) -> bool:
        """
        単一条件の評価

        Args:
            condition: 評価する条件
            strategy_instance: 戦略インスタンス

        Returns:
            条件の評価結果
        """
        try:
            # 左オペランドの値を取得
            left_value = self.get_condition_value(
                condition.left_operand, strategy_instance
            )
            right_value = self.get_condition_value(
                condition.right_operand, strategy_instance
            )

            # ML指標の場合はデバッグログを出力
            if isinstance(
                condition.left_operand, str
            ) and condition.left_operand.startswith("ML_"):
                logger.info(
                    f"[ML条件評価デバッグ] {condition.left_operand} {condition.operator} {condition.right_operand} => {left_value} {condition.operator} {right_value}"
                )

            # 両方の値が数値であることを確認
            if not isinstance(left_value, (int, float)) or not isinstance(
                right_value, (int, float)
            ):
                logger.warning(
                    f"比較できない値です: left={left_value}({type(left_value)}), "
                    f"right={right_value}({type(right_value)})"
                )
                return False

            # 条件評価
            result = False
            if condition.operator == ">":
                result = left_value > right_value
            elif condition.operator == "<":
                result = left_value < right_value
            elif condition.operator == ">=":
                result = left_value >= right_value
            elif condition.operator == "<=":
                result = left_value <= right_value
            elif condition.operator == "==":
                result = bool(np.isclose(left_value, right_value))
            elif condition.operator == "!=":
                result = not bool(np.isclose(left_value, right_value))
            else:
                logger.warning(f"未対応の演算子: {condition.operator}")
                return False

            # ML指標の場合は評価結果もログ出力
            if isinstance(
                condition.left_operand, str
            ) and condition.left_operand.startswith("ML_"):
                logger.info(f"[ML条件評価デバッグ] 評価結果: {result}")

            return result

        except Exception as e:
            logger.error(f"条件評価エラー: {e}")
            return False

    def get_condition_value(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> float:
        """
        条件オペランドから値を取得

        Args:
            operand: オペランド（辞書、文字列、数値）
            strategy_instance: 戦略インスタンス

        Returns:
            オペランドの値（末尾の有限値を優先して返す）
        """

        # 値の末尾有限値取得は IndicatorNameResolver に集約

        try:
            # dictオペランドの処理は Resolver に完全委譲
            if isinstance(operand, dict):
                indicator_name = operand.get("indicator")
                if indicator_name:
                    from ..core.indicator_name_resolver import IndicatorNameResolver

                    resolved, value = IndicatorNameResolver.try_resolve_value(
                        indicator_name, strategy_instance
                    )
                    if resolved:
                        return value

            # 数値はそのまま
            if isinstance(operand, (int, float)):
                return float(operand)

            # 文字列はResolverへ
            if isinstance(operand, str):
                from ..core.indicator_name_resolver import IndicatorNameResolver

                resolved, value = IndicatorNameResolver.try_resolve_value(
                    operand, strategy_instance
                )
                if resolved:
                    return value
                # 未解決の場合のみ警告
                logger.warning(
                    f"未対応のオペランド: {operand} (利用可能な属性: "
                    f"{[attr for attr in dir(strategy_instance) if not attr.startswith('_')]})"
                )
                return 0.0

            # それ以外は失敗
            return 0.0
        except Exception as e:
            logger.error(f"オペランド値取得エラー: {e}")
            return -1  # エラーを示す値
