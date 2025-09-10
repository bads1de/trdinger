"""
条件評価器

戦略の条件評価ロジックを担当します。
"""

import logging
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np
from numbers import Real

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

            if not isinstance(conditions, list):
                logger.warning(f"条件リストが不正な型です: {type(conditions)}")
                return False

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

            # 両方の値が数値であることを確認（numpy互換）
            def is_numeric_value(val):
                import numpy as np
                # Pythonの組み込み数値型
                if isinstance(val, (int, float, complex)):
                    return True
                # numpyの数値型
                if hasattr(val, 'dtype') and np.issubdtype(val.dtype, np.number):
                    return True
                # numbersモジュールの数値
                if isinstance(val, Real):
                    return True
                return False

            # 元のオペランドが文字列で、数値変換が失敗した場合のチェック
            left_is_string = isinstance(condition.left_operand, str)
            right_is_string = isinstance(condition.right_operand, str)

            # 値が数値でも、元のオペランドが数値文字列でない場合の警告
            left_is_original_numeric = isinstance(condition.left_operand, (int, float)) or (
                isinstance(condition.left_operand, str) and
                condition.left_operand.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit() and
                '.' not in condition.left_operand[1:]  # 小数点は先頭以外に1つだけ
            )

            right_is_original_numeric = isinstance(condition.right_operand, (int, float)) or (
                isinstance(condition.right_operand, str) and
                condition.right_operand.replace('.', '').replace('-', '').replace('+', '').replace('e', '').replace('E', '').isdigit() and
                '.' not in condition.right_operand[1:]  # 小数点は先頭以外に1つだけ
            )

            if not is_numeric_value(left_value) or not is_numeric_value(right_value):
                logger.warning(
                    f"比較できない値です: left={left_value}({type(left_value)}), "
                    f"right={right_value}({type(right_value)}), "
                    f"original_left={condition.left_operand}, original_right={condition.right_operand}"
                )
                return False
            elif left_is_string and not left_is_original_numeric and left_value == 0.0:
                # 元のオペランドが数値文字列ではなく、値が0.0の場合（属性アクセス失敗）
                logger.warning(
                    f"非数値文字列オペランドが数値に変換されました: '{condition.left_operand}' -> {left_value}, "
                    f"戦略インスタンスの属性が見つかりませんでした"
                )
            elif right_is_string and not right_is_original_numeric and right_value == 0.0:
                # 右オペランドも同様
                logger.warning(
                    f"非数値文字列オペランドが数値に変換されました: '{condition.right_operand}' -> {right_value}, "
                    f"戦略インスタンスの属性が見つかりませんでした"
                )

            # 条件評価（numpy互換）
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
                result = np.isclose(left_value, right_value, atol=1e-8)
            elif condition.operator == "!=":
                result = not np.isclose(left_value, right_value, atol=1e-8)
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

    def _get_final_value(self, value) -> float:
        """配列/シーケンスから末尾の有限値を取得（pandas-ta対応）"""
        try:
            # pandas Series の場合
            if isinstance(value, pd.Series):
                return value.iloc[-1]  # pandas Seriesの末尾値
            # リスト/array対応
            elif hasattr(value, '__getitem__') and not isinstance(value, (str, bytes)):
                val = float(value[-1])
                return val if pd.notna(val) else 0.0
            # スカラー値の場合
            else:
                val = float(value)
                return val if pd.notna(val) else 0.0
        except Exception:
            try:
                return float(value)
            except Exception:
                return 0.0

    def get_condition_value(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> float:
        """
        条件オペランドから値を取得（pandas-ta統合済み）

        Args:
            operand: オペランド（辞書、文字列、数値）
            strategy_instance: 戦略インスタンス

        Returns:
            オペランドの値（末尾の有限値を優先して返す）
        """
        try:
            # 数値はそのまま返す
            if isinstance(operand, (int, float)):
                return float(operand)

            # dictオペランドの処理
            if isinstance(operand, dict):
                indicator_name = operand.get("indicator")
                if indicator_name:
                    # pandas-ta直接アクセス
                    try:
                        value = getattr(strategy_instance, indicator_name)
                        return self._get_final_value(value)
                    except AttributeError:
                        pass  # 処理続行

            # 文字列オペランドの処理
            if isinstance(operand, str):
                # pandas-taの標準指標名で直接アクセス
                try:
                    value = getattr(strategy_instance, operand)
                    return self._get_final_value(value)
                except AttributeError:
                    # 数値文字列として変換を試みる
                    try:
                        # 数値に変換可能か試す
                        numeric_value = float(operand)
                        return numeric_value
                    except (ValueError, TypeError):
                        # 未解決の場合の警告
                        logger.warning(
                            f"未対応のオペランド: {operand} (利用可能な属性: "
                            f"{[attr for attr in dir(strategy_instance) if not attr.startswith('_')]})"
                        )
                        return 0.0

            # それ以外は失敗
            return 0.0
        except Exception as e:
            logger.error(f"オペランド値取得エラー: {e}")
            return 0.0
