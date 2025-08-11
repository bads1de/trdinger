"""
条件評価器

戦略の条件評価ロジックを担当します。
"""

import logging
from typing import Any, Dict, List, Union

import numpy as np

from ..models.gene_strategy import Condition
from ..models.condition_group import ConditionGroup

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

        def _last_finite(x) -> float:
            try:
                import numpy as _np

                # シーケンス/配列
                if hasattr(x, "__getitem__") and not isinstance(x, (str, bytes)):
                    arr = _np.asarray(x, dtype=float)
                    # 後方から有限値を探索
                    for v in arr[::-1]:
                        if _np.isfinite(v):
                            return float(v)
                    return 0.0
                # スカラー
                val = float(x)
                return val if _np.isfinite(val) else 0.0
            except Exception:
                try:
                    return float(x)
                except Exception:
                    return 0.0

        try:
            # 辞書の場合（指標を表す）
            if isinstance(operand, dict):
                indicator_name = operand.get("indicator")
                if indicator_name and hasattr(strategy_instance, indicator_name):
                    indicator_value = getattr(strategy_instance, indicator_name)
                    return _last_finite(indicator_value)

            # 数値の場合
            if isinstance(operand, (int, float)):
                return float(operand)

            # 文字列の場合
            if isinstance(operand, str):
                # 数値文字列の場合
                if operand.replace(".", "").replace("-", "").isdigit():
                    return float(operand)

                # 価格データの場合
                if operand.lower() in ["close", "high", "low", "open"]:
                    # backtesting.pyのStrategyは属性名が小文字の価格を持たないため、data経由を優先
                    if hasattr(strategy_instance, "data"):
                        price_data = getattr(
                            strategy_instance.data, operand.capitalize()
                        )
                        return _last_finite(price_data)
                    # 念の為、戦略インスタンス直下の属性もフェイルセーフで参照
                    if hasattr(strategy_instance, operand.lower()):
                        return _last_finite(getattr(strategy_instance, operand.lower()))

                # 指標の場合（直接属性が最優先: IndicatorCalculatorで登録済み）
                if hasattr(strategy_instance, operand):
                    return _last_finite(getattr(strategy_instance, operand))

                # 複数出力指標や期間付き表記のマッピング
                if "_" in operand:
                    base_indicator = operand.split("_")[0]

                    # STOCH: %Kを使用
                    if base_indicator == "STOCH" and hasattr(
                        strategy_instance, "STOCH_0"
                    ):
                        return _last_finite(getattr(strategy_instance, "STOCH_0"))

                    # CCI, RSI, SMA, EMA: ベース名にマップ
                    if base_indicator in ["CCI", "RSI", "SMA", "EMA"] and hasattr(
                        strategy_instance, base_indicator
                    ):
                        return _last_finite(getattr(strategy_instance, base_indicator))

                    # MACD: メインライン
                    if base_indicator == "MACD" and hasattr(
                        strategy_instance, "MACD_0"
                    ):
                        return _last_finite(getattr(strategy_instance, "MACD_0"))

                    # BB: Upper/Middle/Lower -> 2/1/0
                    if operand.startswith("BB_"):
                        # 想定フォーマット: BB_Upper_{period} / BB_Middle_{period} / BB_Lower_{period}
                        # IndicatorCalculatorの登録順: 0=Upper, 1=Middle, 2=Lower
                        base = operand.split("_")[1]
                        if base == "Upper" and hasattr(strategy_instance, "BB_0"):
                            return _last_finite(getattr(strategy_instance, "BB_0"))
                        if base == "Middle" and hasattr(strategy_instance, "BB_1"):
                            return _last_finite(getattr(strategy_instance, "BB_1"))
                        if base == "Lower" and hasattr(strategy_instance, "BB_2"):
                            return _last_finite(getattr(strategy_instance, "BB_2"))

                # 単純名のマッピング（MACD, BB, STOCH）
                if operand == "MACD" and hasattr(strategy_instance, "MACD_0"):
                    return _last_finite(getattr(strategy_instance, "MACD_0"))
                if operand == "BB" and hasattr(strategy_instance, "BB_1"):
                    return _last_finite(getattr(strategy_instance, "BB_1"))
                if operand == "STOCH" and hasattr(strategy_instance, "STOCH_0"):
                    return _last_finite(getattr(strategy_instance, "STOCH_0"))

            logger.warning(
                f"未対応のオペランド: {operand} (利用可能な属性: {[attr for attr in dir(strategy_instance) if not attr.startswith('_')]})"
            )
            return 0.0

        except Exception as e:
            logger.error(f"オペランド値取得エラー: {e}")
            return -1  # エラーを示す値
