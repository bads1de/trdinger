"""
条件評価器

戦略の条件評価ロジックを担当します。
"""

import logging
import numpy as np
from typing import List, Union, Dict, Any

from ..models.gene_strategy import Condition

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """
    条件評価器

    戦略の条件評価ロジックを担当します。
    """

    def evaluate_conditions(
        self, conditions: List[Condition], strategy_instance
    ) -> bool:
        """
        条件評価

        Args:
            conditions: 評価する条件リスト
            strategy_instance: 戦略インスタンス

        Returns:
            全条件がTrueかどうか
        """
        try:
            if not conditions:
                return True

            for condition in conditions:
                if not self.evaluate_single_condition(condition, strategy_instance):
                    return False
            return True

        except Exception as e:
            logger.error(f"条件評価エラー: {e}")
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
            オペランドの値
        """
        try:
            # 辞書の場合（指標を表す）
            if isinstance(operand, dict):
                indicator_name = operand.get("indicator")
                if indicator_name and hasattr(strategy_instance, indicator_name):
                    indicator_value = getattr(strategy_instance, indicator_name)
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

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
                    if hasattr(strategy_instance, operand.lower()):
                        # 直接属性から取得（backtesting.pyスタイル）
                        return float(getattr(strategy_instance, operand.lower()))
                    elif hasattr(strategy_instance, "data"):
                        # データフレームから取得
                        price_data = getattr(
                            strategy_instance.data, operand.capitalize()
                        )
                        return float(price_data[-1])

                # 指標の場合（Iディクショナリから取得 - 最優先）
                if (
                    hasattr(strategy_instance, "I")
                    and hasattr(strategy_instance.I, "__contains__")
                    and operand in strategy_instance.I
                ):
                    indicator_value = strategy_instance.I[operand]
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

                # 指標の場合（indicatorsディクショナリから取得）
                if (
                    hasattr(strategy_instance, "indicators")
                    and isinstance(strategy_instance.indicators, dict)
                    and operand in strategy_instance.indicators
                ):
                    indicator_value = strategy_instance.indicators[operand]
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

                # 指標の場合（直接属性から取得）
                if hasattr(strategy_instance, operand):
                    indicator_value = getattr(strategy_instance, operand)
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

                # 複数出力指標の場合（MACD_0, BB_0等）
                # 期間付き指標名の処理（例：STOCH_14 -> STOCH_0, CCI_28 -> CCI）
                if "_" in operand:
                    base_indicator = operand.split("_")[0]

                    # STOCHの場合
                    if base_indicator == "STOCH" and hasattr(
                        strategy_instance, "STOCH_0"
                    ):
                        indicator_value = getattr(strategy_instance, "STOCH_0")
                        if hasattr(indicator_value, "__getitem__"):
                            return float(indicator_value[-1])
                        return float(indicator_value)

                    # CCIの場合
                    if base_indicator == "CCI" and hasattr(strategy_instance, "CCI"):
                        indicator_value = getattr(strategy_instance, "CCI")
                        if hasattr(indicator_value, "__getitem__"):
                            return float(indicator_value[-1])
                        return float(indicator_value)

                    # RSIの場合
                    if base_indicator == "RSI" and hasattr(strategy_instance, "RSI"):
                        indicator_value = getattr(strategy_instance, "RSI")
                        if hasattr(indicator_value, "__getitem__"):
                            return float(indicator_value[-1])
                        return float(indicator_value)

                    # SMAの場合
                    if base_indicator == "SMA" and hasattr(strategy_instance, "SMA"):
                        indicator_value = getattr(strategy_instance, "SMA")
                        if hasattr(indicator_value, "__getitem__"):
                            return float(indicator_value[-1])
                        return float(indicator_value)

                    # EMAの場合
                    if base_indicator == "EMA" and hasattr(strategy_instance, "EMA"):
                        indicator_value = getattr(strategy_instance, "EMA")
                        if hasattr(indicator_value, "__getitem__"):
                            return float(indicator_value[-1])
                        return float(indicator_value)

                    # MACDの場合
                    if base_indicator == "MACD" and hasattr(
                        strategy_instance, "MACD_0"
                    ):
                        indicator_value = getattr(strategy_instance, "MACD_0")
                        if hasattr(indicator_value, "__getitem__"):
                            return float(indicator_value[-1])
                        return float(indicator_value)

                    # ボリンジャーバンドの場合
                    if operand.startswith("BB_"):
                        if "Upper" in operand and hasattr(strategy_instance, "BB_2"):
                            indicator_value = getattr(strategy_instance, "BB_2")
                        elif "Middle" in operand and hasattr(strategy_instance, "BB_1"):
                            indicator_value = getattr(strategy_instance, "BB_1")
                        elif "Lower" in operand and hasattr(strategy_instance, "BB_0"):
                            indicator_value = getattr(strategy_instance, "BB_0")
                        else:
                            indicator_value = None

                        if indicator_value is not None:
                            if hasattr(indicator_value, "__getitem__"):
                                return float(indicator_value[-1])
                            return float(indicator_value)

                # MACDの場合、MACD -> MACD_0（メインライン）にマッピング
                if operand == "MACD" and hasattr(strategy_instance, "MACD_0"):
                    # logger.info(f"MACDオペランドをMACD_0にマッピング")
                    indicator_value = getattr(strategy_instance, "MACD_0")
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

                # BBの場合、BB -> BB_1（中央線）にマッピング
                if operand == "BB" and hasattr(strategy_instance, "BB_1"):
                    # logger.info(f"BBオペランドをBB_1にマッピング")
                    indicator_value = getattr(strategy_instance, "BB_1")
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

                # STOCHの場合、STOCH -> STOCH_0（%K）にマッピング
                if operand == "STOCH" and hasattr(strategy_instance, "STOCH_0"):
                    # logger.info(f"STOCHオペランドをSTOCH_0にマッピング")
                    indicator_value = getattr(strategy_instance, "STOCH_0")
                    if hasattr(indicator_value, "__getitem__"):
                        return float(indicator_value[-1])
                    return float(indicator_value)

            logger.warning(
                f"未対応のオペランド: {operand} (利用可能な属性: {[attr for attr in dir(strategy_instance) if not attr.startswith('_')]})"
            )
            return 0.0

        except Exception as e:
            logger.error(f"オペランド値取得エラー: {e}")
            return -1  # エラーを示す値
