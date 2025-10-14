"""
条件評価器

戦略の条件評価ロジックを担当します。
"""

import logging
from typing import Any, Dict, List, Union

import pandas as pd
import numpy as np
from numbers import Real

from app.utils.error_handler import safe_operation

from ..models.strategy_models import Condition, ConditionGroup

logger = logging.getLogger(__name__)


class ConditionEvaluator:
    """
    条件評価器

    戦略の条件評価ロジックを担当します。
    """

    @safe_operation(context="条件評価（AND）", is_api_call=False, default_return=False)
    def evaluate_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]], strategy_instance
    ) -> bool:
        """
        条件評価（AND）

        - 通常のConditionはそのまま評価（AND）
        - ConditionGroupは内部のORで評価し、グループ全体を1つの条件として扱う
        """
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

    @safe_operation(
        context="条件グループ評価（OR）", is_api_call=False, default_return=False
    )
    def _evaluate_condition_group(
        self, group: ConditionGroup, strategy_instance
    ) -> bool:
        """ORグループ: 内部のいずれかがTrueならTrue"""
        if not group or group.is_empty():
            return False
        for c in group.conditions:
            if self.evaluate_single_condition(c, strategy_instance):
                return True
        return False

    @safe_operation(context="単一条件評価", is_api_call=False, default_return=False)
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
        # 左オペランドの値を取得
        left_value = self.get_condition_value(condition.left_operand, strategy_instance)
        right_value = self.get_condition_value(
            condition.right_operand, strategy_instance
        )

        # 両方の値が数値であることを確認（numpy互換）
        def is_numeric_value(val):
            import numpy as np

            # Pythonの組み込み数値型
            if isinstance(val, (int, float, complex)):
                return True
            # numpyの数値型
            if hasattr(val, "dtype") and np.issubdtype(val.dtype, np.number):
                return True
            # numbersモジュールの数値
            if isinstance(val, Real):
                return True
            return False

        # 元のオペランドが文字列で、数値変換が失敗した場合のチェック
        left_is_string = isinstance(condition.left_operand, str)
        right_is_string = isinstance(condition.right_operand, str)

        # 数値チェックのヘルパー関数
        def _is_original_numeric(operand) -> bool:
            """オペランドが本来の数値かどうかを判定"""
            if isinstance(operand, (int, float)):
                return True
            if isinstance(operand, str):
                cleaned = (
                    operand.replace(".", "")
                    .replace("-", "")
                    .replace("+", "")
                    .replace("e", "")
                    .replace("E", "")
                )
                return cleaned.isdigit() and "." not in operand[1:]
            return False

        left_is_original_numeric = _is_original_numeric(condition.left_operand)
        right_is_original_numeric = _is_original_numeric(condition.right_operand)

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

        return result

    @safe_operation(
        context="最終値取得（pandas-ta対応）", is_api_call=False, default_return=0.0
    )
    def _get_final_value(self, value) -> float:
        """配列/シーケンスから末尾の有限値を取得（pandas-ta対応）"""
        # pandas Series の場合
        if isinstance(value, pd.Series):
            return value.iloc[-1]  # pandas Seriesの末尾値
        # リスト/array対応
        elif hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
            val = float(value[-1])
            return val if pd.notna(val) else 0.0
        # スカラー値の場合
        else:
            val = float(value)
            return val if pd.notna(val) else 0.0

    @safe_operation(
        context="条件オペランド値取得", is_api_call=False, default_return=0.0
    )
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
            # OHLCVデータの処理
            if operand.lower() in ["open", "high", "low", "close", "volume"]:
                ohlcv_value = self._get_ohlcv_value(operand, strategy_instance)
                if ohlcv_value is not None:
                    return ohlcv_value

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

    def _get_ohlcv_value(self, operand: str, strategy_instance) -> float | None:
        """
        OHLCVデータの値を取得するヘルパーメソッド

        Args:
            operand: OHLCVオペランド（"open", "high", "low", "close", "volume"）
            strategy_instance: 戦略インスタンス

        Returns:
            OHLCVの値、取得できない場合はNone
        """
        logger.debug(
            f"[OHLCVアクセス] '{operand}' オペランド検出 - strategy_instance type: {type(strategy_instance)}"
        )

        if not hasattr(strategy_instance, "data"):
            logger.warning(
                f"[OHLCVアクセス] strategy_instanceにdata属性がありません。利用可能な属性: "
                f"{[attr for attr in dir(strategy_instance) if not attr.startswith('_')]}"
            )
            return None

        logger.debug(
            f"[OHLCVアクセス] strategy_instance.data 存在: {type(strategy_instance.data)}"
        )

        try:
            # backtesting.pyではカラム名が大文字（Open, High, Low, Close, Volume）
            capitalized_operand = operand.capitalize()

            # pandas DataFrameの場合
            if hasattr(strategy_instance.data, "columns") and hasattr(
                strategy_instance.data, "__getitem__"
            ):
                logger.debug("[OHLCVアクセス] pandas DataFrame検出")
                if capitalized_operand in strategy_instance.data.columns:
                    logger.debug(
                        f"[OHLCVアクセス] '{capitalized_operand}' カラムが見つかりました"
                    )
                    data_value = strategy_instance.data[capitalized_operand]
                    logger.debug(
                        f"[OHLCVアクセス] '{capitalized_operand}' から取得成功: {data_value}"
                    )
                    return self._get_final_value(data_value)
                else:
                    logger.warning(
                        f"[OHLCVアクセス] '{capitalized_operand}' カラムが見つかりません。"
                        f"利用可能なカラム: {list(strategy_instance.data.columns)}"
                    )
                    return None

            # backtesting.pyの特殊なデータアクセス方法
            try:
                data_value = getattr(strategy_instance.data, capitalized_operand)
                logger.debug(
                    f"[OHLCVアクセス] backtesting.pyデータアクセス成功: {data_value}"
                )
                return self._get_final_value(data_value)
            except AttributeError:
                logger.warning(
                    f"[OHLCVアクセス] backtesting.pyデータアクセス失敗: {capitalized_operand}属性なし"
                )
                return None

        except Exception as e:
            logger.warning(f"[OHLCVアクセス] データアクセスエラー: {e}")
            return None
