"""
条件評価器

戦略の条件評価ロジックを担当します。
"""

import logging
from numbers import Real
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

from app.utils.error_handler import safe_operation

from ..models import Condition, ConditionGroup

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
            if not self._evaluate_recursive_item(cond, strategy_instance):
                return False
        return True

    def _evaluate_recursive_item(
        self, item: Union[Condition, ConditionGroup], strategy_instance
    ) -> bool:
        """再帰的に条件アイテムを評価"""
        if isinstance(item, ConditionGroup):
            return self._evaluate_condition_group(item, strategy_instance)
        elif isinstance(item, Condition):
            return self.evaluate_single_condition(item, strategy_instance)

        logger.warning(f"不明な条件タイプ: {type(item)}")
        return False

    @safe_operation(context="条件グループ評価", is_api_call=False, default_return=False)
    def _evaluate_condition_group(
        self, group: ConditionGroup, strategy_instance
    ) -> bool:
        """
        条件グループ評価

        operator="AND" -> 全てTrueならTrue
        operator="OR"  -> いずれかがTrueならTrue
        """
        if not group or group.is_empty():
            return False

        is_and = getattr(group, "operator", "OR") == "AND"

        for c in group.conditions:
            result = self._evaluate_recursive_item(c, strategy_instance)

            if is_and:
                if not result:
                    return False
            else:  # OR
                if result:
                    return True

        return True if is_and else False

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
        # 高頻度で呼ばれるためデバッグログは削除
        # logger.debug(
        #     f"[OHLCVアクセス] '{operand}' オペランド検出 - strategy_instance type: {type(strategy_instance)}"
        # )

        if not hasattr(strategy_instance, "data"):
            # 頻繁に出る可能性があるのでdebugに下げてコメントアウト、またはonceにする
            # logger.warning(...)
            return None

        try:
            # backtesting.pyではカラム名が大文字（Open, High, Low, Close, Volume）
            capitalized_operand = operand.capitalize()

            # pandas DataFrameの場合
            if hasattr(strategy_instance.data, "columns") and hasattr(
                strategy_instance.data, "__getitem__"
            ):
                if capitalized_operand in strategy_instance.data.columns:
                    data_value = strategy_instance.data[capitalized_operand]
                    return self._get_final_value(data_value)
                else:
                    return None

            # backtesting.pyの特殊なデータアクセス方法
            try:
                data_value = getattr(strategy_instance.data, capitalized_operand)
                return self._get_final_value(data_value)
            except AttributeError:
                return None

        except Exception:
            return None

    # ========================================
    # StatefulCondition 評価メソッド
    # ========================================

    def evaluate_stateful_condition(
        self,
        stateful_condition,
        strategy_instance,
        state_tracker,
        current_bar: int,
    ) -> bool:
        """
        StatefulCondition を評価

        トリガー条件が過去lookback_bars以内に発生しており、
        かつフォロー条件が現在成立していればTrueを返します。

        Args:
            stateful_condition: StatefulCondition インスタンス
            strategy_instance: 戦略インスタンス
            state_tracker: StateTracker インスタンス
            current_bar: 現在のバーインデックス

        Returns:
            条件成立ならTrue
        """
        from ..models.stateful_condition import StatefulCondition

        if not isinstance(stateful_condition, StatefulCondition):
            logger.warning(f"不正な型: {type(stateful_condition)}")
            return False

        if not stateful_condition.enabled:
            return False

        # トリガーが過去lookback_bars以内に発生したか確認
        event_name = stateful_condition.get_trigger_event_name()
        trigger_in_range = state_tracker.was_triggered_within(
            event_name,
            lookback_bars=stateful_condition.lookback_bars,
            current_bar=current_bar,
        )

        if not trigger_in_range:
            return False

        # フォロー条件を評価
        follow_result = self.evaluate_single_condition(
            stateful_condition.follow_condition, strategy_instance
        )

        return follow_result

    def check_and_record_trigger(
        self,
        stateful_condition,
        strategy_instance,
        state_tracker,
        current_bar: int,
    ) -> bool:
        """
        トリガー条件を評価し、成立していればStateTrackerに記録

        Args:
            stateful_condition: StatefulCondition インスタンス
            strategy_instance: 戦略インスタンス
            state_tracker: StateTracker インスタンス
            current_bar: 現在のバーインデックス

        Returns:
            トリガー条件が成立したか
        """
        from ..models.stateful_condition import StatefulCondition

        if not isinstance(stateful_condition, StatefulCondition):
            return False

        if not stateful_condition.enabled:
            return False

        # トリガー条件を評価
        trigger_result = self.evaluate_single_condition(
            stateful_condition.trigger_condition, strategy_instance
        )

        if trigger_result:
            # StateTrackerにイベントを記録
            event_name = stateful_condition.get_trigger_event_name()
            state_tracker.record_event(event_name, bar_index=current_bar)
            logger.debug(f"トリガー記録: {event_name} at bar {current_bar}")

        return trigger_result
