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

from ..genes import Condition, ConditionGroup

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
        """条件グループ評価（AND/OR対応）"""
        if not group or group.is_empty():
            return False

        eval_fn = self._evaluate_recursive_item
        if getattr(group, "operator", "OR") == "AND":
            return all(eval_fn(c, strategy_instance) for c in group.conditions)
        else:
            return any(eval_fn(c, strategy_instance) for c in group.conditions)

    def evaluate_single_condition(
        self, condition: Condition, strategy_instance
    ) -> bool:
        """単一条件の評価"""
        left_val = self.get_condition_value(condition.left_operand, strategy_instance)
        right_val = self.get_condition_value(condition.right_operand, strategy_instance)

        # 1. 数値妥当性チェック
        if not self._is_comparable(left_val, right_val):
            self._log_comparison_warning(condition, left_val, right_val)
            return False

        # 2. 比較演算の実行
        return self._compare_values(left_val, right_val, condition.operator)

    def _is_comparable(self, v1: Any, v2: Any) -> bool:
        """値が比較可能（数値）かどうかを判定"""
        def check(v):
            return isinstance(v, (int, float, Real, np.number))
        return check(v1) and check(v2)

    def _compare_values(self, v1: float, v2: float, operator: str) -> bool:
        """演算子に応じた比較を実行"""
        import operator as op_module
        
        ops = {
            ">": op_module.gt,
            "<": op_module.lt,
            ">=": op_module.ge,
            "<=": op_module.le,
            "==": lambda x, y: np.isclose(x, y, atol=1e-8),
            "!=": lambda x, y: not np.isclose(x, y, atol=1e-8)
        }
        
        func = ops.get(operator)
        if func:
            return bool(func(v1, v2))
        
        logger.warning(f"未対応の演算子: {operator}")
        return False

    def _log_comparison_warning(self, condition, left_val, right_val):
        """比較失敗時のログ出力"""
        logger.warning(
            f"比較できない値です: left={left_val}({type(left_val)}), "
            f"right={right_val}({type(right_val)}), "
            f"original={condition.left_operand} {condition.operator} {condition.right_operand}"
        )

    def _get_final_value(self, value: Any) -> float:
        """型に応じて末尾の有限値を取得"""
        try:
            if isinstance(value, pd.Series):
                val = value.iloc[-1]
            elif hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
                val = value[-1]
            else:
                val = value
            
            f_val = float(val)
            return f_val if np.isfinite(f_val) else 0.0
        except (TypeError, ValueError, IndexError):
            return 0.0

    def get_condition_value(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> float:
        """オペランドから値を取得"""
        if isinstance(operand, (int, float)):
            return float(operand)

        target_attr = operand.get("indicator") if isinstance(operand, dict) else str(operand)
        
        # 1. OHLCVデータ（文字列の場合）
        if isinstance(operand, str) and operand.lower() in ["open", "high", "low", "close", "volume"]:
            val = self._get_ohlcv_value(operand, strategy_instance)
            if val is not None: return val

        # 2. 戦略インスタンスの属性（pandas-ta指標など）
        try:
            attr_val = getattr(strategy_instance, target_attr)
            return self._get_final_value(attr_val)
        except AttributeError:
            # 3. 数値文字列への変換
            try:
                if isinstance(operand, str): return float(operand)
            except (ValueError, TypeError):
                pass

        logger.warning(f"オペランド解決失敗: {target_attr}")
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
        from ..genes.conditions import StatefulCondition

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
        from ..genes.conditions import StatefulCondition

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





