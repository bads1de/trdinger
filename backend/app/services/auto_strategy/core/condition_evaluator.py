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

    def __init__(self):
        """
        初期化
        """
        # オペランド名から戦略インスタンス上の属性名への解決結果をキャッシュ
        self._attr_cache: Dict[str, str] = {}
        # OHLCV名のマッピング
        self._ohlcv_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

    @safe_operation(context="条件評価（AND）", is_api_call=False, default_return=False)
    def evaluate_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]], strategy_instance
    ) -> bool:
        """
        条件リストの全体評価を実行（論理積：AND）

        リスト内の各条件（単一条件または条件グループ）を順次評価し、
        すべての条件が真である場合にのみTrueを返します。短絡評価を行います。

        ConditionGroupが含まれる場合、その内部（通常はOR）で評価された結果を
        一つの条件として扱います。

        Args:
            conditions: 評価対象の条件リスト
            strategy_instance: パラメータやデータへのアクセスを提供する戦略インスタンス

        Returns:
            すべての条件が成立していればTrue
        """
        if not conditions:
            return True

        # strategy_instance がキャッシュを持っているか確認（高速化のため）
        # Mockオブジェクトの場合はキャッシュ作成をスキップ
        from unittest.mock import Mock

        if not isinstance(strategy_instance, Mock) and not hasattr(
            strategy_instance, "_val_accessor_cache"
        ):
            strategy_instance._val_accessor_cache = {}

        if not isinstance(conditions, list):
            # logger.warning(f"条件リストが不正な型です: {type(conditions)}")
            return False

        for cond in conditions:
            if not self._evaluate_recursive_item(cond, strategy_instance):
                return False
        return True

    def calculate_conditions_vectorized(
        self, conditions: List[Union[Condition, ConditionGroup]], strategy_instance
    ) -> Union[pd.Series, np.ndarray, None]:
        """
        条件リストの全体評価をベクトル化して実行（論理積：AND）

        Args:
            conditions: 評価対象の条件リスト
            strategy_instance: 戦略インスタンス

        Returns:
            全期間の評価結果を含むBoolean Series/Array。計算不可能な場合はNone。
        """
        if not conditions:
            # 条件がない場合は常にTrue
            if hasattr(strategy_instance, "data") and hasattr(
                strategy_instance.data, "index"
            ):
                # backtesting.pyのデータ構造に対応
                idx = strategy_instance.data.index
                return pd.Series(True, index=idx)
            return None

        final_mask = None

        for cond in conditions:
            result = self._evaluate_recursive_item_vectorized(cond, strategy_instance)
            if result is None:
                return None  # ベクトル化不可能

            if final_mask is None:
                final_mask = result
            else:
                final_mask = final_mask & result

        return final_mask

    def _evaluate_recursive_item_vectorized(
        self, item: Union[Condition, ConditionGroup], strategy_instance
    ) -> Union[pd.Series, np.ndarray, None]:
        """再帰的なベクトル化評価"""
        if isinstance(item, ConditionGroup):
            return self._evaluate_condition_group_vectorized(item, strategy_instance)
        elif isinstance(item, Condition):
            return self.evaluate_single_condition_vectorized(item, strategy_instance)
        return None

    def _evaluate_condition_group_vectorized(
        self, group: ConditionGroup, strategy_instance
    ) -> Union[pd.Series, np.ndarray, None]:
        """条件グループのベクトル化評価"""
        if not group or group.is_empty():
            return None

        results = []
        for c in group.conditions:
            res = self._evaluate_recursive_item_vectorized(c, strategy_instance)
            if res is None:
                return None
            results.append(res)

        if not results:
            return None

        # Combine
        combined = results[0]
        is_and = getattr(group, "operator", "OR") == "AND"

        for r in results[1:]:
            if is_and:
                combined = combined & r
            else:
                combined = combined | r

        return combined

    def evaluate_single_condition_vectorized(
        self, condition: Condition, strategy_instance
    ) -> Union[pd.Series, np.ndarray, None]:
        """単一条件のベクトル化評価"""
        try:
            left_val = self.get_condition_vector(
                condition.left_operand, strategy_instance
            )
            right_val = self.get_condition_vector(
                condition.right_operand, strategy_instance
            )

            if left_val is None or right_val is None:
                return None

            # 比較実行 (pandas/numpyのオーバーロード演算子を利用)
            # 値の長さが違う場合のブロードキャストはライブラリ任せ
            op = condition.operator
            if op == ">":
                return left_val > right_val
            if op == "<":
                return left_val < right_val
            if op == ">=":
                return left_val >= right_val
            if op == "<=":
                return left_val <= right_val
            if op == "==":
                return left_val == right_val
            if op == "!=":
                return left_val != right_val
            
            # クロス判定 (pandas Seriesを想定)
            if op == "CROSS_UP":
                # (今回 > 今回) かつ (前回 <= 前回)
                # left_val, right_valがSeriesであることを期待
                try:
                    # shift(1)するために Series であるか確認。スカラーならそのまま
                    l_curr = left_val
                    r_curr = right_val
                    
                    l_prev = l_curr.shift(1) if hasattr(l_curr, "shift") else l_curr
                    r_prev = r_curr.shift(1) if hasattr(r_curr, "shift") else r_curr
                    
                    return (l_curr > r_curr) & (l_prev <= r_prev)
                except Exception:
                    return None
            
            if op == "CROSS_DOWN":
                # (今回 < 今回) かつ (前回 >= 前回)
                try:
                    l_curr = left_val
                    r_curr = right_val
                    
                    l_prev = l_curr.shift(1) if hasattr(l_curr, "shift") else l_curr
                    r_prev = r_curr.shift(1) if hasattr(r_curr, "shift") else r_curr
                    
                    return (l_curr < r_curr) & (l_prev >= r_prev)
                except Exception:
                    return None

            return None
        except Exception:
            return None

    def get_condition_vector(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> Union[pd.Series, np.ndarray, float, None]:
        """オペランドからベクトル（またはスカラー）を取得"""
        # 1. スカラー
        if isinstance(operand, (int, float, np.number)):
            return float(operand)

        # 2. 識別子解決
        operand_str = (
            operand.get("indicator") if isinstance(operand, dict) else str(operand)
        )

        # OHLCV
        val = self._get_ohlcv_vector(operand_str, strategy_instance)
        if val is not None:
            return val

        # Indicators
        if hasattr(strategy_instance, "indicators") and isinstance(
            strategy_instance.indicators, dict
        ):
            if operand_str in strategy_instance.indicators:
                return strategy_instance.indicators[operand_str]

        # Attributes
        # strategy.sma_20 のように直接属性として持っている場合も考慮
        if hasattr(strategy_instance, operand_str):
            val = getattr(strategy_instance, operand_str)
            if isinstance(val, (pd.Series, np.ndarray, int, float)):
                return val

        # Try float conversion
        try:
            return float(operand_str)
        except (ValueError, TypeError):
            return None

    def _get_ohlcv_vector(
        self, operand: str, strategy_instance
    ) -> Union[pd.Series, np.ndarray, None]:
        """OHLCVの全データを取得"""
        search_key = operand.lower()
        if search_key not in self._ohlcv_map:
            return None

        if not hasattr(strategy_instance, "data"):
            return None

        attr_name = self._ohlcv_map[search_key]
        data = strategy_instance.data

        if hasattr(data, attr_name):
            # backtesting.pyの_Arrayはnumpy array互換
            return getattr(data, attr_name)

        if isinstance(data, pd.DataFrame):
            if attr_name in data.columns:
                return data[attr_name]
            if search_key in data.columns:
                return data[search_key]

        return None

    def _evaluate_recursive_item(
        self, item: Union[Condition, ConditionGroup], strategy_instance
    ) -> bool:
        """
        条件アイテムの種類（単一またはグループ）を判定して再帰的に評価

        Args:
            item: 評価対象のConditionまたはConditionGroup
            strategy_instance: 戦略インスタンス

        Returns:
            評価結果
        """
        if isinstance(item, ConditionGroup):
            return self._evaluate_condition_group(item, strategy_instance)
        elif isinstance(item, Condition):
            return self.evaluate_single_condition(item, strategy_instance)

        # logger.warning(f"不明な条件タイプ: {type(item)}")
        return False

    @safe_operation(context="条件グループ評価", is_api_call=False, default_return=False)
    def _evaluate_condition_group(
        self, group: ConditionGroup, strategy_instance
    ) -> bool:
        """
        条件グループ内の全条件を評価（AND/OR対応）

        グループが保持する演算子（デフォルトはOR）に従って、
        内部の条件リストに対する評価を統合します。

        Args:
            group: 評価対象のConditionGroup
            strategy_instance: 戦略インスタンス

        Returns:
            グループ全体としての評価結果
        """
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
        """
        最末端の単一条件を評価

        左辺と右辺のオペランドの値を解決（数値化）し、
        指定された比較演算子を適用します。

        Args:
            condition: 評価対象のConditionオブジェクト
            strategy_instance: 戦略インスタンス

        Returns:
            比較結果（数値化不可能等の場合はFalse）
        """
        left_val = self.get_condition_value(condition.left_operand, strategy_instance)
        right_val = self.get_condition_value(condition.right_operand, strategy_instance)

        # 1. 数値妥当性チェック
        if not self._is_comparable(left_val, right_val):
            self._log_comparison_warning(condition, left_val, right_val)
            return False

        # 2. クロス判定の特別処理
        if condition.operator in ["CROSS_UP", "CROSS_DOWN"]:
            # 前回値の取得
            prev_left = self._get_previous_value(
                condition.left_operand, strategy_instance
            )
            prev_right = self._get_previous_value(
                condition.right_operand, strategy_instance
            )

            # 前回値が取得できない、または数値でない場合はFalse
            if not self._is_comparable(prev_left, prev_right):
                return False

            if condition.operator == "CROSS_UP":
                # (今回 > 今回) かつ (前回 <= 前回)
                return (left_val > right_val) and (prev_left <= prev_right)
            elif condition.operator == "CROSS_DOWN":
                # (今回 < 今回) かつ (前回 >= 前回)
                return (left_val < right_val) and (prev_left >= prev_right)

        # 3. 通常の比較演算の実行
        return self._compare_values(left_val, right_val, condition.operator)

    def _get_previous_value(self, operand, strategy_instance) -> float:
        """オペランドの1つ前の値を取得（可能な場合）"""
        # 数値リテラルの場合は変わらない
        if isinstance(operand, (int, float, np.number)):
            return float(operand)

        operand_str = (
            operand.get("indicator") if isinstance(operand, dict) else str(operand)
        )

        val = None
        # 1. OHLCVデータ
        temp_val = self._get_ohlcv_vector(operand_str, strategy_instance)
        if temp_val is not None:
            val = temp_val
        
        # 2. Indicators
        elif hasattr(strategy_instance, "indicators") and isinstance(strategy_instance.indicators, dict):
            if operand_str in strategy_instance.indicators:
                val = strategy_instance.indicators[operand_str]

        # 3. Attributes
        elif hasattr(strategy_instance, operand_str):
            val = getattr(strategy_instance, operand_str)
        
        # 値取得成功した場合、前回値の抽出を試みる
        if val is not None:
            try:
                # pandas Series / DataFrame
                if hasattr(val, "iloc"):
                    if len(val) >= 2:
                        return float(val.iloc[-2])
                
                # numpy array / list / backtesting._Array
                # __getitem__ を持つものを汎用的に扱う
                elif hasattr(val, "__getitem__"):
                    if len(val) >= 2:
                        return float(val[-2])
            except (IndexError, ValueError, TypeError):
                pass
        
        # 取得できない場合はNaNを返す（_is_comparableで弾かれる）
        return float("nan")

    def _is_comparable(self, v1: Any, v2: Any) -> bool:
        """
        値が比較可能（数値かつ非NaN）かどうかを判定します。

        Args:
            v1: 比較する値1
            v2: 比較する値2

        Returns:
            両方の値が有効な数値型であり、かつNaNでない場合はTrue
        """

        def check(v):
            if not isinstance(v, (int, float, Real, np.number)):
                return False
            # NaN（非数）は比較不可とする
            try:
                return not np.isnan(float(v))
            except (TypeError, ValueError):
                return False

        return check(v1) and check(v2)

    def _compare_values(self, v1: float, v2: float, operator: str) -> bool:
        """
        演算子に応じた比較を実行します。

        Args:
            v1: 左辺の値
            v2: 右辺の値
            operator: 比較演算子 (">", "<", ">=", "<=", "==", "!=")

        Returns:
            比較結果
        """
        import operator as op_module

        ops = {
            ">": op_module.gt,
            "<": op_module.lt,
            ">=": op_module.ge,
            "<=": op_module.le,
            "==": lambda x, y: np.isclose(x, y, atol=1e-8),
            "!=": lambda x, y: not np.isclose(x, y, atol=1e-8),
        }

        func = ops.get(operator)
        if func:
            return bool(func(v1, v2))

        logger.warning(f"未対応の演算子: {operator}")
        return False

    def _log_comparison_warning(self, condition, left_val, right_val):
        """
        比較失敗時のログを出力します。

        Args:
            condition: 評価中の条件オブジェクト
            left_val: 左辺の解決された値
            right_val: 右辺の解決された値
        """
        # logger.warning(
        #     f"比較できない値です: left={left_val}({type(left_val)}), "
        #     f"right={right_val}({type(right_val)}), "
        #     f"original={condition.left_operand} {condition.operator} {condition.right_operand}"
        # )
        pass

    def _get_final_value(self, value: Any) -> float:
        """
        型に応じて末尾の有限値を取得します。

        pandas.Seriesやリストの場合は最後の要素を取得し、floatに変換します。
        要素がさらにコレクションの場合は、再帰的に解決します。

        Args:
            value: 取得対象の値

        Returns:
            変換されたfloat値。失敗時や非有限値の場合はnp.nan
        """
        try:
            # 1. すでに数値の場合はそのまま返す
            if isinstance(value, (int, float, np.number)):
                f_val = float(value)
                return f_val if np.isfinite(f_val) else 0.0

            # 2. pandas.Series の場合
            if isinstance(value, pd.Series):
                if value.empty:
                    return 0.0
                return float(value.iloc[-1])

            # 3. リスト、タプル、ndarray などのコレクションの場合
            if hasattr(value, "__getitem__") and not isinstance(value, (str, bytes)):
                if len(value) == 0:
                    return 0.0
                # 最後の要素を再帰的に解決
                return self._get_final_value(value[-1])

            # 4. 文字列などの場合は数値変換を試みる
            f_val = float(value)
            return f_val if np.isfinite(f_val) else 0.0
        except (TypeError, ValueError, IndexError, AttributeError):
            return 0.0

    def get_condition_value(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> float:
        """
        オペランドから具体的な数値を取得します。

        Args:
            operand: オペランド（数値、文字列、または辞書形式の指標指定）
            strategy_instance: 戦略インスタンス（属性アクセス用）

        Returns:
            取得された数値（解決失敗時はnp.nan）
        """
        if isinstance(operand, (int, float, np.number)):
            return float(operand)

        # 辞書形式（{"indicator": "NAME", ...}）の場合は indicator キーを使用
        operand_str = (
            operand.get("indicator") if isinstance(operand, dict) else str(operand)
        )

        # 1. OHLCVデータ（文字列の場合、大文字小文字を問わずチェック）
        if isinstance(operand, str):
            val = self._get_ohlcv_value(operand, strategy_instance)
            if val is not None:
                return val

        # 2. 戦略インスタンスからの取得（キャッシュと辞書を活用）
        # まずは indicators 辞書を確認（属性アクセスより速い）
        # Mockの場合を考慮して isinstance チェックを追加
        if hasattr(strategy_instance, "indicators") and isinstance(
            strategy_instance.indicators, dict
        ):
            if operand_str in strategy_instance.indicators:
                return self._get_final_value(strategy_instance.indicators[operand_str])

        # 次に属性キャッシュを確認
        cached_attr = self._attr_cache.get(operand_str)
        if cached_attr and hasattr(strategy_instance, cached_attr):
            return self._get_final_value(getattr(strategy_instance, cached_attr))

        # 属性名を解決してキャッシュに保存
        try:
            for variant in [operand_str, operand_str.lower(), operand_str.upper()]:
                if hasattr(strategy_instance, variant):
                    self._attr_cache[operand_str] = variant
                    attr_val = getattr(strategy_instance, variant)
                    return self._get_final_value(attr_val)
        except (AttributeError, Exception):
            pass

        # 3. 数値文字列への変換
        try:
            if isinstance(operand, str):
                f_val = float(operand)
                return f_val if np.isfinite(f_val) else 0.0
        except (ValueError, TypeError):
            pass

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
        search_key = operand.lower()
        if search_key not in self._ohlcv_map:
            return None

        if not hasattr(strategy_instance, "data"):
            return None

        try:
            data = strategy_instance.data
            attr_name = self._ohlcv_map[search_key]

            # backtesting.py のデータ構造 (Open, High, Low, Close, Volume) を優先
            if hasattr(data, attr_name):
                # 毎回のアクセスで iloc[-1] または [-1] を取得
                # Mockの場合は特別な処理を避ける
                val = getattr(data, attr_name)
                # 高速化のため、Seriesなら直接 iloc[-1]
                if isinstance(val, pd.Series):
                    return float(val.iloc[-1])
                # ndarrayなら [-1]
                if isinstance(val, np.ndarray):
                    return float(val[-1])
                # その他は再帰的に解決
                return self._get_final_value(val)

            # DataFrameとしてのアクセス
            if isinstance(data, pd.DataFrame):
                if attr_name in data.columns:
                    return float(data[attr_name].iloc[-1])
                if search_key in data.columns:
                    return float(data[search_key].iloc[-1])

        except Exception:
            pass

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
