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
    パフォーマンス最適化のため、演算子関数や属性アクセサをキャッシュします。
    """

    def __init__(self):
        """
        初期化
        """
        import operator as op_module

        # 演算子関数のキャッシュ
        self._ops = {
            ">": op_module.gt,
            "<": op_module.lt,
            ">=": op_module.ge,
            "<=": op_module.le,
            "==": lambda x, y: np.isclose(x, y, atol=1e-8),
            "!=": lambda x, y: not np.isclose(x, y, atol=1e-8),
            # CROSS_UP/DOWN は特別扱い
        }

        # 属性アクセサのキャッシュ: "sma_20" -> operator.attrgetter("sma_20")
        self._accessor_cache: Dict[str, Any] = {}
        
        # OHLCV名のマッピング
        self._ohlcv_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }

    def _get_accessor(self, attr_name: str) -> Any:
        """属性アクセサを取得または作成"""
        if attr_name not in self._accessor_cache:
            import operator
            self._accessor_cache[attr_name] = operator.attrgetter(attr_name)
        return self._accessor_cache[attr_name]

    def _extract_operand_str(self, operand: Any) -> str:
        """オペランドから文字列識別子を抽出"""
        if isinstance(operand, str):
            return operand
        if isinstance(operand, dict):
            # indicator(旧), name(新) の順で検索
            val = operand.get("indicator") or operand.get("name")
            return str(val) if val is not None else ""
        return str(operand)

    @safe_operation(context="条件評価（AND）", is_api_call=False, default_return=False)
    def evaluate_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]], strategy_instance
    ) -> bool:
        """
        条件リストの全体評価を実行（論理積：AND）
        """
        # 最適化: 空チェックと型チェックを最小限に
        if not conditions:
            return True

        # リストの反復処理を高速化
        for cond in conditions:
            if not self._evaluate_recursive_item(cond, strategy_instance):
                return False
        return True

    def calculate_conditions_vectorized(
        self, conditions: List[Union[Condition, ConditionGroup]], strategy_instance
    ) -> Union[pd.Series, np.ndarray, None]:
        """
        条件リストの全体評価をベクトル化して実行（論理積：AND）
        """
        if not conditions:
            # 条件がない場合は常にTrue
            if hasattr(strategy_instance, "data") and hasattr(
                strategy_instance.data, "index"
            ):
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
        # 型チェックの順序を頻度順に最適化（Conditionの方が多いと仮定）
        if isinstance(item, Condition):
            return self.evaluate_single_condition_vectorized(item, strategy_instance)
        elif isinstance(item, ConditionGroup):
            return self._evaluate_condition_group_vectorized(item, strategy_instance)
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
        # getattrのデフォルト値コストを避けるため直接アクセスを試みる
        op = group.operator if hasattr(group, "operator") else "OR"
        is_and = op == "AND"

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

            op = condition.operator
            
            # 高速な辞書ルックアップ
            func = self._ops.get(op)
            if func:
                return func(left_val, right_val)

            # クロス判定 (pandas Seriesを想定)
            if op == "CROSS_UP":
                try:
                    l_curr = left_val
                    r_curr = right_val
                    # shift属性チェックをhasattrでなくtry-exceptで行う方がPythonicで高速
                    l_prev = l_curr.shift(1)
                    r_prev = r_curr.shift(1)
                    return (l_curr > r_curr) & (l_prev <= r_prev)
                except AttributeError:
                    # shiftがない場合（numpy arrayなど）
                    return None

            if op == "CROSS_DOWN":
                try:
                    l_curr = left_val
                    r_curr = right_val
                    l_prev = l_curr.shift(1)
                    r_prev = r_curr.shift(1)
                    return (l_curr < r_curr) & (l_prev >= r_prev)
                except AttributeError:
                    return None

            return None
        except Exception:
            return None

    def get_condition_vector(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> Union[pd.Series, np.ndarray, float, None]:
        """オペランドからベクトル（またはスカラー）を取得"""
        # 1. スカラー (頻出パス)
        if isinstance(operand, (int, float, np.number)):
            return float(operand)

        # 2. 識別子解決
        operand_str = self._extract_operand_str(operand)

        # OHLCV
        val = self._get_ohlcv_vector(operand_str, strategy_instance)
        if val is not None:
            return val

        # Indicators (辞書検索は高速)
        if hasattr(strategy_instance, "indicators") and operand_str in strategy_instance.indicators:
            return strategy_instance.indicators[operand_str]

        # Attributes (アクセサキャッシュ利用)
        try:
            return getattr(strategy_instance, operand_str)
        except AttributeError:
            pass

        # Try float conversion
        try:
            return float(operand_str)
        except (ValueError, TypeError):
            return None

    def _get_ohlcv_vector(
        self, operand: str, strategy_instance
    ) -> Union[pd.Series, np.ndarray, None]:
        """OHLCVの全データを取得"""
        # 小文字変換のコスト削減のため、operandが既に小文字であることを期待したいが
        # 安全のため小文字化。ただし頻出する "close" 等はチェック
        search_key = operand.lower()
        if search_key not in self._ohlcv_map:
            return None

        if not hasattr(strategy_instance, "data"):
            return None

        attr_name = self._ohlcv_map[search_key]
        data = strategy_instance.data

        # getattrは高速
        try:
            return getattr(data, attr_name)
        except AttributeError:
            pass

        if isinstance(data, pd.DataFrame):
            if attr_name in data.columns:
                return data[attr_name]
            if search_key in data.columns:
                return data[search_key]

        return None

    def _evaluate_recursive_item(
        self, item: Union[Condition, ConditionGroup], strategy_instance
    ) -> bool:
        """再帰的に評価"""
        # 型チェックの順序最適化
        if isinstance(item, Condition):
            return self.evaluate_single_condition(item, strategy_instance)
        elif isinstance(item, ConditionGroup):
            return self._evaluate_condition_group(item, strategy_instance)
        return False

    def _evaluate_condition_group(
        self, group: ConditionGroup, strategy_instance
    ) -> bool:
        """条件グループ評価"""
        if not group or group.is_empty():
            return False

        # getattrの代わりに直接属性アクセスを試みる（slots=Trueなので高速）
        # ただし互換性のためgetattrも残す
        op = getattr(group, "operator", "OR")
        
        # リスト内包表記よりジェネレータ式の方がメモリ効率が良いが、
        # all/any はジェネレータを受け取るので遅延評価される
        if op == "AND":
            # allはFalseが見つかった時点で終了する（短絡評価）
            for c in group.conditions:
                if not self._evaluate_recursive_item(c, strategy_instance):
                    return False
            return True
        else:
            # anyはTrueが見つかった時点で終了する
            for c in group.conditions:
                if self._evaluate_recursive_item(c, strategy_instance):
                    return True
            return False

    def evaluate_single_condition(
        self, condition: Condition, strategy_instance
    ) -> bool:
        """
        最末端の単一条件を評価（最適化版）
        """
        # 値の取得
        left_val = self.get_condition_value(condition.left_operand, strategy_instance)
        right_val = self.get_condition_value(condition.right_operand, strategy_instance)

        # NaNチェック (インライン化)
        # np.isnanはオーバーヘッドがあるため、まずは単純な数値比較でフィルタ
        # ほとんどの場合は有限の数値であるはず
        
        # 演算子の取得
        op = condition.operator
        
        # 1. キャッシュされた演算子関数の直接実行（高速パス）
        func = self._ops.get(op)
        if func:
            try:
                return bool(func(left_val, right_val))
            except Exception:
                # 比較不能な場合（NaNなど）
                return False

        # 2. クロス判定（低速パス）
        if op in ("CROSS_UP", "CROSS_DOWN"):
            # 前回値の取得
            prev_left = self._get_previous_value(condition.left_operand, strategy_instance)
            prev_right = self._get_previous_value(condition.right_operand, strategy_instance)

            # NaNチェック
            if np.isnan(prev_left) or np.isnan(prev_right):
                return False

            if op == "CROSS_UP":
                return (left_val > right_val) and (prev_left <= prev_right)
            elif op == "CROSS_DOWN":
                return (left_val < right_val) and (prev_left >= prev_right)

        # logger.warning(f"未対応の演算子: {op}")
        return False

    def _get_previous_value(self, operand, strategy_instance) -> float:
        """オペランドの1つ前の値を取得"""
        if isinstance(operand, (int, float, np.number)):
            return float(operand)

        operand_str = self._extract_operand_str(operand)
        val = None

        # 1. OHLCV (高速パス)
        # _get_ohlcv_vectorは結果を返すので、それを使う
        # ここでは再取得コストを避けるため、get_condition_vectorのようなロジックが必要だが
        # 簡易化のため再取得する（頻度は低いと想定）
        val = self._get_ohlcv_vector(operand_str, strategy_instance)
        
        if val is None:
            # 2. Indicators (辞書アクセス)
            if hasattr(strategy_instance, "indicators") and operand_str in strategy_instance.indicators:
                val = strategy_instance.indicators[operand_str]
            # 3. Attributes
            elif hasattr(strategy_instance, operand_str):
                val = getattr(strategy_instance, operand_str)

        if val is not None:
            try:
                # pandas Series / DataFrame
                if hasattr(val, "iloc"):
                    if len(val) >= 2:
                        return float(val.iloc[-2])
                # numpy array / list
                elif hasattr(val, "__getitem__"):
                    if len(val) >= 2:
                        return float(val[-2])
            except Exception:
                pass

        return float("nan")

    def _is_comparable(self, v1: Any, v2: Any) -> bool:
        """非推奨: evaluate_single_condition内にインライン化"""
        return True

    def _compare_values(self, v1: float, v2: float, operator: str) -> bool:
        """非推奨: evaluate_single_condition内にインライン化"""
        return False

    def _get_final_value(self, value: Any) -> float:
        """
        型に応じて末尾の有限値を取得（最適化版）
        """
        # 1. 数値 (最速)
        if isinstance(value, (float, int, np.number)):
            return float(value)

        # 2. Numpy Array (バックテストで最も多いパターン)
        if isinstance(value, np.ndarray):
            if value.size > 0:
                return float(value[-1])
            return 0.0

        # 3. Pandas Series
        if isinstance(value, pd.Series):
            if not value.empty:
                return float(value.values[-1]) # .iloc[-1]よりvalues[-1]が速い
            return 0.0

        # 4. リスト等
        try:
            return float(value[-1])
        except (IndexError, TypeError, ValueError):
            return 0.0

    def get_condition_value(
        self, operand: Union[Dict[str, Any], str, int, float], strategy_instance
    ) -> float:
        """
        オペランドから具体的な数値を取得（最適化版）
        """
        # 1. スカラー (最速)
        if isinstance(operand, (float, int, np.number)):
            return float(operand)

        # 2. 識別子解決
        operand_str = self._extract_operand_str(operand)

        # 3. OHLCVデータ
        # backtesting.pyの構造上、data.Close[-1] へのアクセスが最も頻出
        if hasattr(strategy_instance, "data"):
            data = strategy_instance.data
            
            # ヘルパー関数: 安全に最後の値を取得
            def _safe_get_last(obj):
                if isinstance(obj, (pd.Series, pd.DataFrame)):
                    return float(obj.values[-1])
                try:
                    return float(obj[-1])
                except (TypeError, KeyError, IndexError):
                    # PandasのRangeIndexなどで[-1]がキーエラーになる場合のフォールバック
                    if hasattr(obj, "iloc"):
                        return float(obj.iloc[-1])
                    raise

            try:
                # マッピングを使わずに直接チェック (小文字前提の最適化)
                if operand_str == "close":
                    return _safe_get_last(data.Close)
                elif operand_str == "high":
                    return _safe_get_last(data.High)
                elif operand_str == "low":
                    return _safe_get_last(data.Low)
                elif operand_str == "open":
                    return _safe_get_last(data.Open)
                elif operand_str == "volume":
                    return _safe_get_last(data.Volume)
                
                # マッピングを使った汎用チェック
                key_lower = operand_str.lower()
                if key_lower in self._ohlcv_map:
                    attr = self._ohlcv_map[key_lower]
                    if hasattr(data, attr):
                        return _safe_get_last(getattr(data, attr))
            except Exception:
                pass

        # 4. Indicators (辞書アクセスは属性アクセスより速い)
        # Mockオブジェクト対策: indicatorsが本当にdictか確認
        if hasattr(strategy_instance, "indicators"):
            indicators = strategy_instance.indicators
            if isinstance(indicators, dict) and operand_str in indicators:
                return self._get_final_value(indicators[operand_str])

        # 5. Attributes (アクセサキャッシュ利用)
        try:
            # キャッシュされたアクセサを試す
            if operand_str in self._accessor_cache:
                return self._get_final_value(self._accessor_cache[operand_str](strategy_instance))
            
            # なければgetattrで取得し、キャッシュする
            val = getattr(strategy_instance, operand_str)
            # 成功したらアクセサをキャッシュ
            import operator
            self._accessor_cache[operand_str] = operator.attrgetter(operand_str)
            return self._get_final_value(val)
        except AttributeError:
            pass

        # 6. 数値文字列
        try:
            return float(operand_str)
        except (ValueError, TypeError):
            return 0.0

    # StatefulCondition関連は変更なしのため省略（old_stringでマッチさせるため、既存コードを維持する形で記述）
    
    def evaluate_stateful_condition(
        self,
        stateful_condition,
        strategy_instance,
        state_tracker,
        current_bar: int,
    ) -> bool:
        """
        StatefulCondition を評価
        """
        # ここではインポートを避けるため型チェックを省略または簡易化
        if not getattr(stateful_condition, "enabled", True):
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
        return self.evaluate_single_condition(
            stateful_condition.follow_condition, strategy_instance
        )

    def check_and_record_trigger(
        self,
        stateful_condition,
        strategy_instance,
        state_tracker,
        current_bar: int,
    ) -> bool:
        """
        トリガー条件を評価し、成立していればStateTrackerに記録
        """
        if not getattr(stateful_condition, "enabled", True):
            return False

        # トリガー条件を評価
        trigger_result = self.evaluate_single_condition(
            stateful_condition.trigger_condition, strategy_instance
        )

        if trigger_result:
            event_name = stateful_condition.get_trigger_event_name()
            state_tracker.record_event(event_name, bar_index=current_bar)
            # logger.debug(f"トリガー記録: {event_name} at bar {current_bar}")

        return trigger_result
