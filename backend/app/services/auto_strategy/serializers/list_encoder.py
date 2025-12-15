"""
リストエンコードコンポーネント

戦略遺伝子を数値リストにエンコードする機能を担当します。
"""

import logging
from typing import Any, List, Tuple, Union

from ..genes.conditions import Condition, ConditionGroup
from ..utils.gene_utils import GeneUtils

logger = logging.getLogger(__name__)


class NormalizationConstants:
    """正規化定数"""

    INDICATOR_ID_DIVISOR = 100.0
    ATR_PERIOD_DIVISOR = 100.0
    LOOKBACK_PERIOD_DIVISOR = 1000.0
    PS_LOOKBACK_DIVISOR = 100.0
    TIMEFRAME_ID_DIVISOR = 20.0
    PARAM_DIVISOR = 1000.0  # 汎用パラメータ正規化除数

    # Encoding Lengths
    # Indicator: [ID, TimeframeID, Param1, Param2, Param3]
    INDICATOR_BLOCK_SIZE = 5

    # Condition: [LogicOp, Operator, LeftType, LeftVal, RightType, RightVal]
    CONDITION_BLOCK_SIZE = 6
    MAX_CONDITIONS = 5  # Entry/Exit それぞれの最大エンコード数

    # Stateful Condition: [Enabled, Lookback, Cooldown, Direction, Trigger(6), Follow(6)]
    STATEFUL_BLOCK_SIZE = 4 + 2 * CONDITION_BLOCK_SIZE
    MAX_STATEFUL_CONDITIONS = 2

    TPSL_BLOCK_SIZE = 8
    POSITION_SIZING_BLOCK_SIZE = 8


class ListEncoder:
    """
    リストエンコードクラス

    戦略遺伝子を数値リストにエンコードする機能を担当します。
    """

    def __init__(self):
        """初期化"""
        self._indicator_map = {
            "SMA": 1,
            "EMA": 2,
            "RSI": 3,
            "MACD": 4,
            "BB": 5,
            "STOCH": 6,
            "ADX": 7,
            "CCI": 8,
            "MFI": 9,
            "ROC": 10,
            "ATR": 11,
            "WMA": 12,
            "HMA": 13,
            "MOM": 14,
            "STDDEV": 15,
        }
        self._timeframe_map = {
            "1m": 1,
            "3m": 2,
            "5m": 3,
            "15m": 4,
            "30m": 5,
            "1h": 6,
            "2h": 7,
            "4h": 8,
            "6h": 9,
            "8h": 10,
            "12h": 11,
            "1d": 12,
            "3d": 13,
            "1w": 14,
        }
        self._operator_map = {
            ">": 1,
            "<": 2,
            ">=": 3,
            "<=": 4,
            "==": 5,
            "!=": 6,
            "CROSS_OVER": 7,
            "CROSS_UNDER": 8,
        }
        self._logic_map = {"AND": 1, "OR": 2}
        self._direction_map = {"long": 1, "short": -1}

    def to_list(self, strategy_gene) -> List[float]:
        """
        戦略遺伝子を固定長の数値リストにエンコード

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            エンコードされた数値リスト
        """
        try:
            from ..config import GAConfig

            encoded = []
            # GAConfigが正しく読み込めない場合のフォールバック
            try:
                max_indicators = GAConfig().max_indicators
            except Exception:
                max_indicators = 5

            # 1. 指標部分
            for i in range(max_indicators):
                if (
                    i < len(strategy_gene.indicators)
                    and strategy_gene.indicators[i].enabled
                ):
                    indicator_list = self._encode_indicator(strategy_gene.indicators[i])
                else:
                    indicator_list = [0.0] * NormalizationConstants.INDICATOR_BLOCK_SIZE
                encoded.extend(indicator_list)

            # 2. エントリー条件 (Flattened Tree) - 主にロング（または共通）条件を使用
            # 将来的にはロング/ショート分離エンコードを検討すべきだが、現在はEffective条件を採用
            effective_entry = strategy_gene.get_effective_long_conditions()
            encoded.extend(
                self._encode_condition_tree(
                    effective_entry,
                    NormalizationConstants.MAX_CONDITIONS,
                )
            )

            # 3. イグジット条件 (Flattened Tree)
            encoded.extend(
                self._encode_condition_tree(
                    strategy_gene.exit_conditions,
                    NormalizationConstants.MAX_CONDITIONS,
                )
            )

            # 4. ステートフル条件
            encoded.extend(
                self._encode_stateful_conditions(
                    strategy_gene.stateful_conditions,
                    NormalizationConstants.MAX_STATEFUL_CONDITIONS,
                )
            )

            # 5. TP/SL遺伝子
            encoded.extend(self._encode_tpsl_gene(strategy_gene.tpsl_gene))

            # 5. ポジションサイジング遺伝子
            encoded.extend(
                self._encode_position_sizing_gene(
                    getattr(strategy_gene, "position_sizing_gene", None)
                )
            )

            return encoded

        except Exception as e:
            logger.error(f"戦略遺伝子エンコードエラー: {e}")
            # フォールバック: 全てゼロの固定長リストを返す
            # max_indicatorsはここでも取得が必要だが、エラー時はデフォルト値を使用
            return self._get_fallback_encoding(5)

    def _encode_indicator(self, indicator) -> List[float]:
        """指標単体をエンコード"""
        try:
            # Type ID
            type_id = (
                self._indicator_map.get(indicator.type, 0)
                / NormalizationConstants.INDICATOR_ID_DIVISOR
            )

            # Timeframe ID
            tf_str = indicator.timeframe or "1h"  # Default fallback
            tf_id = (
                self._timeframe_map.get(tf_str, 0)
                / NormalizationConstants.TIMEFRAME_ID_DIVISOR
            )

            # Parameters (Top 3 by sorted key to be deterministic)
            params = []
            if indicator.parameters:
                sorted_keys = sorted(indicator.parameters.keys())
                for k in sorted_keys:
                    v = indicator.parameters[k]
                    if isinstance(v, (int, float)):
                        params.append(GeneUtils.normalize_parameter(v))

            # Pad or truncate to 3 parameters
            if len(params) < 3:
                params.extend([0.0] * (3 - len(params)))
            else:
                params = params[:3]

            return [type_id, tf_id] + params
        except Exception:
            return [0.0] * NormalizationConstants.INDICATOR_BLOCK_SIZE

    def _encode_condition_tree(
        self, conditions: List[Union[Condition, ConditionGroup]], max_count: int
    ) -> List[float]:
        """条件リスト（ツリー構造）をフラットなリストにエンコード"""
        flat_conditions = self._flatten_conditions(conditions)

        encoded = []
        count = 0
        for logic_op, condition in flat_conditions:
            if count >= max_count:
                break
            encoded.extend(self._encode_flat_condition(logic_op, condition))
            count += 1

        # パディング
        remaining = max_count - count
        if remaining > 0:
            encoded.extend(
                [0.0] * (remaining * NormalizationConstants.CONDITION_BLOCK_SIZE)
            )

        return encoded

    def _flatten_conditions(
        self, conditions: List[Union[Condition, ConditionGroup]], logic_op: str = "AND"
    ) -> List[Tuple[str, Condition]]:
        """条件構造を再帰的にフラット化"""
        flat_list = []
        current_op = logic_op

        if not conditions:
            return []

        for item in conditions:
            if isinstance(item, ConditionGroup):
                # グループの場合、再帰的に展開。グループ内の結合演算子を継承
                flat_list.extend(
                    self._flatten_conditions(item.conditions, item.operator)
                )
            elif isinstance(item, Condition):
                flat_list.append((current_op, item))

            # リスト内の次の要素との結合は、通常親のロジックに従うが、
            # リスト自体が暗黙のANDまたはORグループである場合、その文脈に依存。
            # ここでは簡易的に引数の logic_op を維持する

        return flat_list

    def _encode_flat_condition(
        self, logic_op: str, condition: Condition
    ) -> List[float]:
        """単一条件をエンコード: [LogicOp, Operator, LeftType, LeftVal, RightType, RightVal]"""
        try:
            # 1. Logic Operator
            logic_val = self._logic_map.get(logic_op, 0) / 10.0

            # 2. Comparison Operator
            op_val = self._operator_map.get(condition.operator, 0) / 10.0

            # 3, 4. Left Operand
            l_type, l_val = self._encode_operand(condition.left_operand)

            # 5, 6. Right Operand
            r_type, r_val = self._encode_operand(condition.right_operand)

            return [logic_val, op_val, l_type, l_val, r_type, r_val]
        except Exception:
            return [0.0] * NormalizationConstants.CONDITION_BLOCK_SIZE

    def _encode_operand(self, operand: Any) -> Tuple[float, float]:
        """オペランドをエンコード -> (Type, Value)"""
        # Type maps: 0.1:Const, 0.5:Indicator, 0.9:OHLCV

        if isinstance(operand, (int, float)):
            # Constant
            return 0.1, GeneUtils.normalize_parameter(operand)

        if isinstance(operand, dict) and "indicator" in operand:
            # Indicator Ref
            ind_type = operand.get("indicator", "")
            ind_id = (
                self._indicator_map.get(ind_type, 0)
                / NormalizationConstants.INDICATOR_ID_DIVISOR
            )
            return 0.5, ind_id

        if isinstance(operand, str):
            # OHLCV attribute or Indicator key
            normalized_op = operand.lower()
            if normalized_op in ["open", "high", "low", "close", "volume"]:
                ohlcv_map = {"open": 1, "high": 2, "low": 3, "close": 4, "volume": 5}
                return 0.9, ohlcv_map.get(normalized_op, 0) / 10.0

            # 他の文字列（キー参照など、未解決）
            return 0.5, 0.0

        return 0.0, 0.0

    def _encode_stateful_conditions(
        self, stateful_conditions: List[Any], max_count: int
    ) -> List[float]:
        """ステートフル条件リストをエンコード"""
        encoded = []
        count = 0

        for sc in stateful_conditions:
            if count >= max_count:
                break

            try:
                # 1. Basic properties
                enabled_val = 1.0 if sc.enabled else 0.0
                lookback_val = (
                    sc.lookback_bars / NormalizationConstants.LOOKBACK_PERIOD_DIVISOR
                )
                cooldown_val = (
                    sc.cooldown_bars / NormalizationConstants.LOOKBACK_PERIOD_DIVISOR
                )
                direction_val = (
                    self._direction_map.get(sc.direction, 1)
                    if hasattr(sc, "direction")
                    else 1.0
                )

                basic_props = [enabled_val, lookback_val, cooldown_val, direction_val]

                # 2. Trigger Condition (First logic only)
                trigger_encoded = self._encode_flat_condition(
                    "AND", sc.trigger_condition
                )

                # 3. Follow Condition (First logic only)
                follow_encoded = self._encode_flat_condition("AND", sc.follow_condition)

                encoded.extend(basic_props + trigger_encoded + follow_encoded)
                count += 1
            except Exception:
                encoded.extend([0.0] * NormalizationConstants.STATEFUL_BLOCK_SIZE)

        # パディング
        remaining = max_count - count
        if remaining > 0:
            encoded.extend(
                [0.0] * (remaining * NormalizationConstants.STATEFUL_BLOCK_SIZE)
            )

        return encoded

    def _encode_tpsl_gene(self, tpsl_gene) -> List[float]:
        """TP/SL遺伝子をエンコード"""
        try:
            if not tpsl_gene or not tpsl_gene.enabled:
                return [0.0] * NormalizationConstants.TPSL_BLOCK_SIZE

            encoded = [
                1.0 if tpsl_gene.enabled else 0.0,
                tpsl_gene.stop_loss_pct or 0.0,
                tpsl_gene.take_profit_pct or 0.0,
                (tpsl_gene.risk_reward_ratio or 0.0) / 10.0,  # Normalize
                (tpsl_gene.atr_multiplier_sl or 0.0) / 10.0,
                (tpsl_gene.atr_multiplier_tp or 0.0) / 10.0,
                (tpsl_gene.atr_period or 0) / NormalizationConstants.ATR_PERIOD_DIVISOR,
                (tpsl_gene.lookback_period or 0)
                / NormalizationConstants.LOOKBACK_PERIOD_DIVISOR,
            ]

            # 長さ調整
            if len(encoded) < NormalizationConstants.TPSL_BLOCK_SIZE:
                encoded.extend(
                    [0.0] * (NormalizationConstants.TPSL_BLOCK_SIZE - len(encoded))
                )
            return encoded[: NormalizationConstants.TPSL_BLOCK_SIZE]
        except Exception as e:
            logger.error(f"TP/SL遺伝子エンコードエラー: {e}")
            return [0.0] * NormalizationConstants.TPSL_BLOCK_SIZE

    def _encode_position_sizing_gene(self, ps_gene) -> List[float]:
        """ポジションサイジング遺伝子をエンコード"""
        try:
            if not ps_gene or not ps_gene.enabled:
                return [0.0] * NormalizationConstants.POSITION_SIZING_BLOCK_SIZE

            encoded = [
                1.0 if ps_gene.enabled else 0.0,
                ps_gene.risk_per_trade or 0.0,
                ps_gene.fixed_ratio or 0.0,
                ps_gene.fixed_quantity or 0.0,
                (ps_gene.atr_multiplier or 0.0) / 10.0,
                ps_gene.optimal_f_multiplier or 0.0,
                (ps_gene.lookback_period or 0)
                / NormalizationConstants.PS_LOOKBACK_DIVISOR,
                ps_gene.min_position_size or 0.0,
            ]

            if len(encoded) < NormalizationConstants.POSITION_SIZING_BLOCK_SIZE:
                encoded.extend(
                    [0.0]
                    * (NormalizationConstants.POSITION_SIZING_BLOCK_SIZE - len(encoded))
                )
            return encoded[: NormalizationConstants.POSITION_SIZING_BLOCK_SIZE]
        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子エンコードエラー: {e}")
            return [0.0] * NormalizationConstants.POSITION_SIZING_BLOCK_SIZE

    def _get_fallback_encoding(self, max_indicators: int) -> List[float]:
        """フォールバック用のゼロ埋めリスト生成"""
        expected_len = (
            max_indicators * NormalizationConstants.INDICATOR_BLOCK_SIZE
            + NormalizationConstants.MAX_CONDITIONS
            * NormalizationConstants.CONDITION_BLOCK_SIZE  # Entry
            + NormalizationConstants.MAX_CONDITIONS
            * NormalizationConstants.CONDITION_BLOCK_SIZE  # Exit
            + NormalizationConstants.MAX_STATEFUL_CONDITIONS
            * NormalizationConstants.STATEFUL_BLOCK_SIZE  # Stateful
            + NormalizationConstants.TPSL_BLOCK_SIZE
            + NormalizationConstants.POSITION_SIZING_BLOCK_SIZE
        )
        return [0.0] * expected_len





