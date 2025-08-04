"""
遺伝子デコーダー

GA用の戦略遺伝子デコード機能を担当するモジュール。
"""

import logging
from typing import Dict, List

from ..generators.smart_condition_generator import SmartConditionGenerator
from . import gene_utils
from .gene_position_sizing import PositionSizingGene, PositionSizingMethod
from .gene_strategy import Condition, IndicatorGene
from .gene_tpsl import TPSLGene, TPSLMethod

logger = logging.getLogger(__name__)


class GeneDecoder:
    """
    遺伝子デコーダー

    GA用の戦略遺伝子デコード機能を担当します。
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: SmartConditionGeneratorを使用するか
        """
        self.indicator_ids = gene_utils.get_indicator_ids()
        self.id_to_indicator = gene_utils.get_id_to_indicator(self.indicator_ids)
        self.smart_condition_generator = SmartConditionGenerator(enable_smart_generation)

    def decode_list_to_strategy_gene(self, encoded: List[float], strategy_gene_class):
        """
        数値リストから戦略遺伝子にデコード

        Args:
            encoded: エンコードされた数値リスト
            strategy_gene_class: StrategyGeneクラス

        Returns:
            デコードされた戦略遺伝子オブジェクト
        """
        try:
            max_indicators = 5  # デフォルト値
            indicators = []

            # 指標部分をデコード（多様性を確保）
            for i in range(max_indicators):
                idx = i * 2
                if idx + 1 < len(encoded):
                    if encoded[idx] < 0.01:
                        continue

                    indicator_id = int(encoded[idx] * len(self.indicator_ids))
                    max_id = len(self.indicator_ids) - 1
                    indicator_id = max(1, min(max_id, indicator_id))
                    param_val = encoded[idx + 1]

                    indicator_type = self.id_to_indicator.get(indicator_id, "")
                    if indicator_type and indicator_type != "":
                        parameters = self._generate_indicator_parameters(
                            indicator_type, param_val
                        )
                        indicators.append(
                            IndicatorGene(
                                type=indicator_type,
                                parameters=parameters,
                                enabled=True,
                            )
                        )

            # 条件部分をデコード（SmartConditionGeneratorを使用）
            if indicators:
                long_entry_conditions, short_entry_conditions, exit_conditions = (
                    self.smart_condition_generator.generate_balanced_conditions(indicators)
                )
                # 後方互換性のためのentry_conditions
                entry_conditions = long_entry_conditions
            else:
                # フォールバック条件
                long_entry_conditions = [
                    Condition(left_operand="close", operator=">", right_operand="open")
                ]
                short_entry_conditions = [
                    Condition(left_operand="close", operator="<", right_operand="open")
                ]
                exit_conditions = [
                    Condition(left_operand="close", operator="==", right_operand="open")
                ]
                entry_conditions = long_entry_conditions

            # リスク管理設定
            risk_management = {
                "stop_loss": 0.03,
                "take_profit": 0.15,
                "position_size": 0.1,
            }

            # TP/SL遺伝子をデコード
            tpsl_gene = None
            if len(encoded) >= 24:
                tpsl_encoded = encoded[16:24]
                tpsl_gene = self._decode_tpsl_gene(tpsl_encoded)

            # TP/SL遺伝子が有効な場合はexit_conditionsを空にする
            if tpsl_gene and tpsl_gene.enabled:
                exit_conditions = []

            # ポジションサイジング遺伝子をデコード
            position_sizing_gene = None
            if len(encoded) >= 32:
                position_sizing_encoded = encoded[24:32]
                position_sizing_gene = self._decode_position_sizing_gene(
                    position_sizing_encoded
                )

            # メタデータ
            metadata = {
                "generated_by": "GeneDecoder_decode",
                "source": (
                    "fallback_individual" if len(indicators) <= 1 else "normal_decode"
                ),
                "indicators_count": len(indicators),
                "decoded_from_length": len(encoded),
                "tpsl_gene_included": tpsl_gene is not None,
                "position_sizing_gene_included": position_sizing_gene is not None,
            }

            return strategy_gene_class(
                indicators=indicators,
                entry_conditions=entry_conditions,
                long_entry_conditions=long_entry_conditions,
                short_entry_conditions=short_entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=risk_management,
                tpsl_gene=tpsl_gene,
                position_sizing_gene=position_sizing_gene,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"戦略遺伝子デコードエラー: {e}")
            from ..utils.strategy_gene_utils import create_default_strategy_gene

            return create_default_strategy_gene(strategy_gene_class)

    def _create_conditions_from_indicators(self, indicators: List[IndicatorGene]):
        if not indicators:
            return (
                [Condition(left_operand="close", operator=">", right_operand="open")],
                [Condition(left_operand="close", operator="<", right_operand="open")],
            )

        first_indicator = indicators[0]
        indicator_name = first_indicator.type
        entry_conditions, exit_conditions = self._generate_compatible_conditions(
            indicator_name, first_indicator.type
        )

        if len(indicators) > 1:
            second_indicator = indicators[1]
            second_name = second_indicator.type
            if second_indicator.type in ["RSI", "STOCH", "CCI"]:
                entry_conditions.append(
                    Condition(
                        left_operand=second_name, operator="<", right_operand="50"
                    )
                )
                exit_conditions.append(
                    Condition(
                        left_operand=second_name, operator=">", right_operand="50"
                    )
                )
        return entry_conditions, exit_conditions

    def _generate_indicator_parameters(
        self, indicator_type: str, param_val: float
    ) -> Dict:
        try:
            from app.services.indicators.config.indicator_config import (
                indicator_registry,
            )
            from app.services.indicators.parameter_manager import (
                IndicatorParameterManager,
            )

            config = indicator_registry.get_indicator_config(indicator_type)
            if config:
                manager = IndicatorParameterManager()
                # Note: generate_parameters might need adjustment if it relies on more than just type and config
                return manager.generate_parameters(indicator_type, config)
            else:
                return {}
        except Exception as e:
            logger.error(f"指標 {indicator_type} のパラメータ生成に失敗: {e}")
            return {}

    def _generate_compatible_conditions(self, indicator_name: str, indicator_type: str):
        from .gene_strategy import Condition

        try:
            if indicator_type in ["RSI", "STOCH", "ADX"]:
                return (
                    [
                        Condition(
                            left_operand=indicator_name,
                            operator="<",
                            right_operand=30.0,
                        )
                    ],
                    [
                        Condition(
                            left_operand=indicator_name,
                            operator=">",
                            right_operand=70.0,
                        )
                    ],
                )
            elif indicator_type in ["CCI"]:
                return (
                    [
                        Condition(
                            left_operand=indicator_name,
                            operator="<",
                            right_operand=-100.0,
                        )
                    ],
                    [
                        Condition(
                            left_operand=indicator_name,
                            operator=">",
                            right_operand=100.0,
                        )
                    ],
                )
            elif indicator_type in ["SMA", "EMA", "WMA", "MAMA"]:
                return (
                    [
                        Condition(
                            left_operand="close",
                            operator=">",
                            right_operand=indicator_name,
                        )
                    ],
                    [
                        Condition(
                            left_operand="close",
                            operator="<",
                            right_operand=indicator_name,
                        )
                    ],
                )
            elif indicator_type == "MACD":
                return (
                    [Condition(left_operand="MACD", operator=">", right_operand=0)],
                    [Condition(left_operand="MACD", operator="<", right_operand=0)],
                )
            else:  # OBV, ATR, etc.
                logger.warning(
                    f"未知または未対応の指標タイプ: {indicator_type}, デフォルトの数値閾値を使用"
                )
                return (
                    [
                        Condition(
                            left_operand=indicator_name,
                            operator=">",
                            right_operand=50.0,
                        )
                    ],
                    [
                        Condition(
                            left_operand=indicator_name,
                            operator="<",
                            right_operand=50.0,
                        )
                    ],
                )
        except Exception as e:
            logger.error(f"互換性条件生成エラー: {e}")
            return (
                [Condition(left_operand="close", operator=">", right_operand="open")],
                [Condition(left_operand="close", operator="<", right_operand="open")],
            )

    def _generate_long_short_conditions(self, indicator_name: str, indicator_type: str):
        """
        指標タイプに基づいてロング・ショート条件を分離して生成

        Args:
            indicator_name: 指標名
            indicator_type: 指標タイプ

        Returns:
            (long_entry_conditions, short_entry_conditions, exit_conditions)のタプル
        """
        from .gene_strategy import Condition

        try:
            long_entry_conditions = []
            short_entry_conditions = []
            exit_conditions = []

            if indicator_type == "RSI":
                # RSI条件：売られすぎでロング、買われすぎでショート
                long_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="<", right_operand=30
                    )
                ]
                short_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=70
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=50
                    )
                ]

            elif indicator_type in ["SMA", "EMA", "MAMA"]:
                # 移動平均条件：価格が上でロング、下でショート
                long_entry_conditions = [
                    Condition(
                        left_operand="close", operator=">", right_operand=indicator_name
                    )
                ]
                short_entry_conditions = [
                    Condition(
                        left_operand="close", operator="<", right_operand=indicator_name
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand="close",
                        operator="==",
                        right_operand=indicator_name,
                    )
                ]

            elif indicator_type == "CCI":
                # CCI条件：-100以下でロング、+100以上でショート
                long_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="<", right_operand=-100.0
                    )
                ]
                short_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=100.0
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="==", right_operand=0.0
                    )
                ]

            elif indicator_type == "ADX":
                # ADX条件：トレンド強度に基づく（他の指標と組み合わせて使用）
                long_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=25.0
                    )
                ]
                short_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=25.0
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="<", right_operand=20.0
                    )
                ]

            else:
                # デフォルト条件
                long_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=50.0
                    )
                ]
                short_entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="<", right_operand=50.0
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="==", right_operand=50.0
                    )
                ]

            return long_entry_conditions, short_entry_conditions, exit_conditions

        except Exception as e:
            logger.error(f"ロング・ショート条件生成エラー: {e}")
            return (
                [Condition(left_operand="close", operator=">", right_operand="open")],
                [Condition(left_operand="close", operator="<", right_operand="open")],
                [Condition(left_operand="close", operator="==", right_operand="open")],
            )

    def _decode_tpsl_gene(self, encoded: List[float]) -> TPSLGene:
        try:
            if len(encoded) < 8:
                encoded.extend([0.0] * (8 - len(encoded)))

            method_value = encoded[0]
            if method_value <= 0.2:
                method = TPSLMethod.FIXED_PERCENTAGE
            elif method_value <= 0.4:
                method = TPSLMethod.RISK_REWARD_RATIO
            elif method_value <= 0.6:
                method = TPSLMethod.VOLATILITY_BASED
            elif method_value <= 0.8:
                method = TPSLMethod.STATISTICAL
            else:
                method = TPSLMethod.ADAPTIVE

            stop_loss_pct = encoded[1] * 0.15
            take_profit_pct = encoded[2] * 0.3
            risk_reward_ratio = encoded[3] * 9.5 + 0.5
            base_stop_loss = encoded[4] * 0.15
            atr_multiplier_sl = encoded[5] * 4.5 + 0.5
            atr_multiplier_tp = encoded[6] * 9.0 + 1.0
            priority = encoded[7] * 2.0

            return TPSLGene(
                method=method,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                risk_reward_ratio=risk_reward_ratio,
                base_stop_loss=base_stop_loss,
                atr_multiplier_sl=atr_multiplier_sl,
                atr_multiplier_tp=atr_multiplier_tp,
                priority=priority,
            )
        except Exception as e:
            logger.error(f"TP/SL遺伝子デコードエラー: {e}")
            return TPSLGene()  # Return default

    def _decode_position_sizing_gene(self, encoded: List[float]) -> PositionSizingGene:
        try:
            if len(encoded) < 8:
                default_values = [0.5] * 8
                encoded.extend(default_values[len(encoded) :])

            method_value = encoded[0]
            if method_value <= 0.25:
                method = PositionSizingMethod.HALF_OPTIMAL_F
            elif method_value <= 0.5:
                method = PositionSizingMethod.VOLATILITY_BASED
            elif method_value <= 0.75:
                method = PositionSizingMethod.FIXED_RATIO
            else:
                method = PositionSizingMethod.FIXED_QUANTITY

            lookback_period = int(50 + encoded[1] * 150)
            optimal_f_multiplier = 0.25 + encoded[2] * 0.5
            atr_period = int(10 + encoded[3] * 20)
            atr_multiplier = 1.0 + encoded[4] * 3.0
            risk_per_trade = 0.01 + encoded[5] * 0.04
            fixed_ratio = 0.05 + encoded[6] * 0.25
            fixed_quantity = 0.1 + encoded[7] * 4.9
            priority = 0.5 + encoded[7] * 1.0

            return PositionSizingGene(
                method=method,
                lookback_period=lookback_period,
                optimal_f_multiplier=optimal_f_multiplier,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
                risk_per_trade=risk_per_trade,
                fixed_ratio=fixed_ratio,
                fixed_quantity=fixed_quantity,
                priority=priority,
            )
        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子デコードエラー: {e}")
            return PositionSizingGene()  # Return default


