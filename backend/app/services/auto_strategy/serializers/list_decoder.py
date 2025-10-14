"""
リストデコードコンポーネント

数値リストから戦略遺伝子にデコードする機能を担当します。
"""

import logging
from typing import Any, Dict, List

from ..models.strategy_models import (
    PositionSizingGene,
    PositionSizingMethod,
    TPSLGene,
    TPSLMethod,
)
from ..utils.gene_utils import GeneUtils

logger = logging.getLogger(__name__)


class ListDecoder:
    """
    リストデコードクラス

    数値リストから戦略遺伝子にデコードする機能を担当します。
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: ConditionGeneratorを使用するか
        """
        self.enable_smart_generation = enable_smart_generation
        self._smart_condition_generator = None

    @property
    def smart_condition_generator(self):
        """ConditionGeneratorの遅延初期化"""
        if self._smart_condition_generator is None and self.enable_smart_generation:
            from ..generators.condition_generator import ConditionGenerator

            self._smart_condition_generator = ConditionGenerator(True)
        return self._smart_condition_generator

    def from_list(self, encoded: List[float], strategy_gene_class):
        """
        数値リストから戦略遺伝子にデコード（旧decode_list_to_strategy_gene）

        Args:
            encoded: エンコードされた数値リスト
            strategy_gene_class: StrategyGeneクラス

        Returns:
            デコードされた戦略遺伝子オブジェクト
        """
        try:
            # エンコードされたリストが短すぎる場合は、デフォルト遺伝子を返す
            if not encoded or len(encoded) < 10:
                logger.warning(
                    f"エンコードされたリストが短すぎるため、デフォルト遺伝子を生成します: length={len(encoded)}"
                )
                return GeneUtils.create_default_strategy_gene(strategy_gene_class)

            max_indicators = 5  # デフォルト値
            indicators = []

            # 指標部分をデコード（多様性を確保）
            for i in range(max_indicators):
                idx = i * 2
                if idx + 1 < len(encoded):
                    if encoded[idx] < 0.01:
                        continue

                    indicator_id = int(encoded[idx] * 100)  # 指標IDの範囲を想定
                    max_id = 10  # 利用可能な指標数の最大値
                    indicator_id = max(1, min(max_id, indicator_id))
                    param_val = encoded[idx + 1]

                    indicator_type = self._get_indicator_type(indicator_id)
                    if indicator_type and indicator_type != "":
                        parameters = self._generate_indicator_parameters(
                            indicator_type, param_val
                        )
                        from ..models.strategy_models import IndicatorGene

                        indicators.append(
                            IndicatorGene(
                                type=indicator_type,
                                parameters=parameters,
                                enabled=True,
                            )
                        )

            # 条件部分をデコード（ConditionGeneratorを使用）
            if indicators:
                if self.smart_condition_generator:
                    long_entry_conditions, short_entry_conditions, exit_conditions = (
                        self.smart_condition_generator.generate_balanced_conditions(
                            indicators
                        )
                    )
                else:
                    # ConditionGeneratorが無効な場合のフォールバック
                    from ..models.strategy_models import Condition

                    long_entry_conditions = [
                        Condition(
                            left_operand="close", operator=">", right_operand="open"
                        )
                    ]
                    short_entry_conditions = [
                        Condition(
                            left_operand="close", operator="<", right_operand="open"
                        )
                    ]
                    exit_conditions = [
                        Condition(
                            left_operand="close", operator="==", right_operand="open"
                        )
                    ]
                # 後方互換性のためのentry_conditions
                entry_conditions = long_entry_conditions
            else:
                # フォールバック条件
                from ..models.strategy_models import Condition

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
                "generated_by": "ListDecoder_decode",
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
            return GeneUtils.create_default_strategy_gene(strategy_gene_class)

    def _get_indicator_type(self, indicator_id: int) -> str:
        """
        指標IDからタイプを取得

        Args:
            indicator_id: 指標ID

        Returns:
            指標タイプ
        """
        # 簡易的な指標IDマッピング（実際の実装では詳細化が必要）
        indicator_map = {
            1: "SMA",
            2: "EMA",
            3: "RSI",
            4: "MACD",
            5: "BB",
            6: "STOCH",
            7: "ADX",
            8: "CCI",
            9: "MFI",
            10: "ROC",
        }
        return indicator_map.get(indicator_id, "")

    def _generate_indicator_parameters(
        self, indicator_type: str, param_val: float
    ) -> Dict[str, Any]:
        """指標パラメータを生成"""
        try:
            # 基本的なパラメータ生成
            period = max(1, min(200, int(param_val * 200)))

            # 指標タイプ別の特別なパラメータ
            if indicator_type in ["BB", "KELTNER"]:
                return {"period": period, "std_dev": 2.0}
            elif indicator_type in ["MACD"]:
                return {"fast_period": 12, "slow_period": 26, "signal_period": 9}
            elif indicator_type in ["STOCH", "STOCHRSI"]:
                return {"k_period": period, "d_period": 3}
            else:
                return {"period": period}

        except Exception as e:
            logger.error(f"指標パラメータ生成エラー: {e}")
            return {"period": 20}

    def _decode_tpsl_gene(self, encoded: List[float]):
        """TP/SL遺伝子をデコード"""
        try:
            if len(encoded) < 8 or encoded[0] < 0.5:
                return None

            return TPSLGene(
                enabled=True,
                method=TPSLMethod.RISK_REWARD_RATIO,
                stop_loss_pct=encoded[1],
                take_profit_pct=encoded[2],
                risk_reward_ratio=encoded[3],
                atr_multiplier_sl=encoded[4],
                atr_multiplier_tp=encoded[5],
                atr_period=int(encoded[6] * 100),
                lookback_period=int(encoded[7] * 1000),
            )
        except Exception as e:
            logger.error(f"TP/SL遺伝子デコードエラー: {e}")
            return None

    def _decode_position_sizing_gene(self, encoded: List[float]):
        """ポジションサイジング遺伝子をデコード"""
        try:
            if len(encoded) < 8 or encoded[0] < 0.5:
                return None

            return PositionSizingGene(
                enabled=True,
                method=PositionSizingMethod.VOLATILITY_BASED,
                risk_per_trade=encoded[1],
                fixed_ratio=encoded[2],
                fixed_quantity=encoded[3],
                atr_multiplier=encoded[4],
                optimal_f_multiplier=encoded[5],
                lookback_period=int(encoded[6] * 100),
                min_position_size=encoded[7],
            )
        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子デコードエラー: {e}")
            return None
