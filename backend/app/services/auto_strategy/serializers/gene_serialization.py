"""
統一遺伝子シリアライゼーション

戦略遺伝子のシリアライゼーション・デシリアライゼーション、エンコード・デコードを担当するモジュール。
GeneEncoder、GeneDecoder、GeneSerializerの機能を統合しています。
"""

import json
import logging
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models.strategy_models import Condition


from ..models.strategy_models import (
    IndicatorGene,
    TPSLGene,
    TPSLMethod,
    PositionSizingGene,
    PositionSizingMethod,
)
from ..utils.indicator_utils import (
    get_all_indicator_ids,
    get_id_to_indicator_mapping,
)
from ..utils.gene_utils import GeneUtils
import uuid

logger = logging.getLogger(__name__)


class GeneSerializer:
    """
    統一遺伝子シリアライザー

    戦略遺伝子のシリアライゼーション・デシリアライゼーション、エンコード・デコードを担当します。
    GeneEncoder、GeneDecoder、GeneSerializerの機能を統合しています。
    """

    def __init__(self, enable_smart_generation: bool = True):
        """
        初期化

        Args:
            enable_smart_generation: ConditionGeneratorを使用するか
        """
        self.indicator_ids = get_all_indicator_ids()
        self.id_to_indicator = get_id_to_indicator_mapping(self.indicator_ids)

        # ConditionGeneratorの遅延インポート（循環インポート回避）
        self.enable_smart_generation = enable_smart_generation
        self._smart_condition_generator = None

    def strategy_gene_to_dict(self, strategy_gene) -> Dict[str, Any]:
        """
        戦略遺伝子を辞書形式に変換

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            # risk_managementからTP/SL関連の設定を除外
            clean_risk_management = self._clean_risk_management(
                strategy_gene.risk_management
            )

            return {
                "id": strategy_gene.id,
                "indicators": [
                    self.indicator_gene_to_dict(ind) for ind in strategy_gene.indicators
                ],
                "entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.entry_conditions
                ],
                "long_entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.long_entry_conditions
                ],
                "short_entry_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.short_entry_conditions
                ],
                "exit_conditions": [
                    self.condition_or_group_to_dict(cond)
                    for cond in strategy_gene.exit_conditions
                ],
                "risk_management": clean_risk_management,
                "tpsl_gene": (
                    self.tpsl_gene_to_dict(strategy_gene.tpsl_gene)
                    if strategy_gene.tpsl_gene
                    else None
                ),
                "position_sizing_gene": (
                    self.position_sizing_gene_to_dict(
                        strategy_gene.position_sizing_gene
                    )
                    if getattr(strategy_gene, "position_sizing_gene", None)
                    else None
                ),
                "metadata": strategy_gene.metadata,
            }

        except Exception as e:
            logger.error(f"戦略遺伝子辞書変換エラー: {e}")
            raise ValueError(f"戦略遺伝子の辞書変換に失敗: {e}")

    def indicator_gene_to_dict(self, indicator_gene) -> Dict[str, Any]:
        """
        指標遺伝子を辞書形式に変換

        Args:
            indicator_gene: 指標遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            return {
                "type": indicator_gene.type,
                "parameters": indicator_gene.parameters,
                "enabled": indicator_gene.enabled,
            }

        except Exception as e:
            logger.error(f"指標遺伝子辞書変換エラー: {e}")
            raise ValueError(f"指標遺伝子の辞書変換に失敗: {e}")

    def dict_to_indicator_gene(self, data: Dict[str, Any]) -> "IndicatorGene":
        """
        辞書形式から指標遺伝子を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            指標遺伝子オブジェクト
        """
        try:
            # IndicatorGeneクラスを動的にインポート
            # IndicatorGeneクラスを動的にインポート
            from ..models.strategy_models import IndicatorGene

            return IndicatorGene(
                type=data["type"],
                parameters=data["parameters"],
                enabled=data.get("enabled", True),
            )

        except Exception as e:
            logger.error(f"指標遺伝子復元エラー: {e}")
            raise ValueError(f"指標遺伝子の復元に失敗: {e}")

    def condition_to_dict(self, condition) -> Dict[str, Any]:
        """
        条件を辞書形式に変換

        Args:
            condition: 条件オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            return {
                "left_operand": condition.left_operand,
                "operator": condition.operator,
                "right_operand": condition.right_operand,
            }

        except Exception as e:
            logger.error(f"条件辞書変換エラー: {e}")
            raise ValueError(f"条件の辞書変換に失敗: {e}")

    def condition_or_group_to_dict(self, obj) -> Dict[str, Any]:
        """Condition または ConditionGroup を辞書に変換"""
        try:
            from ..models.strategy_models import ConditionGroup, Condition

            if isinstance(obj, ConditionGroup):
                return {
                    "type": "OR_GROUP",
                    "conditions": [self.condition_to_dict(c) for c in obj.conditions],
                }
            elif isinstance(obj, Condition) or hasattr(obj, 'left_operand'):
                return self.condition_to_dict(obj)
            else:
                raise TypeError(f"未知の条件型: {type(obj)}")
        except Exception as e:
            logger.error(f"条件/グループ辞書変換エラー: {e}")
            raise ValueError(f"条件の辞書変換に失敗: {e}")

    def tpsl_gene_to_dict(self, tpsl_gene) -> Optional[Dict[str, Any]]:
        """
        TP/SL遺伝子を辞書形式に変換

        Args:
            tpsl_gene: TP/SL遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            if tpsl_gene is None:
                return None

            return tpsl_gene.to_dict()

        except Exception as e:
            logger.error(f"TP/SL遺伝子辞書変換エラー: {e}")
            raise ValueError(f"TP/SL遺伝子の辞書変換に失敗: {e}")

    def dict_to_tpsl_gene(self, data: Dict[str, Any]) -> Optional["TPSLGene"]:
        """
        辞書形式からTP/SL遺伝子を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            TP/SL遺伝子オブジェクト
        """
        try:
            if data is None:
                return None

            from ..models.strategy_models import TPSLGene

            return TPSLGene.from_dict(data)  # type: ignore[cSpell] # TPSL is a valid trading acronym

        except Exception as e:
            logger.error(f"TP/SL遺伝子復元エラー: {e}")
            raise ValueError(f"TP/SL遺伝子の復元に失敗: {e}")

    def position_sizing_gene_to_dict(
        self, position_sizing_gene
    ) -> Optional[Dict[str, Any]]:
        """
        ポジションサイジング遺伝子を辞書形式に変換

        Args:
            position_sizing_gene: ポジションサイジング遺伝子オブジェクト

        Returns:
            辞書形式のデータ
        """
        try:
            if position_sizing_gene is None:
                return None

            return position_sizing_gene.to_dict()

        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子辞書変換エラー: {e}")
            raise ValueError(f"ポジションサイジング遺伝子の辞書変換に失敗: {e}")

    def dict_to_position_sizing_gene(self, data: Dict[str, Any]) -> Optional["PositionSizingGene"]:
        """
        辞書形式からポジションサイジング遺伝子を復元

        Args:
            data: 辞書形式のデータ

        Returns:
            ポジションサイジング遺伝子オブジェクト
        """
        try:
            if data is None:
                return None

            # PositionSizingGeneクラスを動的にインポート
            from ..models.strategy_models import PositionSizingGene

            return PositionSizingGene.from_dict(data)

        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子復元エラー: {e}")
            raise ValueError(f"ポジションサイジング遺伝子の復元に失敗: {e}")

    def _clean_risk_management(self, risk_management: Dict[str, Any]) -> Dict[str, Any]:
        """
        risk_managementからTP/SL関連の設定を除外

        Args:
            risk_management: 元のリスク管理設定

        Returns:
            TP/SL関連設定を除外したリスク管理設定
        """
        # TP/SL関連のキーを除外
        tpsl_keys = {
            "stop_loss",
            "take_profit",
            "stop_loss_pct",
            "take_profit_pct",
            "risk_reward_ratio",
            "atr_multiplier_sl",
            "atr_multiplier_tp",
            "_tpsl_strategy",
            "_tpsl_method",
        }

        clean_risk_management = {}
        for key, value in risk_management.items():
            if key not in tpsl_keys:
                # 数値の場合は適切な桁数に丸める
                if isinstance(value, float):
                    if key == "position_size":
                        # BTCトレードでは小数点以下の精度が重要なため、6桁の精度を保持
                        clean_risk_management[key] = round(value, 6)
                    else:
                        clean_risk_management[key] = round(value, 4)
                else:
                    clean_risk_management[key] = value

        return clean_risk_management

    def strategy_gene_to_json(self, strategy_gene) -> str:
        """
        戦略遺伝子をJSON文字列に変換

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            JSON文字列
        """
        try:
            data = self.strategy_gene_to_dict(strategy_gene)
            return json.dumps(data, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"戦略遺伝子JSON変換エラー: {e}")
            raise ValueError(f"戦略遺伝子のJSON変換に失敗: {e}")


    # 以下、旧GeneEncoderから統合されたメソッド

    @property
    def smart_condition_generator(self):
        """ConditionGeneratorの遅延初期化"""
        if self._smart_condition_generator is None and self.enable_smart_generation:
            from ..generators.condition_generator import ConditionGenerator

            self._smart_condition_generator = ConditionGenerator(True)
        return self._smart_condition_generator

    def to_list(self, strategy_gene) -> List[float]:
        """
        戦略遺伝子を固定長の数値リストにエンコード（旧encode_strategy_gene_to_list）

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            エンコードされた数値リスト
        """
        try:
            encoded = []
            max_indicators = getattr(strategy_gene, "MAX_INDICATORS", 5)

            # 指標部分（指標数 × 2値 = 10要素）
            for i in range(max_indicators):
                if (
                    i < len(strategy_gene.indicators)
                    and strategy_gene.indicators[i].enabled
                ):
                    indicator = strategy_gene.indicators[i]
                    indicator_id = self.indicator_ids.get(indicator.type, 0)
                    # パラメータを正規化（期間の場合は1-200を0-1に変換）
                    param_val = GeneUtils.normalize_parameter(
                        indicator.parameters.get("period", 20)
                    )
                else:
                    indicator_id = 0  # 未使用
                    param_val = 0.0

                # 指標IDを0-1の範囲に正規化してエンコード
                normalized_id = (
                    indicator_id / len(self.indicator_ids) if indicator_id > 0 else 0.0
                )
                encoded.extend([normalized_id, param_val])

            # エントリー条件（簡略化: 最初の条件のみ）
            if strategy_gene.entry_conditions:
                entry_cond = strategy_gene.entry_conditions[0]
                entry_encoded = self._encode_condition(entry_cond)
            else:
                entry_encoded = [0, 0, 0]  # デフォルト

            # イグジット条件（簡略化: 最初の条件のみ）
            if strategy_gene.exit_conditions:
                exit_cond = strategy_gene.exit_conditions[0]
                exit_encoded = self._encode_condition(exit_cond)
            else:
                exit_encoded = [0, 0, 0]  # デフォルト

            encoded.extend(entry_encoded)
            encoded.extend(exit_encoded)

            # TP/SL遺伝子のエンコード
            tpsl_encoded = self._encode_tpsl_gene(strategy_gene.tpsl_gene)
            encoded.extend(tpsl_encoded)

            # ポジションサイジング遺伝子のエンコード
            position_sizing_encoded = self._encode_position_sizing_gene(
                getattr(strategy_gene, "position_sizing_gene", None)
            )
            encoded.extend(position_sizing_encoded)

            return encoded

        except Exception as e:
            logger.error(f"戦略遺伝子エンコードエラー: {e}")
            # エラー時はデフォルトエンコードを返す
            return [0.0] * 32  # 5指標×2 + 条件×6 + TP/SL×8 + ポジションサイジング×8

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
                logger.warning(f"エンコードされたリストが短すぎるため、デフォルト遺伝子を生成します: length={len(encoded)}")
                return GeneUtils.create_default_strategy_gene(strategy_gene_class)

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
                "generated_by": "GeneSerializer_decode",
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

    def _generate_indicator_parameters(
        self, indicator_type: str, param_val: float
    ) -> Dict[str, Any]:
        """指標パラメータを生成"""
        try:
            # 基本的なパラメータ生成
            period = max(1, min(200, int(param_val * 200)))

            # 指標タイプ別の特別なパラメータ
            if indicator_type in ["BBANDS", "KELTNER"]:
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

    def _encode_condition(self, condition) -> List[float]:
        """条件をエンコード（簡略化）"""
        try:
            # 簡単な条件エンコード（実際の実装では詳細化が必要）
            if hasattr(condition, "operator"):
                if condition.operator == ">":
                    return [1.0, 0.0, 0.0]
                elif condition.operator == "<":
                    return [0.0, 1.0, 0.0]
                else:
                    return [0.0, 0.0, 1.0]
            return [0.5, 0.5, 0.0]
        except Exception as e:
            logger.error(f"条件エンコードエラー: {e}")
            return [0.0, 0.0, 0.0]

    def _encode_tpsl_gene(self, tpsl_gene) -> List[float]:
        """TP/SL遺伝子をエンコード"""
        try:
            if not tpsl_gene or not tpsl_gene.enabled:
                return [0.0] * 8

            # 基本的なエンコード
            encoded = [
                1.0 if tpsl_gene.enabled else 0.0,
                tpsl_gene.stop_loss_pct or 0.03,
                tpsl_gene.take_profit_pct or 0.06,
                tpsl_gene.risk_reward_ratio or 2.0,
                tpsl_gene.atr_multiplier_sl or 2.0,
                tpsl_gene.atr_multiplier_tp or 3.0,
                (tpsl_gene.atr_period or 14) / 100.0,
                (tpsl_gene.lookback_period or 100) / 1000.0,
            ]
            return encoded[:8]
        except Exception as e:
            logger.error(f"TP/SL遺伝子エンコードエラー: {e}")
            return [0.0] * 8

    def _encode_position_sizing_gene(self, ps_gene) -> List[float]:
        """ポジションサイジング遺伝子をエンコード"""
        try:
            if not ps_gene or not ps_gene.enabled:
                return [0.0] * 8

            # 基本的なエンコード
            encoded = [
                1.0 if ps_gene.enabled else 0.0,
                ps_gene.risk_per_trade or 0.02,
                ps_gene.fixed_ratio or 0.1,
                ps_gene.fixed_quantity or 0.01,
                ps_gene.atr_multiplier or 2.0,
                ps_gene.optimal_f_multiplier or 0.5,
                (ps_gene.lookback_period or 30) / 100.0,
                ps_gene.min_position_size or 0.001,
            ]
            return encoded[:8]
        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子エンコードエラー: {e}")
            return [0.0] * 8

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

    def dict_to_strategy_gene(self, data: Dict[str, Any], strategy_gene_class):
        """
        辞書形式から戦略遺伝子を復元

        Args:
            data: 辞書形式の戦略遺伝子データ
            strategy_gene_class: StrategyGeneクラス

        Returns:
            戦略遺伝子オブジェクト
        """
        try:
            # 入力データがNoneまたは空でないことを確認
            if not data:
                logger.warning(
                    "戦略遺伝子データが空です。デフォルト戦略遺伝子を返します。"
                )
                return GeneUtils.create_default_strategy_gene(strategy_gene_class)

            # 指標遺伝子の復元
            indicators = [
                self.dict_to_indicator_gene(ind_data)
                for ind_data in data.get("indicators", [])
            ]

            # 条件の復元
            from ..models.strategy_models import ConditionGroup

            def parse_condition_or_group(cond_data):
                if isinstance(cond_data, dict) and cond_data.get("type") == "OR_GROUP":
                    conditions = [
                        parse_condition_or_group(c)
                        for c in cond_data.get("conditions", [])
                    ]
                    return ConditionGroup(conditions=conditions)
                else:
                    return self.dict_to_condition(cond_data)

            entry_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("entry_conditions", [])
            ]

            long_entry_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("long_entry_conditions", [])
            ]

            short_entry_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("short_entry_conditions", [])
            ]

            exit_conditions = [
                parse_condition_or_group(cond_data)
                for cond_data in data.get("exit_conditions", [])
            ]

            # リスク管理設定
            risk_management = data.get("risk_management", {})

            # TP/SL遺伝子の復元
            tpsl_gene = None
            if "tpsl_gene" in data and data["tpsl_gene"]:
                tpsl_gene = self.dict_to_tpsl_gene(data["tpsl_gene"])

            # ポジションサイジング遺伝子の復元
            position_sizing_gene = None
            if "position_sizing_gene" in data and data["position_sizing_gene"]:
                position_sizing_gene = self.dict_to_position_sizing_gene(
                    data["position_sizing_gene"]
                )

            # メタデータ
            metadata = data.get("metadata", {})

            # 後方互換性のための処理
            if not long_entry_conditions and entry_conditions:
                long_entry_conditions = entry_conditions
            if not short_entry_conditions and entry_conditions:
                short_entry_conditions = entry_conditions

            return strategy_gene_class(
                id=data.get("id", str(uuid.uuid4())),
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
            logger.error(f"戦略遺伝子辞書復元エラー: {e}")
            # エラー時はデフォルト戦略遺伝子を返す
            return GeneUtils.create_default_strategy_gene(strategy_gene_class)

    def dict_to_condition(self, data: Dict[str, Any]) -> "Condition":
        """
        辞書形式から条件を復元

        Args:
            data: 辞書形式の条件データ

        Returns:
            条件オブジェクト
        """
        try:
            from ..models.strategy_models import Condition

            return Condition(
                left_operand=data["left_operand"],
                operator=data["operator"],
                right_operand=data["right_operand"],
            )

        except Exception as e:
            logger.error(f"条件辞書復元エラー: {e}")
            raise ValueError(f"条件の復元に失敗: {e}")

    def decode_list_to_strategy_gene(self, encoded: List[float], strategy_gene_class):
        """
        数値リストから戦略遺伝子にデコード（旧GeneDecoder.decode_list_to_strategy_gene）

        Args:
            encoded: エンコードされた数値リスト
            strategy_gene_class: StrategyGeneクラス

        Returns:
            デコードされた戦略遺伝子オブジェクト
        """
        return self.from_list(encoded, strategy_gene_class)

    def encode_strategy_gene_to_list(self, strategy_gene):
        """
        戦略遺伝子を数値リストにエンコード（旧GeneEncoder.encode_strategy_gene_to_list）

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            エンコードされた数値リスト
        """
        return self.to_list(strategy_gene)
