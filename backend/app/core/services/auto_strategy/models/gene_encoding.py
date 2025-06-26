"""
遺伝子エンコーディング

GA用の戦略遺伝子エンコード/デコード機能を担当するモジュール。
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class GeneEncoder:
    """
    遺伝子エンコーダー

    GA用の戦略遺伝子エンコード/デコード機能を担当します。
    """

    def __init__(self):
        """初期化"""
        self.indicator_ids = self._get_indicator_ids()
        self.id_to_indicator = self._get_id_to_indicator()

    def _get_indicator_ids(self) -> Dict[str, int]:
        """指標IDマッピングを取得"""
        try:
            from app.core.services.indicators.constants import ALL_INDICATORS

            indicator_ids = {"": 0}  # 未使用
            for i, indicator in enumerate(ALL_INDICATORS, 1):
                indicator_ids[indicator] = i

            return indicator_ids

        except ImportError as e:
            logger.error(f"指標定数インポートエラー: {e}")
            # フォールバック: オートストラテジー用10個の指標のみ
            return {
                "": 0,
                "SMA": 1,
                "EMA": 2,
                "MACD": 3,
                "BB": 4,
                "RSI": 5,
                "STOCH": 6,
                "CCI": 7,
                "ADX": 8,
                "ATR": 9,
                "OBV": 10,
            }

    def _get_id_to_indicator(self) -> Dict[int, str]:
        """ID→指標の逆引きマッピングを取得"""
        return {v: k for k, v in self.indicator_ids.items()}

    def encode_strategy_gene_to_list(self, strategy_gene) -> List[float]:
        """
        戦略遺伝子を固定長の数値リストにエンコード

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
                    param_val = self._normalize_parameter(
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

            return encoded

        except Exception as e:
            logger.error(f"戦略遺伝子エンコードエラー: {e}")
            # エラー時はデフォルトエンコードを返す
            return [0.0] * 16  # 5指標×2 + 条件×6

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
                    # 指標IDマッピングの改善：正規化された値から元のIDを復元
                    # 0.01未満は無効指標として扱う
                    if encoded[idx] < 0.01:
                        continue

                    # 正規化された値から元の指標IDを復元
                    indicator_id = int(encoded[idx] * len(self.indicator_ids))

                    # 指標IDが範囲外の場合は調整（1以上、最大ID以下）
                    max_id = len(self.indicator_ids) - 1
                    indicator_id = max(1, min(max_id, indicator_id))
                    param_val = encoded[idx + 1]

                    indicator_type = self.id_to_indicator.get(indicator_id, "")
                    if indicator_type and indicator_type != "":
                        # 指標タイプに応じたパラメータ生成
                        parameters = self._generate_indicator_parameters(
                            indicator_type, param_val
                        )

                        # IndicatorGeneクラスを動的にインポート
                        from .strategy_gene import IndicatorGene

                        indicators.append(
                            IndicatorGene(
                                type=indicator_type,
                                parameters=parameters,
                                enabled=True,
                            )
                        )

                        logger.debug(
                            f"デコード指標: {indicator_type} (ID: {indicator_id}, 元値: {encoded[idx]:.3f})"
                        )

            # 条件部分をデコード（確実に条件を生成）
            entry_conditions = []
            exit_conditions = []

            if indicators:
                # 複数の指標を使用してより多様な条件を作成
                from .strategy_gene import Condition

                # 最初の指標を使用
                first_indicator = indicators[0]
                indicator_name = f"{first_indicator.type}_{first_indicator.parameters.get('period', 20)}"

                # 基本的な条件を生成
                if first_indicator.type in ["SMA", "EMA", "WMA", "MAMA"]:
                    # 移動平均系（MAMAを含む）
                    entry_conditions = [
                        Condition(
                            left_operand="close",
                            operator="cross_above",
                            right_operand=indicator_name,
                        )
                    ]
                    exit_conditions = [
                        Condition(
                            left_operand="close",
                            operator="cross_below",
                            right_operand=indicator_name,
                        )
                    ]
                elif first_indicator.type == "RSI":
                    # RSI系
                    entry_conditions = [
                        Condition(
                            left_operand=indicator_name,
                            operator="<",
                            right_operand="30",
                        )
                    ]
                    exit_conditions = [
                        Condition(
                            left_operand=indicator_name,
                            operator=">",
                            right_operand="70",
                        )
                    ]
                elif first_indicator.type == "MACD":
                    # MACD系
                    entry_conditions = [
                        Condition(
                            left_operand="MACD_line",
                            operator="cross_above",
                            right_operand="MACD_signal",
                        )
                    ]
                    exit_conditions = [
                        Condition(
                            left_operand="MACD_line",
                            operator="cross_below",
                            right_operand="MACD_signal",
                        )
                    ]
                else:
                    # その他の指標
                    entry_conditions = [
                        Condition(
                            left_operand="close",
                            operator=">",
                            right_operand=indicator_name,
                        )
                    ]
                    exit_conditions = [
                        Condition(
                            left_operand="close",
                            operator="<",
                            right_operand=indicator_name,
                        )
                    ]

                # 複数指標がある場合は追加条件を生成
                if len(indicators) > 1:
                    second_indicator = indicators[1]
                    second_name = f"{second_indicator.type}_{second_indicator.parameters.get('period', 20)}"

                    if second_indicator.type in ["RSI", "STOCH", "CCI"]:
                        # オシレーター系の追加条件
                        entry_conditions.append(
                            Condition(
                                left_operand=second_name,
                                operator="<",
                                right_operand="50",
                            )
                        )
                        exit_conditions.append(
                            Condition(
                                left_operand=second_name,
                                operator=">",
                                right_operand="50",
                            )
                        )

                logger.info(
                    f"生成された条件: エントリー={len(entry_conditions)}, エグジット={len(exit_conditions)}"
                )
            else:
                # 指標がない場合はデフォルト条件（価格ベース）
                from .strategy_gene import Condition

                entry_conditions = [
                    Condition(left_operand="close", operator=">", right_operand="open")
                ]
                exit_conditions = [
                    Condition(left_operand="close", operator="<", right_operand="open")
                ]
                logger.warning("指標なしでデフォルト条件を使用")

            # リスク管理設定を追加
            risk_management = {
                "stop_loss": 0.03,
                "take_profit": 0.15,
                "position_size": 0.1,
            }

            # メタデータを追加
            metadata = {
                "generated_by": "GeneEncoder_decode",
                "source": (
                    "fallback_individual" if len(indicators) <= 1 else "normal_decode"
                ),
                "indicators_count": len(indicators),
                "decoded_from_length": len(encoded),
            }

            return strategy_gene_class(
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
                risk_management=risk_management,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"戦略遺伝子デコードエラー: {e}")
            # エラー時はデフォルト戦略遺伝子を返す
            return self._create_default_strategy_gene(strategy_gene_class)

    def _normalize_parameter(
        self, value: float, min_val: float = 1, max_val: float = 200
    ) -> float:
        """パラメータを0-1の範囲に正規化"""
        try:
            return (value - min_val) / (max_val - min_val)
        except ZeroDivisionError:
            return 0.0

    def _denormalize_parameter(
        self, normalized_val: float, min_val: float = 1, max_val: float = 200
    ) -> int:
        """正規化されたパラメータを元の範囲に戻す"""
        try:
            value = min_val + normalized_val * (max_val - min_val)
            return int(max(min_val, min(max_val, int(value))))
        except Exception:
            return int(min_val)

    def _encode_condition(self, condition) -> List[float]:
        """条件を数値リストにエンコード（簡略化）"""
        # 簡略化: 固定の条件パターンのみを表現
        return [1.0, 0.0, 1.0]  # プレースホルダー値

    def _generate_indicator_parameters(
        self, indicator_type: str, param_val: float
    ) -> Dict:
        """指標タイプに応じたパラメータを生成"""
        try:
            # 基本的な期間パラメータ
            if indicator_type in [
                "SMA",
                "EMA",
                "RSI",
                "ATR",
                "CCI",
                "WILLIAMS",
                "MFI",
                "MOMENTUM",
                "ROC",
            ]:
                period = self._denormalize_parameter(param_val, min_val=5, max_val=50)
                return {"period": period}

            # 複数パラメータを持つ指標
            elif indicator_type == "MACD":
                fast_period = self._denormalize_parameter(
                    param_val, min_val=5, max_val=20
                )
                slow_period = fast_period + self._denormalize_parameter(
                    param_val, min_val=5, max_val=15
                )
                signal_period = self._denormalize_parameter(
                    param_val, min_val=5, max_val=15
                )
                return {
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                }

            elif indicator_type == "BB":
                period = self._denormalize_parameter(param_val, min_val=10, max_val=30)
                std_dev = 1.5 + param_val * 1.0  # 1.5-2.5
                return {"period": period, "std_dev": std_dev}

            elif indicator_type == "STOCH":
                k_period = self._denormalize_parameter(param_val, min_val=5, max_val=20)
                d_period = self._denormalize_parameter(param_val, min_val=3, max_val=10)
                return {"k_period": k_period, "d_period": d_period}

            elif indicator_type == "MAMA":
                # MAMA指標のパラメータ
                fast_limit = 0.2 + param_val * 0.3  # 0.2-0.5の範囲
                slow_limit = 0.01 + param_val * 0.09  # 0.01-0.1の範囲
                return {"fast_limit": fast_limit, "slow_limit": slow_limit}

            else:
                # デフォルト（期間のみ）
                period = self._denormalize_parameter(param_val, min_val=5, max_val=50)
                return {"period": period}

        except Exception as e:
            logger.error(f"指標パラメータ生成エラー: {e}")
            return {"period": 20}  # デフォルト値

    def _generate_indicator_specific_conditions(self, indicator, indicator_name):
        """指標に応じた条件を生成"""
        try:
            from .strategy_gene import Condition

            entry_conditions = []
            exit_conditions = []

            if indicator.type == "RSI":
                # RSI条件
                entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="<", right_operand=30
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=70
                    )
                ]

            elif indicator.type in ["SMA", "EMA", "MAMA"]:
                # 移動平均条件（MAMAを含む）
                entry_conditions = [
                    Condition(
                        left_operand="close", operator=">", right_operand=indicator_name
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand="close", operator="<", right_operand=indicator_name
                    )
                ]

            elif indicator.type == "MACD":
                # MACD条件
                entry_conditions = [
                    Condition(
                        left_operand="MACD_line",
                        operator=">",
                        right_operand="MACD_signal",
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand="MACD_line",
                        operator="<",
                        right_operand="MACD_signal",
                    )
                ]

            else:
                # デフォルト条件
                entry_conditions = [
                    Condition(
                        left_operand=indicator_name, operator=">", right_operand=50
                    )
                ]
                exit_conditions = [
                    Condition(
                        left_operand=indicator_name, operator="<", right_operand=50
                    )
                ]

            return entry_conditions, exit_conditions

        except Exception as e:
            logger.error(f"指標固有条件生成エラー: {e}")
            # エラー時はデフォルト条件を返す
            from .strategy_gene import Condition

            return (
                [Condition(left_operand="close", operator=">", right_operand="close")],
                [Condition(left_operand="close", operator="<", right_operand="close")],
            )

    def _create_default_strategy_gene(self, strategy_gene_class):
        """デフォルト戦略遺伝子を作成"""
        try:
            from .strategy_gene import IndicatorGene, Condition

            # デフォルト指標
            indicators = [
                IndicatorGene(type="SMA", parameters={"period": 20}, enabled=True),
                IndicatorGene(type="RSI", parameters={"period": 14}, enabled=True),
            ]

            # デフォルト条件
            entry_conditions = [
                Condition(left_operand="RSI_14", operator="<", right_operand=30)
            ]
            exit_conditions = [
                Condition(left_operand="RSI_14", operator=">", right_operand=70)
            ]

            return strategy_gene_class(
                indicators=indicators,
                entry_conditions=entry_conditions,
                exit_conditions=exit_conditions,
            )

        except Exception as e:
            logger.error(f"デフォルト戦略遺伝子作成エラー: {e}")
            # 最小限の戦略遺伝子を返す
            return strategy_gene_class()

    def get_encoding_info(self) -> Dict:
        """エンコーディング情報を取得"""
        return {
            "indicator_count": len(self.indicator_ids) - 1,  # 0を除く
            "max_indicators": 5,
            "encoding_length": 16,  # 5指標×2 + 条件×6
            "supported_indicators": list(self.indicator_ids.keys())[1:],  # 空文字を除く
        }
