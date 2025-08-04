"""
遺伝子エンコーダー

GA用の戦略遺伝子エンコード機能を担当するモジュール。
"""

import logging
from typing import List, Optional

from . import gene_utils
from .gene_position_sizing import PositionSizingGene, PositionSizingMethod
from .gene_tpsl import TPSLGene, TPSLMethod

logger = logging.getLogger(__name__)


class GeneEncoder:
    """
    遺伝子エンコーダー

    GA用の戦略遺伝子エンコード機能を担当します。
    """

    def __init__(self):
        """初期化"""
        self.indicator_ids = gene_utils.get_indicator_ids()

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
                    param_val = gene_utils.normalize_parameter(
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

    def _encode_condition(self, condition) -> List[float]:
        """条件を数値リストにエンコード（簡略化）"""
        # 簡略化: 固定の条件パターンのみを表現
        return [1.0, 0.0, 1.0]  # プレースホルダー値

    def _encode_tpsl_gene(self, tpsl_gene: Optional[TPSLGene]) -> List[float]:
        """
        TP/SL遺伝子をエンコード

        Args:
            tpsl_gene: TP/SL遺伝子

        Returns:
            エンコードされた数値リスト（8要素）
        """
        if not tpsl_gene:
            return [0.0] * 8  # デフォルト値

        try:
            # メソッドをエンコード（0-1の範囲）
            method_mapping = {
                TPSLMethod.FIXED_PERCENTAGE: 0.2,
                TPSLMethod.RISK_REWARD_RATIO: 0.4,
                TPSLMethod.VOLATILITY_BASED: 0.6,
                TPSLMethod.STATISTICAL: 0.8,
                TPSLMethod.ADAPTIVE: 1.0,
            }
            method_encoded = method_mapping.get(tpsl_gene.method, 0.4)

            # パラメータを正規化（0-1の範囲）
            sl_pct_norm = min(
                max(tpsl_gene.stop_loss_pct / 0.15, 0.0), 1.0
            )  # 0-15%を0-1に
            tp_pct_norm = min(
                max(tpsl_gene.take_profit_pct / 0.3, 0.0), 1.0
            )  # 0-30%を0-1に
            rr_ratio_norm = min(
                max((tpsl_gene.risk_reward_ratio - 0.5) / 9.5, 0.0), 1.0
            )  # 0.5-10を0-1に
            base_sl_norm = min(
                max(tpsl_gene.base_stop_loss / 0.15, 0.0), 1.0
            )  # 0-15%を0-1に
            atr_sl_norm = min(
                max((tpsl_gene.atr_multiplier_sl - 0.5) / 4.5, 0.0), 1.0
            )  # 0.5-5を0-1に
            atr_tp_norm = min(
                max((tpsl_gene.atr_multiplier_tp - 1.0) / 9.0, 0.0), 1.0
            )  # 1-10を0-1に
            priority_norm = min(max(tpsl_gene.priority / 2.0, 0.0), 1.0)  # 0-2を0-1に

            return [
                method_encoded,
                sl_pct_norm,
                tp_pct_norm,
                rr_ratio_norm,
                base_sl_norm,
                atr_sl_norm,
                atr_tp_norm,
                priority_norm,
            ]

        except Exception as e:
            logger.error(f"TP/SL遺伝子エンコードエラー: {e}")
            return [0.4, 0.2, 0.2, 0.3, 0.2, 0.4, 0.3, 0.5]  # デフォルト値

    def _encode_position_sizing_gene(
        self, position_sizing_gene: Optional[PositionSizingGene]
    ) -> List[float]:
        """
        ポジションサイジング遺伝子をエンコード

        Args:
            position_sizing_gene: ポジションサイジング遺伝子

        Returns:
            エンコードされた数値リスト（8要素）
        """
        if not position_sizing_gene:
            return [0.0] * 8  # デフォルト値

        try:
            # メソッドをエンコード（0-1の範囲）
            method_mapping = {
                PositionSizingMethod.HALF_OPTIMAL_F: 0.25,
                PositionSizingMethod.VOLATILITY_BASED: 0.5,
                PositionSizingMethod.FIXED_RATIO: 0.75,
                PositionSizingMethod.FIXED_QUANTITY: 1.0,
            }
            # デフォルトをボラティリティベース（0.5）に変更
            method_encoded = method_mapping.get(position_sizing_gene.method, 0.5)

            # パラメータを正規化
            lookback_norm = (position_sizing_gene.lookback_period - 50) / (200 - 50)
            optimal_f_norm = (position_sizing_gene.optimal_f_multiplier - 0.25) / (
                0.75 - 0.25
            )
            atr_period_norm = (position_sizing_gene.atr_period - 10) / (30 - 10)
            atr_multiplier_norm = (position_sizing_gene.atr_multiplier - 1.0) / (
                4.0 - 1.0
            )
            risk_per_trade_norm = (position_sizing_gene.risk_per_trade - 0.01) / (
                0.05 - 0.01
            )
            fixed_ratio_norm = (position_sizing_gene.fixed_ratio - 0.05) / (0.3 - 0.05)
            fixed_quantity_norm = (position_sizing_gene.fixed_quantity - 0.1) / (
                5.0 - 0.1
            )
            priority_norm = (position_sizing_gene.priority - 0.5) / (1.5 - 0.5)

            # 0-1の範囲にクリップ
            lookback_norm = max(0, min(1, lookback_norm))
            optimal_f_norm = max(0, min(1, optimal_f_norm))
            atr_period_norm = max(0, min(1, atr_period_norm))
            atr_multiplier_norm = max(0, min(1, atr_multiplier_norm))
            risk_per_trade_norm = max(0, min(1, risk_per_trade_norm))
            fixed_ratio_norm = max(0, min(1, fixed_ratio_norm))
            fixed_quantity_norm = max(0, min(1, fixed_quantity_norm))
            priority_norm = max(0, min(1, priority_norm))

            return [
                method_encoded,
                lookback_norm,
                optimal_f_norm,
                atr_period_norm,
                atr_multiplier_norm,
                risk_per_trade_norm,
                fixed_ratio_norm,
                fixed_quantity_norm,
            ]

        except Exception as e:
            logger.error(f"ポジションサイジング遺伝子エンコードエラー: {e}")
            return [0.75, 0.5, 0.5, 0.2, 0.33, 0.25, 0.2, 0.2]  # デフォルト値
