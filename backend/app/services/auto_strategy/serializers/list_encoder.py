"""
リストエンコードコンポーネント

戦略遺伝子を数値リストにエンコードする機能を担当します。
"""

import logging
from typing import List

from ..utils.gene_utils import GeneUtils

logger = logging.getLogger(__name__)


class ListEncoder:
    """
    リストエンコードクラス

    戦略遺伝子を数値リストにエンコードする機能を担当します。
    """

    def __init__(self):
        """初期化"""
        pass

    def to_list(self, strategy_gene) -> List[float]:
        """
        戦略遺伝子を固定長の数値リストにエンコード（旧encode_strategy_gene_to_list）

        Args:
            strategy_gene: 戦略遺伝子オブジェクト

        Returns:
            エンコードされた数値リスト
        """
        try:
            from ..config import GAConfig

            encoded = []
            max_indicators = GAConfig().max_indicators

            # 指標部分（指標数 × 2値 = 10要素）
            for i in range(max_indicators):
                if (
                    i < len(strategy_gene.indicators)
                    and strategy_gene.indicators[i].enabled
                ):
                    indicator = strategy_gene.indicators[i]
                    indicator_id = self._get_indicator_id(indicator.type)
                    # パラメータを正規化（期間の場合は1-200を0-1に変換）
                    param_val = GeneUtils.normalize_parameter(
                        indicator.parameters.get("period", 20)
                    )
                else:
                    indicator_id = 0  # 未使用
                    param_val = 0.0

                # 指標IDを0-1の範囲に正規化してエンコード
                normalized_id = (
                    indicator_id / 100.0
                    if indicator_id > 0
                    else 0.0  # 指標IDの範囲を想定
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
            # 長さ = 指標数*2 + エントリー*3 + イグジット*3 + TPSL*8 + ポジション*8
            expected_len = max_indicators * 2 + 3 + 3 + 8 + 8
            return [0.0] * expected_len

    def _get_indicator_id(self, indicator_type: str) -> int:
        """
        指標タイプからIDを取得

        Args:
            indicator_type: 指標タイプ

        Returns:
            指標ID
        """
        # 簡易的な指標IDマッピング（実際の実装では詳細化が必要）
        indicator_map = {
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
        }
        return indicator_map.get(indicator_type, 0)

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
