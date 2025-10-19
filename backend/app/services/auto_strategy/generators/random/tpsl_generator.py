"""
TP/SL生成器

ランダム戦略のTP/SL遺伝子を生成する専門ジェネレーター
"""

import logging
import random

from ...models.strategy_models import TPSLGene, create_random_tpsl_gene
from ...models.strategy_models import TPSLMethod

logger = logging.getLogger(__name__)


class TPSLGenerator:
    """
    TP/SL遺伝子の生成を担当するクラス
    """

    def __init__(self, config: any):
        """
        初期化

        Args:
            config: GA設定オブジェクト
        """
        self.config = config

    def generate_tpsl_gene(self) -> TPSLGene:
        """
        TP/SL遺伝子を生成（GA最適化対象）

        Returns:
            生成されたTP/SL遺伝子
        """
        try:
            # Anyの設定範囲内でランダムなTP/SL遺伝子を生成
            tpsl_gene = create_random_tpsl_gene()

            # Anyの制約を適用（設定されている場合）
            if hasattr(self.config, "tpsl_method_constraints"):
                # 許可されたメソッドのみを使用
                allowed_methods = self.config.tpsl_method_constraints
                if allowed_methods:
                    tpsl_gene.method = random.choice(
                        [TPSLMethod(m) for m in allowed_methods]
                    )

            if (
                hasattr(self.config, "tpsl_sl_range")
                and self.config.tpsl_sl_range is not None
            ):
                sl_min, sl_max = self.config.tpsl_sl_range
                tpsl_gene.stop_loss_pct = random.uniform(sl_min, sl_max)
                tpsl_gene.base_stop_loss = random.uniform(sl_min, sl_max)

            if (
                hasattr(self.config, "tpsl_tp_range")
                and self.config.tpsl_tp_range is not None
            ):
                # TP範囲制約
                tp_min, tp_max = self.config.tpsl_tp_range
                tpsl_gene.take_profit_pct = random.uniform(tp_min, tp_max)

            if (
                hasattr(self.config, "tpsl_rr_range")
                and self.config.tpsl_rr_range is not None
            ):
                # リスクリワード比範囲制約
                rr_min, rr_max = self.config.tpsl_rr_range
                tpsl_gene.risk_reward_ratio = random.uniform(rr_min, rr_max)

            if (
                hasattr(self.config, "tpsl_atr_multiplier_range")
                and self.config.tpsl_atr_multiplier_range is not None
            ):
                # ATR倍率範囲制約
                atr_min, atr_max = self.config.tpsl_atr_multiplier_range
                tpsl_gene.atr_multiplier_sl = random.uniform(atr_min, atr_max)
                tpsl_gene.atr_multiplier_tp = random.uniform(
                    atr_min * 1.5, atr_max * 2.0
                )

            return tpsl_gene

        except Exception as e:
            logger.error(f"TP/SL遺伝子生成エラー: {e}")
            # フォールバック: デフォルトのTP/SL遺伝子
            return TPSLGene(
                method=TPSLMethod.RISK_REWARD_RATIO,
                stop_loss_pct=0.03,
                take_profit_pct=0.06,
                risk_reward_ratio=2.0,
                base_stop_loss=0.03,
                enabled=True,
            )
