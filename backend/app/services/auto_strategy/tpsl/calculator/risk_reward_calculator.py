"""
Risk Reward Calculator

リスクリワード比方式のTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional

from ...models.strategy_models import TPSLGene
from ...models.tpsl_result import TPSLResult
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class RiskRewardCalculator(BaseTPSLCalculator):
    """
    リスクリワード比方式のTP/SL計算器

    ベースとなるSL割合と目標RR比からTPを計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__("risk_reward_ratio")

    def calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
        **kwargs
    ) -> TPSLResult:
        """
        リスクリワード比方式でTP/SLを計算

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子
            market_data: 市場データ（使用しない）
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）
            **kwargs: 追加パラメータ

        Returns:
            TPSLResult: 計算結果
        """
        try:
            # パラメータ取得
            if tpsl_gene:
                base_stop_loss = tpsl_gene.base_stop_loss or tpsl_gene.stop_loss_pct
                target_ratio = tpsl_gene.risk_reward_ratio
            else:
                base_stop_loss = kwargs.get("stop_loss_pct", kwargs.get("base_stop_loss", 0.03))
                target_ratio = kwargs.get("target_ratio", kwargs.get("risk_reward_ratio", 2.0))

            # リスクリワード比でTPを計算
            take_profit_pct = base_stop_loss * target_ratio

            return self._create_result(
                stop_loss_pct=base_stop_loss,
                take_profit_pct=take_profit_pct,
                confidence_score=0.85,
                expected_performance={
                    "type": "risk_reward_ratio",
                    "risk_reward_ratio": target_ratio,
                    "base_stop_loss": base_stop_loss,
                },
            )

        except Exception as e:
            logger.error(f"リスクリワード比計算エラー: {e}")
            # フォールバック
            return self._create_fallback_result()

    def _create_fallback_result(self) -> TPSLResult:
        """フォールバック結果を作成"""
        return self._create_result(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            confidence_score=0.5,
            expected_performance={
                "type": "risk_reward_fallback",
                "risk_reward_ratio": 2.0
            },
        )