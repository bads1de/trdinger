"""
Fixed Percentage Calculator

固定パーセンテージ方式のTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional

from ...genes import TPSLGene
from ...genes.tpsl_gene import TPSLResult
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class FixedPercentageCalculator(BaseTPSLCalculator):
    """
    固定パーセンテージ方式のTP/SL計算器

    指定された固定割合でTP/SLを計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__("fixed_percentage")

    def calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
        **kwargs,
    ) -> TPSLResult:
        """
        固定パーセンテージ方式でTP/SLを計算

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
            # TP/SL遺伝子からパラメータを取得
            if tpsl_gene:
                stop_loss_pct = tpsl_gene.stop_loss_pct
                take_profit_pct = tpsl_gene.take_profit_pct
            else:
                # デフォルト値
                stop_loss_pct = kwargs.get("stop_loss_pct", 0.03)
                take_profit_pct = kwargs.get("take_profit_pct", 0.06)

            # 価格計算
            sl_price, tp_price = self._make_prices(
                current_price, stop_loss_pct, take_profit_pct, position_direction
            )

            return self._create_result(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                confidence_score=0.95,  # 固定方式は高信頼性
                expected_performance={
                    "type": "fixed_percentage",
                    "sl_price": sl_price,
                    "tp_price": tp_price,
                },
            )

        except Exception as e:
            logger.error(f"固定パーセンテージ計算エラー: {e}")
            # フォールバック
            return self._create_fallback_result()

    def _create_fallback_result(self) -> TPSLResult:
        """フォールバック結果を作成"""
        return self._create_result(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            confidence_score=0.5,
            expected_performance={"type": "fixed_percentage_fallback"},
        )





