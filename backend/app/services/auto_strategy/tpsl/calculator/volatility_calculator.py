"""
Volatility Calculator

ボラティリティベースのTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional

from ...genes import TPSLGene
from ...genes.tpsl import TPSLResult
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class VolatilityCalculator(BaseTPSLCalculator):
    """
    ボラティリティベースのTP/SL計算器

    ATR（Average True Range）に基づいてTP/SLを計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__("volatility_based")

    def calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
        **kwargs,
    ) -> TPSLResult:
        """
        ボラティリティベースでTP/SLを計算

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子
            market_data: 市場データ（ATR値を含む）
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）
            **kwargs: 追加パラメータ

        Returns:
            TPSLResult: 計算結果
        """
        try:
            # パラメータ取得
            if tpsl_gene:
                atr_period = tpsl_gene.atr_period or 21
                atr_multiplier_sl = tpsl_gene.atr_multiplier_sl or 1.5
                atr_multiplier_tp = tpsl_gene.atr_multiplier_tp or 3.0
            else:
                atr_period = kwargs.get("atr_period", 21)
                atr_multiplier_sl = kwargs.get("atr_multiplier_sl", 1.5)
                atr_multiplier_tp = kwargs.get("atr_multiplier_tp", 3.0)

            # ATR値を取得（または計算）
            atr_value = self._get_atr_value(market_data, atr_period, current_price)

            # ATRベースの割合を計算
            base_atr_pct = atr_value / current_price if atr_value else 0.02

            stop_loss_pct = base_atr_pct * atr_multiplier_sl
            take_profit_pct = base_atr_pct * atr_multiplier_tp

            return self._create_result(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                confidence_score=0.9,
                expected_performance={
                    "type": "volatility_based",
                    "atr_period": atr_period,
                    "atr_multiplier_sl": atr_multiplier_sl,
                    "atr_multiplier_tp": atr_multiplier_tp,
                    "atr_value": atr_value,
                    "base_atr_pct": base_atr_pct,
                },
            )

        except Exception as e:
            logger.error(f"ボラティリティベース計算エラー: {e}")
            # フォールバック
            return self._create_fallback_result()

    def _get_atr_value(
        self,
        market_data: Optional[Dict[str, Any]],
        atr_period: int,
        current_price: float,
    ) -> Optional[float]:
        """ATR値を取得または計算"""
        if not market_data:
            return None

        # 直接ATR値が提供されている場合
        if "atr" in market_data:
            return market_data["atr"]

        # OHLCデータからATRを計算
        if "ohlc_data" in market_data:
            return self._calculate_atr_from_ohlc(market_data["ohlc_data"], atr_period)

        # ボラティリティから推定
        if "volatility" in market_data:
            return current_price * market_data["volatility"]

        return None

    def _calculate_atr_from_ohlc(
        self, ohlc_data: list, atr_period: int
    ) -> Optional[float]:
        """OHLCデータからATRを計算"""
        try:
            if len(ohlc_data) < atr_period:
                return None

            true_ranges = []
            for i in range(1, len(ohlc_data)):
                high = ohlc_data[i]["high"]
                low = ohlc_data[i]["low"]
                prev_close = ohlc_data[i - 1]["close"]

                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                true_ranges.append(tr)

            # ATR = 平均True Range
            atr = sum(true_ranges[-atr_period:]) / atr_period
            return atr

        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")
            return None

    def _create_fallback_result(self) -> TPSLResult:
        """フォールバック結果を作成"""
        return self._create_result(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            confidence_score=0.5,
            expected_performance={
                "type": "volatility_fallback",
                "atr_period": 21,
            },
        )





