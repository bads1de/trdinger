"""
Volatility Calculator

ボラティリティベースのTP/SL計算器
"""

import logging
from typing import Any, cast

import pandas as pd

from app.services.ml.common.utils import calculate_true_range

from ...genes import TPSLGene
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class VolatilityCalculator(BaseTPSLCalculator):
    """
    ボラティリティベースのTP/SL計算器
    """

    def __init__(self) -> None:
        """初期化"""
        super().__init__("volatility_based")

    def _do_calculate(
        self,
        current_price: float,
        tpsl_gene: TPSLGene | None,
        market_data: dict[str, Any] | None,
        position_direction: float,
        **kwargs: Any,
    ) -> tuple[float, float, float, dict[str, Any]]:
        """
        ボラティリティ方式（ATR）によるTP/SL計算の実装

        市場のボラティリティ（ATR: Average True Range）に基づいて、
        値動きの大きさに応じたTP/SLレベルを動的に計算します。

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子
            market_data: 市場データ
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）
            **kwargs: 追加引数（atr_period, atr_multiplier_sl, atr_multiplier_tp）

        Returns:
            (SL割合, TP割合, 信頼度スコア, 実行詳細データ)のタプル
        """
        # 1. パラメータ取得
        if tpsl_gene:
            atr_period = tpsl_gene.atr_period or 21
            atr_multiplier_sl = tpsl_gene.atr_multiplier_sl or 1.5
            atr_multiplier_tp = tpsl_gene.atr_multiplier_tp or 3.0
        else:
            atr_period = kwargs.get("atr_period", 21)
            atr_multiplier_sl = kwargs.get("atr_multiplier_sl", 1.5)
            atr_multiplier_tp = kwargs.get("atr_multiplier_tp", 3.0)

        # 2. ATR値を取得（または計算）
        atr_value = self._get_atr_value(market_data, atr_period, current_price)

        # 3. ATRベースの割合を計算
        base_atr_pct = atr_value / current_price if atr_value else 0.02
        sl_pct = base_atr_pct * atr_multiplier_sl
        tp_pct = base_atr_pct * atr_multiplier_tp

        return (
            sl_pct,
            tp_pct,
            0.9,
            {
                "atr_period": atr_period,
                "atr_multiplier_sl": atr_multiplier_sl,
                "atr_multiplier_tp": atr_multiplier_tp,
                "atr_value": atr_value,
                "base_atr_pct": base_atr_pct,
            },
        )

    def _get_atr_value(
        self,
        market_data: dict[str, Any] | None,
        atr_period: int,
        current_price: float,
    ) -> float | None:
        """ATR値を取得または計算"""
        if not market_data:
            return None

        # 直接ATR値が提供されている場合
        if "atr" in market_data:
            return cast(float, market_data["atr"])

        # OHLCデータからATRを計算
        if "ohlc_data" in market_data:
            return self._calculate_atr_from_ohlc(market_data["ohlc_data"], atr_period)

        # ボラティリティから推定
        if "volatility" in market_data:
            return cast(float, current_price * market_data["volatility"])

        return None

    def _calculate_atr_from_ohlc(
        self, ohlc_data: list, atr_period: int
    ) -> float | None:
        """OHLCデータからATRを計算"""
        try:
            if len(ohlc_data) < atr_period:
                return None

            # True Range 計算は共通ユーティリティに委譲（手書きループを排除）
            # 先頭バーは前回Closeが無いため除外（元実装のループ開始位置 i=1 と同じ）
            df = pd.DataFrame(ohlc_data)
            # ohlc_data が未型付けのため pandas-stubs が列型を推論できず、Series に明示キャストする
            true_ranges = calculate_true_range(
                cast(pd.Series, df["high"]),
                cast(pd.Series, df["low"]),
                cast(pd.Series, df["close"]),
            ).iloc[1:]

            # ATR = 直近 atr_period 個の True Range の平均（元実装の除算仕様を維持）
            last_trs = true_ranges.iloc[-(atr_period):]
            if last_trs.empty:
                return None
            return float(last_trs.sum() / atr_period)

        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")
            return None
