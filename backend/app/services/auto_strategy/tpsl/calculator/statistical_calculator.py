"""
Statistical Calculator

統計的分析ベースのTP/SL計算器
"""

import logging
from typing import Any, Dict, Optional, Tuple

from ...models import TPSLGene
from ...models.tpsl_gene import TPSLResult
from .base_calculator import BaseTPSLCalculator

logger = logging.getLogger(__name__)


class StatisticalCalculator(BaseTPSLCalculator):
    """
    統計的分析ベースのTP/SL計算器

    過去の価格データに基づいて統計的に最適なTP/SLを計算します。
    """

    def __init__(self):
        """初期化"""
        super().__init__("statistical")

    def calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
        **kwargs,
    ) -> TPSLResult:
        """
        統計的分析でTP/SLを計算

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子
            market_data: 市場データ（過去価格データを含む）
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）
            **kwargs: 追加パラメータ

        Returns:
            TPSLResult: 計算結果
        """
        try:
            # パラメータ取得
            if tpsl_gene:
                lookback_period = tpsl_gene.lookback_period or 150
                confidence_threshold = tpsl_gene.confidence_threshold or 0.95
            else:
                lookback_period = kwargs.get(
                    "lookback_period_days", kwargs.get("lookback_period", 150)
                )
                confidence_threshold = kwargs.get("confidence_threshold", 0.95)

            # 統計分析で最適なTP/SLを計算
            stop_loss_pct, take_profit_pct = self._calculate_statistical_levels(
                market_data, lookback_period, confidence_threshold, current_price
            )

            return self._create_result(
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                confidence_score=0.7,
                expected_performance={
                    "type": "statistical",
                    "lookback_period": lookback_period,
                    "confidence_threshold": confidence_threshold,
                },
            )

        except Exception as e:
            logger.error(f"統計的計算エラー: {e}")
            # フォールバック
            return self._create_fallback_result()

    def _calculate_statistical_levels(
        self,
        market_data: Optional[Dict[str, Any]],
        lookback_period: int,
        confidence_threshold: float,
        current_price: float,
    ) -> Tuple[float, float]:
        """統計的な最適レベルを計算"""
        try:
            if not market_data or "historical_prices" not in market_data:
                # データがない場合のデフォルト値
                return 0.03, 0.06

            historical_prices = market_data["historical_prices"]
            if len(historical_prices) < lookback_period:
                return 0.03, 0.06

            # 過去データの統計分析
            recent_prices = historical_prices[-lookback_period:]

            # 価格変化の分布を計算
            price_changes = []
            for i in range(1, len(recent_prices)):
                change_pct = (recent_prices[i] - recent_prices[i - 1]) / recent_prices[
                    i - 1
                ]
                price_changes.append(change_pct)

            if not price_changes:
                return 0.03, 0.06

            # 標準偏差を計算
            mean_change = sum(price_changes) / len(price_changes)
            variance = sum((x - mean_change) ** 2 for x in price_changes) / len(
                price_changes
            )
            std_dev = variance**0.5

            # 信頼区間に基づいてSL/TPを設定
            # 95%信頼区間を使用（正規分布を仮定）
            z_score = 1.96  # 95%信頼区間

            sl_std_devs = z_score
            tp_std_devs = z_score * 2  # TPはSLの2倍の変動を想定

            stop_loss_pct = abs(std_dev * sl_std_devs)
            take_profit_pct = abs(std_dev * tp_std_devs)

            # 最小/最大値の制約
            stop_loss_pct = max(0.01, min(stop_loss_pct, 0.1))  # 1% - 10%
            take_profit_pct = max(0.02, min(take_profit_pct, 0.2))  # 2% - 20%

            return stop_loss_pct, take_profit_pct

        except Exception as e:
            logger.error(f"統計レベル計算エラー: {e}")
            return 0.03, 0.06

    def _create_fallback_result(self) -> TPSLResult:
        """フォールバック結果を作成"""
        return self._create_result(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            confidence_score=0.5,
            expected_performance={
                "type": "statistical_fallback",
                "lookback_period": 150,
            },
        )


