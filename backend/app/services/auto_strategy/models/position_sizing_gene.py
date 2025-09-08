"""
ポジションサイジング遺伝子
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .enums import PositionSizingMethod
from ..utils.gene_utils import BaseGene


@dataclass
class PositionSizingGene(BaseGene):
    """
    ポジションサイジング遺伝子

    GA最適化対象としてのポジションサイジング設定を表現します。
    BaseGeneを継承して共通機能を活用します。
    """

    method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_BASED
    lookback_period: int = 100
    optimal_f_multiplier: float = 0.5
    atr_period: int = 14
    atr_multiplier: float = 2.0
    risk_per_trade: float = 0.02
    fixed_ratio: float = 0.1
    fixed_quantity: float = 1.0
    min_position_size: float = 0.01
    max_position_size: float = 9999.0
    enabled: bool = True
    priority: float = 1.0

    def _validate_parameters(self, errors: List[str]) -> None:
        """パラメータ固有の検証を実装"""
        try:
            from ..constants import POSITION_SIZING_LIMITS

            lb_min, lb_max = POSITION_SIZING_LIMITS["lookback_period"]
            if not (lb_min <= self.lookback_period <= lb_max):
                errors.append(
                    f"lookback_periodは{lb_min}-{lb_max}の範囲である必要があります"
                )

            # 他のパラメータ検証も実装可能
            self._validate_range(
                self.risk_per_trade, 0.001, 0.1, "risk_per_trade", errors
            )
            self._validate_range(self.fixed_ratio, 0.001, 1.0, "fixed_ratio", errors)
            self._validate_range(
                self.atr_multiplier, 0.1, 5.0, "atr_multiplier", errors
            )

        except ImportError:
            # 定数が利用できない場合の基本検証
            if not (50 <= self.lookback_period <= 200):
                errors.append("lookback_periodは50-200の範囲である必要があります")

            # 基本的な範囲検証
            if not (0.001 <= self.risk_per_trade <= 0.1):
                errors.append("risk_per_tradeは0.001-0.1の範囲である必要があります")
            if not (0.001 <= self.fixed_ratio <= 1.0):
                errors.append("fixed_ratioは0.001-1.0の範囲である必要があります")
