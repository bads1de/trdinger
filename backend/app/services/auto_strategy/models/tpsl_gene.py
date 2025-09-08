"""
TP/SL 遺伝子
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .enums import TPSLMethod
from ..utils.gene_utils import BaseGene


@dataclass
class TPSLGene(BaseGene):
    """
    TP/SL遺伝子

    GA最適化対象としてのTP/SL設定を表現します。
    BaseGeneを継承して共通機能を活用します。
    """

    method: TPSLMethod = TPSLMethod.RISK_REWARD_RATIO
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    risk_reward_ratio: float = 2.0
    base_stop_loss: float = 0.03
    atr_multiplier_sl: float = 2.0
    atr_multiplier_tp: float = 3.0
    atr_period: int = 14
    lookback_period: int = 100
    confidence_threshold: float = 0.7
    method_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "fixed": 0.25,
            "risk_reward": 0.35,
            "volatility": 0.25,
            "statistical": 0.15,
        }
    )
    enabled: bool = True
    priority: float = 1.0

    def _validate_parameters(self, errors: List[str]) -> None:
        """パラメータ固有の検証を実装"""
        try:
            from ..constants import TPSL_LIMITS

            sl_min, sl_max = TPSL_LIMITS["stop_loss_pct"]
            if not (sl_min <= self.stop_loss_pct <= sl_max):
                errors.append(
                    f"stop_loss_pct must be between {sl_min*100:.1f}% and {sl_max*100:.0f}%"
                )

            tp_min, tp_max = TPSL_LIMITS["take_profit_pct"]
            if not (tp_min <= self.take_profit_pct <= tp_max):
                errors.append(
                    f"take_profit_pct must be between {tp_min*100:.1f}% and {tp_max*100:.0f}%"
                )

            # 他のパラメータ検証
            self._validate_range(
                self.risk_reward_ratio, 1.0, 10.0, "risk_reward_ratio", errors
            )
            self._validate_range(
                self.confidence_threshold, 0.0, 1.0, "confidence_threshold", errors
            )
            self._validate_range(
                self.atr_multiplier_sl, 0.1, 5.0, "atr_multiplier_sl", errors
            )
            self._validate_range(
                self.atr_multiplier_tp, 0.1, 10.0, "atr_multiplier_tp", errors
            )

            # method_weightsの検証
            total_weight = sum(self.method_weights.values())
            if not (0.99 <= total_weight <= 1.01):  # 浮動小数点誤差考慮
                errors.append("method_weightsの合計は1.0である必要があります")

        except ImportError:
            # 定数が利用できない場合の基本検証
            if not (0.005 <= self.stop_loss_pct <= 0.15):
                errors.append("stop_loss_pct must be between 0.5% and 15%")

            if not (0.01 <= self.take_profit_pct <= 0.3):
                errors.append("take_profit_pct must be between 1% and 30%")
