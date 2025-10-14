"""
TP/SL 計算結果モデル
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class TPSLResult:
    """TP/SL計算結果

    TPSLGeneratorとTPSLServiceでの計算結果を統一して表現します。
    """

    stop_loss_pct: float
    take_profit_pct: float
    method_used: str
    confidence_score: float = 0.0
    expected_performance: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """後処理"""
        if self.expected_performance is None:
            self.expected_performance = {}
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "method_used": self.method_used,
            "confidence_score": self.confidence_score,
            "expected_performance": self.expected_performance,
            "metadata": self.metadata,
        }
