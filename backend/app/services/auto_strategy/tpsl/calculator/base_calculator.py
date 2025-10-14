"""
Base TPSL Calculator

TP/SL計算器の基底クラスを定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from ...models.strategy_models import TPSLGene
from ...models.tpsl_result import TPSLResult

logger = logging.getLogger(__name__)


class BaseTPSLCalculator(ABC):
    """
    TP/SL計算器の基底クラス

    各計算方式の共通インターフェースを提供します。
    """

    def __init__(self, method_name: str):
        """初期化"""
        self.method_name = method_name

    @abstractmethod
    def calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
        **kwargs,
    ) -> TPSLResult:
        """
        TP/SLを計算

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子
            market_data: 市場データ
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）
            **kwargs: 追加パラメータ

        Returns:
            TPSLResult: 計算結果
        """
        pass

    def _make_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """割合からSL/TP価格を生成（共通ユーティリティ）"""
        sl_price: Optional[float] = None
        tp_price: Optional[float] = None

        if stop_loss_pct is not None:
            if stop_loss_pct == 0:
                sl_price = current_price
            else:
                if position_direction > 0:
                    sl_price = current_price * (1 - stop_loss_pct)
                else:
                    sl_price = current_price * (1 + stop_loss_pct)

        if take_profit_pct is not None:
            if take_profit_pct == 0:
                tp_price = current_price
            else:
                if position_direction > 0:
                    tp_price = current_price * (1 + take_profit_pct)
                else:
                    tp_price = current_price * (1 - take_profit_pct)

        return sl_price, tp_price

    def _create_result(
        self,
        stop_loss_pct: float,
        take_profit_pct: float,
        confidence_score: float = 0.0,
        expected_performance: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TPSLResult:
        """TPSLResultオブジェクトを作成"""
        return TPSLResult(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            method_used=self.method_name,
            confidence_score=confidence_score,
            expected_performance=expected_performance or {},
            metadata=metadata or {},
        )
