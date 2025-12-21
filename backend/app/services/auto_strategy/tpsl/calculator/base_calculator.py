"""
Base TPSL Calculator

TP/SL計算器の基底クラスを定義します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from ...genes import TPSLGene
from ...genes.tpsl import TPSLResult

logger = logging.getLogger(__name__)


class BaseTPSLCalculator(ABC):
    """
    TP/SL計算器の基底クラス

    各計算方式の共通インターフェースを提供します。
    テンプレートメソッドパターンを使用して、共通の例外処理と結果作成を実現します。
    """

    def __init__(self, method_name: str):
        """初期化"""
        self.method_name = method_name

    def calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
        **kwargs,
    ) -> TPSLResult:
        """
        TP/SLを計算（テンプレートメソッド）
        """
        try:
            # サブクラス固有の計算を実行
            sl_pct, tp_pct, confidence, metrics = self._do_calculate(
                current_price, tpsl_gene, market_data, position_direction, **kwargs
            )

            return self._create_result(
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                confidence_score=confidence,
                expected_performance=metrics,
                metadata={"method": self.method_name},
            )
        except Exception as e:
            logger.error(f"{self.method_name} 計算エラー: {e}")
            return self._create_fallback_result()

    @abstractmethod
    def _do_calculate(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene],
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
        **kwargs,
    ) -> Tuple[float, float, float, Dict[str, Any]]:
        """固有の計算ロジック（(sl_pct, tp_pct, confidence, metrics) を返す）"""

    def _create_fallback_result(self) -> TPSLResult:
        """デフォルトのフォールバック結果"""
        return self._create_result(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            confidence_score=0.5,
            expected_performance={"fallback": True},
        )

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
                sl_price = current_price * (
                    1 + (position_direction * -1.0 * stop_loss_pct)
                )

        if take_profit_pct is not None:
            if take_profit_pct == 0:
                tp_price = current_price
            else:
                tp_price = current_price * (1 + (position_direction * take_profit_pct))

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
