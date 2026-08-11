"""
統一ポジションサイジングサービス

PositionSizingGeneに基づいて実際のポジションサイズを計算するサービスです。
バックテストループ内での使用に最適化された高速計算を提供します。
"""

import logging
from typing import Any

from ..utils.normalization import normalize_enum_name
from .calculators.calculator_factory import CalculatorFactory

logger = logging.getLogger(__name__)


class PositionSizingService:
    """
    ポジションサイジング計算サービス

    PositionSizingGeneの設定に基づいて、実際のポジションサイズを高速に計算します。
    """

    def __init__(self) -> None:
        """初期化"""
        self.logger = logging.getLogger(__name__)
        self._calculator_factory = CalculatorFactory()

    def _calculate_with_calculator(
        self,
        *,
        gene: Any,
        account_balance: float,
        current_price: float,
        market_data: dict[str, Any] | None,
        trade_history: list[dict[str, Any]] | None,
    ) -> tuple[str, dict[str, Any]]:
        """遺伝子の手法に対応する計算機を実行する"""
        method_val = normalize_enum_name(gene.method)
        calculator = self._calculator_factory.create_calculator(method_val)
        result = calculator.calculate(
            gene,
            account_balance,
            current_price,
            market_data=market_data or {},
            trade_history=trade_history,
        )
        return method_val, result

    def calculate_position_size_fast(
        self,
        gene: Any,
        account_balance: float,
        current_price: float,
        market_data: dict[str, Any] | None = None,
    ) -> float:
        """
        高速ポジションサイズ計算（バックテスト用）

        詳細なリスクメトリクス計算（VaR、Expected Shortfall等）をスキップし、
        純粋なポジションサイズのみを高速に計算します。
        バックテストループ内での使用に最適化されています。

        Args:
            gene: ポジションサイジング遺伝子
            account_balance: 口座残高
            current_price: 現在価格
            market_data: 市場データ（ATRなど、事前計算済みの値を渡す）

        Returns:
            ポジションサイズ（数量）

        Note:
            エラー時は0.0を返します。呼び出し元は戻り値が0.0の場合、
            計算エラーまたは無効な入力として扱う必要があります。
        """
        try:
            # 入力値の簡易検証（フルバリデーションをスキップ）
            if not gene or account_balance <= 0 or current_price <= 0:
                return 0.0  # エラー時は0を返す

            # 計算機の選択と実行（市場データなしで高速実行）
            _, result = self._calculate_with_calculator(
                gene=gene,
                account_balance=account_balance,
                current_price=current_price,
                market_data=market_data,
                trade_history=None,
            )

            return float(result["position_size"])

        except Exception as e:
            self.logger.warning(f"高速ポジションサイズ計算エラー: {e}")
            return 0.0  # エラー時は0を返す
