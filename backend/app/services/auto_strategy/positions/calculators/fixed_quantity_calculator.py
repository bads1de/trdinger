"""
固定枚数方式計算クラス
"""

from typing import Any, Dict

from .base_calculator import BaseCalculator


class FixedQuantityCalculator(BaseCalculator):
    """固定枚数方式計算クラス"""

    def calculate(
        self, gene, account_balance: float, current_price: float, **kwargs
    ) -> Dict[str, Any]:
        """固定枚数方式の拡張計算"""
        details: Dict[str, Any] = {"method": "fixed_quantity"}

        # ポジションサイズの計算
        position_size = gene.fixed_quantity

        # 詳細情報の更新
        details.update({"fixed_quantity": gene.fixed_quantity})

        # 統一された最終処理（重複コード除去）
        return self._apply_size_limits_and_finalize(position_size, details, [], gene)
