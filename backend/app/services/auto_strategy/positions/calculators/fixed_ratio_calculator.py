"""
固定比率方式計算クラス
"""

from typing import Any, Dict

from .base_calculator import BaseCalculator


class FixedRatioCalculator(BaseCalculator):
    """固定比率方式計算クラス"""

    def calculate(
        self, gene, account_balance: float, current_price: float, **kwargs
    ) -> Dict[str, Any]:
        """固定比率方式の拡張計算"""
        details: Dict[str, Any] = {"method": "fixed_ratio"}

        # ポジションサイズの計算
        position_amount = account_balance * gene.fixed_ratio
        position_size = self._safe_calculate_with_price_check(
            lambda: position_amount / current_price,
            current_price,
            0,
            "現在価格が無効",
            None,
        )

        # 詳細情報の更新
        details.update(
            {
                "fixed_ratio": gene.fixed_ratio,
                "account_balance": account_balance,
                "calculated_amount": position_amount,
            }
        )

        # 統一された最終処理（重複コード除去）
        return self._apply_size_limits_and_finalize(position_size, details, [], gene)
