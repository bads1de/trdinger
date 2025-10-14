"""
ベースポジションサイジング計算クラス

共通の計算ロジックとユーティリティを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable

logger = logging.getLogger(__name__)


class BaseCalculator(ABC):
    """ベースポジションサイジング計算クラス"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abstractmethod
    def calculate(
        self, gene, account_balance: float, current_price: float, **kwargs
    ) -> Dict[str, Any]:
        """計算実行（サブクラスで実装）"""
        pass

    def _apply_size_limits_and_finalize(
        self, position_size: float, details: Dict[str, Any], warnings: List[str], gene
    ) -> Dict[str, Any]:
        """統一された最終処理（共通重複コード除去）"""
        position_size = max(gene.min_position_size, position_size)
        details["final_position_size"] = position_size

        return {
            "position_size": position_size,
            "details": details,
            "warnings": warnings,
        }

    def _safe_calculate_with_price_check(
        self,
        calculator_fn: Callable[[], float],
        current_price: float,
        fallback_value: float = 0,
        warning_msg: str = "現在価格が無効",
        warnings_list: Optional[List[str]] = None,
    ) -> float:
        """価格チェック共通処理（共通重複コード除去）"""
        if current_price > 0:
            return calculator_fn()
        else:
            if warnings_list is not None:
                warnings_list.append(warning_msg)
            return fallback_value

    def _create_calculation_result(
        self, position_size: float, details: Dict[str, Any], warnings: List[str], gene
    ) -> Dict[str, Any]:
        """計算結果の統一作成（結果構造重複除去）"""
        return self._apply_size_limits_and_finalize(
            position_size, details, warnings, gene
        )
