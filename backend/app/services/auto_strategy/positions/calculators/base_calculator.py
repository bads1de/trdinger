"""
ベースポジションサイジング計算クラス

共通の計算ロジックとユーティリティを提供します。
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

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

    def _get_param(self, gene, attr_name: str, default: Any) -> Any:
        """遺伝子から安全にパラメータを取得"""
        return getattr(gene, attr_name, default)

    def _get_risk_params(self, gene) -> Dict[str, Any]:
        """共通のリスク管理パラメータを一括取得"""
        return {
            "var_confidence": self._get_param(gene, "var_confidence", 0.95),
            "var_lookback": self._get_param(gene, "var_lookback", 100),
            "max_var_ratio": self._get_param(gene, "max_var_ratio", 0.0),
            "max_es_ratio": self._get_param(gene, "max_expected_shortfall_ratio", 0.0),
        }

    def _apply_size_limits_and_finalize(
        self, position_size: float, details: Dict[str, Any], warnings: List[str], gene
    ) -> Dict[str, Any]:
        """
        計算されたサイズに対して最小・最大制限を適用し、最終結果を構築

        Args:
            position_size: 計算された生ポジションサイズ
            details: 計算の詳細情報
            warnings: 警告メッセージのリスト
            gene: ポジションサイジング遺伝子

        Returns:
            正規化された最終出力辞書
        """
        # 最小/最大サイズ制限の適用
        min_size = self._get_param(gene, "min_position_size", 0.001)
        max_size = self._get_param(gene, "max_position_size", 9999.0)

        position_size = max(min_size, min(max_size, position_size))
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
        """
        現在価格の妥当性をチェックしてから計算を実行

        価格が0以下の場合は計算をスキップし、フォールバック値を返します。

        Args:
            calculator_fn: 実行する計算関数
            current_price: 現在の市場価格
            fallback_value: 価格無効時の戻り値
            warning_msg: 価格無効時の警告メッセージ
            warnings_list: 警告を追記するリスト

        Returns:
            計算結果またはフォールバック値
        """
        if current_price > 0:
            return calculator_fn()
        else:
            if warnings_list is not None:
                warnings_list.append(warning_msg)
            return fallback_value

    def _create_calculation_result(
        self, position_size: float, details: Dict[str, Any], warnings: List[str], gene
    ) -> Dict[str, Any]:
        """計算結果の統一作成"""
        return self._apply_size_limits_and_finalize(
            position_size, details, warnings, gene
        )
