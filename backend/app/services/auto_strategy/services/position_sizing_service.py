"""
統一ポジションサイジングサービス

ポジションサイジング計算ロジックを一元化し、異なる計算方式を統一的なインターフェースで提供します。
"""

import logging
from typing import Any, Dict, List, Optional

from ..calculators.position_sizing_calculator import (
    PositionSizingCalculatorService,
    PositionSizingResult,
)
from ..models.gene_position_sizing import PositionSizingGene, PositionSizingMethod

logger = logging.getLogger(__name__)


class PositionSizingService:
    """
    統一ポジションサイジングサービス

    異なるポジションサイジング計算方式を統一的なインターフェースで提供し、
    計算ロジックの一元化を実現します。
    """

    def __init__(self):
        """サービスを初期化"""
        self.position_sizing_calculator = PositionSizingCalculatorService()

    def calculate_position_size(
        self,
        position_sizing_gene: Optional[PositionSizingGene] = None,
        account_balance: float = 100000.0,
        current_price: float = 50000.0,
        symbol: str = "BTCUSDT",
        market_data: Optional[Dict[str, Any]] = None,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> PositionSizingResult:
        """
        統一的なポジションサイズ計算

        Args:
            position_sizing_gene: ポジションサイジング遺伝子（GA最適化対象）
            account_balance: 口座残高
            current_price: 現在価格
            symbol: 取引ペア
            market_data: 市場データ
            trade_history: 取引履歴
            use_cache: キャッシュを使用するか
            **kwargs: その他のパラメータ

        Returns:
            ポジションサイジング結果
        """
        try:
            # ポジションサイジング遺伝子が利用可能な場合（GA最適化対象）
            if position_sizing_gene and position_sizing_gene.enabled:
                return self._calculate_from_gene(
                    position_sizing_gene,
                    account_balance,
                    current_price,
                    symbol,
                    market_data,
                    trade_history,
                    use_cache,
                )

            # 従来方式の場合（デフォルト遺伝子を作成）
            default_gene = self._create_default_gene(**kwargs)
            return self._calculate_from_gene(
                default_gene,
                account_balance,
                current_price,
                symbol,
                market_data,
                trade_history,
                use_cache,
            )

        except Exception as e:
            logger.error(f"ポジションサイズ計算エラー: {e}")
            # フォールバック: 固定比率
            return self._calculate_fallback(account_balance, current_price)

    def _calculate_from_gene(
        self,
        position_sizing_gene: PositionSizingGene,
        account_balance: float,
        current_price: float,
        symbol: str,
        market_data: Optional[Dict[str, Any]],
        trade_history: Optional[List[Dict[str, Any]]],
        use_cache: bool,
    ) -> PositionSizingResult:
        """ポジションサイジング遺伝子からポジションサイズを計算"""
        try:
            # 既存のPositionSizingCalculatorを使用
            return self.position_sizing_calculator.calculate_position_size(
                gene=position_sizing_gene,
                account_balance=account_balance,
                current_price=current_price,
                symbol=symbol,
                market_data=market_data,
                trade_history=trade_history,
                use_cache=use_cache,
            )

        except Exception as e:
            logger.error(f"遺伝子ベースポジションサイズ計算エラー: {e}")
            return self._calculate_fallback(account_balance, current_price)

    def _create_default_gene(self, **kwargs) -> PositionSizingGene:
        """デフォルトのポジションサイジング遺伝子を作成"""
        # kwargsから設定を取得
        method_str = kwargs.get("method", "volatility_based")

        # 文字列からenumに変換
        method_mapping = {
            "half_optimal_f": PositionSizingMethod.HALF_OPTIMAL_F,
            "volatility_based": PositionSizingMethod.VOLATILITY_BASED,
            "fixed_ratio": PositionSizingMethod.FIXED_RATIO,
            "fixed_quantity": PositionSizingMethod.FIXED_QUANTITY,
        }

        method = method_mapping.get(method_str, PositionSizingMethod.VOLATILITY_BASED)

        return PositionSizingGene(
            method=method,
            lookback_period=kwargs.get("lookback_period", 100),
            optimal_f_multiplier=kwargs.get("optimal_f_multiplier", 0.5),
            atr_period=kwargs.get("atr_period", 14),
            atr_multiplier=kwargs.get("atr_multiplier", 2.0),
            risk_per_trade=kwargs.get("risk_per_trade", 0.02),
            fixed_ratio=kwargs.get("fixed_ratio", 0.1),
            fixed_quantity=kwargs.get("fixed_quantity", 1.0),
            enabled=True,
        )

    def _calculate_fallback(
        self,
        account_balance: float,
        current_price: float,
    ) -> PositionSizingResult:
        """フォールバック計算（固定比率）"""
        from datetime import datetime

        default_ratio = 0.1  # 10%
        position_amount = account_balance * default_ratio
        position_size = position_amount / current_price if current_price > 0 else 0

        return PositionSizingResult(
            position_size=position_size,
            method_used="fixed_ratio_fallback",
            calculation_details={
                "account_balance": account_balance,
                "current_price": current_price,
                "fixed_ratio": default_ratio,
                "position_amount": position_amount,
                "fallback": True,
            },
            confidence_score=0.5,
            risk_metrics={
                "position_value": position_amount,
                "account_exposure": default_ratio,
                "risk_level": "medium",
            },
            warnings=["フォールバック計算を使用"],
            timestamp=datetime.now(),
        )

    def calculate_position_size_simple(
        self,
        method: str = "volatility_based",
        account_balance: float = 100000.0,
        current_price: float = 50000.0,
        **kwargs,
    ) -> float:
        """
        簡易ポジションサイズ計算（後方互換性用）

        Args:
            method: 計算方式
            account_balance: 口座残高
            current_price: 現在価格
            **kwargs: その他のパラメータ

        Returns:
            ポジションサイズ（数量）
        """
        try:
            result = self.calculate_position_size(
                position_sizing_gene=None,
                account_balance=account_balance,
                current_price=current_price,
                method=method,
                **kwargs,
            )
            return result.position_size

        except Exception as e:
            logger.error(f"簡易ポジションサイズ計算エラー: {e}")
            # フォールバック: 固定比率
            default_ratio = kwargs.get("fixed_ratio", 0.1)
            position_amount = account_balance * default_ratio
            return position_amount / current_price if current_price > 0 else 0

    def validate_position_size(
        self,
        position_size: float,
        account_balance: float,
        current_price: float,
        max_position_ratio: float = 0.5,
    ) -> bool:
        """
        ポジションサイズの妥当性を検証

        Args:
            position_size: ポジションサイズ
            account_balance: 口座残高
            current_price: 現在価格
            max_position_ratio: 最大ポジション比率

        Returns:
            妥当性（True/False）
        """
        try:
            if position_size <= 0:
                return False

            position_value = position_size * current_price
            position_ratio = position_value / account_balance

            # 最大ポジション比率チェック
            if position_ratio > max_position_ratio:
                return False

            # 最小ポジションサイズチェック（0.001単位以上）
            if position_size < 0.001:
                return False

            return True

        except Exception as e:
            logger.error(f"ポジションサイズ検証エラー: {e}")
            return False

    def get_recommended_method(
        self,
        account_balance: float,
        trade_history: Optional[List[Dict[str, Any]]] = None,
        market_volatility: str = "medium",
    ) -> str:
        """
        推奨計算方式を取得

        Args:
            account_balance: 口座残高
            trade_history: 取引履歴
            market_volatility: 市場ボラティリティ

        Returns:
            推奨方式名
        """
        try:
            # 取引履歴が十分にある場合
            if trade_history and len(trade_history) >= 50:
                return "half_optimal_f"

            # 高ボラティリティ市場の場合
            if market_volatility == "high":
                return "volatility_based"

            # 小額口座の場合
            if account_balance < 10000:
                return "fixed_quantity"

            # デフォルト
            return "volatility_based"

        except Exception as e:
            logger.error(f"推奨方式取得エラー: {e}")
            return "volatility_based"
