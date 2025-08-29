"""
統合TPSLジェネレーター
3つのTPSL生成器を統一したインターフェースを提供します。
"""

import logging
from typing import Any, Dict, Optional

from ..models.strategy_models import TPSLMethod, TPSLResult

logger = logging.getLogger(__name__)


class TPSLStrategy:
    """TP/SL戦略の基底クラス"""

    def generate(self, **kwargs) -> TPSLResult:
        """TP/SLを生成"""
        raise NotImplementedError


class RiskRewardStrategy(TPSLStrategy):
    """リスクリワード比ベース戦略"""

    def generate(self, **kwargs) -> TPSLResult:
        # 簡単な実装 - 後で詳細化
        stop_loss_pct = kwargs.get("stop_loss_pct", 0.03)
        target_ratio = kwargs.get("target_ratio", 2.0)

        take_profit_pct = stop_loss_pct * target_ratio

        return TPSLResult(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            method_used="risk_reward",
            confidence_score=0.8,
            expected_performance={"risk_reward_ratio": target_ratio},
        )


class StatisticalStrategy(TPSLStrategy):
    """統計的優位性ベース戦略"""

    def generate(self, **kwargs) -> TPSLResult:
        # 簡単な実装 - 後で詳細化
        return TPSLResult(
            stop_loss_pct=0.03,
            take_profit_pct=0.06,
            method_used="statistical",
            confidence_score=0.7,
            expected_performance={"sharpe_ratio": 0.5},
        )


class VolatilityStrategy(TPSLStrategy):
    """ボラティリティベース戦略"""

    def generate(self, **kwargs) -> TPSLResult:
        # 簡単な実装 - 後で詳細化
        base_atr_pct = kwargs.get("base_atr_pct", 0.02)

        return TPSLResult(
            stop_loss_pct=base_atr_pct * 1.5,
            take_profit_pct=base_atr_pct * 3.0,
            method_used="volatility",
            confidence_score=0.9,
            expected_performance={"volatility_adjustment": "applied"},
        )


class AdaptiveStrategy(TPSLStrategy):
    """適応的戦略（市場条件に基づいて最適な戦略を選択）"""

    def generate(self, **kwargs) -> TPSLResult:
        # 市場条件に基づいて適切な戦略を選択

        tpsl_gene = kwargs.get("tpsl_gene")
        if tpsl_gene and hasattr(tpsl_gene, "method"):
            # TPSLGeneのmethodに基づいて選択
            method = tpsl_gene.method
            if method.name == "VOLATILITY_BASED":
                return VolatilityStrategy().generate(**kwargs)
            elif method.name == "RISK_REWARD_RATIO":
                return RiskRewardStrategy().generate(**kwargs)
            elif method.name == "STATISTICAL":
                return StatisticalStrategy().generate(**kwargs)
            else:
                return FixedPercentageStrategy().generate(**kwargs)
        else:
            # デフォルトは固定パーセンテージ
            return FixedPercentageStrategy().generate(**kwargs)


class FixedPercentageStrategy(TPSLStrategy):
    """固定パーセンテージ戦略"""

    def generate(self, **kwargs) -> TPSLResult:
        stop_loss_pct = kwargs.get("stop_loss_pct", 0.03)
        take_profit_pct = kwargs.get("take_profit_pct", 0.06)

        return TPSLResult(
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            method_used="fixed_percentage",
            confidence_score=0.95,
            expected_performance={"type": "fixed"},
        )


class UnifiedTPSLGenerator:
    """
    統合TP/SL生成器

    複数の計算方式を統一的なインターフェースで提供します。
    """

    def __init__(self):
        """初期化"""
        # AdaptiveStrategyを事前に作成してから辞書に追加
        adaptive_strategy = self._create_adaptive_strategy()

        self.strategies = {
            TPSLMethod.RISK_REWARD_RATIO: RiskRewardStrategy(),
            TPSLMethod.STATISTICAL: StatisticalStrategy(),
            TPSLMethod.VOLATILITY_BASED: VolatilityStrategy(),
            TPSLMethod.FIXED_PERCENTAGE: FixedPercentageStrategy(),
            TPSLMethod.ADAPTIVE: adaptive_strategy,
        }

    def _create_adaptive_strategy(self) -> TPSLStrategy:
        """適応的戦略を作成"""
        return FixedPercentageStrategy()  # 適応的には固定パーセンテージをデフォルトに

    def generate_tpsl(self, method: str, **kwargs) -> TPSLResult:
        """
        指定された手法でTP/SLを生成

        Args:
            method: 生成手法（"risk_reward", "statistical", "volatility", "fixed_percentage"）
            **kwargs: 手法固有のパラメータ

        Returns:
            TPSLResult: 生成されたTP/SL結果

        Raises:
            ValueError: 未知の手法が指定された場合
        """
        if method in ["risk_reward_ratio", "risk_reward"]:
            return self.strategies[TPSLMethod.RISK_REWARD_RATIO].generate(**kwargs)
        elif method == "statistical":
            return self.strategies[TPSLMethod.STATISTICAL].generate(**kwargs)
        elif method in ["volatility_based", "volatility"]:
            return self.strategies[TPSLMethod.VOLATILITY_BASED].generate(**kwargs)
        elif method in ["fixed_percentage", "fixed"]:
            return self.strategies[TPSLMethod.FIXED_PERCENTAGE].generate(**kwargs)
        elif method == "adaptive":
            return self.strategies[TPSLMethod.ADAPTIVE].generate(**kwargs)
        else:
            raise ValueError(f"Unknown TPSL method: {method}")

    def generate_adaptive_tpsl(
        self, market_conditions: Dict[str, Any], **kwargs
    ) -> TPSLResult:
        """
        市場条件に基づいて自動的に最適な手法を選択してTP/SLを生成

        Args:
            market_conditions: 市場条件データ
            **kwargs: 追加パラメータ

        Returns:
            TPSLResult: 生成されたTP/SL結果
        """
        # 簡易的な手法選択ロジック
        volatility = market_conditions.get("volatility", "normal")
        trend = market_conditions.get("trend", "neutral")

        if volatility == "high":
            method = "volatility"
        elif trend in ["strong_up", "strong_down"]:
            method = "risk_reward"
        elif market_conditions.get("historical_data_available", False):
            method = "statistical"
        else:
            method = "fixed_percentage"

        return self.generate_tpsl(method, **kwargs)
