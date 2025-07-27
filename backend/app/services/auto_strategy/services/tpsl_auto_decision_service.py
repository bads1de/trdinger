"""
TP/SL自動決定サービス

このサービスは、複数の戦略を使用してテイクプロフィット（TP）と
ストップロス（SL）を自動的に決定する機能を提供します。
"""

import logging
import random
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class TPSLStrategy(Enum):
    """TP/SL決定戦略の種類"""

    RANDOM = "random"
    RISK_REWARD = "risk_reward"
    VOLATILITY_ADAPTIVE = "volatility_adaptive"
    STATISTICAL = "statistical"
    AUTO_OPTIMAL = "auto_optimal"


@dataclass
class TPSLConfig:
    """TP/SL自動決定の設定"""

    strategy: TPSLStrategy
    max_risk_per_trade: float = 0.03  # 1取引あたりの最大リスク（3%）
    preferred_risk_reward_ratio: float = 2.0  # 希望するリスクリワード比（1:2）
    volatility_sensitivity: str = "medium"  # low, medium, high
    min_stop_loss: float = 0.005  # 最小SL（0.5%）
    max_stop_loss: float = 0.1  # 最大SL（10%）
    min_take_profit: float = 0.01  # 最小TP（1%）
    max_take_profit: float = 0.2  # 最大TP（20%）


@dataclass
class TPSLResult:
    """TP/SL決定結果"""

    stop_loss_pct: float
    take_profit_pct: float
    risk_reward_ratio: float
    strategy_used: str
    confidence_score: float = 1.0  # 決定の信頼度（0.0-1.0）
    metadata: Dict[str, Any] = field(default_factory=dict)


class TPSLAutoDecisionService:
    """
    TP/SL自動決定サービス

    複数の戦略を使用してテイクプロフィットとストップロスを
    自動的に決定する機能を提供します。
    """

    def __init__(self):
        """サービスを初期化"""
        # self.logger = logging.getLogger(__name__)

        # ボラティリティ感度設定
        self.volatility_multipliers = {
            "low": {"sl": 1.0, "tp": 1.0},
            "medium": {"sl": 1.5, "tp": 1.5},
            "high": {"sl": 2.0, "tp": 2.0},
        }

        # 統計的データ（将来的にはデータベースから取得）
        self.statistical_data = {
            "optimal_rr_ratios": [1.5, 2.0, 2.5, 3.0],
            "optimal_sl_ranges": [0.02, 0.03, 0.04, 0.05],
            "win_rates": {"conservative": 0.65, "balanced": 0.55, "aggressive": 0.45},
        }

    def generate_tpsl_values(
        self,
        config: TPSLConfig,
        market_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ) -> TPSLResult:
        """
        指定された戦略に基づいてTP/SL値を生成

        Args:
            config: TP/SL決定設定
            market_data: 市場データ（ATR値など）
            symbol: 取引シンボル

        Returns:
            TP/SL決定結果
        """
        try:
            # self.logger.info(f"TP/SL自動決定開始: 戦略={config.strategy.value}")

            if config.strategy == TPSLStrategy.RANDOM:
                return self._generate_random_tpsl(config)
            elif config.strategy == TPSLStrategy.RISK_REWARD:
                return self._generate_risk_reward_tpsl(config)
            elif config.strategy == TPSLStrategy.VOLATILITY_ADAPTIVE:
                return self._generate_volatility_adaptive_tpsl(config, market_data)
            elif config.strategy == TPSLStrategy.STATISTICAL:
                return self._generate_statistical_tpsl(config, symbol)
            elif config.strategy == TPSLStrategy.AUTO_OPTIMAL:
                return self._generate_auto_optimal_tpsl(config, market_data, symbol)
            else:
                raise ValueError(f"未サポートの戦略: {config.strategy}")

        except Exception as e:
            logger.error(f"TP/SL自動決定エラー: {e}", exc_info=True)
            # フォールバック: デフォルト値を返す
            return self._generate_fallback_tpsl(config)

    def _generate_random_tpsl(self, config: TPSLConfig) -> TPSLResult:
        """ランダム戦略でTP/SLを生成"""
        # SLをランダムに決定
        sl_pct = random.uniform(config.min_stop_loss, config.max_stop_loss)

        # TPをランダムに決定（ただし、最小限のRR比は確保）
        min_tp = sl_pct * 1.2  # 最低1.2倍のリターン
        max_tp = min(config.max_take_profit, sl_pct * 5.0)  # 最大5倍のリターン
        tp_pct = random.uniform(min_tp, max_tp)

        rr_ratio = tp_pct / sl_pct

        return TPSLResult(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_reward_ratio=rr_ratio,
            strategy_used="random",
            confidence_score=0.5,
            metadata={"method": "uniform_random"},
        )

    def _generate_risk_reward_tpsl(self, config: TPSLConfig) -> TPSLResult:
        """リスクリワード比ベース戦略でTP/SLを生成"""
        # 最大リスクに基づいてSLを決定
        sl_pct = min(config.max_risk_per_trade, config.max_stop_loss)

        # 希望するリスクリワード比でTPを計算
        tp_pct = sl_pct * config.preferred_risk_reward_ratio

        # TP上限チェック
        if tp_pct > config.max_take_profit:
            tp_pct = config.max_take_profit
            # TPが制限された場合、実際のRR比を再計算
            actual_rr_ratio = tp_pct / sl_pct
        else:
            actual_rr_ratio = config.preferred_risk_reward_ratio

        return TPSLResult(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_reward_ratio=actual_rr_ratio,
            strategy_used="risk_reward",
            confidence_score=0.8,
            metadata={
                "target_rr_ratio": config.preferred_risk_reward_ratio,
                "actual_rr_ratio": actual_rr_ratio,
            },
        )

    def _generate_volatility_adaptive_tpsl(
        self, config: TPSLConfig, market_data: Optional[Dict[str, Any]]
    ) -> TPSLResult:
        """ボラティリティ適応戦略でTP/SLを生成"""
        # デフォルトのボラティリティ倍率を取得
        multiplier = self.volatility_multipliers.get(
            config.volatility_sensitivity, self.volatility_multipliers["medium"]
        )

        # 基本SL（最大リスクの80%）
        base_sl = config.max_risk_per_trade * 0.8

        # ボラティリティに基づいてSLを調整
        if market_data and "atr_pct" in market_data:
            # ATRが利用可能な場合
            atr_pct = market_data["atr_pct"]
            sl_pct = min(atr_pct * multiplier["sl"], config.max_stop_loss)
        else:
            # ATRが利用できない場合はデフォルト値を使用
            sl_pct = base_sl * multiplier["sl"]

        # SL範囲チェック
        sl_pct = max(config.min_stop_loss, min(sl_pct, config.max_stop_loss))

        # TPをリスクリワード比で計算
        tp_pct = sl_pct * config.preferred_risk_reward_ratio * multiplier["tp"]
        tp_pct = min(tp_pct, config.max_take_profit)

        rr_ratio = tp_pct / sl_pct

        return TPSLResult(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_reward_ratio=rr_ratio,
            strategy_used="volatility_adaptive",
            confidence_score=0.7,
            metadata={
                "volatility_sensitivity": config.volatility_sensitivity,
                "atr_available": market_data is not None and "atr_pct" in market_data,
                "multiplier_used": multiplier,
            },
        )

    def _generate_statistical_tpsl(
        self, config: TPSLConfig, symbol: Optional[str]
    ) -> TPSLResult:
        """統計的優位性ベース戦略でTP/SLを生成"""
        # 統計データから最適値を選択（現在はサンプルデータ）
        optimal_sl = random.choice(self.statistical_data["optimal_sl_ranges"])
        optimal_rr = random.choice(self.statistical_data["optimal_rr_ratios"])

        # 設定範囲内に調整
        sl_pct = max(config.min_stop_loss, min(optimal_sl, config.max_stop_loss))
        tp_pct = sl_pct * optimal_rr
        tp_pct = min(tp_pct, config.max_take_profit)

        rr_ratio = tp_pct / sl_pct

        return TPSLResult(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_reward_ratio=rr_ratio,
            strategy_used="statistical",
            confidence_score=0.9,
            metadata={
                "symbol": symbol,
                "statistical_sl": optimal_sl,
                "statistical_rr": optimal_rr,
            },
        )

    def _generate_auto_optimal_tpsl(
        self,
        config: TPSLConfig,
        market_data: Optional[Dict[str, Any]],
        symbol: Optional[str],
    ) -> TPSLResult:
        """自動最適化戦略でTP/SLを生成"""
        # 複数の戦略を試して最適なものを選択
        strategies = [
            TPSLStrategy.RISK_REWARD,
            TPSLStrategy.VOLATILITY_ADAPTIVE,
            TPSLStrategy.STATISTICAL,
        ]

        results = []
        for strategy in strategies:
            temp_config = TPSLConfig(
                strategy=strategy,
                max_risk_per_trade=config.max_risk_per_trade,
                preferred_risk_reward_ratio=config.preferred_risk_reward_ratio,
                volatility_sensitivity=config.volatility_sensitivity,
            )
            result = self.generate_tpsl_values(temp_config, market_data, symbol)
            results.append(result)

        # 最も信頼度の高い結果を選択
        best_result = max(results, key=lambda r: r.confidence_score)
        best_result.strategy_used = "auto_optimal"
        best_result.metadata = {
            "selected_from": [r.strategy_used for r in results],
            "confidence_scores": [r.confidence_score for r in results],
            "original_metadata": best_result.metadata,
        }

        return best_result

    def _generate_fallback_tpsl(self, config: TPSLConfig) -> TPSLResult:
        """フォールバック用のデフォルトTP/SL値を生成"""
        sl_pct = config.max_risk_per_trade
        tp_pct = sl_pct * 2.0  # デフォルト1:2のリスクリワード比

        return TPSLResult(
            stop_loss_pct=sl_pct,
            take_profit_pct=tp_pct,
            risk_reward_ratio=2.0,
            strategy_used="fallback",
            confidence_score=0.3,
            metadata={"reason": "fallback_due_to_error"},
        )

    def validate_tpsl_values(self, result: TPSLResult, config: TPSLConfig) -> bool:
        """TP/SL値の妥当性を検証"""
        try:
            # 基本的な範囲チェック
            if not (
                config.min_stop_loss <= result.stop_loss_pct <= config.max_stop_loss
            ):
                return False

            if not (
                config.min_take_profit
                <= result.take_profit_pct
                <= config.max_take_profit
            ):
                return False

            # リスクリワード比の妥当性チェック
            if result.risk_reward_ratio < 1.0:  # 最低1:1は確保
                return False

            # 信頼度スコアの範囲チェック
            if not (0.0 <= result.confidence_score <= 1.0):
                return False

            return True

        except Exception as e:
            logger.error(f"TP/SL値検証エラー: {e}")
            return False
