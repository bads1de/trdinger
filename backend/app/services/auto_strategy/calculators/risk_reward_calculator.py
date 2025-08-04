"""
リスクリワード比ベースのTP計算機能

このモジュールは、ストップロス（SL）が決定された後に、
指定されたリスクリワード比に基づいてテイクプロフィット（TP）を
自動計算する機能を提供します。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RiskRewardProfile(Enum):
    """リスクリワードプロファイルの種類"""

    CONSERVATIVE = "conservative"  # 1:1.5 - 1:2
    BALANCED = "balanced"  # 1:2 - 1:3
    AGGRESSIVE = "aggressive"  # 1:3 - 1:5


@dataclass
class RiskRewardConfig:
    """リスクリワード計算の設定"""

    target_ratio: float = 2.0  # 目標リスクリワード比
    min_ratio: float = 1.0  # 最小リスクリワード比
    max_ratio: float = 5.0  # 最大リスクリワード比
    profile: RiskRewardProfile = RiskRewardProfile.BALANCED
    allow_partial_tp: bool = True  # 部分利確を許可するか
    max_tp_limit: float = 0.2  # TP上限（20%）
    min_tp_limit: float = 0.01  # TP下限（1%）


@dataclass
class RiskRewardResult:
    """リスクリワード計算結果"""

    take_profit_pct: float
    actual_risk_reward_ratio: float
    target_ratio: float
    is_ratio_achieved: bool
    adjustment_reason: Optional[str] = None
    partial_tp_levels: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class RiskRewardCalculator:
    """
    リスクリワード比ベースのTP計算機能

    ストップロスが決定された後に、指定されたリスクリワード比に基づいて
    テイクプロフィットを自動計算します。
    """

    def __init__(self):
        """計算機を初期化"""
        self.logger = logging.getLogger(__name__)

        # プロファイル別のデフォルト設定
        self.profile_settings = {
            RiskRewardProfile.CONSERVATIVE: {
                "default_ratio": 1.5,
                "ratio_range": (1.2, 2.0),
                "partial_tp_ratios": [0.5, 1.0, 1.5],
            },
            RiskRewardProfile.BALANCED: {
                "default_ratio": 2.0,
                "ratio_range": (1.5, 3.0),
                "partial_tp_ratios": [1.0, 2.0, 3.0],
            },
            RiskRewardProfile.AGGRESSIVE: {
                "default_ratio": 3.0,
                "ratio_range": (2.0, 5.0),
                "partial_tp_ratios": [1.5, 3.0, 4.5],
            },
        }

    def calculate_take_profit(
        self, stop_loss_pct: float, config: RiskRewardConfig
    ) -> RiskRewardResult:
        """
        リスクリワード比に基づいてテイクプロフィットを計算

        Args:
            stop_loss_pct: ストップロス割合（例: 0.03 = 3%）
            config: リスクリワード計算設定

        Returns:
            リスクリワード計算結果
        """
        try:
            self.logger.info(
                f"リスクリワード計算開始: SL={stop_loss_pct:.3f}, "
                f"目標比率={config.target_ratio}"
            )

            # 基本的なTP計算
            basic_tp = stop_loss_pct * config.target_ratio

            # TP制限チェックと調整
            adjusted_tp, adjustment_reason = self._apply_tp_limits(basic_tp, config)

            # 実際のリスクリワード比を計算
            actual_ratio = adjusted_tp / stop_loss_pct

            # 目標比率が達成されたかチェック
            is_ratio_achieved = abs(actual_ratio - config.target_ratio) < 0.1

            # 部分利確レベルの計算
            partial_tp_levels = None
            if config.allow_partial_tp:
                partial_tp_levels = self._calculate_partial_tp_levels(
                    stop_loss_pct, adjusted_tp, config
                )

            result = RiskRewardResult(
                take_profit_pct=adjusted_tp,
                actual_risk_reward_ratio=actual_ratio,
                target_ratio=config.target_ratio,
                is_ratio_achieved=is_ratio_achieved,
                adjustment_reason=adjustment_reason,
                partial_tp_levels=partial_tp_levels,
                metadata={
                    "original_tp": basic_tp,
                    "profile": config.profile.value,
                    "sl_input": stop_loss_pct,
                },
            )

            self.logger.info(
                f"リスクリワード計算完了: TP={adjusted_tp:.3f}, "
                f"実際比率={actual_ratio:.2f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"リスクリワード計算エラー: {e}", exc_info=True)
            # フォールバック計算
            return self._calculate_fallback_tp(stop_loss_pct, config)

    def calculate_optimal_ratio(
        self,
        stop_loss_pct: float,
        market_conditions: Optional[Dict[str, Any]] = None,
        historical_performance: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        市場条件と過去のパフォーマンスに基づいて最適なリスクリワード比を計算

        Args:
            stop_loss_pct: ストップロス割合
            market_conditions: 市場条件（ボラティリティなど）
            historical_performance: 過去のパフォーマンスデータ

        Returns:
            最適なリスクリワード比
        """
        try:
            # デフォルト比率
            base_ratio = 2.0

            # 市場条件による調整
            if market_conditions:
                volatility = market_conditions.get("volatility", "medium")
                trend_strength = market_conditions.get("trend_strength", 0.5)

                # ボラティリティによる調整
                if volatility == "high":
                    base_ratio *= 1.2  # 高ボラティリティ時は高いリターンを狙う
                elif volatility == "low":
                    base_ratio *= 0.8  # 低ボラティリティ時は控えめに

                # トレンド強度による調整
                base_ratio *= 0.8 + trend_strength * 0.4

            # 過去のパフォーマンスによる調整
            if historical_performance:
                win_rate = historical_performance.get("win_rate", 0.5)
                avg_return = historical_performance.get("avg_return", 0.02)

                # 勝率が高い場合は控えめに、低い場合は積極的に
                if win_rate > 0.6:
                    base_ratio *= 0.9
                elif win_rate < 0.4:
                    base_ratio *= 1.1

                # 平均リターンによる調整
                # 平均リターンが高い場合は、より高いリスクリワード比を推奨
                if avg_return > 0.03:  # 例: 3%以上の平均リターン
                    base_ratio *= 1.05
                elif avg_return < 0.01:  # 例: 1%未満の平均リターン
                    base_ratio *= 0.95

            # 範囲制限
            optimal_ratio = max(1.0, min(5.0, base_ratio))

            self.logger.info(f"最適リスクリワード比計算: {optimal_ratio:.2f}")
            return optimal_ratio

        except Exception as e:
            self.logger.error(f"最適比率計算エラー: {e}")
            return 2.0  # デフォルト値

    def calculate_multiple_tp_levels(
        self, stop_loss_pct: float, ratios: List[float]
    ) -> List[Tuple[float, float]]:
        """
        複数のリスクリワード比に対応するTP レベルを計算

        Args:
            stop_loss_pct: ストップロス割合
            ratios: リスクリワード比のリスト

        Returns:
            (TP割合, リスクリワード比)のタプルのリスト
        """
        try:
            tp_levels = []
            for ratio in ratios:
                tp_pct = stop_loss_pct * ratio
                tp_levels.append((tp_pct, ratio))

            # TP割合でソート
            tp_levels.sort(key=lambda x: x[0])

            self.logger.info(f"複数TPレベル計算: {len(tp_levels)}レベル")
            return tp_levels

        except Exception as e:
            self.logger.error(f"複数TPレベル計算エラー: {e}")
            return [(stop_loss_pct * 2.0, 2.0)]  # デフォルト

    def _apply_tp_limits(
        self, tp_pct: float, config: RiskRewardConfig
    ) -> Tuple[float, Optional[str]]:
        """TP制限を適用して調整"""
        adjustment_reason = None

        if tp_pct > config.max_tp_limit:
            tp_pct = config.max_tp_limit
            adjustment_reason = f"TP上限制限適用: {config.max_tp_limit:.1%}"
        elif tp_pct < config.min_tp_limit:
            tp_pct = config.min_tp_limit
            adjustment_reason = f"TP下限制限適用: {config.min_tp_limit:.1%}"

        return tp_pct, adjustment_reason

    def _calculate_partial_tp_levels(
        self, stop_loss_pct: float, final_tp_pct: float, config: RiskRewardConfig
    ) -> List[float]:
        """部分利確レベルを計算"""
        try:
            profile_settings = self.profile_settings.get(
                config.profile, self.profile_settings[RiskRewardProfile.BALANCED]
            )

            partial_ratios = profile_settings["partial_tp_ratios"]
            partial_levels = []

            for ratio in partial_ratios:
                tp_level = stop_loss_pct * ratio
                if tp_level <= final_tp_pct:
                    partial_levels.append(tp_level)

            return partial_levels

        except Exception as e:
            self.logger.error(f"部分利確レベル計算エラー: {e}")
            return []

    def _calculate_fallback_tp(
        self, stop_loss_pct: float, config: RiskRewardConfig
    ) -> RiskRewardResult:
        """フォールバック用のTP計算"""
        fallback_ratio = 2.0
        tp_pct = stop_loss_pct * fallback_ratio

        return RiskRewardResult(
            take_profit_pct=tp_pct,
            actual_risk_reward_ratio=fallback_ratio,
            target_ratio=config.target_ratio,
            is_ratio_achieved=False,
            adjustment_reason="フォールバック計算",
            metadata={"fallback": True},
        )

    def validate_risk_reward_ratio(self, ratio: float) -> bool:
        """リスクリワード比の妥当性を検証"""
        return 0.5 <= ratio <= 10.0  # 合理的な範囲

    def get_recommended_ratio_for_profile(self, profile: RiskRewardProfile) -> float:
        """プロファイルに基づく推奨リスクリワード比を取得"""
        settings = self.profile_settings.get(
            profile, self.profile_settings[RiskRewardProfile.BALANCED]
        )
        return settings["default_ratio"]
