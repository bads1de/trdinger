"""
リスクリワード比ベースのTP/SL生成機能

TP/SLのうち、SLが与えられた前提で、指定したリスクリワード比に基づいて
テイクプロフィット（TP）を決定するジェネレーター。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..config.constants import GA_DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class RiskRewardProfile(Enum):
    """リスクリワードプロファイルの種類"""

    CONSERVATIVE = "conservative"  # 1:1.5 - 1:2
    BALANCED = "balanced"  # 1:2 - 1:3
    AGGRESSIVE = "aggressive"  # 1:3 - 1:5


@dataclass
class RiskRewardConfig:
    """リスクリワード計算の設定"""

    target_ratio: float = GA_DEFAULT_CONFIG.get("max_indicators", 3) * 1.0  # max_indicators * 1.0 を目標比率に
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


class RiskRewardTPSLGenerator:
    """
    リスクリワード比ベースのTP/SL生成ジェネレーター

    与えられたSL割合と設定からTP割合を導出します。
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

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

    def generate_risk_reward_tpsl(
        self, stop_loss_pct: float, config: RiskRewardConfig
    ) -> RiskRewardResult:
        """リスクリワードに基づいてTP割合を生成"""
        try:
            self.logger.info(
                f"RR TP生成開始: SL={stop_loss_pct:.3f}, 目標={config.target_ratio}"
            )

            basic_tp = stop_loss_pct * config.target_ratio
            adjusted_tp, reason = self._apply_tp_limits(basic_tp, config)
            actual_ratio = adjusted_tp / stop_loss_pct if stop_loss_pct > 0 else 0.0

            partial_tp_levels = None
            if config.allow_partial_tp:
                partial_tp_levels = self._calculate_partial_tp_levels(
                    stop_loss_pct, adjusted_tp, config
                )

            return RiskRewardResult(
                take_profit_pct=adjusted_tp,
                actual_risk_reward_ratio=actual_ratio,
                target_ratio=config.target_ratio,
                is_ratio_achieved=abs(actual_ratio - config.target_ratio) < 0.1,
                adjustment_reason=reason,
                partial_tp_levels=partial_tp_levels,
                metadata={
                    "original_tp": basic_tp,
                    "profile": config.profile.value,
                    "sl_input": stop_loss_pct,
                },
            )

        except Exception as e:
            logger.error(f"RR TP生成エラー: {e}", exc_info=True)
            # シンプルなフォールバック
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

    def get_recommended_ratio_for_profile(self, profile: RiskRewardProfile) -> float:
        settings = self.profile_settings.get(
            profile, self.profile_settings[RiskRewardProfile.BALANCED]
        )
        return settings["default_ratio"]

    def _apply_tp_limits(
        self, tp_pct: float, config: RiskRewardConfig
    ) -> tuple[float, Optional[str]]:
        reason = None
        if tp_pct > config.max_tp_limit:
            tp_pct = config.max_tp_limit
            reason = f"TP上限制限適用: {config.max_tp_limit:.1%}"
        elif tp_pct < config.min_tp_limit:
            tp_pct = config.min_tp_limit
            reason = f"TP下限制限適用: {config.min_tp_limit:.1%}"
        return tp_pct, reason

    def _calculate_partial_tp_levels(
        self, stop_loss_pct: float, final_tp_pct: float, config: RiskRewardConfig
    ) -> List[float]:
        try:
            partial_ratios = self.profile_settings[config.profile]["partial_tp_ratios"]
            levels: List[float] = []
            for ratio in partial_ratios:
                level = stop_loss_pct * ratio
                if level <= final_tp_pct:
                    levels.append(level)
            return levels
        except Exception:
            return []

