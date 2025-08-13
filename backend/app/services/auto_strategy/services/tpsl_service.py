"""
TP/SL計算サービス

TP/SL計算ロジックを一元化し、異なる計算方式を統一的なインターフェースで提供します。
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple, List
from enum import Enum
from dataclasses import dataclass, field
from ..generators.statistical_tpsl_generator import (
    StatisticalTPSLGenerator,
    StatisticalConfig,
)
from ..generators.volatility_based_generator import (
    VolatilityBasedGenerator,
    VolatilityConfig,
)
from ..models.gene_tpsl import TPSLGene, TPSLMethod

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


class TPSLCalculationMethod(Enum):
    """TP/SL計算方式"""

    FIXED_PERCENTAGE = "fixed_percentage"
    RISK_REWARD_RATIO = "risk_reward_ratio"
    VOLATILITY_BASED = "volatility_based"
    STATISTICAL = "statistical"


class TPSLResult:
    """TP/SL計算結果"""

    def __init__(
        self,
        stop_loss_pct: float,
        take_profit_pct: float,
        method: TPSLCalculationMethod,
        confidence_score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.method = method
        self.confidence_score = confidence_score
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return {
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "method": self.method.value,
            "confidence_score": self.confidence_score,
            "metadata": self.metadata,
        }


class TPSLService:
    """
    TP/SL計算サービス

    複数の計算方式を統一的なインターフェースで提供します。
    """

    def __init__(self):
        """初期化"""
        self.risk_reward_calculator = RiskRewardCalculator()
        self.statistical_generator = StatisticalTPSLGenerator()
        self.volatility_generator = VolatilityBasedGenerator()

    def calculate_tpsl_prices(
        self,
        current_price: float,
        tpsl_gene: Optional[TPSLGene] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        risk_management: Optional[Dict[str, Any]] = None,
        market_data: Optional[Dict[str, Any]] = None,
        position_direction: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        統一的なTP/SL価格計算

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子（GA最適化対象）
            stop_loss_pct: ストップロス割合（従来方式）
            take_profit_pct: テイクプロフィット割合（従来方式）
            risk_management: リスク管理設定
            market_data: 市場データ
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）

        Returns:
            (SL価格, TP価格)のタプル
        """
        try:
            # TP/SL遺伝子が利用可能な場合（GA最適化対象）
            if tpsl_gene and hasattr(tpsl_gene, "enabled") and tpsl_gene.enabled:
                return self._calculate_from_gene(
                    current_price, tpsl_gene, market_data, position_direction
                )

            # 従来方式の場合
            return self._calculate_basic_tpsl_prices(
                current_price=current_price,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                risk_management=risk_management or {},
                position_direction=position_direction,
            )

        except Exception as e:
            logger.error(f"TP/SL価格計算エラー: {e}")
            # フォールバック: 基本計算
            return self._calculate_fallback(current_price, position_direction)

    def _make_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """割合からSL/TP価格を生成（共通ユーティリティ）"""
        try:
            sl_price: Optional[float] = None
            tp_price: Optional[float] = None

            if stop_loss_pct is not None:
                if stop_loss_pct == 0:
                    sl_price = current_price
                else:
                    if position_direction > 0:
                        sl_price = current_price * (1 - stop_loss_pct)
                    else:
                        sl_price = current_price * (1 + stop_loss_pct)

            if take_profit_pct is not None:
                if take_profit_pct == 0:
                    tp_price = current_price
                else:
                    if position_direction > 0:
                        tp_price = current_price * (1 + take_profit_pct)
                    else:
                        tp_price = current_price * (1 - take_profit_pct)

            return sl_price, tp_price
        except Exception as e:
            logger.error(f"価格生成エラー: {e}")
            return None, None

    def _calculate_from_gene(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """TP/SL遺伝子からTP/SL価格を計算"""
        try:
            if tpsl_gene.method == TPSLMethod.FIXED_PERCENTAGE:
                return self._calculate_fixed_percentage(
                    current_price, tpsl_gene, position_direction
                )

            elif tpsl_gene.method == TPSLMethod.RISK_REWARD_RATIO:
                return self._calculate_risk_reward_ratio(
                    current_price, tpsl_gene, position_direction
                )

            elif tpsl_gene.method == TPSLMethod.VOLATILITY_BASED:
                return self._calculate_volatility_based(
                    current_price, tpsl_gene, market_data, position_direction
                )

            elif tpsl_gene.method == TPSLMethod.STATISTICAL:
                return self._calculate_statistical(
                    current_price, tpsl_gene, market_data, position_direction
                )

            elif tpsl_gene.method == TPSLMethod.ADAPTIVE:
                return self._calculate_adaptive(
                    current_price, tpsl_gene, market_data, position_direction
                )

            else:
                # 未知の方式の場合はフォールバック
                logger.warning(f"未知のTP/SL方式: {tpsl_gene.method}")
                return self._calculate_fallback(current_price, position_direction)

        except Exception as e:
            logger.error(f"遺伝子ベースTP/SL計算エラー: {e}")
            return self._calculate_fallback(current_price, position_direction)

    def _calculate_fixed_percentage(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """固定パーセンテージ方式"""
        return self._make_prices(
            current_price,
            tpsl_gene.stop_loss_pct,
            tpsl_gene.take_profit_pct,
            position_direction,
        )

    def _calculate_risk_reward_ratio(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """リスクリワード比方式"""
        try:
            config = RiskRewardConfig(target_ratio=tpsl_gene.risk_reward_ratio)
            result = self.risk_reward_calculator.calculate_take_profit(
                tpsl_gene.base_stop_loss, config
            )

            return self._make_prices(
                current_price,
                tpsl_gene.base_stop_loss,
                result.take_profit_pct,
                position_direction,
            )

        except Exception as e:
            logger.error(f"リスクリワード比計算エラー: {e}")
            return self._calculate_fixed_percentage(
                current_price, tpsl_gene, position_direction
            )

    def _calculate_volatility_based(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """ボラティリティベース方式"""
        try:
            config = VolatilityConfig(
                atr_period=tpsl_gene.atr_period,
                atr_multiplier_sl=tpsl_gene.atr_multiplier_sl,
                atr_multiplier_tp=tpsl_gene.atr_multiplier_tp,
            )

            result = self.volatility_generator.generate_volatility_based_tpsl(
                market_data or {}, config, current_price
            )

            return self._make_prices(
                current_price,
                result.stop_loss_pct,
                result.take_profit_pct,
                position_direction,
            )

        except Exception as e:
            logger.error(f"ボラティリティベース計算エラー: {e}")
            return self._calculate_fixed_percentage(
                current_price, tpsl_gene, position_direction
            )

    def _calculate_statistical(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """統計的方式"""
        try:
            config = StatisticalConfig(
                lookback_period_days=tpsl_gene.lookback_period,
                confidence_threshold=tpsl_gene.confidence_threshold,
            )

            result = self.statistical_generator.generate_statistical_tpsl(
                config, market_conditions=market_data
            )

            return self._make_prices(
                current_price,
                result.stop_loss_pct,
                result.take_profit_pct,
                position_direction,
            )

        except Exception as e:
            logger.error(f"統計的計算エラー: {e}")
            return self._calculate_fixed_percentage(
                current_price, tpsl_gene, position_direction
            )

    def _calculate_adaptive(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """適応的方式（複数方式の組み合わせ）"""
        try:
            # 複数の方式を組み合わせて最適な値を選択
            # 現在は簡易実装として、ボラティリティベースを使用
            return self._calculate_volatility_based(
                current_price, tpsl_gene, market_data, position_direction
            )

        except Exception as e:
            logger.error(f"適応的計算エラー: {e}")
            return self._calculate_fixed_percentage(
                current_price, tpsl_gene, position_direction
            )

    def _calculate_fallback(
        self,
        current_price: float,
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """フォールバック計算（デフォルト値）"""
        default_sl_pct = 0.03  # 3%
        default_tp_pct = 0.06  # 6%

        return self._make_prices(
            current_price,
            default_sl_pct,
            default_tp_pct,
            position_direction,
        )

    def _calculate_basic_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        risk_management: Dict[str, Any],
        position_direction: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        基本的なTP/SL価格計算（従来方式）

        Args:
            current_price: 現在価格
            stop_loss_pct: ストップロス割合
            take_profit_pct: テイクプロフィット割合
            risk_management: リスク管理設定
            position_direction: ポジション方向（1.0=ロング, -1.0=ショート）

        Returns:
            (SL価格, TP価格)のタプル
        """
        try:
            # 高度なTP/SL計算方式が使用されているかチェック
            if self._is_advanced_tpsl_used(risk_management):
                return self._calculate_advanced_tpsl_prices(
                    current_price,
                    stop_loss_pct,
                    take_profit_pct,
                    risk_management,
                    position_direction,
                )
            else:
                # 基本的なTP/SL計算
                return self._calculate_simple_tpsl_prices(
                    current_price, stop_loss_pct, take_profit_pct, position_direction
                )

        except Exception as e:
            logger.error(f"基本TP/SL価格計算エラー: {e}")
            # フォールバック: 基本的な計算方式
            return self._calculate_simple_tpsl_prices(
                current_price, stop_loss_pct, take_profit_pct, position_direction
            )

    def _calculate_simple_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        position_direction: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """基本的なTP/SL価格計算（エラーハンドリング強化版）"""
        try:
            # 入力値検証
            if not self._validate_price(current_price):
                logger.warning(f"不正な価格: {current_price}")
                return None, None

            valid_sl_pct = None
            if stop_loss_pct is not None:
                if self._validate_percentage(stop_loss_pct, "SL"):
                    valid_sl_pct = stop_loss_pct
                else:
                    logger.warning(f"不正なSL割合: {stop_loss_pct}")

            valid_tp_pct = None
            if take_profit_pct is not None:
                if self._validate_percentage(take_profit_pct, "TP"):
                    valid_tp_pct = take_profit_pct
                else:
                    logger.warning(f"不正なTP割合: {take_profit_pct}")

            return self._make_prices(
                current_price, valid_sl_pct, valid_tp_pct, position_direction
            )

        except Exception as e:
            logger.error(f"基本TP/SL計算エラー: {e}")
            # フォールバック
            return self._calculate_fallback(current_price, position_direction)

    def _is_advanced_tpsl_used(self, risk_management: Dict[str, Any]) -> bool:
        """高度なTP/SL計算方式が使用されているかチェック"""
        return any(
            key in risk_management
            for key in [
                "_tpsl_strategy",
                "_risk_reward_ratio",
                "_volatility_adaptive",
                "_statistical_tpsl",
            ]
        )

    def _calculate_advanced_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        risk_management: Dict[str, Any],
        position_direction: float = 1.0,
    ) -> Tuple[Optional[float], Optional[float]]:
        """高度なTP/SL価格計算（リスクリワード比、ボラティリティベースなど）"""
        try:
            # 使用された戦略を取得
            strategy_used = risk_management.get("_tpsl_strategy", "unknown")

            # 基本的な価格計算（ポジション方向を考慮）
            sl_price, tp_price = self._make_prices(
                current_price, stop_loss_pct, take_profit_pct, position_direction
            )

            # 戦略固有の調整
            if strategy_used == "volatility_adaptive":
                # ボラティリティベースの場合、追加の調整を適用
                sl_price, tp_price = self._apply_volatility_adjustments(
                    current_price, sl_price, tp_price, risk_management
                )
            elif strategy_used == "risk_reward":
                # リスクリワード比ベースの場合、比率の整合性をチェック
                sl_price, tp_price = self._apply_risk_reward_adjustments(
                    current_price, sl_price, tp_price, risk_management
                )

            return sl_price, tp_price

        except Exception as e:
            logger.error(f"高度なTP/SL価格計算エラー: {e}")
            # フォールバック
            return self._calculate_simple_tpsl_prices(
                current_price, stop_loss_pct, take_profit_pct, position_direction
            )

    def _apply_volatility_adjustments(
        self,
        current_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        risk_management: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """ボラティリティベース調整を適用"""
        # 現在は基本実装のみ（将来的にATRベース調整を追加）
        return sl_price, tp_price

    def _apply_risk_reward_adjustments(
        self,
        current_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        risk_management: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """リスクリワード比ベース調整を適用"""
        try:
            target_rr_ratio = risk_management.get("_risk_reward_ratio")

            if target_rr_ratio and sl_price:
                # SLが設定されている場合、RR比に基づいてTPを再計算
                sl_distance = current_price - sl_price
                tp_distance = sl_distance * target_rr_ratio
                adjusted_tp_price = current_price + tp_distance

                logger.debug(
                    f"RR比調整: 目標比率={target_rr_ratio}, "
                    f"調整後TP={adjusted_tp_price}"
                )

                return sl_price, adjusted_tp_price

            return sl_price, tp_price

        except Exception as e:
            logger.error(f"RR比調整エラー: {e}")
            return sl_price, tp_price

    def _validate_price(self, price: float) -> bool:
        """価格の妥当性を検証"""
        if price is None:
            return False
        if not isinstance(price, (int, float)):
            return False
        if not math.isfinite(price):
            return False
        if price <= 0:
            return False
        return True

    def _validate_percentage(self, percentage: float, label: str) -> bool:
        """割合の妥当性を検証"""
        if percentage is None:
            return False
        if not isinstance(percentage, (int, float)):
            return False
        if not math.isfinite(percentage):
            return False

        # SLは0-100%、TPは0-1000%まで許容
        if label == "SL":
            return 0 <= percentage <= 1.0
        elif label == "TP":
            return 0 <= percentage <= 10.0
        else:
            return 0 <= percentage <= 1.0
