"""
TP/SL計算サービス

TP/SL計算ロジックを一元化し、異なる計算方式を統一的なインターフェースで提供します。
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple
from enum import Enum

from ..calculators.risk_reward_calculator import RiskRewardCalculator, RiskRewardConfig
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

    def _convert_to_prices(
        self, result: TPSLResult, current_price: float, position_direction: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """TPSLResultを価格に変換"""
        try:
            if position_direction > 0:  # ロング
                sl_price = current_price * (1 - result.stop_loss_pct)
                tp_price = current_price * (1 + result.take_profit_pct)
            else:  # ショート
                sl_price = current_price * (1 + result.stop_loss_pct)
                tp_price = current_price * (1 - result.take_profit_pct)
            return sl_price, tp_price
        except Exception as e:
            logger.error(f"価格変換エラー: {e}")
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
        if position_direction > 0:  # ロング
            sl_price = current_price * (1 - tpsl_gene.stop_loss_pct)
            tp_price = current_price * (1 + tpsl_gene.take_profit_pct)
        else:  # ショート
            sl_price = current_price * (1 + tpsl_gene.stop_loss_pct)
            tp_price = current_price * (1 - tpsl_gene.take_profit_pct)

        return sl_price, tp_price

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

            if position_direction > 0:  # ロング
                sl_price = current_price * (1 - tpsl_gene.base_stop_loss)
                tp_price = current_price * (1 + result.take_profit_pct)
            else:  # ショート
                sl_price = current_price * (1 + tpsl_gene.base_stop_loss)
                tp_price = current_price * (1 - result.take_profit_pct)

            return sl_price, tp_price

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

            if position_direction > 0:  # ロング
                sl_price = current_price * (1 - result.stop_loss_pct)
                tp_price = current_price * (1 + result.take_profit_pct)
            else:  # ショート
                sl_price = current_price * (1 + result.stop_loss_pct)
                tp_price = current_price * (1 - result.take_profit_pct)

            return sl_price, tp_price

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

            if position_direction > 0:  # ロング
                sl_price = current_price * (1 - result.stop_loss_pct)
                tp_price = current_price * (1 + result.take_profit_pct)
            else:  # ショート
                sl_price = current_price * (1 + result.stop_loss_pct)
                tp_price = current_price * (1 - result.take_profit_pct)

            return sl_price, tp_price

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

        if position_direction > 0:  # ロング
            sl_price = current_price * (1 - default_sl_pct)
            tp_price = current_price * (1 + default_tp_pct)
        else:  # ショート
            sl_price = current_price * (1 + default_sl_pct)
            tp_price = current_price * (1 - default_tp_pct)

        return sl_price, tp_price

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

            # SL割合の検証と処理
            sl_price = None
            if stop_loss_pct is not None:
                if self._validate_percentage(stop_loss_pct, "SL"):
                    if stop_loss_pct == 0:
                        sl_price = current_price  # 0%の場合は現在価格と同じ
                    else:
                        if position_direction > 0:  # ロングポジション
                            sl_price = current_price * (1 - stop_loss_pct)
                        else:  # ショートポジション
                            sl_price = current_price * (1 + stop_loss_pct)
                else:
                    logger.warning(f"不正なSL割合: {stop_loss_pct}")

            # TP割合の検証と処理
            tp_price = None
            if take_profit_pct is not None:
                if self._validate_percentage(take_profit_pct, "TP"):
                    if take_profit_pct == 0:
                        tp_price = current_price  # 0%の場合は現在価格と同じ
                    else:
                        if position_direction > 0:  # ロングポジション
                            tp_price = current_price * (1 + take_profit_pct)
                        else:  # ショートポジション
                            tp_price = current_price * (1 - take_profit_pct)
                else:
                    logger.warning(f"不正なTP割合: {take_profit_pct}")

            return sl_price, tp_price

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
            if position_direction > 0:  # ロングポジション
                sl_price = (
                    current_price * (1 - stop_loss_pct) if stop_loss_pct else None
                )
                tp_price = (
                    current_price * (1 + take_profit_pct) if take_profit_pct else None
                )
            else:  # ショートポジション
                sl_price = (
                    current_price * (1 + stop_loss_pct) if stop_loss_pct else None
                )
                tp_price = (
                    current_price * (1 - take_profit_pct) if take_profit_pct else None
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
