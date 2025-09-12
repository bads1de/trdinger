"""
TP/SL計算サービス

リファクタリング済み: Calculatorパターンを使用したTP/SL計算サービス
各計算方式を個別のCalculatorクラスとして実装しています。
"""

import logging
import math
from typing import Any, Dict, Optional, Tuple

from .calculator import (
    AdaptiveCalculator,
    FixedPercentageCalculator,
    RiskRewardCalculator,
    StatisticalCalculator,
    VolatilityCalculator,
)
from ..models.strategy_models import TPSLGene, TPSLMethod

logger = logging.getLogger(__name__)


class TPSLService:
    """
    TP/SL計算サービス

    Calculatorパターンを使用して、各計算方式を個別のクラスとして実装しています。
    """

    def __init__(self):
        """初期化"""
        # Calculatorインスタンスを作成
        self.fixed_percentage_calculator = FixedPercentageCalculator()
        self.risk_reward_calculator = RiskRewardCalculator()
        self.volatility_calculator = VolatilityCalculator()
        self.statistical_calculator = StatisticalCalculator()
        self.adaptive_calculator = AdaptiveCalculator()

        # Calculatorマッピング
        self.calculators = {
            TPSLMethod.FIXED_PERCENTAGE: self.fixed_percentage_calculator,
            TPSLMethod.RISK_REWARD_RATIO: self.risk_reward_calculator,
            TPSLMethod.VOLATILITY_BASED: self.volatility_calculator,
            TPSLMethod.STATISTICAL: self.statistical_calculator,
            TPSLMethod.ADAPTIVE: self.adaptive_calculator,
        }

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
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="TP/SL価格計算",
            is_api_call=False,
            default_return=self._calculate_fallback(current_price, position_direction),
        )
        def _calculate_tpsl_prices():
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

        return _calculate_tpsl_prices()


    def _calculate_from_gene(
        self,
        current_price: float,
        tpsl_gene: TPSLGene,
        market_data: Optional[Dict[str, Any]],
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """TP/SL遺伝子からTP/SL価格を計算（リファクタリング済み）"""
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="CalculatorベースTP/SL計算",
            is_api_call=False,
            default_return=self._calculate_fallback(current_price, position_direction),
        )
        def _calculate_from_gene():
            # Calculatorマッピングから適切なCalculatorを取得
            calculator = self.calculators.get(tpsl_gene.method)

            if calculator:
                # Calculatorを使用して計算
                result = calculator.calculate(
                    current_price=current_price,
                    tpsl_gene=tpsl_gene,
                    market_data=market_data,
                    position_direction=position_direction,
                )

                # TPSLResultから価格を抽出
                sl_price, tp_price = calculator._make_prices(
                    current_price,
                    result.stop_loss_pct,
                    result.take_profit_pct,
                    position_direction,
                )

                return sl_price, tp_price
            else:
                # 未知の方式の場合はフォールバック
                logger.warning(f"未知のTP/SL方式: {tpsl_gene.method}")
                return self._calculate_fallback(current_price, position_direction)

        return _calculate_from_gene()


    def _calculate_fallback(
        self,
        current_price: float,
        position_direction: float,
    ) -> Tuple[Optional[float], Optional[float]]:
        """フォールバック計算（デフォルト値）"""
        default_sl_pct = 0.03  # 3%
        default_tp_pct = 0.06  # 6%

        return self.fixed_percentage_calculator._make_prices(
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
        from app.utils.error_handler import safe_operation

        @safe_operation(
            context="基本TP/SL価格計算",
            is_api_call=False,
            default_return=self._calculate_simple_tpsl_prices(
                current_price, stop_loss_pct, take_profit_pct, position_direction
            ),
        )
        def _calculate_basic_tpsl_prices():
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

        return _calculate_basic_tpsl_prices()

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

            return self.fixed_percentage_calculator._make_prices(
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
            sl_price, tp_price = self.fixed_percentage_calculator._make_prices(
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

        # 定数で定義された妥当範囲に統一
        try:
            from ..constants import TPSL_LIMITS

            if label == "SL":
                min_v, max_v = TPSL_LIMITS["stop_loss_pct"]
            elif label == "TP":
                min_v, max_v = TPSL_LIMITS["take_profit_pct"]
            else:
                # デフォルトは 0-1 の保守的な範囲
                min_v, max_v = 0.0, 1.0

            return min_v <= float(percentage) <= max_v
        except Exception:
            # フォールバック: 安全なデフォルト
            if label == "SL":
                return 0 <= percentage <= 1.0
            elif label == "TP":
                return 0 <= percentage <= 10.0
            else:
                return 0 <= percentage <= 1.0
