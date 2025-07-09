"""
TP/SL計算器

Take Profit/Stop Loss価格の計算を担当します。
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple

from ..models.gene_tpsl import TPSLGene

logger = logging.getLogger(__name__)


class TPSLCalculator:
    """
    TP/SL計算器

    Take Profit/Stop Loss価格の計算を担当します。
    """

    def calculate_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        risk_management: Dict[str, Any],
        gene: Optional[Any] = None,
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        TP/SL価格を計算（従来方式と新方式の両方をサポート）

        Args:
            current_price: 現在価格
            stop_loss_pct: ストップロス割合
            take_profit_pct: テイクプロフィット割合
            risk_management: リスク管理設定
            gene: 戦略遺伝子（オプション）

        Returns:
            (SL価格, TP価格)のタプル
        """
        try:
            # TP/SL遺伝子が利用可能かチェック（GA最適化対象）
            if gene and hasattr(gene, "tpsl_gene") and gene.tpsl_gene:
                return self.calculate_tpsl_from_gene(current_price, gene.tpsl_gene)
            # 新しいTP/SL計算方式が使用されているかチェック（従来の高度機能）
            elif self.is_advanced_tpsl_used(risk_management):
                return self.calculate_advanced_tpsl_prices(
                    current_price, stop_loss_pct, take_profit_pct, risk_management
                )
            else:
                # 従来の固定割合ベース計算
                return self.calculate_legacy_tpsl_prices(
                    current_price, stop_loss_pct, take_profit_pct
                )

        except Exception as e:
            logger.error(f"TP/SL価格計算エラー: {e}")
            # フォールバック: 従来方式
            return self.calculate_legacy_tpsl_prices(
                current_price, stop_loss_pct, take_profit_pct
            )

    def is_advanced_tpsl_used(self, risk_management: Dict[str, Any]) -> bool:
        """高度なTP/SL機能が使用されているかチェック"""
        return any(key.startswith("_tpsl_") for key in risk_management.keys())

    def calculate_legacy_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
    ) -> Tuple[Optional[float], Optional[float]]:
        """従来の固定割合ベースTP/SL価格計算（エラーハンドリング強化版）"""
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
                        sl_price = current_price * (1 - stop_loss_pct)
                else:
                    logger.warning(f"不正なSL割合: {stop_loss_pct}")

            # TP割合の検証と処理
            tp_price = None
            if take_profit_pct is not None:
                if self._validate_percentage(take_profit_pct, "TP"):
                    if take_profit_pct == 0:
                        tp_price = current_price  # 0%の場合は現在価格と同じ
                    else:
                        tp_price = current_price * (1 + take_profit_pct)
                else:
                    logger.warning(f"不正なTP割合: {take_profit_pct}")

            return sl_price, tp_price

        except Exception as e:
            logger.error(f"TP/SL価格計算エラー: {e}")
            return None, None

    def calculate_advanced_tpsl_prices(
        self,
        current_price: float,
        stop_loss_pct: Optional[float],
        take_profit_pct: Optional[float],
        risk_management: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """高度なTP/SL価格計算（リスクリワード比、ボラティリティベースなど）"""
        try:
            # 使用された戦略を取得
            strategy_used = risk_management.get("_tpsl_strategy", "unknown")

            # 基本的な価格計算
            sl_price = current_price * (1 - stop_loss_pct) if stop_loss_pct else None
            tp_price = (
                current_price * (1 + take_profit_pct) if take_profit_pct else None
            )

            # 戦略固有の調整
            if strategy_used == "volatility_adaptive":
                # ボラティリティベースの場合、追加の調整を適用
                sl_price, tp_price = self.apply_volatility_adjustments(
                    current_price, sl_price, tp_price, risk_management
                )
            elif strategy_used == "risk_reward":
                # リスクリワード比ベースの場合、比率の整合性をチェック
                sl_price, tp_price = self.apply_risk_reward_adjustments(
                    current_price, sl_price, tp_price, risk_management
                )

            return sl_price, tp_price

        except Exception as e:
            logger.error(f"高度なTP/SL価格計算エラー: {e}")
            # フォールバック
            return self.calculate_legacy_tpsl_prices(
                current_price, stop_loss_pct, take_profit_pct
            )

    def apply_volatility_adjustments(
        self,
        current_price: float,
        sl_price: Optional[float],
        tp_price: Optional[float],
        risk_management: Dict[str, Any],
    ) -> Tuple[Optional[float], Optional[float]]:
        """ボラティリティベース調整を適用"""
        # 現在は基本実装のみ（将来的にATRベース調整を追加）
        return sl_price, tp_price

    def apply_risk_reward_adjustments(
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

    def calculate_tpsl_from_gene(
        self, current_price: float, tpsl_gene: TPSLGene
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        TP/SL遺伝子からTP/SL価格を計算（GA最適化対象）

        Args:
            current_price: 現在価格
            tpsl_gene: TP/SL遺伝子

        Returns:
            (SL価格, TP価格)のタプル
        """
        try:
            # TP/SL遺伝子から値を計算
            tpsl_values = tpsl_gene.calculate_tpsl_values()

            sl_pct = tpsl_values.get("stop_loss", 0.03)
            tp_pct = tpsl_values.get("take_profit", 0.06)

            # 価格に変換
            sl_price = current_price * (1 - sl_pct)
            tp_price = current_price * (1 + tp_pct)

            return sl_price, tp_price

        except Exception as e:
            logger.error(f"TP/SL遺伝子計算エラー: {e}")
            # フォールバック
            return current_price * 0.97, current_price * 1.06  # デフォルト3%SL, 6%TP

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
