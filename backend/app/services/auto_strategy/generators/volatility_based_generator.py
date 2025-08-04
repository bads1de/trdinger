"""
ボラティリティベースのTP/SL生成機能

このモジュールは、ATR（Average True Range）やその他のボラティリティ指標を使用して、
市場のボラティリティに応じたテイクプロフィット（TP）とストップロス（SL）を
動的に設定する機能を提供します。
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VolatilityRegime(Enum):
    """ボラティリティレジームの種類"""

    VERY_LOW = "very_low"  # 非常に低ボラティリティ
    LOW = "low"  # 低ボラティリティ
    NORMAL = "normal"  # 通常ボラティリティ
    HIGH = "high"  # 高ボラティリティ
    VERY_HIGH = "very_high"  # 非常に高ボラティリティ


@dataclass
class VolatilityConfig:
    """ボラティリティベース計算の設定"""

    atr_period: int = 14  # ATR計算期間
    atr_multiplier_sl: float = 2.0  # SL用ATR倍率
    atr_multiplier_tp: float = 3.0  # TP用ATR倍率
    volatility_sensitivity: str = "medium"  # 感度設定
    adaptive_multiplier: bool = True  # 適応的倍率調整
    min_sl_pct: float = 0.005  # 最小SL（0.5%）
    max_sl_pct: float = 0.1  # 最大SL（10%）
    min_tp_pct: float = 0.01  # 最小TP（1%）
    max_tp_pct: float = 0.2  # 最大TP（20%）
    regime_lookback: int = 50  # ボラティリティレジーム判定期間


@dataclass
class VolatilityResult:
    """ボラティリティベース計算結果"""

    stop_loss_pct: float
    take_profit_pct: float
    atr_value: float
    atr_pct: float
    volatility_regime: VolatilityRegime
    multipliers_used: Dict[str, float]
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class VolatilityBasedGenerator:
    """
    ボラティリティベースのTP/SL生成機能

    ATRやその他のボラティリティ指標を使用して、市場の状況に応じた
    動的なTP/SL設定を提供します。
    """

    def __init__(self):
        """ジェネレーターを初期化"""
        # self.logger = logging.getLogger(__name__)

        # ボラティリティレジーム別の設定
        self.regime_settings = {
            VolatilityRegime.VERY_LOW: {
                "sl_multiplier": 1.5,
                "tp_multiplier": 2.0,
                "confidence": 0.6,
            },
            VolatilityRegime.LOW: {
                "sl_multiplier": 1.8,
                "tp_multiplier": 2.5,
                "confidence": 0.7,
            },
            VolatilityRegime.NORMAL: {
                "sl_multiplier": 2.0,
                "tp_multiplier": 3.0,
                "confidence": 0.8,
            },
            VolatilityRegime.HIGH: {
                "sl_multiplier": 2.5,
                "tp_multiplier": 3.5,
                "confidence": 0.7,
            },
            VolatilityRegime.VERY_HIGH: {
                "sl_multiplier": 3.0,
                "tp_multiplier": 4.0,
                "confidence": 0.6,
            },
        }

        # 感度設定別の調整係数
        self.sensitivity_adjustments = {
            "low": {"sl": 0.8, "tp": 0.8},
            "medium": {"sl": 1.0, "tp": 1.0},
            "high": {"sl": 1.2, "tp": 1.2},
        }

    def generate_volatility_based_tpsl(
        self,
        market_data: Dict[str, Any],
        config: VolatilityConfig,
        current_price: float,
    ) -> VolatilityResult:
        """
        ボラティリティに基づいてTP/SLを生成

        Args:
            market_data: 市場データ（OHLCV、ATR値など）
            config: ボラティリティ設定
            current_price: 現在価格

        Returns:
            ボラティリティベース計算結果
        """
        try:
            # self.logger.info("ボラティリティベースTP/SL生成開始")

            # ATR値の取得または計算
            atr_value, atr_pct = self._get_or_calculate_atr(
                market_data, config, current_price
            )

            # ボラティリティレジームの判定
            volatility_regime = self._determine_volatility_regime(
                market_data, config, atr_pct
            )

            # レジーム別の倍率を取得
            regime_multipliers = self.regime_settings[volatility_regime]

            # 感度調整の適用
            sensitivity_adj = self.sensitivity_adjustments.get(
                config.volatility_sensitivity, self.sensitivity_adjustments["medium"]
            )

            # 最終的な倍率を計算
            final_sl_multiplier = (
                regime_multipliers["sl_multiplier"] * sensitivity_adj["sl"]
            )
            final_tp_multiplier = (
                regime_multipliers["tp_multiplier"] * sensitivity_adj["tp"]
            )

            # 適応的倍率調整
            if config.adaptive_multiplier:
                final_sl_multiplier, final_tp_multiplier = (
                    self._apply_adaptive_adjustment(
                        final_sl_multiplier,
                        final_tp_multiplier,
                        market_data,
                        volatility_regime,
                    )
                )

            # TP/SL割合を計算
            sl_pct = atr_pct * final_sl_multiplier
            tp_pct = atr_pct * final_tp_multiplier

            # 制限範囲内に調整
            sl_pct = max(config.min_sl_pct, min(sl_pct, config.max_sl_pct))
            tp_pct = max(config.min_tp_pct, min(tp_pct, config.max_tp_pct))

            # 信頼度スコアを計算
            confidence_score = self._calculate_confidence_score(
                volatility_regime, market_data, config
            )

            result = VolatilityResult(
                stop_loss_pct=sl_pct,
                take_profit_pct=tp_pct,
                atr_value=atr_value,
                atr_pct=atr_pct,
                volatility_regime=volatility_regime,
                multipliers_used={
                    "sl_multiplier": final_sl_multiplier,
                    "tp_multiplier": final_tp_multiplier,
                },
                confidence_score=confidence_score,
                metadata={
                    "current_price": current_price,
                    "sensitivity": config.volatility_sensitivity,
                    "adaptive_adjustment": config.adaptive_multiplier,
                    "regime_confidence": regime_multipliers["confidence"],
                },
            )

            # self.logger.info(
            #     f"ボラティリティベース生成完了: SL={sl_pct:.3f}, TP={tp_pct:.3f}, "
            #     f"レジーム={volatility_regime.value}"
            # )

            return result

        except Exception as e:
            logger.error(f"ボラティリティベース生成エラー: {e}", exc_info=True)
            return self._generate_fallback_result(config, current_price)

    def _get_or_calculate_atr(
        self,
        market_data: Dict[str, Any],
        config: VolatilityConfig,
        current_price: float,
    ) -> Tuple[float, float]:
        """ATR値を取得または計算"""
        try:
            # 既にATR値が提供されている場合
            if "atr" in market_data:
                atr_value = market_data["atr"]
                atr_pct = atr_value / current_price
                return atr_value, atr_pct

            # OHLC データからATRを計算
            if all(key in market_data for key in ["high", "low", "close"]):
                atr_value = self._calculate_atr(
                    market_data["high"],
                    market_data["low"],
                    market_data["close"],
                    config.atr_period,
                )
                atr_pct = atr_value / current_price
                return atr_value, atr_pct

            # フォールバック: 推定ATR
            estimated_atr_pct = 0.02  # 2%のデフォルト
            estimated_atr_value = estimated_atr_pct * current_price

            logger.warning("ATR計算用データ不足、推定値を使用")
            return estimated_atr_value, estimated_atr_pct

        except Exception as e:
            logger.error(f"ATR取得/計算エラー: {e}")
            # エラー時のフォールバック
            fallback_atr_pct = 0.02
            return fallback_atr_pct * current_price, fallback_atr_pct

    def _calculate_atr(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int
    ) -> float:
        """ATRを計算"""
        try:
            # True Range の計算
            high_low = high - low
            high_close_prev = np.abs(high - np.roll(close, 1))
            low_close_prev = np.abs(low - np.roll(close, 1))

            true_range = np.maximum(
                high_low, np.maximum(high_close_prev, low_close_prev)
            )

            # ATRの計算（単純移動平均）
            atr = np.mean(true_range[-period:])

            return float(atr)

        except Exception as e:
            logger.error(f"ATR計算エラー: {e}")
            return 0.02 * close[-1]  # フォールバック

    def _determine_volatility_regime(
        self,
        market_data: Dict[str, Any],
        config: VolatilityConfig,
        current_atr_pct: float,
    ) -> VolatilityRegime:
        """ボラティリティレジームを判定"""
        try:
            # 過去のATR値から基準を計算
            if "atr_history" in market_data:
                atr_history = market_data["atr_history"]
                atr_mean = np.mean(atr_history)
                atr_std = np.std(atr_history)
            else:
                # デフォルト基準値
                atr_mean = 0.02  # 2%
                atr_std = 0.01  # 1%

            # Z-スコアベースの判定
            z_score = (current_atr_pct - atr_mean) / atr_std if atr_std > 0 else 0

            if z_score < -1.5:
                return VolatilityRegime.VERY_LOW
            elif z_score < -0.5:
                return VolatilityRegime.LOW
            elif z_score < 0.5:
                return VolatilityRegime.NORMAL
            elif z_score < 1.5:
                return VolatilityRegime.HIGH
            else:
                return VolatilityRegime.VERY_HIGH

        except Exception as e:
            logger.error(f"ボラティリティレジーム判定エラー: {e}")
            return VolatilityRegime.NORMAL  # デフォルト

    def _apply_adaptive_adjustment(
        self,
        sl_multiplier: float,
        tp_multiplier: float,
        market_data: Dict[str, Any],
        regime: VolatilityRegime,
    ) -> Tuple[float, float]:
        """適応的倍率調整を適用"""
        try:
            # トレンド強度による調整
            trend_strength = market_data.get("trend_strength", 0.5)

            # 強いトレンドの場合はTPを拡大、SLを縮小
            if trend_strength > 0.7:
                tp_multiplier *= 1.1
                sl_multiplier *= 0.9
            elif trend_strength < 0.3:
                tp_multiplier *= 0.9
                sl_multiplier *= 1.1

            # ボリュームによる調整
            volume_ratio = market_data.get("volume_ratio", 1.0)
            if volume_ratio > 1.5:  # 高ボリューム
                # 高ボリューム時は信頼度が高いので倍率を調整
                tp_multiplier *= 1.05
            elif volume_ratio < 0.5:  # 低ボリューム
                # 低ボリューム時は保守的に
                sl_multiplier *= 0.95
                tp_multiplier *= 0.95

            return sl_multiplier, tp_multiplier

        except Exception as e:
            logger.error(f"適応的調整エラー: {e}")
            return sl_multiplier, tp_multiplier

    def _calculate_confidence_score(
        self,
        regime: VolatilityRegime,
        market_data: Dict[str, Any],
        config: VolatilityConfig,
    ) -> float:
        """信頼度スコアを計算"""
        try:
            base_confidence = self.regime_settings[regime]["confidence"]

            # データ品質による調整
            data_quality = 1.0
            if "atr" not in market_data:
                data_quality *= 0.8  # ATR値が直接提供されていない

            if not all(key in market_data for key in ["high", "low", "close"]):
                data_quality *= 0.7  # OHLC データが不完全

            # 最終信頼度
            final_confidence = base_confidence * data_quality
            return max(0.1, min(1.0, final_confidence))

        except Exception as e:
            logger.error(f"信頼度計算エラー: {e}")
            return 0.5  # デフォルト

    def _generate_fallback_result(
        self, config: VolatilityConfig, current_price: float
    ) -> VolatilityResult:
        """フォールバック結果を生成"""
        fallback_atr_pct = 0.02
        fallback_atr_value = fallback_atr_pct * current_price

        return VolatilityResult(
            stop_loss_pct=fallback_atr_pct * 2.0,
            take_profit_pct=fallback_atr_pct * 3.0,
            atr_value=fallback_atr_value,
            atr_pct=fallback_atr_pct,
            volatility_regime=VolatilityRegime.NORMAL,
            multipliers_used={"sl_multiplier": 2.0, "tp_multiplier": 3.0},
            confidence_score=0.3,
            metadata={"fallback": True},
        )
