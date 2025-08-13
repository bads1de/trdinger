"""
市場レジーム検出・適応学習サービス

市場の状態変化を検出し、モデルの適応的学習を実行します。
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass


logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """市場レジーム"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CALM = "calm"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"


@dataclass
class RegimeDetectionResult:
    """レジーム検出結果"""

    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    timestamp: datetime
    duration: Optional[timedelta] = None


class MarketRegimeDetector:
    """
    市場レジーム検出器

    複数の指標を組み合わせて市場の状態を検出し、
    適応的学習のトリガーを提供します。
    """

    def __init__(
        self,
        lookback_period: int = 100,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.1,
    ):
        """
        初期化

        Args:
            lookback_period: 分析期間
            volatility_threshold: ボラティリティ閾値
            trend_threshold: トレンド強度閾値
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.regime_history: List[RegimeDetectionResult] = []

    def detect_regime(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """
        市場レジームを検出

        Args:
            data: OHLCV価格データ

        Returns:
            レジーム検出結果
        """
        try:
            if len(data) < self.lookback_period:
                logger.warning(f"データ不足: {len(data)} < {self.lookback_period}")
                return RegimeDetectionResult(
                    regime=MarketRegime.RANGING,
                    confidence=0.5,
                    indicators={},
                    timestamp=datetime.now(),
                )

            # 各指標を計算
            indicators = self._calculate_regime_indicators(data)

            # レジーム判定
            regime, confidence = self._classify_regime(indicators)

            result = RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                indicators=indicators,
                timestamp=datetime.now(),
            )

            # 履歴に追加
            self.regime_history.append(result)

            # 履歴サイズ制限
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-500:]

            logger.info(f"市場レジーム検出: {regime.value} (信頼度: {confidence:.2f})")
            return result

        except Exception as e:
            logger.error(f"レジーム検出エラー: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                indicators={},
                timestamp=datetime.now(),
            )

    def _calculate_regime_indicators(self, data: pd.DataFrame) -> Dict[str, float]:
        """レジーム判定指標を計算"""
        indicators = {}

        try:
            # 価格データ
            close = data["Close"].iloc[-self.lookback_period :]
            high = data["High"].iloc[-self.lookback_period :]
            low = data["Low"].iloc[-self.lookback_period :]
            volume = data["Volume"].iloc[-self.lookback_period :]

            # 1. トレンド強度
            returns = close.pct_change().dropna()
            indicators["trend_strength"] = (
                abs(returns.mean()) / returns.std() if returns.std() > 0 else 0
            )
            indicators["trend_direction"] = np.sign(returns.mean())

            # 2. ボラティリティ
            indicators["volatility"] = returns.std()
            indicators["volatility_percentile"] = self._calculate_percentile(
                returns.rolling(20).std().dropna(), returns.std()
            )

            # 3. レンジ指標
            atr = self._calculate_atr(high, low, close, period=14)
            price_range = (high.max() - low.min()) / close.iloc[-1]
            indicators["atr_normalized"] = atr / close.iloc[-1]
            indicators["price_range"] = price_range

            # 4. モメンタム指標
            rsi = self._calculate_rsi(close, period=14)
            indicators["rsi"] = rsi
            indicators["rsi_extreme"] = 1 if rsi > 70 or rsi < 30 else 0

            # 5. ボリューム指標
            volume_ma = volume.rolling(20).mean()
            indicators["volume_ratio"] = (
                volume.iloc[-1] / volume_ma.iloc[-1] if volume_ma.iloc[-1] > 0 else 1
            )

            # 6. ブレイクアウト指標
            bb_upper, bb_lower = self._calculate_bollinger_bands(
                close, period=20, std_dev=2
            )
            indicators["bb_position"] = (
                (close.iloc[-1] - bb_lower) / (bb_upper - bb_lower)
                if bb_upper != bb_lower
                else 0.5
            )
            indicators["bb_squeeze"] = (bb_upper - bb_lower) / close.iloc[-1]

            # 7. 反転指標
            indicators["price_deviation"] = (
                close.iloc[-1] - close.rolling(20).mean().iloc[-1]
            ) / close.rolling(20).std().iloc[-1]

        except Exception as e:
            logger.error(f"指標計算エラー: {e}")

        return indicators

    def _classify_regime(
        self, indicators: Dict[str, float]
    ) -> Tuple[MarketRegime, float]:
        """指標からレジームを分類"""
        try:
            # 各レジームのスコアを計算
            scores = {}

            # トレンド上昇
            scores[MarketRegime.TRENDING_UP] = (
                max(0, indicators.get("trend_strength", 0) - self.trend_threshold) * 2
                + max(0, indicators.get("trend_direction", 0)) * 3
                + max(0, indicators.get("volume_ratio", 1) - 1) * 1
            )

            # トレンド下降
            scores[MarketRegime.TRENDING_DOWN] = (
                max(0, indicators.get("trend_strength", 0) - self.trend_threshold) * 2
                + max(0, -indicators.get("trend_direction", 0)) * 3
                + max(0, indicators.get("volume_ratio", 1) - 1) * 1
            )

            # レンジ相場
            scores[MarketRegime.RANGING] = (
                max(0, self.trend_threshold - indicators.get("trend_strength", 0)) * 2
                + max(0, 0.5 - abs(indicators.get("bb_position", 0.5) - 0.5)) * 2
                + max(0, self.volatility_threshold - indicators.get("volatility", 0))
                * 1
            )

            # 高ボラティリティ
            scores[MarketRegime.VOLATILE] = (
                max(0, indicators.get("volatility", 0) - self.volatility_threshold * 2)
                * 3
                + max(0, indicators.get("volatility_percentile", 0.5) - 0.7) * 2
            )

            # 低ボラティリティ
            scores[MarketRegime.CALM] = (
                max(
                    0, self.volatility_threshold * 0.5 - indicators.get("volatility", 0)
                )
                * 3
                + max(0, 0.3 - indicators.get("volatility_percentile", 0.5)) * 2
            )

            # ブレイクアウト
            scores[MarketRegime.BREAKOUT] = (
                indicators.get("rsi_extreme", 0) * 2
                + max(0, abs(indicators.get("bb_position", 0.5) - 0.5) - 0.4) * 3
                + max(0, indicators.get("volume_ratio", 1) - 1.5) * 2
            )

            # 反転
            scores[MarketRegime.REVERSAL] = (
                indicators.get("rsi_extreme", 0) * 2
                + max(0, abs(indicators.get("price_deviation", 0)) - 2) * 3
                + max(0, indicators.get("volume_ratio", 1) - 1.2) * 1
            )

            # 最高スコアのレジームを選択
            best_regime = max(scores, key=scores.get)
            max_score = scores[best_regime]

            # 信頼度を計算（0-1の範囲に正規化）
            total_score = sum(scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
            confidence = min(1.0, max(0.0, confidence))

            return best_regime, confidence

        except Exception as e:
            logger.error(f"レジーム分類エラー: {e}")
            return MarketRegime.RANGING, 0.0

    def _calculate_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> float:
        """ATR（Average True Range）を計算（pandas-ta使用）"""
        import pandas_ta as ta

        atr = ta.atr(high=high, low=low, close=close, length=period)
        return float(atr.iloc[-1])

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """RSI（Relative Strength Index）を計算（pandas-ta使用）"""
        import pandas_ta as ta

        rsi = ta.rsi(close, length=period)
        return float(rsi.iloc[-1])

    def _calculate_bollinger_bands(
        self, close: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[float, float]:
        """ボリンジャーバンドを計算（pandas-ta使用）"""
        import pandas_ta as ta

        bb_result = ta.bbands(close, length=period, std=std_dev)
        upper = bb_result[f"BBU_{period}_{std_dev}"].iloc[-1]
        lower = bb_result[f"BBL_{period}_{std_dev}"].iloc[-1]
        return float(upper), float(lower)

    def _calculate_percentile(self, series: pd.Series, value: float) -> float:
        """値のパーセンタイルを計算"""
        try:
            return (series <= value).mean()
        except Exception:
            return 0.5

    def get_regime_stability(self, window: int = 10) -> float:
        """レジームの安定性を計算"""
        if len(self.regime_history) < window:
            return 0.5

        recent_regimes = [r.regime for r in self.regime_history[-window:]]
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        stability = recent_regimes.count(most_common) / len(recent_regimes)

        return stability

    def should_retrain_model(self, stability_threshold: float = 0.7) -> bool:
        """モデル再学習が必要かを判定"""
        if len(self.regime_history) < 5:
            return False

        # レジーム安定性をチェック
        stability = self.get_regime_stability()

        # 最近のレジーム変化をチェック
        recent_regimes = [r.regime for r in self.regime_history[-5:]]
        regime_changes = len(set(recent_regimes))

        # 再学習条件
        should_retrain = (
            stability < stability_threshold  # 不安定
            or regime_changes >= 3  # 頻繁な変化
        )

        if should_retrain:
            logger.info(
                f"モデル再学習推奨: 安定性={stability:.2f}, 変化数={regime_changes}"
            )

        return should_retrain
