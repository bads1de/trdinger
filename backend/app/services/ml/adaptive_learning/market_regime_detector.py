"""
統合市場レジーム検出・適応学習サービス

ルールベースと機械学習ベースのアプローチを組み合わせた
高度な市場の状態変化検出と適応的学習を提供します。
"""

import logging
import warnings
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, cast, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm

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


class RegimeDetectionMethod(Enum):
    """レジーム検出方法"""

    RULE_BASED = "rule_based"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HMM = "hmm"
    ENSEMBLE = "ensemble"


@dataclass
class RegimeDetectionResult:
    """レジーム検出結果"""

    regime: MarketRegime
    confidence: float
    indicators: Dict[str, float]
    timestamp: datetime
    duration: Optional[timedelta] = None
    method: Optional[str] = None
    cluster: Optional[int] = None
    votes: Optional[Dict[str, float]] = None


class MarketRegimeDetector:
    """
    統合市場レジーム検出器

    ルールベースと機械学習ベースのアプローチを組み合わせた
    高度な市場レジーム検出を提供します。
    """

    def __init__(
        self,
        lookback_period: int = 100,
        volatility_threshold: float = 0.02,
        trend_threshold: float = 0.1,
        detection_method: RegimeDetectionMethod = RegimeDetectionMethod.RULE_BASED,
        n_clusters: int = 4,
        hmm_states: int = 3,
    ):
        """
        初期化

        Args:
            lookback_period: 分析期間
            volatility_threshold: ボラティリティ閾値
            trend_threshold: トレンド強度閾値
            detection_method: 検出方法
            n_clusters: クラスタ数（KMeans用）
            hmm_states: HMM状態数
        """
        self.lookback_period = lookback_period
        self.volatility_threshold = volatility_threshold
        self.trend_threshold = trend_threshold
        self.detection_method = detection_method
        self.n_clusters = n_clusters
        self.hmm_states = hmm_states

        # 機械学習モデル
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.hmm_model = None

        # 履歴
        self.regime_history: List[RegimeDetectionResult] = []
        self.feature_history: List[np.ndarray] = []

        logger.info(f"統合レジーム検出器初期化: method={detection_method.value}")

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
                    method="default",
                )

            # 検出方法に応じて処理
            if self.detection_method == RegimeDetectionMethod.RULE_BASED:
                result = self._rule_based_detection(data)
            elif self.detection_method == RegimeDetectionMethod.KMEANS:
                result = self._kmeans_detection(data)
            elif self.detection_method == RegimeDetectionMethod.DBSCAN:
                result = self._dbscan_detection(data)
            elif self.detection_method == RegimeDetectionMethod.HMM:
                result = self._hmm_detection(data)
            elif self.detection_method == RegimeDetectionMethod.ENSEMBLE:
                result = self._ensemble_detection(data)
            else:
                result = self._rule_based_detection(data)

            # 履歴に追加
            self.regime_history.append(result)

            # 履歴サイズ制限
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-500:]

            logger.info(
                f"市場レジーム検出: {result.regime.value} (信頼度: {result.confidence:.2f}, 手法: {result.method})"
            )
            return result

        except Exception as e:
            logger.error(f"レジーム検出エラー: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                indicators={},
                timestamp=datetime.now(),
                method="error",
            )

    def _rule_based_detection(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """ルールベースのレジーム検出（従来の方法）"""
        try:
            # 各指標を計算
            indicators = self._calculate_regime_indicators(data)

            # レジーム判定
            regime, confidence = self._classify_regime(indicators)

            return RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                indicators=indicators,
                timestamp=datetime.now(),
                method="rule_based",
            )

        except Exception as e:
            logger.error(f"ルールベース検出エラー: {e}")
            return RegimeDetectionResult(
                regime=MarketRegime.RANGING,
                confidence=0.0,
                indicators={},
                timestamp=datetime.now(),
                method="rule_based",
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
            best_regime = max(scores.items(), key=lambda x: x[1])[0]
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

        atr = cast(
            Union[pd.Series, pd.DataFrame, None],
            ta.atr(high=high, low=low, close=close, length=period),
        )
        if atr is None or not hasattr(atr, "iloc"):
            return 0.0
        return float(atr.iloc[-1])

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> float:
        """RSI（Relative Strength Index）を計算（pandas-ta使用）"""
        import pandas_ta as ta

        rsi_raw = ta.rsi(close, length=period)  # type: ignore

        if rsi_raw is None:
            return 50.0

        # numpy ndarray を pandas Series に変換
        if isinstance(rsi_raw, np.ndarray):
            rsi = pd.Series(rsi_raw, index=close.index[-len(rsi_raw):])
        else:
            rsi = rsi_raw

        if rsi is None or not hasattr(rsi, "iloc"):
            return 50.0

        return float(rsi.iloc[-1])

    def _calculate_bollinger_bands(
        self, close: pd.Series, period: int = 20, std_dev: float = 2
    ) -> Tuple[float, float]:
        """ボリンジャーバンドを計算（pandas-ta使用）"""
        import pandas_ta as ta

        bb_result = ta.bbands(close, length=period, std=std_dev)

        if bb_result is None:
            # ta.bbandsがNoneを返す場合のフォールバック
            current_price = float(close.iloc[-1])
            return current_price, current_price

        upper = bb_result[f"BBU_{period}_{std_dev}"].iloc[-1]
        lower = bb_result[f"BBL_{period}_{std_dev}"].iloc[-1]
        return float(upper), float(lower)

    def _calculate_percentile(self, series: pd.Series, value: float) -> float:
        """値のパーセンタイルを計算"""
        try:
            return (series <= value).mean()
        except Exception:
            return 0.5

    def _calculate_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """レジーム判定用特徴量を計算"""
        try:
            close = data["Close"].iloc[-self.lookback_period :]
            high = data["High"].iloc[-self.lookback_period :]
            low = data["Low"].iloc[-self.lookback_period :]
            volume = data["Volume"].iloc[-self.lookback_period :]

            returns = close.pct_change().dropna()

            if len(returns) < 20:
                return None

            features = []
            window = 20

            for i in range(window, len(close)):
                window_returns = returns.iloc[i - window : i]
                window_volume = volume.iloc[i - window : i]
                window_high = high.iloc[i - window : i]
                window_low = low.iloc[i - window : i]
                window_close = close.iloc[i - window : i]

                # 基本統計量
                volatility = window_returns.std()
                mean_return = window_returns.mean()
                skewness = window_returns.skew()
                kurtosis = window_returns.kurtosis()

                # トレンド指標
                trend_strength = abs(mean_return) / volatility if volatility > 0 else 0

                # ボリューム指標
                volume_ratio = (
                    window_volume.iloc[-5:].mean() / window_volume.mean()
                    if window_volume.mean() > 0
                    else 1
                )

                # 価格レンジ指標
                price_range = (
                    (window_high.max() - window_low.min()) / window_close.mean()
                    if window_close.mean() > 0
                    else 0
                )

                # ボラティリティ変化
                vol_change = (
                    volatility / window_returns.iloc[:10].std()
                    if window_returns.iloc[:10].std() > 0
                    else 1
                )

                features.append(
                    [
                        volatility,
                        trend_strength,
                        volume_ratio,
                        price_range,
                        vol_change,
                        skewness,
                        kurtosis,
                        mean_return,
                    ]
                )

            features_array = np.array(features)

            # 特徴量の正規化
            features_scaled: np.ndarray
            if len(self.feature_history) == 0:
                # 初回は現在のデータで正規化
                features_scaled = cast(
                    np.ndarray, self.scaler.fit_transform(features_array)
                )
            else:
                # 既存のスケーラーを使用
                features_scaled = cast(
                    np.ndarray, self.scaler.transform(features_array)
                )

            return features_scaled

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            return None

    def _kmeans_detection(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """KMeansクラスタリングによるレジーム検出"""
        try:
            features = self._calculate_features(data)
            if features is None or len(features) == 0:
                return self._create_default_result()

            if self.kmeans_model is None:
                self.kmeans_model = KMeans(
                    n_clusters=self.n_clusters, random_state=42, n_init="auto"  # type: ignore
                )

            # 十分なデータがある場合のみ学習
            if len(features) >= self.n_clusters * 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.kmeans_model.fit(features)

            # 最新の特徴量でクラスタを予測
            cluster: int = cast(int, self.kmeans_model.predict([features[-1]])[0])

            # クラスタからレジームにマッピング
            regime_mapping = {
                0: MarketRegime.TRENDING_UP,
                1: MarketRegime.TRENDING_DOWN,
                2: MarketRegime.RANGING,
                3: MarketRegime.VOLATILE,
            }

            regime = regime_mapping.get(int(cluster), MarketRegime.RANGING)

            # 信頼度計算（クラスタ中心からの距離ベース）
            distances = self.kmeans_model.transform([features[-1]])[0]
            confidence = 1.0 / (1.0 + distances[cluster])
            confidence = min(1.0, max(0.0, confidence))

            # 履歴に追加
            self.feature_history.append(features[-1])

            return RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                indicators={},
                timestamp=datetime.now(),
                method="kmeans",
                cluster=int(cluster),
            )

        except Exception as e:
            logger.error(f"KMeans検出エラー: {e}")
            return self._create_default_result()

    def _dbscan_detection(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """DBSCANクラスタリングによるレジーム検出"""
        try:
            features = self._calculate_features(data)
            if features is None or len(features) == 0:
                return self._create_default_result()

            if self.dbscan_model is None:
                self.dbscan_model = DBSCAN(eps=0.5, min_samples=5)

            # 十分なデータがある場合のみクラスタリング実行
            if len(features) >= 10:
                labels = self.dbscan_model.fit_predict(features)

                # 最新データのクラスタを取得
                latest_cluster = labels[-1]

                # ノイズ点の場合
                if latest_cluster == -1:
                    regime = MarketRegime.VOLATILE
                    confidence = 0.3
                else:
                    # クラスタサイズベースでレジーム判定
                    cluster_size = np.sum(labels == latest_cluster)
                    total_points = len(labels[labels != -1])

                    if cluster_size / total_points > 0.4:
                        regime = MarketRegime.RANGING
                    elif cluster_size / total_points > 0.2:
                        regime = MarketRegime.TRENDING_UP
                    else:
                        regime = MarketRegime.VOLATILE

                    confidence = min(1.0, cluster_size / 20)
            else:
                regime = MarketRegime.RANGING
                confidence = 0.5
                latest_cluster = 0

            # 履歴に追加
            self.feature_history.append(features[-1])

            return RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                indicators={},
                timestamp=datetime.now(),
                method="dbscan",
                cluster=int(latest_cluster),
            )

        except Exception as e:
            logger.error(f"DBSCAN検出エラー: {e}")
            return self._create_default_result()

    def _hmm_detection(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """HMMによるレジーム検出"""
        try:
            returns = data["Close"].pct_change().dropna().iloc[-self.lookback_period :]

            if len(returns) < self.hmm_states * 10:
                return self._create_default_result()

            # HMMモデルの初期化
            if self.hmm_model is None:
                self.hmm_model = hmm.GaussianHMM(
                    n_components=self.hmm_states,
                    covariance_type="full",
                    random_state=42,
                )

            # データを準備
            returns_array = returns.values.reshape(-1, 1)

            # モデル学習
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.hmm_model.fit(returns_array)

            # 状態予測
            states = self.hmm_model.predict(returns_array)
            latest_state = states[-1]

            # 状態からレジームにマッピング
            state_means = self.hmm_model.means_.flatten()

            # covars_ の安全なアクセス（Noneチェックと範囲チェック）
            if (
                self.hmm_model is not None
                and hasattr(self.hmm_model, "covars_")
                and self.hmm_model.covars_ is not None
                and 0 <= latest_state < len(self.hmm_model.covars_)
            ):
                covar_matrix = self.hmm_model.covars_[latest_state]
                if hasattr(covar_matrix, "ndim") and covar_matrix.ndim == 2:
                    state_vars = np.diag(covar_matrix)
                else:
                    state_vars = (
                        np.array([covar_matrix])
                        if hasattr(covar_matrix, "ndim") and covar_matrix.ndim == 0
                        else np.diag(covar_matrix)
                    )
            else:
                # フォールバック: デフォルトのボラティリティを使用
                logger.warning(f"covars_ のアクセスに失敗、latest_state={latest_state}")
                state_vars = np.array([0.02])  # デフォルトボラティリティ

            mean_return = state_means[latest_state]
            volatility = np.sqrt(state_vars[0]) if len(state_vars) > 0 else 0.02

            if volatility > 0.03:
                regime = MarketRegime.VOLATILE
            elif volatility < 0.01:
                regime = MarketRegime.CALM
            elif mean_return > 0.001:
                regime = MarketRegime.TRENDING_UP
            elif mean_return < -0.001:
                regime = MarketRegime.TRENDING_DOWN
            else:
                regime = MarketRegime.RANGING

            # 信頼度計算
            state_probs = self.hmm_model.predict_proba(returns_array)
            confidence = state_probs[-1, latest_state]

            return RegimeDetectionResult(
                regime=regime,
                confidence=confidence,
                indicators={},
                timestamp=datetime.now(),
                method="hmm",
                cluster=int(latest_state),
            )

        except Exception as e:
            logger.error(f"HMM検出エラー: {e}")
            return self._create_default_result()

    def _ensemble_detection(self, data: pd.DataFrame) -> RegimeDetectionResult:
        """アンサンブル手法によるレジーム検出"""
        try:
            features = self._calculate_features(data)
            if features is None or len(features) == 0:
                return self._create_default_result()

            results = []

            # 各手法の結果を取得
            kmeans_result = self._kmeans_detection(data)
            dbscan_result = self._dbscan_detection(data)

            results.append((kmeans_result, 0.4))  # 重み
            results.append((dbscan_result, 0.3))

            # HMMの結果を追加
            hmm_result = self._hmm_detection(data)
            results.append((hmm_result, 0.3))

            # 投票による最終判定
            regime_votes = {}
            total_confidence = 0

            for result, weight in results:
                regime = result.regime.value
                confidence = result.confidence

                if regime not in regime_votes:
                    regime_votes[regime] = 0
                regime_votes[regime] += confidence * weight
                total_confidence += confidence * weight

            # 最高得票のレジームを選択
            final_regime = max(regime_votes.items(), key=lambda x: x[1])[0]
            final_confidence = (
                regime_votes[final_regime] / len(results) if len(results) > 0 else 0.5
            )
            final_confidence = min(1.0, max(0.0, final_confidence))

            return RegimeDetectionResult(
                regime=MarketRegime(final_regime),
                confidence=final_confidence,
                indicators={},
                timestamp=datetime.now(),
                method="ensemble",
                votes=regime_votes,
            )

        except Exception as e:
            logger.error(f"アンサンブル検出エラー: {e}")
            return self._create_default_result()

    def _create_default_result(self) -> RegimeDetectionResult:
        """デフォルト結果を作成"""
        return RegimeDetectionResult(
            regime=MarketRegime.RANGING,
            confidence=0.5,
            indicators={},
            timestamp=datetime.now(),
            method="default",
        )

    def get_regime_stability(self, window: int = 10) -> float:
        """レジームの安定性を計算"""
        if len(self.regime_history) < window:
            return 0.5

        recent_regimes = [r.regime for r in self.regime_history[-window:]]
        most_common = max(set(recent_regimes), key=recent_regimes.count)
        stability = recent_regimes.count(most_common) / len(recent_regimes)

        return stability
