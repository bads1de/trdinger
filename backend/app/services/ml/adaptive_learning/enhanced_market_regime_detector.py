"""
強化された市場レジーム検出器

scikit-learnのクラスタリング（KMeans、DBSCAN）とhmmlearnのHMMを使用した
データ駆動アプローチによる市場レジーム判定を提供します。
"""

import logging
import warnings
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler


try:
    from hmmlearn import hmm

    HMM_AVAILABLE = True
except Exception:
    hmm = None
    HMM_AVAILABLE = False


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


class EnhancedMarketRegimeDetector:
    """
    強化された市場レジーム検出器

    複数のアルゴリズムを使用してデータ駆動な市場レジーム判定を行います。
    """

    def __init__(
        self,
        lookback_period: int = 100,
        detection_method: RegimeDetectionMethod = RegimeDetectionMethod.ENSEMBLE,
        n_clusters: int = 4,
        hmm_states: int = 3,
    ):
        """
        初期化

        Args:
            lookback_period: 分析期間
            detection_method: 検出方法
            n_clusters: クラスタ数（KMeans用）
            hmm_states: HMM状態数
        """
        self.lookback_period = lookback_period
        self.detection_method = detection_method
        self.n_clusters = n_clusters
        self.hmm_states = hmm_states

        # モデルの初期化
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.dbscan_model = None
        self.hmm_model = None

        # 履歴
        self.regime_history: List[Dict] = []
        self.feature_history: List[np.ndarray] = []

        logger.info(f"強化レジーム検出器初期化: method={detection_method.value}")

    def detect_regime(self, data: pd.DataFrame) -> Dict:
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
                return self._create_default_result()

            # 特徴量を計算
            features = self._calculate_features(data)

            if features is None or len(features) == 0:
                return self._create_default_result()

            # 検出方法に応じてレジームを判定
            if self.detection_method == RegimeDetectionMethod.RULE_BASED:
                result = self._rule_based_detection(features, data)
            elif self.detection_method == RegimeDetectionMethod.KMEANS:
                result = self._kmeans_detection(features)
            elif self.detection_method == RegimeDetectionMethod.DBSCAN:
                result = self._dbscan_detection(features)
            elif self.detection_method == RegimeDetectionMethod.HMM:
                result = self._hmm_detection(data)
            elif self.detection_method == RegimeDetectionMethod.ENSEMBLE:
                result = self._ensemble_detection(features, data)
            else:
                result = self._create_default_result()

            # 履歴に追加
            self.regime_history.append(result)
            self.feature_history.append(
                features[-1] if len(features) > 0 else np.array([])
            )

            # 履歴サイズ制限
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-500:]
                self.feature_history = self.feature_history[-500:]

            logger.info(
                f"レジーム検出完了: {result['regime']} (信頼度: {result['confidence']:.2f})"
            )
            return result

        except Exception as e:
            logger.error(f"レジーム検出エラー: {e}")
            return self._create_default_result()

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
            if len(self.feature_history) == 0:
                # 初回は現在のデータで正規化
                features_scaled = self.scaler.fit_transform(features_array)
            else:
                # 既存のスケーラーを使用
                features_scaled = self.scaler.transform(features_array)

            return features_scaled

        except Exception as e:
            logger.error(f"特徴量計算エラー: {e}")
            return None

    def _kmeans_detection(self, features: np.ndarray) -> Dict:
        """KMeansクラスタリングによるレジーム検出"""
        try:
            if self.kmeans_model is None:
                self.kmeans_model = KMeans(
                    n_clusters=self.n_clusters, random_state=42, n_init=10
                )

            # 十分なデータがある場合のみ学習
            if len(features) >= self.n_clusters * 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.kmeans_model.fit(features)

            # 最新の特徴量でクラスタを予測
            cluster = self.kmeans_model.predict([features[-1]])[0]

            # クラスタからレジームにマッピング
            regime_mapping = {
                0: MarketRegime.TRENDING_UP,
                1: MarketRegime.TRENDING_DOWN,
                2: MarketRegime.RANGING,
                3: MarketRegime.VOLATILE,
            }

            regime = regime_mapping.get(cluster, MarketRegime.RANGING)

            # 信頼度計算（クラスタ中心からの距離ベース）
            distances = self.kmeans_model.transform([features[-1]])[0]
            confidence = 1.0 / (1.0 + distances[cluster])

            return {
                "regime": regime.value,
                "confidence": min(1.0, max(0.0, confidence)),
                "method": "kmeans",
                "cluster": int(cluster),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"KMeans検出エラー: {e}")
            return self._create_default_result()

    def _dbscan_detection(self, features: np.ndarray) -> Dict:
        """DBSCANクラスタリングによるレジーム検出"""
        try:
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

            return {
                "regime": regime.value,
                "confidence": confidence,
                "method": "dbscan",
                "cluster": int(latest_cluster),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"DBSCAN検出エラー: {e}")
            return self._create_default_result()

    def _hmm_detection(self, data: pd.DataFrame) -> Dict:
        """HMMによるレジーム検出"""
        if not HMM_AVAILABLE:
            logger.warning("HMMライブラリが利用できません。デフォルト結果を返します。")
            return self._create_default_result()

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
            state_vars = np.diag(self.hmm_model.covars_[latest_state])

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

            return {
                "regime": regime.value,
                "confidence": confidence,
                "method": "hmm",
                "state": int(latest_state),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"HMM検出エラー: {e}")
            return self._create_default_result()

    def _ensemble_detection(self, features: np.ndarray, data: pd.DataFrame) -> Dict:
        """アンサンブル手法によるレジーム検出"""
        try:
            results = []

            # 各手法の結果を取得
            kmeans_result = self._kmeans_detection(features)
            dbscan_result = self._dbscan_detection(features)

            results.append((kmeans_result, 0.4))  # 重み
            results.append((dbscan_result, 0.3))

            # HMMが利用可能な場合は追加
            if HMM_AVAILABLE:
                hmm_result = self._hmm_detection(data)
                results.append((hmm_result, 0.3))
            else:
                # ルールベースをフォールバック
                rule_result = self._rule_based_detection(features, data)
                results.append((rule_result, 0.3))

            # 投票による最終判定
            regime_votes = {}
            total_confidence = 0

            for result, weight in results:
                regime = result["regime"]
                confidence = result["confidence"]

                if regime not in regime_votes:
                    regime_votes[regime] = 0
                regime_votes[regime] += confidence * weight
                total_confidence += confidence * weight

            # 最高得票のレジームを選択
            final_regime = max(regime_votes, key=regime_votes.get)
            final_confidence = (
                regime_votes[final_regime] / len(results) if len(results) > 0 else 0.5
            )

            return {
                "regime": final_regime,
                "confidence": min(1.0, max(0.0, final_confidence)),
                "method": "ensemble",
                "votes": regime_votes,
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"アンサンブル検出エラー: {e}")
            return self._create_default_result()

    def _rule_based_detection(self, features: np.ndarray, data: pd.DataFrame) -> Dict:
        """従来のルールベース検出（フォールバック用）"""
        try:
            if len(features) == 0:
                return self._create_default_result()

            latest_features = features[-1]

            # 簡単なルールベース判定
            volatility = latest_features[0] if len(latest_features) > 0 else 0.02
            trend_strength = latest_features[1] if len(latest_features) > 1 else 0

            if volatility > 0.03:
                regime = MarketRegime.VOLATILE
                confidence = 0.7
            elif volatility < 0.01:
                regime = MarketRegime.CALM
                confidence = 0.7
            elif trend_strength > 0.1:
                regime = MarketRegime.TRENDING_UP
                confidence = 0.6
            elif trend_strength < -0.1:
                regime = MarketRegime.TRENDING_DOWN
                confidence = 0.6
            else:
                regime = MarketRegime.RANGING
                confidence = 0.5

            return {
                "regime": regime.value,
                "confidence": confidence,
                "method": "rule_based",
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"ルールベース検出エラー: {e}")
            return self._create_default_result()

    def _create_default_result(self) -> Dict:
        """デフォルト結果を作成"""
        return {
            "regime": MarketRegime.RANGING.value,
            "confidence": 0.5,
            "method": "default",
            "timestamp": datetime.now(),
        }

    def get_regime_stability(self) -> float:
        """レジーム安定性を取得"""
        if len(self.regime_history) < 5:
            return 1.0

        recent_regimes = [r["regime"] for r in self.regime_history[-10:]]
        unique_regimes = len(set(recent_regimes))

        return max(0.0, 1.0 - (unique_regimes - 1) / 9.0)

    def should_retrain_model(self, stability_threshold: float = 0.7) -> bool:
        """モデル再学習が必要かを判定"""
        if len(self.regime_history) < 5:
            return False

        stability = self.get_regime_stability()
        recent_confidences = [r["confidence"] for r in self.regime_history[-5:]]
        avg_confidence = np.mean(recent_confidences)

        should_retrain = stability < stability_threshold or avg_confidence < 0.6

        if should_retrain:
            logger.info(
                f"モデル再学習推奨: 安定性={stability:.2f}, 平均信頼度={avg_confidence:.2f}"
            )

        return should_retrain
