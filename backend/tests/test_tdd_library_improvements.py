"""
ライブラリ化改善のTDDテストファイル

市場レジーム判定ロジックと正規化・標準化の手動実装を
ライブラリ化する修正のテスト駆動開発用テストファイル
"""

import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestMarketRegimeDetectorImprovement(unittest.TestCase):
    """市場レジーム判定ロジック改善のテスト"""

    def setUp(self):
        """テストデータの準備"""
        # サンプル価格データを生成
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='1h')
        
        # 異なる市場レジームを模擬したデータ
        base_price = 50000
        
        # トレンド上昇期間
        trend_up = np.cumsum(np.random.normal(0.001, 0.01, 50)) + base_price
        
        # レンジ相場期間
        range_market = np.random.normal(0, 0.005, 50) + trend_up[-1]
        
        # 高ボラティリティ期間
        volatile = np.cumsum(np.random.normal(0, 0.03, 50)) + range_market[-1]
        
        # 低ボラティリティ期間
        calm = np.random.normal(0, 0.002, 50) + volatile[-1]
        
        prices = np.concatenate([trend_up, range_market, volatile, calm])
        
        self.test_data = pd.DataFrame({
            'Open': prices,
            'High': prices * (1 + np.abs(np.random.normal(0, 0.01, 200))),
            'Low': prices * (1 - np.abs(np.random.normal(0, 0.01, 200))),
            'Close': prices,
            'Volume': np.random.uniform(100, 1000, 200)
        }, index=dates)
        
        logger.info(f"テストデータ作成完了: {len(self.test_data)} rows")

    def test_clustering_based_regime_detection(self):
        """クラスタリングベースのレジーム判定テスト"""
        logger.info("クラスタリングベースレジーム判定テスト開始")
        
        # 特徴量計算
        features = self._calculate_regime_features(self.test_data)
        
        # KMeansクラスタリング
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(features)
        
        # DBSCANクラスタリング
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(features)
        
        # 基本的な検証
        self.assertEqual(len(kmeans_labels), len(features))
        self.assertEqual(len(dbscan_labels), len(features))
        
        # クラスタ数の確認
        n_kmeans_clusters = len(np.unique(kmeans_labels))
        n_dbscan_clusters = len(np.unique(dbscan_labels[dbscan_labels != -1]))
        
        self.assertEqual(n_kmeans_clusters, 4)  # KMeansは指定した4クラスタ
        self.assertGreater(n_dbscan_clusters, 0)  # DBSCANは少なくとも1クラスタ
        
        logger.info(f"KMeansクラスタ数: {n_kmeans_clusters}")
        logger.info(f"DBSCANクラスタ数: {n_dbscan_clusters}")
        logger.info("クラスタリングベースレジーム判定テスト成功")

    def test_hmm_based_regime_detection(self):
        """HMMベースのレジーム判定テスト"""
        logger.info("HMMベースレジーム判定テスト開始")
        
        try:
            from hmmlearn import hmm
            
            # 価格変化率を観測データとして使用
            returns = self.test_data['Close'].pct_change().dropna().values.reshape(-1, 1)
            
            # HMMモデル（3状態）
            model = hmm.GaussianHMM(n_components=3, covariance_type="full", random_state=42)
            model.fit(returns)
            
            # 状態予測
            states = model.predict(returns)
            
            # 基本的な検証
            self.assertEqual(len(states), len(returns))
            self.assertGreaterEqual(states.min(), 0)
            self.assertLessEqual(states.max(), 2)
            
            # 状態遷移確率の確認
            self.assertEqual(model.transmat_.shape, (3, 3))
            
            logger.info(f"HMM状態数: {len(np.unique(states))}")
            logger.info("HMMベースレジーム判定テスト成功")
            
        except ImportError:
            logger.warning("hmmlearnライブラリが利用できません。テストをスキップします。")
            self.skipTest("hmmlearn not available")

    def _calculate_regime_features(self, data):
        """レジーム判定用特徴量を計算"""
        close = data['Close']
        returns = close.pct_change().dropna()
        
        features = []
        window = 20
        
        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]
            window_returns = returns.iloc[i-window:i]
            
            # 特徴量計算
            volatility = window_returns.std()
            trend_strength = abs(window_returns.mean()) / volatility if volatility > 0 else 0
            volume_ratio = window_data['Volume'].iloc[-5:].mean() / window_data['Volume'].mean()
            
            features.append([volatility, trend_strength, volume_ratio])
        
        return np.array(features)


class TestNormalizationStandardizationImprovement(unittest.TestCase):
    """正規化・標準化改善のテスト"""

    def setUp(self):
        """テストデータの準備"""
        np.random.seed(42)
        self.test_data = np.random.randn(100) * 10 + 50
        self.test_series = pd.Series(self.test_data)
        
        # 異常値を含むデータ
        self.outlier_data = self.test_data.copy()
        self.outlier_data[10] = 1000  # 異常値
        self.outlier_data[50] = -500  # 異常値
        
        logger.info("正規化・標準化テストデータ準備完了")

    def test_sklearn_minmax_scaler(self):
        """scikit-learn MinMaxScalerのテスト"""
        logger.info("MinMaxScalerテスト開始")
        
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(self.test_data.reshape(-1, 1)).flatten()
        
        # 基本的な検証
        self.assertAlmostEqual(scaled_data.min(), 0.0, places=5)
        self.assertAlmostEqual(scaled_data.max(), 1.0, places=5)
        self.assertEqual(len(scaled_data), len(self.test_data))
        
        # 手動実装との比較
        manual_scaled = self._manual_minmax_normalize(self.test_data)
        np.testing.assert_array_almost_equal(scaled_data, manual_scaled, decimal=5)
        
        logger.info("MinMaxScalerテスト成功")

    def test_sklearn_standard_scaler(self):
        """scikit-learn StandardScalerのテスト"""
        logger.info("StandardScalerテスト開始")
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.test_data.reshape(-1, 1)).flatten()
        
        # 基本的な検証
        self.assertAlmostEqual(scaled_data.mean(), 0.0, places=5)
        self.assertAlmostEqual(scaled_data.std(), 1.0, places=5)
        self.assertEqual(len(scaled_data), len(self.test_data))
        
        # 手動実装との比較
        manual_scaled = self._manual_standard_normalize(self.test_data)
        np.testing.assert_array_almost_equal(scaled_data, manual_scaled, decimal=5)
        
        logger.info("StandardScalerテスト成功")

    def test_sklearn_robust_scaler(self):
        """scikit-learn RobustScalerのテスト"""
        logger.info("RobustScalerテスト開始")
        
        scaler = RobustScaler()
        scaled_data = scaler.fit_transform(self.outlier_data.reshape(-1, 1)).flatten()
        
        # 基本的な検証
        self.assertEqual(len(scaled_data), len(self.outlier_data))
        
        # 中央値が0付近になることを確認
        self.assertAlmostEqual(np.median(scaled_data), 0.0, places=1)
        
        # 外れ値に対する堅牢性を確認
        # RobustScalerは外れ値の影響を受けにくい
        q75, q25 = np.percentile(scaled_data, [75, 25])
        iqr = q75 - q25
        self.assertAlmostEqual(iqr, 1.0, places=1)
        
        logger.info("RobustScalerテスト成功")

    def test_edge_cases(self):
        """エッジケースのテスト"""
        logger.info("エッジケーステスト開始")
        
        # 定数データ
        constant_data = np.array([5.0] * 10)
        
        # MinMaxScaler（定数データ）
        scaler = MinMaxScaler()
        scaled_constant = scaler.fit_transform(constant_data.reshape(-1, 1)).flatten()
        np.testing.assert_array_equal(scaled_constant, np.zeros(10))
        
        # StandardScaler（定数データ）
        scaler = StandardScaler()
        scaled_constant = scaler.fit_transform(constant_data.reshape(-1, 1)).flatten()
        np.testing.assert_array_equal(scaled_constant, np.zeros(10))
        
        logger.info("エッジケーステスト成功")

    def _manual_minmax_normalize(self, data):
        """手動Min-Max正規化（比較用）"""
        min_val = data.min()
        max_val = data.max()
        if max_val == min_val:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)

    def _manual_standard_normalize(self, data):
        """手動標準化（比較用）"""
        mean_val = data.mean()
        std_val = data.std()
        if std_val == 0:
            return np.zeros_like(data)
        return (data - mean_val) / std_val


def run_tests():
    """テスト実行関数"""
    logger.info("ライブラリ化改善TDDテスト開始")
    
    # テストスイート作成
    suite = unittest.TestSuite()
    
    # 市場レジーム判定テスト
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestMarketRegimeDetectorImprovement))
    
    # 正規化・標準化テスト
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestNormalizationStandardizationImprovement))
    
    # テスト実行
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 結果サマリー
    if result.wasSuccessful():
        logger.info("すべてのTDDテストが成功しました！")
        return True
    else:
        logger.error(f"TDDテスト失敗: {len(result.failures)} failures, {len(result.errors)} errors")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
