
import unittest
import pandas as pd
import numpy as np
from backend.app.services.ml.feature_selection.dynamic_meta_selector import DynamicMetaSelector

class TestDynamicMetaSelector(unittest.TestCase):
    def setUp(self):
        self.selector = DynamicMetaSelector(clustering_threshold=0.5)
        
        # テストデータの生成
        np.random.seed(42)
        n_samples = 100
        
        # グループ1: 強い相関がある3つの特徴量
        base1 = np.random.randn(n_samples)
        f1 = base1 + 0.01 * np.random.randn(n_samples)
        f2 = base1 + 0.01 * np.random.randn(n_samples)
        f3 = base1 + 0.01 * np.random.randn(n_samples)
        
        # グループ2: 別の相関グループ
        base2 = np.random.randn(n_samples)
        f4 = base2 + 0.01 * np.random.randn(n_samples)
        f5 = base2 + 0.01 * np.random.randn(n_samples)
        
        # 独立した特徴量
        f6 = np.random.randn(n_samples)
        
        self.X = pd.DataFrame({
            'group1_a': f1, 'group1_b': f2, 'group1_c': f3,
            'group2_a': f4, 'group2_b': f5,
            'independent': f6
        })
        # ターゲット（適当）
        self.y = (f1 + f4 > 0).astype(int)

    def test_cluster_features(self):
        """相関が高い特徴量が正しくグループ化されるかテスト"""
        clusters = self.selector._cluster_features(self.X)
        self.assertGreaterEqual(len(clusters), 3)
        found_group1 = False
        for c_features in clusters.values():
            if 'group1_a' in c_features and 'group1_b' in c_features:
                found_group1 = True
        self.assertTrue(found_group1, "相関の高いgroup1が同じクラスターにまとめられていません")

    def test_fit_process(self):
        """fit メソッド全体の流れをテスト"""
        # 一次モデルの予測確率を模した列を追加
        X_with_proba = self.X.copy()
        X_with_proba['primary_proba'] = np.random.uniform(0, 1, len(self.X))
        
        self.selector.fit(X_with_proba, self.y)
        
        # 選択された特徴量が存在すること
        self.assertIsNotNone(self.selector.selected_features_)
        self.assertGreater(len(self.selector.selected_features_), 0)
        
        # 'primary_proba' が常に含まれていること
        self.assertIn('primary_proba', self.selector.selected_features_)
        
        # マスクの形状が正しいこと
        self.assertEqual(len(self.selector.support_mask_), X_with_proba.shape[1])

if __name__ == '__main__':
    unittest.main()
