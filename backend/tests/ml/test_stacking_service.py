import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss
from app.services.ml.stacking_service import StackingService

class TestStackingService:
    def test_ridge_nnls_weighting(self):
        """
        Ridge(NNLS)が単純平均よりも適切な重みを学習できるかテスト
        モデルA: 非常に優秀 (正解に近い)
        モデルB: ランダムに近い
        期待: モデルAの重み >> モデルBの重み となり、Lossが単純平均より低くなること
        """
        np.random.seed(42)
        n_samples = 1000
        
        # 正解ラベル
        y_true = np.random.randint(0, 2, n_samples)
        
        # モデルAの予測 (正解にノイズを少し乗せる -> 高精度)
        # 正解が1なら0.7~0.9, 0なら0.1~0.3
        pred_a = np.where(
            y_true == 1,
            np.random.uniform(0.7, 0.9, n_samples),
            np.random.uniform(0.1, 0.3, n_samples)
        )
        
        # モデルBの予測 (ほぼランダム -> 低精度)
        pred_b = np.random.uniform(0.4, 0.6, n_samples)
        
        # スタッキング用データフレーム
        X_meta = pd.DataFrame({
            'Model_A': pred_a,
            'Model_B': pred_b
        })
        
        service = StackingService()
        
        # 学習
        service.train(X_meta, y_true)
        
        # 重みの確認
        weights = service.get_weights()
        print(f"\nLearned Weights: {weights}")
        
        assert 'Model_A' in weights
        assert 'Model_B' in weights
        assert weights['Model_A'] > weights['Model_B'], "優秀なモデルAの重みが大きくなるべき"
        assert weights['Model_A'] > 0.8, "モデルAが支配的になるべき"
        
        # 予測と評価
        y_pred_stack = service.predict(X_meta)
        y_pred_avg = X_meta.mean(axis=1)
        
        loss_stack = log_loss(y_true, y_pred_stack)
        loss_avg = log_loss(y_true, y_pred_avg)
        
        print(f"Stacking Loss: {loss_stack:.4f}")
        print(f"Average Loss:  {loss_avg:.4f}")
        
        assert loss_stack < loss_avg, "スタッキングは単純平均より改善するべき"

    def test_non_negative_constraint(self):
        """重みが負にならないことを確認 (NNLS)"""
        np.random.seed(42)
        n_samples = 100
        y_true = np.random.randint(0, 2, n_samples)
        
        # 逆相関のモデル（わざと間違えるモデル）を作っても、
        # 通常の回帰なら負の重みをつけるが、NNLSなら0になるはず（または正の重みで調整）
        # ここでは単純にランダムな予測でテスト
        X_meta = pd.DataFrame({
            'M1': np.random.rand(n_samples),
            'M2': np.random.rand(n_samples)
        })
        
        service = StackingService()
        service.train(X_meta, y_true)
        weights = service.get_weights()
        
        for model, w in weights.items():
            assert w >= 0, f"重みは非負であるべき: {model}={w}"

    def test_prediction_shape(self):
        """予測の形状確認"""
        X_meta = pd.DataFrame({
            'A': [0.1, 0.9],
            'B': [0.2, 0.8]
        })
        y_true = np.array([0, 1])
        
        service = StackingService()
        service.train(X_meta, y_true)
        preds = service.predict(X_meta)
        
        assert len(preds) == 2
        assert np.all((preds >= 0) & (preds <= 1))
